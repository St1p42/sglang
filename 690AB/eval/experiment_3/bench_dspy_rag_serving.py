import argparse
import asyncio
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import requests
from transformers import AutoTokenizer

import dspy
from dspy.evaluate import answer_exact_match

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

try:
    import pynvml
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

# ---------------------------------------------------------------------------
# GPU Monitor (unchanged)
# ---------------------------------------------------------------------------

class GPUMonitor:
    def __init__(self, interval_s: float = 0.1, device_index: int = 0):
        self.interval_s = interval_s
        self.device_index = device_index
        self.samples: List[float] = []
        self._task: Optional[asyncio.Task] = None

    async def _sample_loop(self) -> None:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        try:
            while True:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.samples.append(float(util.gpu))
                await asyncio.sleep(self.interval_s)
        except asyncio.CancelledError:
            pass
        finally:
            pynvml.nvmlShutdown()

    def start(self) -> None:
        self._task = asyncio.create_task(self._sample_loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def get_metrics(self) -> Dict[str, float]:
        if not self.samples:
            return {}
        return {
            'gpu_avg_utilization_pct': float(np.mean(self.samples)),
            'gpu_max_utilization_pct': float(np.max(self.samples)),
            'gpu_samples': len(self.samples),
        }

# ---------------------------------------------------------------------------
# Data / result containers (unchanged)
# ---------------------------------------------------------------------------

@dataclass
class RequestFuncOutput:
    success: bool = False
    error: str = ''
    prompt_len: int = 0
    output_len: int = 0
    ttft: float = 0.0
    latency: float = 0.0
    start_time: float = 0.0
    generated_text: str = ''
    itl: List[float] = None
    def __post_init__(self):
        if self.itl is None:
            self.itl = []

@dataclass
class DatasetRow:
    prompt: str
    prompt_len: int
    output_len: int

@dataclass
class WorkloadRow:
    request_id: str
    group_id: str
    variant_id: int
    prompt: str
    answer: str

@dataclass
class Metrics:
    request_throughput: float = 0.0
    input_throughput: float = 0.0
    output_throughput: float = 0.0
    total_throughput: float = 0.0
    mean_e2e_latency_ms: float = 0.0
    mean_ttft_ms: float = 0.0
    mean_tpot_ms: float = 0.0
    concurrency: float = 0.0

# ---------------------------------------------------------------------------
# Helpers (unchanged)
# ---------------------------------------------------------------------------

def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix):] if text.startswith(prefix) else text


def compute_extended_metrics(outputs: List[RequestFuncOutput]) -> Dict[str, float]:
    successful = [o for o in outputs if o.success]
    if not successful:
        return {}
    extended: Dict[str, float] = {}
    ttfts_ms = np.array([o.ttft for o in successful if o.ttft > 0]) * 1000
    if len(ttfts_ms) > 0:
        extended['p50_ttft_ms'] = float(np.percentile(ttfts_ms, 50))
        extended['p95_ttft_ms'] = float(np.percentile(ttfts_ms, 95))
        extended['p99_ttft_ms'] = float(np.percentile(ttfts_ms, 99))
        extended['std_ttft_ms'] = float(np.std(ttfts_ms))
        extended['max_ttft_ms'] = float(np.max(ttfts_ms))
    lat_ms = np.array([o.latency for o in successful if o.latency > 0]) * 1000
    if len(lat_ms) > 0:
        extended['p50_e2e_latency_ms'] = float(np.percentile(lat_ms, 50))
        extended['p95_e2e_latency_ms'] = float(np.percentile(lat_ms, 95))
        extended['p99_e2e_latency_ms'] = float(np.percentile(lat_ms, 99))
        extended['std_e2e_latency_ms'] = float(np.std(lat_ms))
    return extended


def compute_scaling_efficiency(request_throughput: float, mean_e2e_latency_ms: float, parallel: int) -> float:
    if mean_e2e_latency_ms <= 0 or parallel <= 0:
        return 0.0
    return request_throughput * (mean_e2e_latency_ms / 1000.0) / parallel

# ---------------------------------------------------------------------------
# RAG: Embedding-based local retriever  ← REDESIGNED
# ---------------------------------------------------------------------------

class EmbeddingRetriever:
    """
    Dense retriever using HuggingFace transformers directly — 
    no sentence-transformers dependency. Mean-pools the last hidden 
    state and L2-normalises for cosine similarity.
    """
    def __init__(
        self,
        corpus: List[str],
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        k: int = 4,
        device: str = 'cpu',
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer as AT

        self.corpus = corpus
        self.k = k
        self.device = device
        self._torch = torch
        self._tokenizer = AT.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(device)
        self._model.eval()

        # Encode corpus once at init
        self._corpus_embeddings = self._encode(corpus)  # (N, D) numpy

    def _encode(self, texts: List[str]) -> np.ndarray:
        torch = self._torch
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt',
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            out = self._model(**encoded)
        # Mean pool over token dimension
        mask = encoded['attention_mask'].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()

    def __call__(self, query: str) -> List[str]:
        query_emb = self._encode([query])  # (1, D)
        scores = (self._corpus_embeddings @ query_emb.T).squeeze(axis=1)
        top_k_idx = np.argpartition(scores, -self.k)[-self.k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        return [self.corpus[i] for i in top_k_idx]

# ---------------------------------------------------------------------------
# RAG: External retriever with embedding fallback  ← REDESIGNED
# ---------------------------------------------------------------------------

class ExternalRetriever:
    """
    Calls a teammate's retrieval service (POST /query → {passages: [...]}).
    Falls back to a local EmbeddingRetriever if the endpoint is unreachable
    or returns a non-200 response.

    The fallback is any callable that accepts a query string and returns
    List[str], so it can be an EmbeddingRetriever or anything else.
    """

    def __init__(
        self,
        endpoint: str,
        timeout_s: int = 10,
        fallback: Optional[callable] = None,
    ):
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.fallback = fallback

    def __call__(self, query: str) -> List[str]:
        try:
            r = requests.post(
                self.endpoint,
                json={'query': query},
                timeout=self.timeout_s,
            )
            r.raise_for_status()
            data = r.json()

            # Accept several common response shapes
            if isinstance(data, dict):
                for key in ('passages', 'documents', 'chunks', 'results'):
                    if key in data and isinstance(data[key], list):
                        return [str(x) for x in data[key]]
            if isinstance(data, list):
                return [str(x) for x in data]

            raise ValueError(f'Unexpected retriever response shape: {type(data)}')

        except Exception as exc:
            if self.fallback is not None:
                return self.fallback(query)
            raise RuntimeError(
                f'External retriever failed and no fallback is configured: {exc}'
            ) from exc

# ---------------------------------------------------------------------------
# RAG: DSPy module that wires retriever → generator  ← REDESIGNED
# ---------------------------------------------------------------------------

class DSPyRAG(dspy.Module):
    """
    Proper RAG module:
      1. Retrieve top-k passages for the question via `retriever`.
      2. Pass context + question to a DSPy ChainOfThought generator.

    The generator uses whatever LM is configured on the dspy.settings
    context (set externally before calling forward).  Passing `lm`
    explicitly is supported for testing.
    """

    def __init__(self, retriever: callable, lm=None):
        super().__init__()
        self.retriever = retriever
        # ChainOfThought signature: context + question → answer
        self.generate = dspy.ChainOfThought('context, question -> answer')
        self._lm = lm  # reserved for explicit LM override if needed

    def forward(self, question: str) -> dspy.Prediction:
        passages: List[str] = self.retriever(question)
        context = '\n\n'.join(passages)
        pred = self.generate(context=context, question=question)
        return dspy.Prediction(answer=pred.answer, passages=passages)

# ---------------------------------------------------------------------------
# Arg parsing (unchanged)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='DSPy RAG benchmark with embedding retriever')
    p.add_argument('--backend', type=str, default='sglang', choices=['sglang', 'vllm'])
    p.add_argument('--host', type=str, default='127.0.0.1')
    p.add_argument('--port', type=int, default=None)
    p.add_argument('--model', type=str, required=True)
    p.add_argument('--parallel', type=int, default=8)
    p.add_argument('--num-questions', type=int, default=64,
                   help='Number of synthetic questions to use when --dataset-file is not provided.')
    p.add_argument('--dataset-file', type=str, default='',
                   help='Optional JSONL workload file with fields: request_id, group_id, variant_id, prompt, answer. '
                        'Rows are read in file order and submitted in that order.')
    p.add_argument('--max-requests', type=int, default=None,
                   help='Optional limit on rows read from --dataset-file. By default, all rows are used.')
    p.add_argument('--retriever-endpoint', type=str, default='',
                   help='URL of teammate retrieval service; falls back to local embedding retriever if empty or unreachable.')
    p.add_argument('--embed-model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                   help='sentence-transformers checkpoint for local EmbeddingRetriever.')
    p.add_argument('--embed-device', type=str, default='cuda',
                   choices=['cpu', 'cuda', 'mps'],
                   help='Device for sentence-transformers encoding.')
    p.add_argument('--retriever-k', type=int, default=4,
                   help='Number of passages to retrieve per query.')
    p.add_argument('--result-file', type=str, default='result_serving.jsonl')
    p.add_argument('--raw-result-file', type=str, default=None)
    p.add_argument('--run-id', type=str, default=None)
    args = p.parse_args()
    if args.port is None:
        args.port = 30000 if args.backend == 'sglang' else 21000
    return args


def load_workload_jsonl(path: str, max_requests: Optional[int] = None) -> List[WorkloadRow]:
    """
    Load the slim JSONL workload in file order.

    Expected schema per line:
      {
        "request_id": "doc_00000_q00",
        "group_id": "doc_00000",
        "variant_id": 0,
        "prompt": "Context:\\n...\\n\\nQuestion: ...\\nAnswer:",
        "answer": "..."
      }

    The order of rows in the returned list is exactly the order of lines in the file.
    """
    rows: List[WorkloadRow] = []

    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            missing = [k for k in ('prompt', 'answer') if k not in obj]
            if missing:
                raise ValueError(f'{path}:{line_no} missing required field(s): {missing}')

            # request_id/group_id/variant_id are useful metadata, but default safely
            # so older files with only prompt/answer still run.
            request_id = str(obj.get('request_id', f'req_{len(rows):06d}'))
            group_id = str(obj.get('group_id', 'unknown_group'))
            variant_id = int(obj.get('variant_id', -1))

            rows.append(WorkloadRow(
                request_id=request_id,
                group_id=group_id,
                variant_id=variant_id,
                prompt=str(obj['prompt']),
                answer=str(obj['answer']),
            ))

            if max_requests is not None and len(rows) >= max_requests:
                break

    if not rows:
        raise ValueError(f'No workload rows loaded from {path}')

    return rows


def resolve_dataset_file(path: str) -> str:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return str(candidate)

    script_dir = Path(__file__).resolve().parent
    fallbacks = [
        script_dir / candidate,
        script_dir / 'workloads' / candidate.name,
    ]

    for fallback in fallbacks:
        if fallback.exists():
            return str(fallback)

    searched = ', '.join(str(p) for p in [candidate, *fallbacks])
    raise FileNotFoundError(
        f'Could not find dataset file {path!r}. Tried: {searched}'
    )

# ---------------------------------------------------------------------------
# Async request loop (unchanged)
# ---------------------------------------------------------------------------

async def async_request_sglang_generate(session, url, prompt, prompt_len, output_len):
    payload = {
        'text': prompt,
        'sampling_params': {'temperature': 0.0, 'max_new_tokens': output_len, 'ignore_eos': True},
        'stream': True,
        'stream_options': {'include_usage': True},
        'return_logprob': False,
        'logprob_start_len': -1,
    }
    output = RequestFuncOutput(prompt_len=prompt_len)
    st = time.perf_counter()
    output.start_time = st
    ttft = 0.0
    generated_text = ''
    last_output_len = 0
    most_recent_timestamp = st
    cached_tokens = 0
    try:
        async with session.post(url=url, json=payload) as response:
            if response.status != 200:
                output.error = response.reason or ''
                return output, cached_tokens
            async for chunk_bytes in response.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue
                chunk = remove_prefix(chunk_bytes.decode('utf-8'), 'data: ')
                if chunk == '[DONE]':
                    continue
                data = json.loads(chunk)
                text = data.get('text', '')
                if not text:
                    continue
                timestamp = time.perf_counter()
                generated_text = text
                output.output_len = (data.get('meta_info') or {}).get('completion_tokens', output_len)
                if ttft == 0.0:
                    ttft = timestamp - st
                    output.ttft = ttft
                    cached_tokens = (data.get('meta_info') or {}).get('cached_tokens', 0)
                else:
                    num_new_tokens = output.output_len - last_output_len
                    if num_new_tokens > 0:
                        chunk_gap = timestamp - most_recent_timestamp
                        output.itl.extend([chunk_gap / num_new_tokens] * num_new_tokens)
                most_recent_timestamp = timestamp
                last_output_len = output.output_len
            output.generated_text = generated_text
            output.success = True
            output.latency = time.perf_counter() - st
            return output, cached_tokens
    except Exception as e:
        output.error = str(e)
        return output, cached_tokens


async def async_request_vllm_generate(session, url, prompt, prompt_len, output_len):
    payload = {
        'prompt': prompt,
        'temperature': 0.0,
        'max_tokens': output_len,
        'n': 1,
    }
    output = RequestFuncOutput(prompt_len=prompt_len)
    st = time.perf_counter()
    output.start_time = st
    try:
        async with session.post(url=url, json=payload) as response:
            if response.status != 200:
                output.error = response.reason or ''
                return output, 0
            data = await response.json()
            text = data.get('text', [''])
            generated = text[0][len(prompt):] if text else ''
            output.generated_text = generated
            output.output_len = output_len
            output.success = True
            output.latency = time.perf_counter() - st
            return output, 0
    except Exception as e:
        output.error = str(e)
        return output, 0


async def run_all(args, questions, rag, tokenizer, gt_answers, workload_rows: Optional[List[WorkloadRow]] = None):
    url = f'http://{args.host}:{args.port}/generate'
    sem = asyncio.Semaphore(args.parallel)
    request_fn = async_request_sglang_generate if args.backend == 'sglang' else async_request_vllm_generate

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        async def one_synthetic(i: int, q: str, gt: str):
            async with sem:
                passages = rag.retriever(q)
                prompt = f'Context:\n{chr(10).join(passages)}\n\nQuestion: {q}\nAnswer:'
                prompt_len = len(tokenizer(prompt).input_ids)
                out, cached = await request_fn(session, url, prompt, prompt_len, 32)
                out.answer = out.generated_text.strip()
                out.ground_truth = gt
                out.passages = passages
                out.cached_tokens = cached
                out.request_id = f'synthetic_{i:06d}'
                out.group_id = None
                out.variant_id = None
                out.answer_em = float(answer_exact_match(dspy.Example(answer=gt), dspy.Prediction(answer=out.answer)))
                return out, prompt_len, cached, prompt

        async def one_workload(i: int, row: WorkloadRow):
            async with sem:
                prompt = row.prompt
                prompt_len = len(tokenizer(prompt).input_ids)
                out, cached = await request_fn(session, url, prompt, prompt_len, 32)
                out.answer = out.generated_text.strip()
                out.ground_truth = row.answer
                out.passages = []  # No retriever is used for prebuilt JSONL prompts.
                out.cached_tokens = cached
                out.request_id = row.request_id
                out.group_id = row.group_id
                out.variant_id = row.variant_id
                out.answer_em = float(answer_exact_match(dspy.Example(answer=row.answer), dspy.Prediction(answer=out.answer)))
                return out, prompt_len, cached, prompt

        if workload_rows is not None:
            # Preserve JSONL order when creating tasks. Concurrency is still controlled
            # by the semaphore; SGLang performs its own dynamic scheduling server-side.
            tasks = [
                asyncio.create_task(one_workload(i, row))
                for i, row in enumerate(workload_rows)
            ]
        else:
            tasks = [
                asyncio.create_task(one_synthetic(i, q, gt))
                for i, (q, gt) in enumerate(zip(questions, gt_answers))
            ]

        gpu_monitor = GPUMonitor() if _NVML_AVAILABLE else None
        if gpu_monitor is not None:
            gpu_monitor.start()
        st = time.perf_counter()
        results = await asyncio.gather(*tasks)
        dur = time.perf_counter() - st
        if gpu_monitor is not None:
            await gpu_monitor.stop()
            gpu_metrics = gpu_monitor.get_metrics()
        else:
            gpu_metrics = {}

    outputs = [r[0] for r in results]
    input_requests = [DatasetRow(prompt=r[3], prompt_len=r[1], output_len=32) for r in results]
    cached_tokens_per_turn = [r[2] for r in results]
    return input_requests, outputs, cached_tokens_per_turn, dur, gpu_metrics

# ---------------------------------------------------------------------------
# Metrics / printing (unchanged)
# ---------------------------------------------------------------------------

def summarize_rounds(questions, outputs):
    grouped = defaultdict(list)
    for i, out in enumerate(outputs):
        turn = i % 4
        grouped[turn].append(out)
    round_summary = {}
    for turn, items in sorted(grouped.items()):
        prompt_sum = sum(o.prompt_len for o in items)
        cached_sum = sum(getattr(o, 'cached_tokens', 0) for o in items)
        round_summary[f'turn_{turn}'] = {
            'requests': len(items),
            'mean_ttft_ms': float(np.mean([o.ttft for o in items]) * 1000) if items else 0.0,
            'mean_e2e_latency_ms': float(np.mean([o.latency for o in items]) * 1000) if items else 0.0,
            'cache_hit_rate': 0.0 if prompt_sum == 0 else cached_sum / prompt_sum,
        }
    return round_summary

def compute_answer_quality(outputs) -> Dict[str, float]:
    """
    Computes generation quality metrics and, when passages are available,
    retrieval quality metrics.

    For JSONL workload mode, prompts are prebuilt and no retriever is called,
    so passages are empty and retrieval_recall / mrr are omitted instead of
    being reported as misleading zeros.
    """
    from rouge_score import rouge_scorer as rs

    successful = [o for o in outputs if o.success]
    if not successful:
        return {}

    scorer = rs.RougeScorer(['rougeL'], use_stemmer=True)

    rouge_l_scores, f1_scores, recall_hits, reciprocal_ranks = [], [], [], []

    for o in successful:
        pred = getattr(o, 'answer', '').strip().lower()
        gt = getattr(o, 'ground_truth', '').strip().lower()
        passages = getattr(o, 'passages', []) or []

        # ── ROUGE-L ──────────────────────────────────────────────────────
        rouge_l_scores.append(scorer.score(gt, pred)['rougeL'].fmeasure)

        # ── Token-level F1 ───────────────────────────────────────────────
        pred_tokens = set(pred.split())
        gt_tokens = set(gt.split())
        if pred_tokens and gt_tokens:
            common = pred_tokens & gt_tokens
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(gt_tokens)
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
        else:
            f1 = 0.0
        f1_scores.append(f1)

        # ── Retrieval metrics only when passages exist ───────────────────
        if passages:
            hit = any(gt in p.lower() for p in passages)
            recall_hits.append(float(hit))

            rr = 0.0
            for rank, p in enumerate(passages, start=1):
                if gt in p.lower():
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

    result = {
        'rouge_l': float(np.mean(rouge_l_scores)),
        'token_f1': float(np.mean(f1_scores)),
    }

    if recall_hits:
        result['retrieval_recall'] = float(np.mean(recall_hits))
    if reciprocal_ranks:
        result['mrr'] = float(np.mean(reciprocal_ranks))

    return result

def print_metrics(metrics, cache_hit_rate, round_summary, extended_metrics=None, gpu_metrics=None, scaling_efficiency=0.0, answer_quality=None):
    print('\n{:=^60}'.format(' DSPy RAG Benchmark Result '))
    print('{:<45} {:<10.2f}'.format('Request Throughput (req/s)', metrics.request_throughput))
    print('{:<45} {:<10.2f}'.format('Input Token Throughput (tok/s)', metrics.input_throughput))
    print('{:<45} {:<10.2f}'.format('Output Token Throughput (tok/s)', metrics.output_throughput))
    print('{:<45} {:<10.2f}'.format('Total Token Throughput (tok/s)', metrics.total_throughput))
    print('{:<45} {:<10.2f}'.format('Concurrency', metrics.concurrency))
    print('{:<45} {:<10.6f}'.format('Cache Hit Rate', cache_hit_rate))
    print('{:-^60}'.format(' Latency '))
    print('{:<45} {:<10.2f}'.format('Mean E2E Latency (ms)', metrics.mean_e2e_latency_ms))
    print('{:<45} {:<10.2f}'.format('Mean TTFT (ms)', metrics.mean_ttft_ms))
    print('{:<45} {:<10.2f}'.format('Mean TPOT (ms)', metrics.mean_tpot_ms))
    if extended_metrics:
        print('{:-^60}'.format(' Tail Latency '))
        for label, keys in [('TTFT', ('p50_ttft_ms','p95_ttft_ms','p99_ttft_ms')), ('E2E Latency', ('p50_e2e_latency_ms','p95_e2e_latency_ms','p99_e2e_latency_ms'))]:
            vals = [extended_metrics.get(k) for k in keys]
            if all(v is not None for v in vals):
                print(f'  {label:<20} p50={vals[0]:>10.2f}  p95={vals[1]:>10.2f}  p99={vals[2]:>10.2f}')
        print('{:-^60}'.format(' Fairness '))
        for k, label in [('std_ttft_ms', 'Std TTFT (ms)'), ('max_ttft_ms', 'Max TTFT (ms)'), ('std_e2e_latency_ms', 'Std E2E Latency (ms)')]:
            if k in extended_metrics:
                print('{:<45} {:<10.2f}'.format(label, extended_metrics[k]))
    if gpu_metrics:
        print('{:-^60}'.format(' GPU Utilization '))
        print('{:<45} {:<10.1f}'.format('Avg GPU Utilization (%)', gpu_metrics['gpu_avg_utilization_pct']))
        print('{:<45} {:<10.1f}'.format('Max GPU Utilization (%)', gpu_metrics['gpu_max_utilization_pct']))
        print('{:<45} {:<10d}'.format('GPU Samples', int(gpu_metrics['gpu_samples'])))
    if scaling_efficiency > 0:
        print('{:-^60}'.format(' Scaling '))
        print('{:<45} {:<10.4f}'.format('Scaling Efficiency', scaling_efficiency))
    if answer_quality is not None:
        print('{:-^60}'.format(' Answer Quality '))
        for key, label in [
            ('rouge_l',          'ROUGE-L'),
            ('token_f1',         'Token F1'),
            ('retrieval_recall', 'Retrieval Recall@k'),
            ('mrr',              'MRR'),
        ]:
            if key in answer_quality:
                print('{:<45} {:<10.4f}'.format(label, answer_quality[key]))
    print('=' * 60)
    if round_summary:
        print('Per-turn summary:')
        for turn_key, item in round_summary.items():
            print(f"  {turn_key}: requests={item['requests']}, mean_ttft_ms={item['mean_ttft_ms']:.2f}, mean_e2e_latency_ms={item['mean_e2e_latency_ms']:.2f}, cache_hit_rate={item['cache_hit_rate']:.6f}")

# ---------------------------------------------------------------------------
# Dataset (unchanged)
# ---------------------------------------------------------------------------

def make_dataset(n: int):
    base = [
        ("Where does Khoa study and when does he graduate?", "UMass Amherst, May 2027"),
        ("What did Khoa build at FPT Software?", "RAG and GraphRAG systems"),
        ("What lab did Khoa do undergraduate research in?", "PLASMA Lab"),
        ("What NeurIPS 2024 paper is Khoa reimplementing?", "SGLang RadixAttention and cache-aware scheduling"),
        ("What performance gains did Khoa achieve on his KV cache project?", "38% latency reduction, 61% throughput gain"),
        ("What does Khoa's agentic search pipeline use?", "React, Claude API, parallel scraping"),
        ("What is Khoa's independent study about?", "benchmarking deep research LLM agents"),
        ("How many records did Khoa process in his QA pipeline?", "100k+"),
        ("What did Jake Murphy score on the SAT?", "1540"),
        ("Where was Jake Murphy admitted after his SAT score?", "Carnegie Mellon University"),
        ("What championship did Sara Okonkwo win in 2024?", "regional robotics championship at Georgia Tech"),
        ("What fellowship funds Sara Okonkwo's PhD?", "NSF fellowship"),
        ("What did Derek Liang build at his trading firm internship?", "latency-optimized order book"),
        ("Where did Derek Liang publish a paper as an undergraduate?", "ICLR 2025"),
        ("Where did Priya Nambiar transfer to in 2022?", "UC Berkeley"),
        ("What is Priya Nambiar researching?", "federated learning for medical imaging"),
        ("What did Marcus Delgado build for his senior capstone?", "a compiler for a subset of Rust"),
        ("What contest did Aisha Thornton place second in?", "ACM ICPC North America regional 2024"),
        ("What is Aisha Thornton's thesis topic?", "Byzantine fault-tolerant consensus protocols"),
        ("Where did Lena Fischer complete her double degree?", "ETH Zurich"),
        ("What does Omar Yusuf's fraud detection system process per day?", "2 million transactions"),
        ("What did Chloe Park win in 2023?", "Kaggle grandmaster title"),
        ("What did Theo Bergmann design for his master thesis?", "a custom RISC-V processor"),
        ("What is Mei Watanabe co-authoring a paper on?", "near-memory computing"),
        ("What did Jordan Ellis research?", "energy-efficient neural network inference on embedded devices"),
        ("What benchmark is Fatima Al-Rashid building?", "multilingual NLP benchmark covering 40 low-resource languages"),
        ("How much did Ravi Subramaniam's startup raise?", "1.2 million dollars"),
        ("What medal did Nina Kozlov win at the IMO?", "silver medal in 2021"),
        ("What did Ben Oduya build and open-source?", "a distributed key-value store"),
        ("What is Isla Mackenzie studying at Edinburgh?", "Human-Computer Interaction"),
        ("What did Carlos Vega implement at his game studio internship?", "real-time ray tracing pipeline"),
        ("What is Yuki Tanaka researching?", "generative AI and creative writing tools"),
        ("What app did Amara Osei build?", "a mobile app for offline health records"),
        ("How many patients use Amara Osei's app?", "3000 patients in rural Ghana"),
        ("What is Leo Hartmann learning?", "formal verification methods for safety-critical systems"),
        ("What did Sofia Reyes defer to work on?", "her startup"),
        ("How many students has Sofia Reyes's tutoring platform served?", "over 10000 students"),
        ("What award did Theo Bergmann receive?", "best thesis award from the computer architecture group"),
        ("Where did Ravi Subramaniam previously work?", "recommendation systems at a major e-commerce platform"),
        ("What did Omar Yusuf complete to get his certifications?", "three AWS certifications"),
        ("What is Chloe Park's current role?", "data scientist at a pharmaceutical company"),
        ("What conference is Mei Watanabe targeting for her paper?", "MICRO 2025"),
        ("What did Jordan Ellis do while completing his PhD?", "ran the Boston Marathon in 3 hours 12 minutes"),
        ("What did Fatima Al-Rashid score on the GRE?", "99th percentile"),
        ("What did Ben Oduya optimize during his internship?", "query planning for analytical workloads"),
        ("What did Isla Mackenzie do during her gap year?", "taught coding to high school students in rural Scotland"),
        ("What did Carlos Vega study?", "competitive programming at his university"),
        ("What did Yuki Tanaka publish alongside her CS degree?", "a short story collection"),
        ("What scholarship did Amara Osei receive?", "Google Generation scholarship"),
        ("What sport did Leo Hartmann complete while working full time?", "Ironman triathlon"),
    ]
    qs, ans = [], []
    for i in range(n):
        q, a = base[i % len(base)]
        qs.append(q)
        ans.append(a)
    return qs, ans

# ---------------------------------------------------------------------------
# Corpus (unchanged content, referenced by EmbeddingRetriever now)
# ---------------------------------------------------------------------------

LOCAL_CORPUS = [
    'Khoa is a first-year MS CS student at UMass Amherst expected to graduate in May 2027.',
    'Khoa interned as an ML Engineer at FPT Software building RAG and GraphRAG systems.',
    'Khoa worked at PLASMA Lab profiling inference latency on edge hardware.',
    'Khoa is reimplementing RadixAttention and cache-aware scheduling from SGLang NeurIPS 2024.',
    'Khoa achieved 38% latency reduction and 61% throughput gain on L4 GPUs with his KV cache project.',
    'Khoa built a multi-stage agentic search pipeline using React, Claude API, and parallel scraping.',
    'Khoa is benchmarking deep research LLM agents with GPU-accelerated evaluation as an independent study.',
    'Khoa processed 100k+ records in his multi-document QA pipeline with 70-85% speedup via LRU caching.',
    'Jake Murphy scored 1540 on the SAT in 2023 and was admitted to Carnegie Mellon University.',
    'Jake Murphy is majoring in Electrical Engineering with a minor in Music Technology.',
    'Sara Okonkwo won the 2024 regional robotics championship with her team at Georgia Tech.',
    'Sara Okonkwo is completing a PhD in Human-Robot Interaction funded by an NSF fellowship.',
    'Derek Liang interned at a quantitative trading firm in Chicago and built a latency-optimized order book.',
    'Derek Liang published a paper on transformer pruning at ICLR 2025 as an undergraduate.',
    'Priya Nambiar transferred from community college to UC Berkeley with a 4.0 GPA in 2022.',
    'Priya Nambiar is researching federated learning for medical imaging at the UCSF partnership lab.',
    'Marcus Delgado built a compiler for a subset of Rust as his senior capstone at UT Austin.',
    'Marcus Delgado received a return offer from a systems software team after his internship at a cloud provider.',
    'Aisha Thornton placed second in the 2024 ACM ICPC North America regional contest.',
    'Aisha Thornton is writing her thesis on Byzantine fault-tolerant consensus protocols.',
    'Lena Fischer completed a double degree in Mathematics and Computer Science at ETH Zurich in 2024.',
    'Lena Fischer joined a Berlin-based ML infrastructure startup as a founding engineer.',
    'Omar Yusuf deployed a real-time fraud detection system processing 2 million transactions per day.',
    'Omar Yusuf holds three AWS certifications and leads cloud architecture at a fintech company.',
    'Chloe Park won a Kaggle grandmaster title in 2023 specializing in tabular data competitions.',
    'Chloe Park is currently a data scientist at a pharmaceutical company working on drug discovery pipelines.',
    'Theo Bergmann designed a custom RISC-V processor as part of his master thesis at TU Munich.',
    'Theo Bergmann received the best thesis award from the computer architecture group in 2024.',
    'Mei Watanabe interned at a semiconductor company and taped out a custom memory controller.',
    'Mei Watanabe is co-authoring a paper on near-memory computing to be submitted to MICRO 2025.',
    'Jordan Ellis ran the Boston Marathon in 3 hours 12 minutes while completing a PhD at MIT.',
    'Jordan Ellis researches energy-efficient neural network inference on embedded devices.',
    'Fatima Al-Rashid scored in the 99th percentile on the GRE and received a full scholarship to Stanford.',
    'Fatima Al-Rashid is building a multilingual NLP benchmark covering 40 low-resource languages.',
    'Ravi Subramaniam cofounded a startup that raised 1.2 million dollars in pre-seed funding in 2024.',
    'Ravi Subramaniam previously worked on recommendation systems at a major e-commerce platform.',
    'Nina Kozlov won a silver medal at the International Mathematical Olympiad in 2021.',
    'Nina Kozlov is pursuing a PhD in theoretical computer science focusing on approximation algorithms.',
    'Ben Oduya built a distributed key-value store from scratch as a side project and open-sourced it.',
    'Ben Oduya interned at a database company and optimized query planning for analytical workloads.',
    'Isla Mackenzie completed a gap year teaching coding to high school students in rural Scotland.',
    'Isla Mackenzie is now studying Human-Computer Interaction at the University of Edinburgh.',
    'Carlos Vega holds the record for fastest solve of a custom competitive programming problem set at his university.',
    'Carlos Vega interned at a video game studio implementing a real-time ray tracing pipeline.',
    'Yuki Tanaka published a short story collection while simultaneously earning a BS in Computer Science.',
    'Yuki Tanaka is researching the intersection of generative AI and creative writing tools.',
    'Amara Osei received a Google Generation scholarship and is studying software engineering in Accra.',
    'Amara Osei built a mobile app for offline health records used by 3000 patients in rural Ghana.',
    'Leo Hartmann completed an Ironman triathlon while working full time as a backend engineer.',
    'Leo Hartmann is learning formal verification methods to apply to safety-critical systems.',
    'Sofia Reyes graduated valedictorian from her high school and deferred her MIT admission to work on her startup.',
    'Sofia Reyes is building an AI tutoring platform that has served over 10000 students.',
]

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)

    workload_rows: Optional[List[WorkloadRow]] = None

    if args.dataset_file:
        dataset_file = resolve_dataset_file(args.dataset_file)
        workload_rows = load_workload_jsonl(dataset_file, max_requests=args.max_requests)
        questions = [row.request_id for row in workload_rows]  # used only for summaries/alignment
        gt_answers = [row.answer for row in workload_rows]
        rag = None
        retriever_info = {
            'type': 'prebuilt_jsonl_prompts',
            'dataset_file': dataset_file,
            'max_requests': args.max_requests,
        }
        print(f'[dataset] Loaded {len(workload_rows)} prebuilt prompts from {dataset_file}')
        print('[dataset] JSONL row order is preserved when creating async tasks.')
    else:
        questions, gt_answers = make_dataset(args.num_questions)

        print(f'[retriever] Loading embedding model: {args.embed_model} on {args.embed_device}')
        local_retriever = EmbeddingRetriever(
            corpus=LOCAL_CORPUS,
            model_name=args.embed_model,
            k=args.retriever_k,
            device=args.embed_device,
        )

        # Use external retriever if endpoint given, with local embedding fallback;
        # otherwise use the local embedding retriever directly.
        if args.retriever_endpoint:
            print(f'[retriever] Using external endpoint: {args.retriever_endpoint} (fallback: local embedding)')
            retriever = ExternalRetriever(
                endpoint=args.retriever_endpoint,
                timeout_s=10,
                fallback=local_retriever,
            )
        else:
            print('[retriever] No external endpoint — using local EmbeddingRetriever')
            retriever = local_retriever

        rag = DSPyRAG(retriever=retriever)
        retriever_info = {
            'type': 'external+embedding_fallback' if args.retriever_endpoint else 'embedding_local',
            'embed_model': args.embed_model,
            'embed_device': args.embed_device,
            'k': args.retriever_k,
            'endpoint': args.retriever_endpoint or None,
        }

    input_requests, outputs, cached_tokens_per_turn, duration, gpu_metrics = asyncio.run(
        run_all(args, questions, rag, tokenizer, gt_answers, workload_rows=workload_rows)
    )

    # ---- aggregate metrics (unchanged) ------------------------------------
    metrics = Metrics()
    successful = [o for o in outputs if o.success]
    metrics.request_throughput = len(successful) / duration if duration > 0 else 0.0
    metrics.input_throughput = sum(o.prompt_len for o in successful) / duration if duration > 0 else 0.0
    metrics.output_throughput = sum(o.output_len for o in successful) / duration if duration > 0 else 0.0
    metrics.total_throughput = metrics.input_throughput + metrics.output_throughput
    metrics.mean_e2e_latency_ms = float(np.mean([o.latency for o in successful]) * 1000) if successful else 0.0
    metrics.mean_ttft_ms = float(np.mean([o.ttft for o in successful]) * 1000) if successful else 0.0
    metrics.mean_tpot_ms = float(np.mean([np.mean(o.itl) for o in successful if o.itl]) * 1000) if any(o.itl for o in successful) else 0.0
    metrics.concurrency = float(args.parallel)

    total_prompt_tokens = sum(o.prompt_len for o in successful if o.prompt_len > 0)
    cache_hit_rate = 0.0 if total_prompt_tokens == 0 else sum(cached_tokens_per_turn) / total_prompt_tokens
    round_summary = summarize_rounds(questions, outputs)
    extended_metrics = compute_extended_metrics(outputs)
    scaling_efficiency = compute_scaling_efficiency(metrics.request_throughput, metrics.mean_e2e_latency_ms, args.parallel)
    answer_quality = compute_answer_quality(outputs)

    print_metrics(
        metrics, cache_hit_rate, round_summary,
        extended_metrics=extended_metrics,
        gpu_metrics=gpu_metrics,
        scaling_efficiency=scaling_efficiency,
        answer_quality=answer_quality,
    )

    result = {
        'task': 'dspy_rag_serving',
        'backend': args.backend,
        'num_gpus': 1,
        'run_id': args.run_id,
        'host': args.host,
        'port': args.port,
        'num_requests': len(input_requests),
        'parallel': args.parallel,
        'duration': duration,
        'cache_hit_rate': cache_hit_rate,
        'metrics': asdict(metrics),
        'extended_metrics': extended_metrics,
        'gpu_metrics': gpu_metrics,
        'scaling_efficiency': scaling_efficiency,
        'round_summary': round_summary,
        'answer_quality': answer_quality,
        'retriever': retriever_info,
        'details': {
            'input_lens':     [request.prompt_len for request in input_requests],
            'output_lens':    [o.output_len for o in outputs],
            'ttfts':          [o.ttft for o in outputs],
            'latencies':      [o.latency for o in outputs],
            'itls':           [o.itl for o in outputs],
            'cached_tokens':  cached_tokens_per_turn,
            'success':        [o.success for o in outputs],
            'errors':         [o.error for o in outputs],
            'request_ids':    [getattr(o, 'request_id', None) for o in outputs],
            'group_ids':      [getattr(o, 'group_id', None) for o in outputs],
            'variant_ids':    [getattr(o, 'variant_id', None) for o in outputs],
        },
    }

    with open(args.result_file, 'a') as f:
        f.write(json.dumps(result) + '\n')
    if args.raw_result_file:
        with open(args.raw_result_file, 'w') as f:
            json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
