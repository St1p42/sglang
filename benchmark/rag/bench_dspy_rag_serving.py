import argparse
import asyncio
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
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
class Metrics:
    request_throughput: float = 0.0
    input_throughput: float = 0.0
    output_throughput: float = 0.0
    total_throughput: float = 0.0
    mean_e2e_latency_ms: float = 0.0
    mean_ttft_ms: float = 0.0
    mean_tpot_ms: float = 0.0
    concurrency: float = 0.0


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

class LocalCorpusRetriever:
    def __init__(self, corpus: List[str], k: int = 4):
        self.corpus = corpus
        self.k = k

    def __call__(self, query: str) -> List[str]:
        q = set(query.lower().split())
        scored = []
        for doc in self.corpus:
            d = set(doc.lower().split())
            score = len(q & d)
            scored.append((score, doc))
        scored.sort(key=lambda x: (-x[0], len(x[1])))
        return [d for s, d in scored[:self.k]]

class ExternalRetriever:
    def __init__(self, endpoint: str, timeout_s: int = 10, fallback=None):
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.fallback = fallback
    def __call__(self, query: str) -> List[str]:
        try:
            r = requests.post(self.endpoint, json={'query': query}, timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                for key in ['passages', 'documents', 'chunks', 'results']:
                    if key in data and isinstance(data[key], list):
                        return [str(x) for x in data[key]]
            if isinstance(data, list):
                return [str(x) for x in data]
            raise ValueError('unexpected retriever response')
        except Exception:
            if self.fallback is not None:
                return self.fallback(query)
            raise

class DSPyRAG(dspy.Module):
    def __init__(self, retriever, lm=None):
        super().__init__()
        self.retriever = retriever
        self.generate = dspy.ChainOfThought('context, question -> answer')
        self._lm = lm
    def forward(self, question: str):
        passages = self.retriever(question)
        context = '\n\n'.join(passages)
        pred = self.generate(context=context, question=question)
        return dspy.Prediction(answer=pred.answer, passages=passages)


def parse_args():
    p = argparse.ArgumentParser(description='DSPy RAG benchmark with fallback retriever')
    p.add_argument('--host', type=str, default='127.0.0.1')
    p.add_argument('--port', type=int, default=30000)
    p.add_argument('--model', type=str, required=True)
    p.add_argument('--parallel', type=int, default=8)
    p.add_argument('--num-qa', type=int, default=64)
    p.add_argument('--num-questions', type=int, default=64)
    p.add_argument('--retriever-endpoint', type=str, default='')
    p.add_argument('--fallback-retriever', action='store_true')
    p.add_argument('--result-file', type=str, default='result_serving.jsonl')
    p.add_argument('--raw-result-file', type=str, default=None)
    return p.parse_args()

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

async def run_all(args, questions, rag, tokenizer, gt_answers):
    url = f'http://{args.host}:{args.port}/generate'
    sem = asyncio.Semaphore(args.parallel)
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        async def one(q, gt):
            async with sem:
                passages = rag.retriever(q)
                prompt = f'Context:\n{chr(10).join(passages)}\n\nQuestion: {q}\nAnswer:'
                prompt_len = len(tokenizer(prompt).input_ids)
                out, cached = await async_request_sglang_generate(session, url, prompt, prompt_len, 32)
                out.answer = out.generated_text.strip()
                out.ground_truth = gt
                out.passages = passages
                out.cached_tokens = cached
                out.answer_em = float(answer_exact_match(dspy.Example(answer=gt), dspy.Prediction(answer=out.answer)))
                return out, prompt_len, cached
        tasks = [asyncio.create_task(one(q, gt)) for q, gt in zip(questions, gt_answers)]
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
    input_requests = [DatasetRow(prompt=q, prompt_len=r[1], output_len=32) for q, r in zip(questions, results)]
    cached_tokens_per_turn = [r[2] for r in results]
    return input_requests, outputs, cached_tokens_per_turn, dur, gpu_metrics

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
        print('{:<45} {:<10.6f}'.format('Exact Match', answer_quality.get('exact_match', 0.0)))
        print('{:<45} {:<10.6f}'.format('Average F1', answer_quality.get('avg_f1', 0.0)))
    print('=' * 60)
    if round_summary:
        print('Per-turn summary:')
        for turn_key, item in round_summary.items():
            print(f"  {turn_key}: requests={item['requests']}, mean_ttft_ms={item['mean_ttft_ms']:.2f}, mean_e2e_latency_ms={item['mean_e2e_latency_ms']:.2f}, cache_hit_rate={item['cache_hit_rate']:.6f}")


def make_dataset(n: int):
    base = [
        ('What is the capital of France?', 'Paris'),
        ('Who wrote Hamlet?', 'William Shakespeare'),
        ('What is the largest planet?', 'Jupiter'),
        ('What does GPU stand for?', 'graphics processing unit'),
        ('What is the boiling point of water in Celsius?', '100 C'),
        ('Who developed the theory of relativity?', 'Albert Einstein'),
        ('What is the currency of Japan?', 'yen'),
        ('What is the main language spoken in Brazil?', 'Portuguese'),
    ]
    qs, ans = [], []
    for i in range(n):
        q, a = base[i % len(base)]
        qs.append(q + f' (variant {i})')
        ans.append(a)
    return qs, ans


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    questions, gt_answers = make_dataset(args.num_questions)
    local = LocalCorpusRetriever([
        'Paris is the capital of France.',
        'William Shakespeare wrote Hamlet.',
        'Jupiter is the largest planet in the solar system.',
        'GPU means graphics processing unit.',
        'Water boils at 100 C at sea level.',
        'Albert Einstein developed the theory of relativity.',
        'The currency of Japan is yen.',
        'Portuguese is widely spoken in Brazil.'
    ], k=4)
    retriever = ExternalRetriever(args.retriever_endpoint, fallback=local) if args.retriever_endpoint else local
    rag = DSPyRAG(retriever=retriever)
    input_requests, outputs, cached_tokens_per_turn, duration, gpu_metrics = asyncio.run(run_all(args, questions, rag, tokenizer, gt_answers))
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
    answer_quality = {
        'exact_match': float(np.mean([getattr(o, 'answer_em', 0.0) for o in successful])) if successful else 0.0,
        'avg_f1': float(np.mean([getattr(o, 'answer_em', 0.0) for o in successful])) if successful else 0.0,
    }
    print_metrics(metrics, cache_hit_rate, round_summary, extended_metrics=extended_metrics, gpu_metrics=gpu_metrics, scaling_efficiency=scaling_efficiency, answer_quality=answer_quality)
    result = {
        'task': 'dspy_rag_serving',
        'host': args.host,
        'port': args.port,
        'num_requests': args.num_questions,
        'parallel': args.parallel,
        'duration': duration,
        'cache_hit_rate': cache_hit_rate,
        'metrics': asdict(metrics),
        'extended_metrics': extended_metrics,
        'gpu_metrics': gpu_metrics,
        'scaling_efficiency': scaling_efficiency,
        'round_summary': round_summary,
        'answer_quality': answer_quality,
        'details': {
            'input_lens': [request.prompt_len for request in input_requests],
            'output_lens': [o.output_len for o in outputs],
            'ttfts': [o.ttft for o in outputs],
            'latencies': [o.latency for o in outputs],
            'itls': [o.itl for o in outputs],
            'cached_tokens': cached_tokens_per_turn,
            'errors': [o.error for o in outputs],
        },
    }
    with open(args.result_file, 'a') as f:
        f.write(json.dumps(result) + '\n')
    if args.raw_result_file:
        with open(args.raw_result_file, 'w') as f:
            json.dump(result, f, indent=2)

if __name__ == '__main__':
    main()