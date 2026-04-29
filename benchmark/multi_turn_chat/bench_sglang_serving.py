import argparse
import asyncio
import json
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np
from data_gen import gen_arguments
from transformers import AutoTokenizer

from sglang.bench_serving import (
    DatasetRow,
    RequestFuncOutput,
    calculate_metrics,
    remove_prefix,
)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

# ---------------------------------------------------------------------------
# GPU utilization monitoring (gracefully skipped when NVML is unavailable)
# ---------------------------------------------------------------------------

_NVML_AVAILABLE = False
try:
    # nvidia-ml-py is the official NVIDIA-maintained package (pip install nvidia-ml-py).
    # It exposes itself as "pynvml" at import time.  The older gpuopenanalytics/pynvml
    # package is deprecated and should not be used.
    import pynvml  # provided by nvidia-ml-py

    _NVML_AVAILABLE = True
except ImportError:
    pass


class GPUMonitor:
    """Async background sampler for GPU utilization via NVML."""

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
            "gpu_avg_utilization_pct": float(np.mean(self.samples)),
            "gpu_max_utilization_pct": float(np.max(self.samples)),
            "gpu_samples": len(self.samples),
        }


# ---------------------------------------------------------------------------
# Extended metrics helpers
# ---------------------------------------------------------------------------


def compute_extended_metrics(outputs: List[RequestFuncOutput]) -> Dict[str, float]:
    """Compute tail-latency and fairness metrics from raw request outputs."""
    successful = [o for o in outputs if o.success]
    if not successful:
        return {}

    extended: Dict[str, float] = {}

    ttfts_ms = np.array([o.ttft for o in successful if o.ttft > 0]) * 1000
    if len(ttfts_ms) > 0:
        extended["p50_ttft_ms"] = float(np.percentile(ttfts_ms, 50))
        extended["p95_ttft_ms"] = float(np.percentile(ttfts_ms, 95))
        extended["p99_ttft_ms"] = float(np.percentile(ttfts_ms, 99))
        extended["std_ttft_ms"] = float(np.std(ttfts_ms))
        extended["max_ttft_ms"] = float(np.max(ttfts_ms))

    lat_ms = np.array([o.latency for o in successful if o.latency > 0]) * 1000
    if len(lat_ms) > 0:
        extended["p50_e2e_latency_ms"] = float(np.percentile(lat_ms, 50))
        extended["p95_e2e_latency_ms"] = float(np.percentile(lat_ms, 95))
        extended["p99_e2e_latency_ms"] = float(np.percentile(lat_ms, 99))
        extended["std_e2e_latency_ms"] = float(np.std(lat_ms))

    return extended


def compute_scaling_efficiency(
    request_throughput: float,
    mean_e2e_latency_ms: float,
    parallel: int,
) -> float:
    """Estimate throughput scaling efficiency via Little's law.

    efficiency = (throughput * mean_latency) / parallelism
    A value of 1.0 means perfect linear scaling; <1.0 indicates contention.
    """
    if mean_e2e_latency_ms <= 0 or parallel <= 0:
        return 0.0
    return request_throughput * (mean_e2e_latency_ms / 1000.0) / parallel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark SGLang multi-turn chat with serving-style metrics."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--num-qa", type=int, default=20)
    parser.add_argument("--parallel", type=int, default=64)
    parser.add_argument("--min-len-q", type=int, default=256)
    parser.add_argument("--max-len-q", type=int, default=512)
    parser.add_argument("--min-len-a", type=int, default=4)
    parser.add_argument("--max-len-a", type=int, default=8)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-ignore-eos", action="store_true")
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--result-file", type=str, default="result_serving.jsonl")
    parser.add_argument("--raw-result-file", type=str, default=None)
    return parser.parse_args()


async def async_request_sglang_generate(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    disable_ignore_eos: bool,
) -> Tuple[RequestFuncOutput, int]:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": not disable_ignore_eos,
        },
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_logprob": False,
        "logprob_start_len": -1,
    }

    output = RequestFuncOutput()
    output.prompt_len = prompt_len
    generated_text = ""
    ttft = 0.0
    cached_tokens = 0
    prompt_tokens = prompt_len
    st = time.perf_counter()
    output.start_time = st
    most_recent_timestamp = st
    last_output_len = 0

    try:
        async with session.post(url=url, json=payload) as response:
            if response.status != 200:
                output.success = False
                output.error = response.reason or ""
                return output, cached_tokens

            async for chunk_bytes in response.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue

                chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                latency = time.perf_counter() - st
                if chunk == "[DONE]":
                    continue

                data = json.loads(chunk)
                text = data.get("text", "")
                if not text:
                    continue

                timestamp = time.perf_counter()
                generated_text = text
                output.output_len = (data.get("meta_info") or {}).get(
                    "completion_tokens", output_len
                )

                if ttft == 0.0:
                    ttft = timestamp - st
                    output.ttft = ttft
                    prompt_tokens = (data.get("meta_info") or {}).get(
                        "prompt_tokens", prompt_len
                    )
                    cached_tokens = (data.get("meta_info") or {}).get(
                        "cached_tokens", 0
                    )
                else:
                    num_new_tokens = output.output_len - last_output_len
                    if num_new_tokens > 0:
                        chunk_gap = timestamp - most_recent_timestamp
                        output.itl.extend([chunk_gap / num_new_tokens] * num_new_tokens)

                most_recent_timestamp = timestamp
                last_output_len = output.output_len

            output.generated_text = generated_text
            output.success = True
            output.latency = latency
            output.prompt_len = prompt_tokens
            return output, cached_tokens
    except Exception as e:
        output.success = False
        output.error = str(e)
        return output, cached_tokens


async def run_one_conversation(
    session: aiohttp.ClientSession,
    url: str,
    conv_id: int,
    qas: List[Dict[str, int]],
    tokenizer,
    disable_ignore_eos: bool,
):
    history = ""
    outputs: List[RequestFuncOutput] = []
    input_requests: List[DatasetRow] = []
    cached_tokens_per_turn: List[int] = []
    round_metrics: List[Dict[str, float]] = []

    for turn_idx, qa in enumerate(qas):
        history += qa["prompt"]
        prompt_len = len(tokenizer(history).input_ids)
        input_requests.append(
            DatasetRow(
                prompt=history,
                prompt_len=prompt_len,
                output_len=qa["new_tokens"],
            )
        )
        output, cached_tokens = await async_request_sglang_generate(
            session=session,
            url=url,
            prompt=history,
            prompt_len=prompt_len,
            output_len=qa["new_tokens"],
            disable_ignore_eos=disable_ignore_eos,
        )
        outputs.append(output)
        cached_tokens_per_turn.append(cached_tokens)
        round_metrics.append(
            {
                "conversation_id": conv_id,
                "turn": turn_idx,
                "ttft": output.ttft,
                "latency": output.latency,
                "output_len": output.output_len,
                "prompt_len": output.prompt_len,
                "cached_tokens": cached_tokens,
                "success": output.success,
            }
        )
        if not output.success:
            break
        history += output.generated_text

    return input_requests, outputs, cached_tokens_per_turn, round_metrics


async def run_all(args, multi_qas, tokenizer):
    url = f"http://{args.host}:{args.port}/generate"
    semaphore = asyncio.Semaphore(args.parallel)

    gpu_monitor: Optional[GPUMonitor] = None
    if _NVML_AVAILABLE:
        gpu_monitor = GPUMonitor(interval_s=0.1)

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        async def run_with_limit(conv_id, qas):
            async with semaphore:
                return await run_one_conversation(
                    session=session,
                    url=url,
                    conv_id=conv_id,
                    qas=qas,
                    tokenizer=tokenizer,
                    disable_ignore_eos=args.disable_ignore_eos,
                )

        tasks = [
            asyncio.create_task(run_with_limit(i, item["qas"]))
            for i, item in enumerate(multi_qas)
        ]
        if gpu_monitor is not None:
            gpu_monitor.start()
        benchmark_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - benchmark_start
        if gpu_monitor is not None:
            await gpu_monitor.stop()

    gpu_metrics = gpu_monitor.get_metrics() if gpu_monitor is not None else {}

    input_requests: List[DatasetRow] = []
    outputs: List[RequestFuncOutput] = []
    cached_tokens_per_turn: List[int] = []
    raw_round_metrics: List[Dict[str, float]] = []
    for reqs, conv_outputs, conv_cached_tokens, conv_round_metrics in results:
        input_requests.extend(reqs)
        outputs.extend(conv_outputs)
        cached_tokens_per_turn.extend(conv_cached_tokens)
        raw_round_metrics.extend(conv_round_metrics)

    return (
        input_requests,
        outputs,
        cached_tokens_per_turn,
        raw_round_metrics,
        duration,
        gpu_metrics,
    )


def summarize_rounds(raw_round_metrics: List[Dict[str, float]]):
    grouped = defaultdict(list)
    for item in raw_round_metrics:
        if item["success"]:
            grouped[item["turn"]].append(item)

    round_summary = {}
    for turn, items in sorted(grouped.items()):
        prompt_sum = sum(item["prompt_len"] for item in items)
        cached_sum = sum(item["cached_tokens"] for item in items)
        round_summary[f"turn_{turn}"] = {
            "requests": len(items),
            "mean_ttft_ms": float(np.mean([item["ttft"] for item in items]) * 1000),
            "mean_e2e_latency_ms": float(
                np.mean([item["latency"] for item in items]) * 1000
            ),
            "cache_hit_rate": 0.0 if prompt_sum == 0 else cached_sum / prompt_sum,
        }
    return round_summary


def print_metrics(
    metrics,
    cache_hit_rate,
    round_summary,
    extended_metrics: Optional[Dict[str, float]] = None,
    gpu_metrics: Optional[Dict[str, float]] = None,
    scaling_efficiency: float = 0.0,
):
    print("\n{:=^60}".format(" Multi-turn Serving Benchmark Result "))

    print("{:<45} {:<10.2f}".format("Request Throughput (req/s)", metrics.request_throughput))
    print("{:<45} {:<10.2f}".format("Input Token Throughput (tok/s)", metrics.input_throughput))
    print("{:<45} {:<10.2f}".format("Output Token Throughput (tok/s)", metrics.output_throughput))
    print("{:<45} {:<10.2f}".format("Total Token Throughput (tok/s)", metrics.total_throughput))
    print("{:<45} {:<10.2f}".format("Concurrency", metrics.concurrency))
    print("{:<45} {:<10.6f}".format("Cache Hit Rate", cache_hit_rate))

    print("{:-^60}".format(" Latency "))
    print("{:<45} {:<10.2f}".format("Mean E2E Latency (ms)", metrics.mean_e2e_latency_ms))
    print("{:<45} {:<10.2f}".format("Mean TTFT (ms)", metrics.mean_ttft_ms))
    print("{:<45} {:<10.2f}".format("Mean TPOT (ms)", metrics.mean_tpot_ms))

    if extended_metrics:
        print("{:-^60}".format(" Tail Latency "))
        for label, keys in [
            ("TTFT", ("p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms")),
            ("E2E Latency", ("p50_e2e_latency_ms", "p95_e2e_latency_ms", "p99_e2e_latency_ms")),
        ]:
            vals = [extended_metrics.get(k) for k in keys]
            if all(v is not None for v in vals):
                print(
                    f"  {label:<20} p50={vals[0]:>10.2f}  "
                    f"p95={vals[1]:>10.2f}  p99={vals[2]:>10.2f}"
                )

        print("{:-^60}".format(" Fairness "))
        if "std_ttft_ms" in extended_metrics:
            print("{:<45} {:<10.2f}".format("Std TTFT (ms)", extended_metrics["std_ttft_ms"]))
        if "max_ttft_ms" in extended_metrics:
            print("{:<45} {:<10.2f}".format("Max TTFT (ms)", extended_metrics["max_ttft_ms"]))
        if "std_e2e_latency_ms" in extended_metrics:
            print("{:<45} {:<10.2f}".format("Std E2E Latency (ms)", extended_metrics["std_e2e_latency_ms"]))

    if gpu_metrics:
        print("{:-^60}".format(" GPU Utilization "))
        print("{:<45} {:<10.1f}".format(
            "Avg GPU Utilization (%)", gpu_metrics["gpu_avg_utilization_pct"]
        ))
        print("{:<45} {:<10.1f}".format(
            "Max GPU Utilization (%)", gpu_metrics["gpu_max_utilization_pct"]
        ))
        print("{:<45} {:<10d}".format(
            "GPU Samples", int(gpu_metrics["gpu_samples"])
        ))

    if scaling_efficiency > 0:
        print("{:-^60}".format(" Scaling "))
        print("{:<45} {:<10.4f}".format("Scaling Efficiency", scaling_efficiency))

    print("=" * 60)

    if round_summary:
        print("Per-turn summary:")
        for turn_key, item in round_summary.items():
            print(
                f"  {turn_key}: requests={item['requests']}, "
                f"mean_ttft_ms={item['mean_ttft_ms']:.2f}, "
                f"mean_e2e_latency_ms={item['mean_e2e_latency_ms']:.2f}, "
                f"cache_hit_rate={item['cache_hit_rate']:.6f}"
            )


def main():
    args = parse_args()
    if args.long:
        args.min_len_a = 256
        args.max_len_a = 512
        args.num_qa = 20

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    multi_qas = gen_arguments(args, tokenizer)

    (
        input_requests,
        outputs,
        cached_tokens_per_turn,
        raw_round_metrics,
        duration,
        gpu_metrics,
    ) = asyncio.run(run_all(args, multi_qas, tokenizer))

    metrics, output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=duration,
        tokenizer=tokenizer,
        backend="sglang",
    )

    total_prompt_tokens = sum(
        output.prompt_len for output in outputs if output.success and output.prompt_len > 0
    )
    cache_hit_rate = (
        0.0 if total_prompt_tokens == 0 else sum(cached_tokens_per_turn) / total_prompt_tokens
    )
    round_summary = summarize_rounds(raw_round_metrics)

    extended_metrics = compute_extended_metrics(outputs)
    scaling_efficiency = compute_scaling_efficiency(
        metrics.request_throughput, metrics.mean_e2e_latency_ms, args.parallel
    )

    print_metrics(
        metrics,
        cache_hit_rate,
        round_summary,
        extended_metrics=extended_metrics,
        gpu_metrics=gpu_metrics,
        scaling_efficiency=scaling_efficiency,
    )

    result = {
        "task": "multi_turn_chat_serving",
        "host": args.host,
        "port": args.port,
        "num_requests": args.num_qa,
        "num_turns": args.turns,
        "parallel": args.parallel,
        "duration": duration,
        "cache_hit_rate": cache_hit_rate,
        "metrics": asdict(metrics),
        "extended_metrics": extended_metrics,
        "gpu_metrics": gpu_metrics,
        "scaling_efficiency": scaling_efficiency,
        "round_summary": round_summary,
        "details": {
            "input_lens": [request.prompt_len for request in input_requests],
            "output_lens": output_lens,
            "ttfts": [output.ttft for output in outputs],
            "latencies": [output.latency for output in outputs],
            "itls": [output.itl for output in outputs],
            "cached_tokens": cached_tokens_per_turn,
            "errors": [output.error for output in outputs],
        },
    }

    with open(args.result_file, "a") as fout:
        fout.write(json.dumps(result) + "\n")

    if args.raw_result_file:
        with open(args.raw_result_file, "w") as fout:
            json.dump(result, fout, indent=2)


if __name__ == "__main__":
    main()
