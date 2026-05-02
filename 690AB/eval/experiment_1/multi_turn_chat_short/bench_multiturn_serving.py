import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests
from tqdm import tqdm
from transformers import AutoTokenizer

from data_gen import gen_arguments


REQUEST_TIMEOUT = 600


@dataclass
class RequestRecord:
    request_id: int
    success: bool
    latency: float
    error: str = ""
    output_chars: int = 0


@dataclass
class Metrics:
    request_throughput: float = 0.0
    mean_e2e_latency_ms: float = 0.0


def call_generate_vllm(prompt, temperature, max_tokens, stop=None, url=None):
    assert url is not None

    data = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
        "n": 1,
    }
    res = requests.post(url, json=data, timeout=REQUEST_TIMEOUT)
    res.raise_for_status()
    return res.json()["text"][0][len(prompt) :]


def call_generate_sglang(prompt, temperature, max_tokens, stop=None, url=None):
    assert url is not None

    data = {
        "text": prompt,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "stop": stop,
        },
        "return_logprob": False,
    }
    res = requests.post(url, json=data, timeout=REQUEST_TIMEOUT)
    res.raise_for_status()
    return res.json()["text"]


def get_call_generate(args):
    if args.backend == "vllm":
        return partial(call_generate_vllm, url=f"{args.host}:{args.port}/generate")
    if args.backend == "sglang":
        return partial(call_generate_sglang, url=f"{args.host}:{args.port}/generate")
    raise ValueError(f"Unsupported backend: {args.backend}")


def multi_turns(generate, qas):
    s = ""
    for qa in qas:
        s += qa["prompt"]
        s += generate(s, max_tokens=qa["new_tokens"])
    return s


def dump_outputs(path: str, states):
    p = Path(path)
    with p.open("w") as fout:
        for idx, state in enumerate(states):
            fout.write(f"=== request {idx} ===\n")
            fout.write(state or "")
            fout.write("\n\n")


def compute_extended_metrics(records: List[RequestRecord]) -> Dict[str, float]:
    successful = [record for record in records if record.success]
    if not successful:
        return {}

    lat_ms = np.array([record.latency for record in successful], dtype=float) * 1000.0
    return {
        "median_e2e_latency_ms": float(np.percentile(lat_ms, 50)),
        "p90_e2e_latency_ms": float(np.percentile(lat_ms, 90)),
        "p95_e2e_latency_ms": float(np.percentile(lat_ms, 95)),
        "p99_e2e_latency_ms": float(np.percentile(lat_ms, 99)),
        "std_e2e_latency_ms": float(np.std(lat_ms)),
        "max_e2e_latency_ms": float(np.max(lat_ms)),
    }


def print_metrics(metrics: Metrics, extended_metrics: Dict[str, float]) -> None:
    print("\n{:=^60}".format(" Multi-turn Chat Benchmark Result "))
    print("{:<45} {:<10.2f}".format("Request Throughput (req/s)", metrics.request_throughput))
    print("{:-^60}".format(" Latency "))
    print("{:<45} {:<10.2f}".format("Mean E2E Latency (ms)", metrics.mean_e2e_latency_ms))
    if extended_metrics:
        print("{:<45} {:<10.2f}".format("Median E2E Latency (ms)", extended_metrics["median_e2e_latency_ms"]))
        print("{:<45} {:<10.2f}".format("P90 E2E Latency (ms)", extended_metrics["p90_e2e_latency_ms"]))
        print("{:<45} {:<10.2f}".format("P95 E2E Latency (ms)", extended_metrics["p95_e2e_latency_ms"]))
        print("{:<45} {:<10.2f}".format("P99 E2E Latency (ms)", extended_metrics["p99_e2e_latency_ms"]))
        print("{:<45} {:<10.2f}".format("Std E2E Latency (ms)", extended_metrics["std_e2e_latency_ms"]))
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, required=True, choices=["sglang", "vllm"])
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--result-file", type=str, default="result.jsonl")
    parser.add_argument("--raw-result-file", type=str, default=None)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--num-qa", type=int, default=20)
    parser.add_argument("--min-len-q", type=int, default=256)
    parser.add_argument("--max-len-q", type=int, default=512)
    parser.add_argument("--min-len-a", type=int, default=4)
    parser.add_argument("--max-len-a", type=int, default=8)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    if args.port is None:
        args.port = 30000 if args.backend == "sglang" else 21000

    if args.long:
        args.min_len_a = 256
        args.max_len_a = 512
        args.num_qa = 20

    return args


def main():
    args = parse_args()
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    multi_qas = gen_arguments(args, tokenizer)
    states: List[Optional[str]] = [None] * args.num_qa
    records: List[Optional[RequestRecord]] = [None] * args.num_qa
    call_generate = partial(get_call_generate(args), temperature=0)

    def get_one_answer(i):
        start = time.perf_counter()
        try:
            state = multi_turns(generate=call_generate, **multi_qas[i])
            latency = time.perf_counter() - start
            states[i] = state
            records[i] = RequestRecord(
                request_id=i,
                success=True,
                latency=latency,
                output_chars=len(state),
            )
        except Exception as exc:
            latency = time.perf_counter() - start
            states[i] = None
            records[i] = RequestRecord(
                request_id=i,
                success=False,
                latency=latency,
                error=str(exc),
            )

    start = time.perf_counter()
    if args.parallel == 1:
        for i in tqdm(range(len(multi_qas))):
            get_one_answer(i)
    else:
        with ThreadPoolExecutor(args.parallel) as executor:
            rets = list(
                tqdm(
                    executor.map(get_one_answer, list(range(len(multi_qas)))),
                    total=len(multi_qas),
                )
            )
            for _ in rets:
                pass
    duration = time.perf_counter() - start

    finalized_records: List[RequestRecord] = [record for record in records if record is not None]
    successful = [record for record in finalized_records if record.success]
    metrics = Metrics(
        request_throughput=0.0 if duration <= 0 else len(successful) / duration,
        mean_e2e_latency_ms=(
            0.0
            if not successful
            else float(np.mean([record.latency for record in successful]) * 1000.0)
        ),
    )
    extended_metrics = compute_extended_metrics(finalized_records)

    print(f"Batch duration: {duration:.3f}")
    print_metrics(metrics, extended_metrics)

    dump_outputs(f"tmp_output_{args.backend}.txt", states)

    result = {
        "task": "multi_turn_chat",
        "backend": args.backend,
        "num_gpus": 1,
        "run_id": args.run_id,
        "host": args.host,
        "port": args.port,
        "num_requests": args.num_qa,
        "num_turns": args.turns,
        "duration": duration,
        "metrics": asdict(metrics),
        "extended_metrics": extended_metrics,
        "other": {
            "parallel": args.parallel,
            "output_mode": "long" if args.long else "short",
            "min_len_q": args.min_len_q,
            "max_len_q": args.max_len_q,
            "min_len_a": args.min_len_a,
            "max_len_a": args.max_len_a,
        },
        "details": {
            "latencies": [record.latency for record in finalized_records],
            "success": [record.success for record in finalized_records],
            "errors": [record.error for record in finalized_records],
            "output_chars": [record.output_chars for record in finalized_records],
            "request_ids": [record.request_id for record in finalized_records],
        },
    }

    with open(args.result_file, "a") as fout:
        fout.write(json.dumps(result) + "\n")
    if args.raw_result_file:
        with open(args.raw_result_file, "w") as fout:
            json.dump(result, fout, indent=2)


if __name__ == "__main__":
    main()
