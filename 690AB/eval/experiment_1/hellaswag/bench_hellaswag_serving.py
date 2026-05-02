import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from typing import Dict, List, Optional

import numpy as np

from sglang.utils import download_and_cache_file, read_jsonl


@dataclass
class RequestRecord:
    request_id: int
    success: bool
    latency: float
    pred: Optional[int] = None
    label: Optional[int] = None
    error: str = ""


@dataclass
class Metrics:
    request_throughput: float = 0.0
    mean_e2e_latency_ms: float = 0.0
    accuracy: float = 0.0


def get_one_example(lines, i, include_answer):
    ret = lines[i]["activity_label"] + ": " + lines[i]["ctx"] + " "
    if include_answer:
        ret += lines[i]["endings"][lines[i]["label"]]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


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
    print("\n{:=^60}".format(" HellaSwag Benchmark Result "))
    print("{:<45} {:<10.2f}".format("Request Throughput (req/s)", metrics.request_throughput))
    print("{:<45} {:<10.4f}".format("Accuracy", metrics.accuracy))
    print("{:-^60}".format(" Latency "))
    print("{:<45} {:<10.2f}".format("Mean E2E Latency (ms)", metrics.mean_e2e_latency_ms))
    if extended_metrics:
        print("{:<45} {:<10.2f}".format("Median E2E Latency (ms)", extended_metrics["median_e2e_latency_ms"]))
        print("{:<45} {:<10.2f}".format("P90 E2E Latency (ms)", extended_metrics["p90_e2e_latency_ms"]))
        print("{:<45} {:<10.2f}".format("P95 E2E Latency (ms)", extended_metrics["p95_e2e_latency_ms"]))
        print("{:<45} {:<10.2f}".format("P99 E2E Latency (ms)", extended_metrics["p99_e2e_latency_ms"]))
        print("{:<45} {:<10.2f}".format("Std E2E Latency (ms)", extended_metrics["std_e2e_latency_ms"]))
    print("=" * 60)


def run_sglang(args, arguments, choices, labels, few_shot_examples):
    import sglang as sgl
    from sglang.lang.api import set_default_backend
    from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

    set_default_backend(RuntimeEndpoint(f"{args.host}:{args.port}"))

    @sgl.function
    def few_shot_hellaswag(s, question, choices):
        s += few_shot_examples + question
        s += sgl.select("answer", choices=choices)

    records: List[Optional[RequestRecord]] = [None] * len(arguments)

    def get_one_answer(i):
        start = time.perf_counter()
        try:
            state = few_shot_hellaswag.run(
                question=arguments[i]["question"],
                choices=arguments[i]["choices"],
                temperature=0,
            )
            latency = time.perf_counter() - start
            pred = choices[i].index(state["answer"])
            records[i] = RequestRecord(
                request_id=i,
                success=True,
                latency=latency,
                pred=pred,
                label=labels[i],
            )
        except Exception as exc:
            latency = time.perf_counter() - start
            records[i] = RequestRecord(
                request_id=i,
                success=False,
                latency=latency,
                label=labels[i],
                error=str(exc),
            )

    start = time.perf_counter()
    if args.parallel == 1:
        for i in range(len(arguments)):
            get_one_answer(i)
    else:
        with ThreadPoolExecutor(args.parallel) as executor:
            list(executor.map(get_one_answer, list(range(len(arguments)))))
    duration = time.perf_counter() - start
    return duration, [record for record in records if record is not None]


def run_vllm(args, questions, choices, labels, few_shot_examples):
    from sglang.test.test_utils import get_call_select

    call_select = get_call_select(args)
    records: List[Optional[RequestRecord]] = [None] * len(labels)

    def get_one_answer(i):
        start = time.perf_counter()
        try:
            pred = call_select(context=few_shot_examples + questions[i], choices=choices[i])
            latency = time.perf_counter() - start
            records[i] = RequestRecord(
                request_id=i,
                success=True,
                latency=latency,
                pred=pred,
                label=labels[i],
            )
        except Exception as exc:
            latency = time.perf_counter() - start
            records[i] = RequestRecord(
                request_id=i,
                success=False,
                latency=latency,
                label=labels[i],
                error=str(exc),
            )

    start = time.perf_counter()
    if args.parallel == 1:
        for i in range(len(questions)):
            get_one_answer(i)
    else:
        with ThreadPoolExecutor(args.parallel) as executor:
            list(executor.map(get_one_answer, list(range(len(questions)))))
    duration = time.perf_counter() - start
    return duration, [record for record in records if record is not None]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, required=True, choices=["sglang", "vllm"])
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--result-file", type=str, default="result.jsonl")
    parser.add_argument("--num-shots", type=int, default=20)
    parser.add_argument("--data-path", type=str, default="hellaswag_val.jsonl")
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--raw-result-file", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()
    if args.port is None:
        args.port = 30000 if args.backend == "sglang" else 21000
    return args


def main():
    args = parse_args()
    print(args)

    data_path = args.data_path
    url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
    if not os.path.isfile(data_path):
        data_path = download_and_cache_file(url)
    lines = list(read_jsonl(data_path))

    few_shot_examples = get_few_shot_examples(lines, args.num_shots)
    questions = []
    choices = []
    labels = []
    for i in range(len(lines[: args.num_questions])):
        questions.append(get_one_example(lines, i, False))
        choices.append(lines[i]["endings"])
        labels.append(lines[i]["label"])
    arguments = [{"question": q, "choices": c} for q, c in zip(questions, choices)]

    if args.backend == "sglang":
        duration, records = run_sglang(args, arguments, choices, labels, few_shot_examples)
    else:
        duration, records = run_vllm(args, questions, choices, labels, few_shot_examples)

    successful = [record for record in records if record.success]
    accuracy = 0.0 if not successful else float(np.mean([record.pred == record.label for record in successful]))
    metrics = Metrics(
        request_throughput=0.0 if duration <= 0 else len(successful) / duration,
        mean_e2e_latency_ms=0.0 if not successful else float(np.mean([record.latency for record in successful]) * 1000.0),
        accuracy=accuracy,
    )
    extended_metrics = compute_extended_metrics(records)
    print(f"Batch duration: {duration:.3f}")
    print_metrics(metrics, extended_metrics)

    result = {
        "task": "hellaswag",
        "backend": args.backend,
        "num_gpus": 1,
        "run_id": args.run_id,
        "host": args.host,
        "port": args.port,
        "num_requests": args.num_questions,
        "duration": duration,
        "metrics": asdict(metrics),
        "extended_metrics": extended_metrics,
        "other": {
            "num_questions": args.num_questions,
            "num_shots": args.num_shots,
            "parallel": args.parallel,
        },
        "details": {
            "latencies": [record.latency for record in records],
            "success": [record.success for record in records],
            "errors": [record.error for record in records],
            "preds": [record.pred for record in records],
            "labels": [record.label for record in records],
            "request_ids": [record.request_id for record in records],
        },
    }

    with open(args.result_file, "a") as fout:
        fout.write(json.dumps(result) + "\n")
    if args.raw_result_file:
        with open(args.raw_result_file, "w") as fout:
            json.dump(result, fout, indent=2)


if __name__ == "__main__":
    main()
