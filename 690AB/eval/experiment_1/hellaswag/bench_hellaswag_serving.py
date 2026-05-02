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
    pred: Optional[int] = None
    label: Optional[int] = None
    error: str = ""


@dataclass
class Metrics:
    request_throughput: float = 0.0
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


def print_metrics(metrics: Metrics, duration: float) -> None:
    print("\n{:=^60}".format(" HellaSwag Benchmark Result "))
    print("{:<45} {:<10.2f}".format("Request Throughput (req/s)", metrics.request_throughput))
    print("{:<45} {:<10.4f}".format("Accuracy", metrics.accuracy))
    print("{:<45} {:<10.2f}".format("Batch Duration (s)", duration))
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

    start = time.perf_counter()
    states = few_shot_hellaswag.run_batch(
        arguments,
        temperature=0,
        num_threads=args.parallel,
        progress_bar=True,
    )
    duration = time.perf_counter() - start
    records: List[RequestRecord] = []
    for i, state in enumerate(states):
        try:
            pred = choices[i].index(state["answer"])
            records.append(
                RequestRecord(
                    request_id=i,
                    success=True,
                    pred=pred,
                    label=labels[i],
                )
            )
        except Exception as exc:
            records.append(
                RequestRecord(
                    request_id=i,
                    success=False,
                    label=labels[i],
                    error=str(exc),
                )
            )
    return duration, records


def run_vllm(args, questions, choices, labels, few_shot_examples):
    from sglang.test.test_utils import get_call_select

    call_select = get_call_select(args)
    records: List[Optional[RequestRecord]] = [None] * len(labels)

    def get_one_answer(i):
        try:
            pred = call_select(context=few_shot_examples + questions[i], choices=choices[i])
            records[i] = RequestRecord(
                request_id=i,
                success=True,
                pred=pred,
                label=labels[i],
            )
        except Exception as exc:
            records[i] = RequestRecord(
                request_id=i,
                success=False,
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
        accuracy=accuracy,
    )
    print(f"Batch duration: {duration:.3f}")
    print_metrics(metrics, duration)

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
        "other": {
            "num_questions": args.num_questions,
            "num_shots": args.num_shots,
            "parallel": args.parallel,
        },
        "details": {
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
