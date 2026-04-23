import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict
from typing import List, Tuple

import aiohttp
import numpy as np
import pandas as pd
import tiktoken
from transformers import AutoTokenizer

from sglang.bench_serving import (
    DatasetRow,
    RequestFuncOutput,
    calculate_metrics,
    remove_prefix,
)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

CHOICES = ["A", "B", "C", "D"]
TIKTOKEN_TOKENIZER = tiktoken.encoding_for_model("gpt-3.5-turbo")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark SGLang MMLU with serving-style metrics."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data-dir", "--data_dir", "-d", type=str, default="data")
    parser.add_argument("--nsub", type=int, default=60)
    parser.add_argument("--parallel", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-ignore-eos", action="store_true")
    parser.add_argument("--result-file", type=str, default="result_serving.jsonl")
    parser.add_argument("--raw-result-file", type=str, default=None)
    return parser.parse_args()


def format_subject(subject):
    parts = subject.split("_")
    return "".join(" " + part for part in parts)


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    num_choices = df.shape[1] - 2
    for j in range(num_choices):
        prompt += "\n{}. {}".format(CHOICES[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, num_choices + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about{}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def load_requests(args, tokenizer) -> Tuple[List[DatasetRow], List[str], List[Tuple[str, int]]]:
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    input_requests = []
    labels = []
    subject_counts = []

    for subject in subjects[: args.nsub]:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )
        subject_counts.append((subject, test_df.shape[0]))

        k = args.ntrain
        few_shot_examples = gen_prompt(dev_df, subject, k)
        while len(TIKTOKEN_TOKENIZER.encode(few_shot_examples)) > 1536:
            k -= 1
            few_shot_examples = gen_prompt(dev_df, subject, k)

        for i in range(test_df.shape[0]):
            prompt = few_shot_examples + format_example(
                test_df, i, include_answer=False
            )
            prompt_len = len(tokenizer(prompt).input_ids)
            input_requests.append(
                DatasetRow(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    output_len=args.max_new_tokens,
                )
            )
            labels.append(test_df.iloc[i, test_df.shape[1] - 1])

    return input_requests, labels, subject_counts


async def async_request_sglang_generate(
    session: aiohttp.ClientSession,
    url: str,
    request: DatasetRow,
    disable_ignore_eos: bool,
) -> Tuple[RequestFuncOutput, int]:
    payload = {
        "text": request.prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": request.output_len,
            "ignore_eos": not disable_ignore_eos,
        },
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_logprob": False,
        "logprob_start_len": -1,
    }

    output = RequestFuncOutput()
    output.prompt_len = request.prompt_len
    generated_text = ""
    cached_tokens = 0
    prompt_tokens = request.prompt_len
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
                if chunk == "[DONE]":
                    continue

                data = json.loads(chunk)
                text = data.get("text", "")
                if not text:
                    continue

                timestamp = time.perf_counter()
                generated_text = text
                meta_info = data.get("meta_info") or {}
                output.output_len = meta_info.get("completion_tokens", output.output_len)

                if output.ttft == 0.0:
                    output.ttft = timestamp - st
                    prompt_tokens = meta_info.get("prompt_tokens", request.prompt_len)
                    cached_tokens = meta_info.get("cached_tokens", 0)
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
            output.prompt_len = prompt_tokens
            if output.output_len == 0:
                output.output_len = request.output_len
            return output, cached_tokens
    except Exception as e:
        output.success = False
        output.error = str(e)
        return output, cached_tokens


async def run_all(args, input_requests: List[DatasetRow]):
    url = f"http://{args.host}:{args.port}/generate"
    semaphore = asyncio.Semaphore(args.parallel)

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        async def run_with_limit(request):
            async with semaphore:
                return await async_request_sglang_generate(
                    session=session,
                    url=url,
                    request=request,
                    disable_ignore_eos=args.disable_ignore_eos,
                )

        benchmark_start = time.perf_counter()
        results = await asyncio.gather(
            *[asyncio.create_task(run_with_limit(request)) for request in input_requests]
        )
        duration = time.perf_counter() - benchmark_start

    outputs = [output for output, _ in results]
    cached_tokens = [cached for _, cached in results]
    return outputs, cached_tokens, duration


def compute_accuracy(outputs, labels, subject_counts):
    preds = [
        output.generated_text.strip()[0]
        if output.success and output.generated_text.strip()
        else ""
        for output in outputs
    ]
    cors = [pred == label for pred, label in zip(preds, labels)]

    subject_accuracy = {}
    pt = 0
    for subject, num_qs in subject_counts:
        subject_accuracy[subject] = float(np.mean(cors[pt : pt + num_qs]))
        pt += num_qs
    assert pt == len(cors)

    weighted_acc = float(np.mean(cors)) if cors else 0.0
    return weighted_acc, preds, subject_accuracy


def print_metrics(metrics, cache_hit_rate, accuracy, subject_accuracy):
    print("\n{:=^50}".format(" MMLU Serving Benchmark Result "))
    print(
        "{:<40} {:<10.2f}".format(
            "Request Throughput (req/s)", metrics.request_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input Token Throughput (tok/s)", metrics.input_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output Token Throughput (tok/s)", metrics.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total Token Throughput (tok/s)", metrics.total_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Mean E2E Latency (ms)", metrics.mean_e2e_latency_ms
        )
    )
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms)", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms)", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Concurrency", metrics.concurrency))
    print("{:<40} {:<10.6f}".format("Cache Hit Rate", cache_hit_rate))
    print("{:<40} {:<10.3f}".format("Average Accuracy", accuracy))
    print("=" * 50)

    for subject, acc in subject_accuracy.items():
        print(f"subject: {subject}, acc: {acc:.3f}")


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    input_requests, labels, subject_counts = load_requests(args, tokenizer)
    outputs, cached_tokens, duration = asyncio.run(run_all(args, input_requests))

    metrics, output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=duration,
        tokenizer=tokenizer,
        backend="sglang",
    )
    total_prompt_tokens = sum(
        output.prompt_len
        for output in outputs
        if output.success and output.prompt_len > 0
    )
    cache_hit_rate = (
        0.0 if total_prompt_tokens == 0 else sum(cached_tokens) / total_prompt_tokens
    )
    accuracy, preds, subject_accuracy = compute_accuracy(
        outputs, labels, subject_counts
    )

    print_metrics(metrics, cache_hit_rate, accuracy, subject_accuracy)

    result = {
        "task": "mmlu_serving",
        "host": args.host,
        "port": args.port,
        "num_requests": len(input_requests),
        "nsub": args.nsub,
        "parallel": args.parallel,
        "duration": duration,
        "accuracy": accuracy,
        "subject_accuracy": subject_accuracy,
        "cache_hit_rate": cache_hit_rate,
        "metrics": asdict(metrics),
        "details": {
            "input_lens": [request.prompt_len for request in input_requests],
            "output_lens": output_lens,
            "ttfts": [output.ttft for output in outputs],
            "latencies": [output.latency for output in outputs],
            "itls": [output.itl for output in outputs],
            "cached_tokens": cached_tokens,
            "preds": preds,
            "labels": labels,
            "answers": [output.generated_text for output in outputs],
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
