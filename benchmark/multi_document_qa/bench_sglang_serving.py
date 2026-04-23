import argparse
import asyncio
import json
import time
from dataclasses import asdict
from typing import List, Tuple

import aiohttp
from transformers import AutoTokenizer

from sglang.bench_serving import (
    DatasetRow,
    RequestFuncOutput,
    calculate_metrics,
    remove_prefix,
)
from sglang.utils import read_jsonl

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

USER_PREFIX = "[INST] "
USER_SUFFIX = " [/INST]"
ASSISTANT_PREFIX = ""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark SGLang multi-document QA with serving-style metrics."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--data-path", type=str, default="questions.jsonl")
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--num-docs", type=int, default=10)
    parser.add_argument("--parallel", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-ignore-eos", action="store_true")
    parser.add_argument("--result-file", type=str, default="result_serving.jsonl")
    parser.add_argument("--raw-result-file", type=str, default=None)
    return parser.parse_args()


def build_prompt(docs: List[str], question: str) -> str:
    prompt = USER_PREFIX
    prompt += "Please answer a question according to given documents.\n"
    prompt += "Question:" + question + "Documents begin.\n"
    prompt += "".join(docs)
    prompt += "\nDocuments end."
    prompt += (
        "\n\nBased on the above documents, please answer this question:\n"
        + question
        + "\nAnswer in three words or fewer."
    )
    prompt += USER_SUFFIX
    prompt += ASSISTANT_PREFIX
    return prompt


def load_requests(args, tokenizer) -> Tuple[List[DatasetRow], List[str]]:
    lines = read_jsonl(args.data_path)
    item = lines[0]

    questions = item["questions"][: args.num_questions]
    answers = item["answers"][: args.num_questions]
    docs = item["documents"][: args.num_docs]

    input_requests = []
    for question in questions:
        prompt = build_prompt(docs, question)
        prompt_len = len(tokenizer(prompt).input_ids)
        input_requests.append(
            DatasetRow(
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=args.max_new_tokens,
            )
        )

    return input_requests, answers


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


def compute_accuracy(outputs: List[RequestFuncOutput], labels: List[str]) -> float:
    if not labels:
        return 0.0

    correct = 0
    for output, label in zip(outputs, labels):
        if not output.success:
            continue
        answer = output.generated_text.lower()
        if all(part in answer for part in label.lower().split(" ")):
            correct += 1
    return correct / len(labels)


def print_metrics(metrics, cache_hit_rate, accuracy):
    print("\n{:=^50}".format(" Multi-document QA Serving Benchmark Result "))
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
    print("{:<40} {:<10.3f}".format("Accuracy", accuracy))
    print("=" * 50)


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    input_requests, labels = load_requests(args, tokenizer)
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
    accuracy = compute_accuracy(outputs, labels)

    print_metrics(metrics, cache_hit_rate, accuracy)

    result = {
        "task": "multi_document_qa_serving",
        "host": args.host,
        "port": args.port,
        "num_requests": len(input_requests),
        "num_docs": args.num_docs,
        "parallel": args.parallel,
        "duration": duration,
        "accuracy": accuracy,
        "cache_hit_rate": cache_hit_rate,
        "metrics": asdict(metrics),
        "details": {
            "input_lens": [request.prompt_len for request in input_requests],
            "output_lens": output_lens,
            "ttfts": [output.ttft for output in outputs],
            "latencies": [output.latency for output in outputs],
            "itls": [output.itl for output in outputs],
            "cached_tokens": cached_tokens,
            "answers": [output.generated_text for output in outputs],
            "labels": labels,
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
