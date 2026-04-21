import argparse
import asyncio
import json
import mimetypes
import os
import time
from dataclasses import asdict
from typing import Dict, List, Tuple

import aiohttp
import numpy as np
from PIL import Image
import requests
from transformers import AutoProcessor

from sglang.bench_serving import (
    DatasetRow,
    RequestFuncOutput,
    calculate_metrics,
    remove_prefix,
)
from sglang.utils import encode_image_base64, read_jsonl

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


def to_jsonable(value):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark SGLang LLaVA image QA with serving-style metrics."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--question-file", type=str, default="questions.jsonl")
    parser.add_argument("--answer-file", type=str, default="answers_serving.jsonl")
    parser.add_argument("--image-folder", type=str, default="./images")
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--parallel", type=int, default=64)
    parser.add_argument("--image-token", type=str, default="<image>")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-ignore-eos", action="store_true")
    parser.add_argument("--result-file", type=str, default="result_serving.jsonl")
    parser.add_argument("--raw-result-file", type=str, default=None)
    return parser.parse_args()


def build_data_uri(image_file: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_file)
    if mime_type is None:
        mime_type = "image/jpeg"
    return f"data:{mime_type};base64,{encode_image_base64(image_file)}"


def fetch_server_model_info(args) -> Dict:
    response = requests.get(f"http://{args.host}:{args.port}/get_model_info", timeout=30)
    response.raise_for_status()
    return response.json()


def build_prompt_and_token_counts(
    args, processor, image: Image.Image, image_data_uri: str, question: str
):
    try:
        content_items = [
            {"type": "image", "image": {"url": image_data_uri}},
            {"type": "text", "text": question},
        ]
        prompt_str = processor.apply_chat_template(
            [{"role": "user", "content": content_items}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        prompt_str = f"{args.image_token}\n{question}"

    prompt_len = processor(
        text=[prompt_str],
        images=[image],
        padding=False,
        return_tensors="pt",
    )["input_ids"].numel()

    try:
        text_only_prompt = processor.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True,
            tokenize=False,
        )
        text_prompt_len = processor(
            text=[text_only_prompt],
            padding=False,
            return_tensors="pt",
        )["input_ids"].numel()
    except Exception:
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        text_prompt_len = len(tokenizer.encode(question))

    return prompt_str, prompt_len, text_prompt_len, prompt_len - text_prompt_len


def load_requests(args, processor):
    lines = list(read_jsonl(args.question_file))[: args.num_questions]
    input_requests = []
    question_metadata = []

    for line in lines:
        image_file = os.path.abspath(os.path.join(args.image_folder, line["image"]))
        image = Image.open(image_file).convert("RGB")
        image_data_uri = build_data_uri(image_file)
        prompt_str, prompt_len, text_prompt_len, vision_prompt_len = (
            build_prompt_and_token_counts(
                args, processor, image, image_data_uri, line["text"]
            )
        )
        input_requests.append(
            DatasetRow(
                prompt=prompt_str,
                prompt_len=prompt_len,
                output_len=args.max_new_tokens,
                text_prompt_len=text_prompt_len,
                vision_prompt_len=vision_prompt_len,
                image_data=[image_data_uri],
            )
        )
        question_metadata.append(
            {
                "question_id": line["question_id"],
                "image": line["image"],
                "question": line["text"],
                "prompt": prompt_str,
                "category": line.get("category"),
            }
        )

    return input_requests, question_metadata


async def async_request_sglang_generate(
    session: aiohttp.ClientSession,
    url: str,
    request: DatasetRow,
    disable_ignore_eos: bool,
) -> Tuple[RequestFuncOutput, int]:
    payload = {
        "text": request.prompt,
        "image_data": request.image_data,
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
    saw_text_chunk = False
    st = time.perf_counter()
    output.start_time = st
    most_recent_timestamp = st
    last_output_len = 0

    try:
        async with session.post(url=url, json=payload) as response:
            if response.status != 200:
                output.success = False
                output.error = await response.text()
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

                saw_text_chunk = True
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
            output.latency = time.perf_counter() - st
            output.prompt_len = prompt_tokens
            if output.output_len == 0 and saw_text_chunk:
                output.output_len = request.output_len
            if saw_text_chunk:
                output.success = True
            else:
                output.success = False
                output.output_len = 0
                output.error = "Server returned 200 OK but emitted no text tokens."
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


def print_metrics(metrics, cache_hit_rate):
    print("\n{:=^50}".format(" LLaVA Image QA Serving Benchmark Result "))
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
    print("=" * 50)


def write_answers(args, question_metadata, outputs, model_id: str):
    with open(args.answer_file, "w") as fout:
        for i, (item, output) in enumerate(zip(question_metadata, outputs)):
            value = {
                "question_id": item["question_id"],
                "prompt": item["question"],
                "text": output.generated_text.strip() if output.success else "",
                "model_id": model_id,
                "answer_id": i,
                "metadata": {
                    "success": output.success,
                    "error": output.error,
                    "category": item["category"],
                    "image": item["image"],
                    "serving_prompt": item["prompt"],
                },
            }
            fout.write(json.dumps(value) + "\n")


def main():
    args = parse_args()
    model_info = fetch_server_model_info(args)
    processor = AutoProcessor.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    input_requests, question_metadata = load_requests(args, processor)
    outputs, cached_tokens, duration = asyncio.run(run_all(args, input_requests))

    successful_outputs = [output for output in outputs if output.success]
    if not successful_outputs:
        print("All image benchmark requests failed.")
        print("Sample request errors:")
        for idx, output in enumerate(outputs[:5]):
            print(f"  request_{idx}: {output.error}")
        write_answers(args, question_metadata, outputs, model_info["model_path"])
        return

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

    print_metrics(metrics, cache_hit_rate)
    write_answers(args, question_metadata, outputs, model_info["model_path"])

    result = {
        "task": "llava_bench_serving",
        "model_path": model_info["model_path"],
        "host": args.host,
        "port": args.port,
        "num_requests": len(input_requests),
        "parallel": args.parallel,
        "duration": duration,
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
            "errors": [output.error for output in outputs],
            "questions": question_metadata,
        },
    }

    with open(args.result_file, "a") as fout:
        fout.write(json.dumps(to_jsonable(result)) + "\n")

    if args.raw_result_file:
        with open(args.raw_result_file, "w") as fout:
            json.dump(to_jsonable(result), fout, indent=2)


if __name__ == "__main__":
    main()
