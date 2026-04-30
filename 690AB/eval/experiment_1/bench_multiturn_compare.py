import json
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import requests
from tqdm import tqdm
from transformers import AutoTokenizer

from data_gen import gen_arguments


def call_generate_vllm(prompt, temperature, max_tokens, stop=None, url=None):
    assert url is not None

    data = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
        "n": 1,
    }
    res = requests.post(url, json=data, timeout=600)
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
    res = requests.post(url, json=data, timeout=600)
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


def main(args):
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    multi_qas = gen_arguments(args, tokenizer)
    states = [None] * args.num_qa
    call_generate = partial(get_call_generate(args), temperature=0)

    def get_one_answer(i):
        states[i] = multi_turns(generate=call_generate, **multi_qas[i])

    tic = time.perf_counter()
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

    latency = time.perf_counter() - tic
    print(f"Latency: {latency:.3f}")

    dump_outputs(f"tmp_output_{args.backend}.txt", states)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "multi_turn_chat",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "num_requests": args.num_qa,
            "num_turns": args.turns,
            "other": {
                "parallel": args.parallel,
                "output_mode": "long" if args.long else "short",
                "min_len_q": args.min_len_q,
                "max_len_q": args.max_len_q,
                "min_len_a": args.min_len_a,
                "max_len_a": args.max_len_a,
                "host": args.host,
                "port": args.port,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--backend", type=str, required=True, choices=["sglang", "vllm"]
    )
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--result-file", type=str, default="result.jsonl")
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--num-qa", type=int, default=20)
    parser.add_argument("--min-len-q", type=int, default=256)
    parser.add_argument("--max-len-q", type=int, default=512)
    parser.add_argument("--min-len-a", type=int, default=4)
    parser.add_argument("--max-len-a", type=int, default=8)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--long", action="store_true")
    args = parser.parse_args()

    if args.port is None:
        args.port = 30000 if args.backend == "sglang" else 21000

    if args.long:
        args.min_len_a = 256
        args.max_len_a = 512
        args.num_qa = 20

    main(args)
