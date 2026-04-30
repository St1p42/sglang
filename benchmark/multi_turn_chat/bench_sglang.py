import json
import time
from argparse import ArgumentParser

from data_gen import gen_arguments
from vllm.transformers_utils.tokenizer import get_tokenizer

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text


@sgl.function
def multi_turns(s, qas):
    for turn_idx, qa in enumerate(qas):
        s += qa["prompt"]
        s += sgl.gen(
            name=f"answer_{turn_idx}",
            max_tokens=qa["new_tokens"],
            ignore_eos=True,
        )


def main(args):
    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)

    multi_qas = gen_arguments(args, tokenizer)

    backend = select_sglang_backend(args)

    tic = time.perf_counter()
    states = multi_turns.run_batch(
        multi_qas,
        temperature=0,
        backend=backend,
        num_threads=args.parallel,
        progress_bar=True,
    )
    latency = time.perf_counter() - tic

    print(f"Latency: {latency:.3f}")

    turn_records = []
    turn_summary = {}
    for turn_idx in range(args.turns):
        turn_key = f"answer_{turn_idx}"
        turn_items = []
        for state_idx, state in enumerate(states):
            meta_info = state.get_meta_info(turn_key)
            if not meta_info:
                continue

            prompt_tokens = meta_info.get("prompt_tokens", 0)
            cached_tokens = meta_info.get("cached_tokens", 0)
            cache_hit_rate = (
                0.0 if prompt_tokens == 0 else cached_tokens / prompt_tokens
            )
            record = {
                "state_idx": state_idx,
                "turn": turn_idx,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "cache_hit_rate": cache_hit_rate,
                "completion_tokens": meta_info.get("completion_tokens", 0),
                "e2e_latency": meta_info.get("e2e_latency"),
            }
            turn_records.append(record)
            turn_items.append(record)
            print(
                "[TURN CACHE] "
                f"turn={turn_idx} state={state_idx} "
                f"prompt_tokens={prompt_tokens} cached_tokens={cached_tokens} "
                f"cache_hit_rate={cache_hit_rate:.6f}"
            )

        if turn_items:
            prompt_sum = sum(item["prompt_tokens"] for item in turn_items)
            cached_sum = sum(item["cached_tokens"] for item in turn_items)
            turn_summary[f"turn_{turn_idx}"] = {
                "requests": len(turn_items),
                "prompt_tokens": prompt_sum,
                "cached_tokens": cached_sum,
                "cache_hit_rate": 0.0 if prompt_sum == 0 else cached_sum / prompt_sum,
            }

    if turn_summary:
        print("Per-turn cache summary:")
        for turn_key, item in turn_summary.items():
            print(
                f"  {turn_key}: requests={item['requests']}, "
                f"prompt_tokens={item['prompt_tokens']}, "
                f"cached_tokens={item['cached_tokens']}, "
                f"cache_hit_rate={item['cache_hit_rate']:.6f}"
            )

    dump_state_text(f"tmp_output_{args.backend}.txt", states)

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
            },
            "turn_summary": turn_summary,
            "turn_records": turn_records,
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--num-qa", type=int, default=20)
    parser.add_argument("--min-len-q", type=int, default=256)
    parser.add_argument("--max-len-q", type=int, default=512)
    parser.add_argument("--min-len-a", type=int, default=4)
    parser.add_argument("--max-len-a", type=int, default=8)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--long", action="store_true")
    args = add_common_sglang_args_and_parse(parser)

    if args.long:
        args.min_len_a = 256
        args.max_len_a = 512
        args.num_qa = 20

    print(args)
    main(args)
