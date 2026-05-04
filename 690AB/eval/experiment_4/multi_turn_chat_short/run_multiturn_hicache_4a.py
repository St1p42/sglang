import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


DEFAULT_RESULTS_PATH = (
    Path(__file__).resolve().parent / "multiturnhicache benchmark_4a.md"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Experiment 4a multiturn HiCache benchmark 5 times and summarize mean/std."
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs.")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Where to write the per-run and mean/std summary.",
    )
    parser.add_argument(
        "--jsonl-file",
        type=Path,
        default=Path("/tmp/multiturn_hicache_4a_runs.jsonl"),
        help="Temporary JSONL file produced by bench_multiturn.py.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--model-path", type=str, default="Qwen/Qwen2.5-7B-Instruct"
    )
    parser.add_argument("--request-rate", type=float, default=2.0)
    parser.add_argument("--distribution", type=str, default="uniform")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-clients", type=int, default=8)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--request-length", type=int, default=1024)
    parser.add_argument("--sub-question-input-length", type=int, default=256)
    parser.add_argument("--output-length", type=int, default=1)
    return parser.parse_args()


METRIC_KEYS = [
    ("total_requests", "Total requests"),
    ("request_rate", "Request rate"),
    ("average_prompt_len", "Average Prompt Length"),
    ("average_output_len", "Average Output Length"),
    ("average_ttft", "Average TTFT"),
    ("p90_ttft", "P90 TTFT"),
    ("median_ttft", "Median TTFT"),
    ("average_latency", "Average latency"),
    ("p90_latency", "P90 latency"),
    ("median_latency", "Median latency"),
    ("input_token_throughput", "Input token throughput"),
    ("output_token_throughput", "Output token throughput"),
    ("throughput", "Request Throughput"),
    ("cache_hit_rate", "Cache Hit Rate"),
]


def read_last_jsonl_row(path: Path):
    lines = path.read_text().strip().splitlines()
    if not lines:
        raise RuntimeError(f"No benchmark results found in {path}")
    return json.loads(lines[-1])


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if not values:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((x - avg) ** 2 for x in values) / len(values))


def fmt(value):
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def build_benchmark_cmd(args, run_idx):
    return [
        sys.executable,
        "-u",
        "benchmark/hicache/bench_multiturn.py",
        "--model-path",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--disable-auto-run",
        "--request-rate",
        str(args.request_rate),
        "--distribution",
        args.distribution,
        "--seed",
        str(args.seed + run_idx - 1),
        "--num-clients",
        str(args.num_clients),
        "--max-parallel",
        str(args.max_parallel),
        "--num-rounds",
        str(args.num_rounds),
        "--request-length",
        str(args.request_length),
        "--sub-question-input-length",
        str(args.sub_question_input_length),
        "--output-length",
        str(args.output_length),
        "--log-file",
        str(args.jsonl_file),
        "--tag",
        f"Run{run_idx}",
    ]


def main():
    args = parse_args()
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    if args.jsonl_file.exists():
        args.jsonl_file.unlink()

    summaries = []
    lines = []
    lines.append("# Multiturn HiCache Benchmark 4A")
    lines.append("")
    lines.append("This file contains 5 benchmark runs plus mean/std over all runs.")
    lines.append("")

    for run_idx in range(1, args.runs + 1):
        cmd = build_benchmark_cmd(args, run_idx)
        print(f"Starting Run{run_idx}...")
        subprocess.run(cmd, check=True)
        row = read_last_jsonl_row(args.jsonl_file)
        summary = row["summary"]
        summaries.append(summary)

        lines.append(f"## Run{run_idx}")
        lines.append("")
        for key, label in METRIC_KEYS:
            lines.append(f"- {label}: {fmt(summary[key])}")
        lines.append("")

    lines.append("## Mean/Std")
    lines.append("")
    for key, label in METRIC_KEYS:
        values = [summary[key] for summary in summaries]
        if key == "total_requests":
            lines.append(f"- {label}: mean={mean(values):.2f}, std={std(values):.2f}")
        else:
            lines.append(f"- {label}: mean={mean(values):.6f}, std={std(values):.6f}")
    lines.append("")

    args.results_file.write_text("\n".join(lines))
    print(f"Wrote results to {args.results_file}")


if __name__ == "__main__":
    main()
