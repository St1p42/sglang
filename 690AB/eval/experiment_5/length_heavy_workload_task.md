# Experiment 5: Length-Heavy Nested Shared-Prefix Workload

This task evaluates baseline HiCache vs. length-gated HiCache using the
long-heavy nested shared-prefix workload.

## Workload

- Dataset: `generated-shared-prefix-nested-bucketed`
- Groups: `25`
- Prompts per group: `20`
- Prefix lengths:
  - short: `512`
  - mid: `2048`
  - long: `4096`
- Variant mix per group:
  - short: `4`
  - mid: `4`
  - long: `12`
- Question length: `64`
- Output length: `32`
- Ordering: `phased`
- Total prompts: `500`
- Request rate: `4`
- Max concurrency: `64`

## Baseline server launch

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --port 30000 \
  --mem-fraction-static 0.75 \
  --max-total-tokens 12000 \
  --radix-cache-impl custom \
  --cache-aware-scheduling custom \
  --enable-hierarchical-cache \
  --hicache-impl custom \
  --hicache-size 2 \
  --hicache-write-policy write_through \
  --enable-cache-report
```

## Length-gated server launch

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --port 30000 \
  --mem-fraction-static 0.75 \
  --max-total-tokens 12000 \
  --radix-cache-impl custom \
  --cache-aware-scheduling custom \
  --enable-hierarchical-cache \
  --hicache-impl custom \
  --hicache-custom-backup-policy length_gated \
  --hicache-min-backup-len 1024 \
  --hicache-size 2 \
  --hicache-write-policy write_through \
  --enable-cache-report
```

## Benchmark command

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset-name generated-shared-prefix-nested-bucketed \
  --gsp-num-groups 25 \
  --gsp-prompts-per-group 20 \
  --gsp-nested-short-len 512 \
  --gsp-nested-mid-len 2048 \
  --gsp-nested-long-len 4096 \
  --gsp-nested-short-variants 4 \
  --gsp-nested-mid-variants 4 \
  --gsp-nested-long-variants 12 \
  --gsp-question-len 64 \
  --gsp-output-len 32 \
  --gsp-nested-order phased \
  --num-prompts 500 \
  --request-rate 4 \
  --max-concurrency 64 \
  --apply-chat-template \
  --output-details \
  --output-file /content/results.json
```

## Metrics of interest

- Request throughput
- Total token throughput
- Mean TTFT
- Mean E2E latency
- Host backup count
- Host backup tokens
- Host backup avg len
- Host backup skipped count
- Host backup skipped tokens
- Host hit tokens
