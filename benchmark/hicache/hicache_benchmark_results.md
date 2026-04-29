# HiCache Benchmark Results

Model: `Qwen/Qwen2.5-1.5B-Instruct`  
Benchmark: `benchmark/hicache/bench_multiturn.py`  
Server settings: `--mem-fraction-static 0.65 --enable-hierarchical-cache --hicache-size 12`

## Workload

- Request rate: `10`
- Seed: `1`
- Num clients: `64`
- Max parallel: `32`
- Num rounds: `3`
- Total requests: `192`
- Input tokens: `32768`, `65536`
- Output tokens: `4096`, `8192`

## Results

| Metric | Vanilla | Custom |
| --- | ---: | ---: |
| Average Prompt Length | 1089.48 | 1089.40 |
| Average Output Length | 64.00 | 64.00 |
| Average TTFT | 1.35 | 0.11 |
| P90 TTFT | 8.16 | 0.16 |
| Median TTFT | 0.11 | 0.10 |
| Average latency | 3.36 | 1.93 |
| P90 latency | 10.22 | 2.48 |
| Median latency | 2.02 | 1.88 |
| Input token throughput | 6665.01 | 9579.95 |
| Output token throughput | 391.53 | 562.80 |
| Request Throughput | 6.12 | 8.79 |
| Cache Hit Rate | 0.490742 | 0.490723 |

## Notes

- `custom` outperformed `vanilla` on TTFT, latency, and throughput.
- Cache hit rate was effectively identical across both runs, so the gain appears to come from implementation efficiency rather than higher cache reuse.
