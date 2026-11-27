# Open Protocol Corpora — Skew, Truncation-Free OBFD Packing & Checker Throughput

| Corpus | Docs | Median / P95 / Max Toks | Skew (`max/med`) | `bin_cap` (0 trunc) | Padded (`B=16`) | Bucketed (`B=16`) | First-Fit Bins (`useful%`) | OBFD Bins (`useful%`) | Batched Checker (proto/s) |
|---|---:|---|---:|---:|---:|---:|---|---|---:|
| `opentrons_shared_data` | 19 | 1058 / 4216 / 4216 | 3.98x | 8192 | 39.0% | 37.6% | 4 (84.0%) | 4 (84.0%) | 5,233.7 |
| `wlp` | 80 | 316 / 1032 / 1301 | 4.11x | 2048 | 37.4% | 71.5% | 15 (95.3%) | 15 (95.3%) | 7,113.0 |
| `bioplanner` | 80 | 450 / 1002 / 1257 | 2.80x | 2048 | 53.6% | 77.5% | 19 (97.2%) | 19 (97.2%) | 8,236.6 |
| `smolinstruct` | 200 | 90 / 185 / 361 | 3.99x | 512 | 47.3% | 89.4% | 40 (96.4%) | 39 (98.8%) | 80,516.4 |
| `ord` | 160 | 746 / 944 / 1085 | 1.45x | 2048 | 78.0% | 96.1% | 68 (86.8%) | 65 (90.8%) | 19,173.4 |
