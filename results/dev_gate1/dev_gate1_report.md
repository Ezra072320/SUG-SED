# Gate 1 Report (Strictly DEV M_02..M_05, Zero Held-Out Touched)

## Event2012 Dev (`M_02..M_05`)

| Group | Method | Tok | Show% | AMI | Mac-$B^3$ | Mic-$B^3$ | Tail-$B^3$ | UnqNet | Wall(s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-LLM Paper 1 | EventX (TKDD'20) | 0 | 0.00% | 87.92 | 62.86 | 83.30 | 14.80 | +0.0 | 3.7 |
| 0-LLM Paper 2 | SBERT-Agglom (EMNLP'19) | 0 | 0.00% | 90.51 | 77.65 | 87.74 | 64.53 | +0.0 | 2.9 |
| 0-LLM Paper 3a | HiSEvent Default (AAAI'24) | 0 | 0.00% | 88.53 | 69.17 | 82.10 | 30.25 | +0.0 | 0.0 |
| 0-LLM Paper 3b | HiSEvent Dev-Tuned (AAAI'24) | 0 | 0.00% | 88.88 | 78.17 | 82.18 | 60.69 | +0.0 | 0.5 |
| 0-LLM Paper 4 | HyperSED (AAAI'25) | 0 | 0.00% | 90.24 | 74.39 | 85.48 | 46.58 | +0.0 | 3.7 |
| Internal 0-Tok | Pure 2D-SE Base (0.50) | 0 | 0.00% | 88.88 | 78.17 | 82.18 | 60.69 | +0.0 | 0.0 |
| Internal 0-Tok | AutoMerge-Only (0.85) | 0 | 0.00% | 91.19 | 79.28 | 87.18 | 60.60 | +0.0 | 0.0 |
| Internal 0-Tok | Meso-Heur-Best (Pure) | 0 | 0.00% | 92.37 | 79.90 | 89.27 | 60.60 | +0.0 | 0.4 |
| LLM Paper 5 | ClusterLLM-NoFT (EMNLP'23) | 39598 | 2.34% | 88.37 | 77.10 | 84.72 | 67.33 | -645.0 | 661.9 |
| LLM Paper 6 | Few-Shot LLM-Post (TACL'24) | 38756 | 5.19% | 88.88 | 78.17 | 82.18 | 60.69 | +0.0 | 159.2 |
| LLM Paper 7a | Cequel-Edge (arXiv'25) | 34865 | 3.68% | 88.74 | 78.09 | 81.98 | 60.69 | -188.5 | 723.2 |
| LLM Paper 7b | Cequel-Transitive (arXiv'25) | 34865 | 3.68% | 92.11 | 79.12 | 89.01 | 60.60 | +374.5 | 723.2 |
| LLM Paper 8 | LSED-Budgeted (ACL'25) | 39681 | 4.35% | 88.93 | 78.13 | 82.11 | 60.66 | +35.5 | 891.1 |
| LLM Paper 9 | RagSEDE-Budgeted (WWW'26) | 38086 | 6.65% | 88.89 | 78.20 | 82.18 | 60.69 | +0.0 | 328.7 |
| LLM Paper 10 | LHG-Budgeted (WWW'26) | 39236 | 4.62% | 88.44 | 78.00 | 81.46 | 60.69 | -134.2 | 501.0 |
| Internal Ctrl | Random (3-seed mean) | 38819 | 5.14% | 88.87 | 78.15 | 82.16 | 60.69 | -28.7 | 0.0 |
| Internal Ctrl | Random-Meso (3-seed mean) | 34907 | 3.42% | 92.71 | 79.50 | 89.79 | 60.54 | +535.4 | 0.0 |
| **OUR METHOD** | **Pure-Star-Meso (Qwen3-8B)** | 32864 | 3.36% | 94.06 | 81.31 | 92.13 | 63.73 | +649.0 | 2.8 |
| Plug-and-Play | Star-Meso on HiSEvent Default | 27514 | 2.66% | 93.64 | 71.44 | 91.58 | 31.92 | +651.2 | 2.1 |

*Note: `ClusterLLM-NoFT (EMNLP'23)` omits contrastive encoder fine-tuning under the zero-training budgeted protocol and computes `UnqNet` relative to its 0-LLM hierarchical base `SBERT-Agglom (0.60)`. `HyperSED` caps `n_anchors <= 360`.*

### Event2012 Pillar Ablations (`Qwen/Qwen3-8B`)

| Variant | Calls | Tok | AMI | Mac-$B^3$ | Tail-$B^3$ | UnqNet |
|---|---:|---:|---:|---:|---:|---:|
| **Full Pure-Star-Meso** | 64 | 32864 | 94.06 | 81.31 | 63.73 | +649.0 |
| w/o AutoMerge (>=0.85) | 67 | 34777 | 94.06 | 81.31 | 63.73 | +643.5 |
| w/o Bayes Band & AutoMerge | 66 | 37857 | 94.05 | 81.06 | 63.40 | +619.0 |
| w/o Star Packing (k=1 Pair) | 135 | 30673 | 93.35 | 81.17 | 63.41 | +667.5 |
| w/o Guarded UF | 66 | 33888 | 92.49 | 80.12 | 63.40 | +631.0 |
| w/o Seam Split | 58 | 29833 | 93.99 | 80.40 | 61.25 | +624.0 |

### Event2012 Dev Cross-Day Frontier (`M_00..M_05`, eval `M_02..M_05`)

| Mode | G-AMI | G-Mac$B^3$ | G-Mic$B^3$ | G-Tail$B^3$ | Q |
|---|---:|---:|---:|---:|---:|
| no_link | 91.59 | 75.66 | 82.72 | 75.20 | 0 |
| proto_0.60 | 93.69 | 74.73 | 88.69 | 63.77 | 0 |
| proto_0.65 | 94.53 | 78.37 | 90.56 | 68.50 | 0 |
| proto_0.72 | 94.36 | 78.35 | 90.43 | 68.51 | 0 |
| proto_0.78 | 94.47 | 78.96 | 90.60 | 68.52 | 0 |
| proto_0.80 | 94.65 | 79.65 | 91.02 | 68.53 | 0 |
| proto_0.82 | 94.67 | 80.26 | 91.07 | 68.53 | 0 |
| proto_0.85 | 94.67 | 80.26 | 91.07 | 68.53 | 0 |
| proto_0.88 | 94.80 | 80.99 | 91.10 | 68.53 | 0 |
| proto_0.90 | 94.64 | 81.47 | 90.82 | 74.69 | 0 |
| proto_0.92 | 94.18 | 80.18 | 89.77 | 74.69 | 0 |
| proto_0.95 | 92.79 | 77.44 | 86.20 | 75.13 | 0 |
| proto_attr_0.80 | 94.67 | 79.66 | 91.05 | 68.53 | 0 |
| proto_attr_0.85 | 94.64 | 79.64 | 91.02 | 68.53 | 0 |
| proto_attr_0.90 | 94.82 | 80.28 | 91.16 | 68.53 | 0 |
| proto_attr_0.95 | 94.61 | 81.07 | 90.73 | 74.69 | 0 |
| sug_cross_star | 94.71 | 81.52 | 91.15 | 74.69 | 31 |
| oracle_band | 95.25 | 83.09 | 92.03 | 74.69 | 33 |

## Event2018 Dev (`M_02..M_05`)

| Group | Method | Tok | Show% | AMI | Mac-$B^3$ | Mic-$B^3$ | Tail-$B^3$ | UnqNet | Wall(s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-LLM Paper 1 | EventX (TKDD'20) | 0 | 0.00% | 75.18 | 66.43 | 73.55 | 36.25 | +0.0 | 9.9 |
| 0-LLM Paper 2 | SBERT-Agglom (EMNLP'19) | 0 | 0.00% | 67.52 | 71.24 | 68.56 | 67.31 | +0.0 | 7.7 |
| 0-LLM Paper 3a | HiSEvent Default (AAAI'24) | 0 | 0.00% | 67.80 | 64.49 | 59.09 | 29.33 | +0.0 | 0.0 |
| 0-LLM Paper 3b | HiSEvent Dev-Tuned (AAAI'24) | 0 | 0.00% | 68.57 | 75.01 | 59.69 | 69.77 | +0.0 | 1.2 |
| 0-LLM Paper 4 | HyperSED (AAAI'25) | 0 | 0.00% | 67.98 | 61.20 | 61.61 | 35.82 | +0.0 | 4.5 |
| Internal 0-Tok | Pure 2D-SE Base (0.50) | 0 | 0.00% | 68.53 | 71.39 | 59.89 | 54.75 | +0.0 | 0.0 |
| Internal 0-Tok | AutoMerge-Only (0.85) | 0 | 0.00% | 69.59 | 72.25 | 62.16 | 54.64 | +0.0 | 0.1 |
| Internal 0-Tok | Meso-Heur-Best (Pure) | 0 | 0.00% | 74.44 | 73.86 | 72.41 | 54.40 | +0.0 | 0.7 |
| LLM Paper 5 | ClusterLLM-NoFT (EMNLP'23) | 39328 | 1.01% | 57.10 | 71.20 | 48.34 | 84.19 | -1899.8 | 521.5 |
| LLM Paper 6 | Few-Shot LLM-Post (TACL'24) | 39032 | 2.48% | 68.68 | 71.46 | 59.97 | 54.76 | +359.8 | 138.3 |
| LLM Paper 7a | Cequel-Edge (arXiv'25) | 39407 | 2.25% | 68.50 | 71.36 | 59.85 | 54.75 | +69.5 | 767.6 |
| LLM Paper 7b | Cequel-Transitive (arXiv'25) | 39407 | 2.25% | 66.94 | 71.44 | 61.07 | 54.39 | -148.2 | 767.6 |
| LLM Paper 8 | LSED-Budgeted (ACL'25) | 39599 | 2.27% | 68.55 | 70.94 | 59.52 | 52.62 | -14.2 | 972.2 |
| LLM Paper 9 | RagSEDE-Budgeted (WWW'26) | 38163 | 3.20% | 68.54 | 71.39 | 59.88 | 54.75 | -238.2 | 4.7 |
| LLM Paper 10 | LHG-Budgeted (WWW'26) | 39435 | 3.20% | 68.66 | 72.48 | 60.03 | 59.55 | +63.5 | 311.3 |
| Internal Ctrl | Random (3-seed mean) | 39302 | 2.52% | 68.54 | 71.41 | 59.89 | 54.76 | +58.9 | 0.0 |
| Internal Ctrl | Random-Meso (3-seed mean) | 38272 | 1.96% | 75.85 | 75.99 | 74.60 | 61.54 | +1175.7 | 0.0 |
| **OUR METHOD** | **Pure-Star-Meso (Qwen3-8B)** | 38297 | 1.92% | 78.80 | 79.05 | 81.01 | 68.70 | +1358.2 | 3.7 |
| Plug-and-Play | Star-Meso on HiSEvent Default | 38623 | 1.82% | 77.91 | 74.13 | 79.28 | 51.22 | +1465.5 | 3.8 |

*Note: `ClusterLLM-NoFT (EMNLP'23)` omits contrastive encoder fine-tuning under the zero-training budgeted protocol and computes `UnqNet` relative to its 0-LLM hierarchical base `SBERT-Agglom (0.60)`. `HyperSED` caps `n_anchors <= 360`.*

### Event2018 Pillar Ablations (`Qwen/Qwen3-8B`)

| Variant | Calls | Tok | AMI | Mac-$B^3$ | Tail-$B^3$ | UnqNet |
|---|---:|---:|---:|---:|---:|---:|
| **Full Pure-Star-Meso** | 49 | 38297 | 78.80 | 79.05 | 68.70 | +1358.2 |
| w/o AutoMerge (>=0.85) | 49 | 38543 | 76.97 | 77.84 | 68.83 | +1431.2 |
| w/o Bayes Band & AutoMerge | 46 | 37324 | 77.28 | 77.93 | 68.82 | +1594.0 |
| w/o Star Packing (k=1 Pair) | 139 | 39394 | 72.44 | 75.93 | 68.77 | +900.5 |
| w/o Guarded UF | 49 | 38399 | 80.06 | 78.10 | 68.70 | +1584.8 |
| w/o Seam Split | 47 | 38563 | 78.73 | 73.85 | 50.34 | +1528.2 |

### Event2018 Dev Cross-Day Frontier (`M_00..M_05`, eval `M_02..M_05`)

| Mode | G-AMI | G-Mac$B^3$ | G-Mic$B^3$ | G-Tail$B^3$ | Q |
|---|---:|---:|---:|---:|---:|
| no_link | 71.77 | 72.10 | 56.63 | 72.85 | 0 |
| proto_0.60 | 78.31 | 64.99 | 75.90 | 62.70 | 0 |
| proto_0.65 | 79.82 | 72.60 | 77.41 | 69.84 | 0 |
| proto_0.72 | 79.85 | 75.57 | 77.57 | 75.85 | 0 |
| proto_0.78 | 80.44 | 76.36 | 78.44 | 75.85 | 0 |
| proto_0.80 | 80.44 | 76.36 | 78.44 | 75.85 | 0 |
| proto_0.82 | 80.14 | 77.13 | 78.06 | 75.85 | 0 |
| proto_0.85 | 80.07 | 76.63 | 77.96 | 75.85 | 0 |
| proto_0.88 | 79.28 | 78.49 | 77.07 | 75.85 | 0 |
| proto_0.90 | 78.98 | 78.49 | 76.68 | 75.85 | 0 |
| proto_0.92 | 76.01 | 77.85 | 68.93 | 75.86 | 0 |
| proto_0.95 | 74.06 | 76.55 | 64.43 | 75.86 | 0 |
| proto_attr_0.80 | 79.98 | 77.43 | 77.97 | 75.85 | 0 |
| proto_attr_0.85 | 80.04 | 77.94 | 78.02 | 75.85 | 0 |
| proto_attr_0.90 | 77.07 | 77.58 | 72.35 | 75.87 | 0 |
| proto_attr_0.95 | 75.66 | 78.16 | 66.87 | 75.87 | 0 |
| sug_cross_star | 80.39 | 80.15 | 78.42 | 75.85 | 40 |
| oracle_band | 80.73 | 81.62 | 78.79 | 75.85 | 40 |

## Cross-Model Dev Comparison (`M_02..M_05`)

| Model | Dataset | Method | Calls | Tok | Show% | Err | AMI | Mac-$B^3$ | Mic-$B^3$ | Tail-$B^3$ | UnqNet |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen2.5-7B-Instruct | Event2012 | clusterllm_emnlp23 | 202 | 39338 | 2.38% | 0 | 87.83 | 75.90 | 84.10 | 66.06 | -611.0 |
| Qwen/Qwen2.5-7B-Instruct | Event2012 | fewshot_post_tacl24 | 74 | 39348 | 5.30% | 0 | 88.88 | 78.17 | 82.18 | 60.69 | +0.0 |
| Qwen/Qwen2.5-7B-Instruct | Event2012 | cequel_edge_25 | 173 | 34574 | 3.79% | 0 | 88.82 | 78.19 | 82.10 | 60.70 | -69.5 |
| Qwen/Qwen2.5-7B-Instruct | Event2012 | cequel_trans_25 | 173 | 34574 | 3.79% | 0 | 91.57 | 79.10 | 87.94 | 60.51 | +411.2 |
| Qwen/Qwen2.5-7B-Instruct | Event2012 | lsed_acl25 | 339 | 39510 | 4.73% | 0 | 88.52 | 78.00 | 81.54 | 60.69 | +11.0 |
| Qwen/Qwen2.5-7B-Instruct | Event2012 | ragsede_www26 | 60 | 38265 | 6.72% | 0 | 88.87 | 78.18 | 82.17 | 60.69 | -23.8 |
| Qwen/Qwen2.5-7B-Instruct | Event2012 | lhg_www26 | 174 | 39319 | 4.71% | 0 | 88.62 | 78.15 | 81.89 | 60.69 | -97.0 |
| Qwen/Qwen2.5-7B-Instruct | Event2012 | ctrl_random_meso_s42 | 70 | 33442 | 3.32% | 0 | 93.49 | 79.95 | 91.66 | 61.06 | +678.0 |
| Qwen/Qwen2.5-7B-Instruct | Event2012 | our_pure_star_meso | 66 | 32820 | 3.36% | 0 | 94.08 | 81.40 | 92.28 | 63.41 | +723.2 |
| Qwen/Qwen2.5-7B-Instruct | Event2018 | clusterllm_emnlp23 | 162 | 39454 | 1.02% | 0 | 58.20 | 72.42 | 51.49 | 82.81 | -1573.8 |
| Qwen/Qwen2.5-7B-Instruct | Event2018 | fewshot_post_tacl24 | 50 | 38649 | 2.48% | 0 | 68.66 | 71.45 | 59.97 | 54.76 | +407.8 |
| Qwen/Qwen2.5-7B-Instruct | Event2018 | cequel_edge_25 | 154 | 39346 | 2.33% | 0 | 68.47 | 71.37 | 59.85 | 54.76 | +19.0 |
| Qwen/Qwen2.5-7B-Instruct | Event2018 | cequel_trans_25 | 154 | 39346 | 2.33% | 0 | 64.76 | 70.52 | 59.28 | 53.90 | -390.8 |
| Qwen/Qwen2.5-7B-Instruct | Event2018 | lsed_acl25 | 278 | 39407 | 2.38% | 0 | 68.65 | 71.64 | 59.84 | 55.08 | +106.8 |
| Qwen/Qwen2.5-7B-Instruct | Event2018 | ragsede_www26 | 41 | 37864 | 3.20% | 0 | 68.55 | 71.40 | 59.90 | 54.76 | +8.8 |
| Qwen/Qwen2.5-7B-Instruct | Event2018 | lhg_www26 | 141 | 39341 | 3.26% | 0 | 68.44 | 72.08 | 59.28 | 59.22 | -255.8 |
| Qwen/Qwen2.5-7B-Instruct | Event2018 | ctrl_random_meso_s42 | 52 | 37979 | 1.99% | 0 | 71.54 | 73.39 | 65.52 | 59.96 | +618.8 |
| Qwen/Qwen2.5-7B-Instruct | Event2018 | our_pure_star_meso | 50 | 37858 | 1.87% | 0 | 75.61 | 76.29 | 76.17 | 60.51 | +1436.8 |