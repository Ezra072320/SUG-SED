# IP&M Stage 2A Causal Qwen/Qwen3-8B Experiment Report
- Generated: `2026-11-24 15:18:08` | Model: `Qwen/Qwen3-8B`
- Ledger Totals across Stage 2A: `{'calls': 18588, 'live': 19, 'tokens': 9909140, 'errors': 0, 'all_journal_errors': 1}`

---

## Event2012
### S2-1A: Dev-Only Prompt Calibration (`M_02..M_05`, `CausalUniformBucket(10000)`)
- `dev_sug_meso_extract`: AMI=0.9165, Mac-B³=0.7072, Mic-B³=0.8826, Tail-B³=0.3315 | Merge={'TP': 15, 'FP': 0, 'TN': 28, 'FN': 10, 'precision': 1.0, 'recall': 0.6, 'specificity': 1.0} | Seam={'TP': 3, 'FP': 5, 'TN': 10, 'FN': 0, 'precision': 0.375, 'recall': 1.0, 'specificity': 0.6667}
- `dev_sug_meso_action`: AMI=0.9367, Mac-B³=0.7158, Mic-B³=0.9187, Tail-B³=0.3264 | Merge={'TP': 24, 'FP': 1, 'TN': 37, 'FN': 3, 'precision': 0.96, 'recall': 0.8889, 'specificity': 0.9737} | Seam={'TP': 3, 'FP': 4, 'TN': 11, 'FN': 0, 'precision': 0.4286, 'recall': 1.0, 'specificity': 0.7333}
- **Selected Dev-Best Schema**: `action_first`

### S2-1B / S2-2 / S2-6: Causal Held-Out Results (`10k/blk`, `all_shown` uncontaminated $U_t$)
| Variant | Prompt | Ctrl | Calls (Shown%) | AMI | Mac-B³ | Tail-B³ | Unq ΔMic | UnqNet/blk | Merge P/R/S | Seam P/R/S | Err |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | ---: |
| `sug_meso_causal_fixed_extract` | extract_compare | fixed | 294 (3.24%) | 0.8854 | 0.6904 | 0.4618 | +0.0760 | +522.1 | 0.97/0.52/0.97 | 0.59/0.89/0.50 | 0 |
| `sug_meso_causal_fixed_action` | action_first | fixed | 353 (3.78%) | 0.8976 | 0.6901 | 0.4602 | +0.0961 | +608.3 | 0.88/0.79/0.87 | 0.65/0.82/0.66 | 0 |
| `sug_meso_causal_carryover_extract` | extract_compare | carryover | 305 (3.36%) | 0.8856 | 0.6909 | 0.4621 | +0.0762 | +529.8 | 0.97/0.52/0.98 | 0.59/0.90/0.51 | 0 |
| `sug_meso_causal_carryover_action` | action_first | carryover | 366 (3.87%) | 0.8986 | 0.6907 | 0.4612 | +0.0970 | +616.5 | 0.89/0.78/0.88 | 0.65/0.83/0.67 | 0 |
| `sug_meso_causal_sliding_action` | action_first | sliding | 310 (3.44%) | 0.8926 | 0.6880 | 0.4605 | +0.0885 | +584.2 | 0.86/0.77/0.81 | 0.66/0.82/0.64 | 0 |
| `random_seed42` | extract_compare | fixed | 185 (3.66%) | 0.8390 | 0.6574 | 0.4118 | -0.0004 | -123.6 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0 |
| `random_seed100` | extract_compare | fixed | 187 (3.61%) | 0.8391 | 0.6590 | 0.4165 | -0.0002 | -80.1 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0 |
| `random_seed200` | extract_compare | fixed | 186 (3.80%) | 0.8392 | 0.6572 | 0.4125 | -0.0002 | +17.0 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0 |
| `causal_merge_only_action` | action_first | fixed | 343 (2.93%) | 0.8954 | 0.6610 | 0.4062 | +0.0942 | +585.9 | 0.85/0.75/0.88 | 0.00/0.00/0.00 | 0 |
| `causal_seam_only_action` | action_first | fixed | 103 (1.89%) | 0.8406 | 0.6817 | 0.4654 | -0.0015 | +38.8 | 0.00/0.00/0.00 | 0.65/0.81/0.68 | 0 |
| `causal_fiedler_seam_action` | action_first | fixed | 352 (3.63%) | 0.8963 | 0.6778 | 0.4358 | +0.0950 | +585.1 | 0.87/0.77/0.87 | 0.62/0.84/0.72 | 0 |
| `causal_fiedler_seam_extract` | extract_compare | fixed | 294 (3.16%) | 0.8854 | 0.6817 | 0.4386 | +0.0762 | +527.9 | 0.97/0.51/0.98 | 0.55/0.92/0.56 | 0 |
| `random_meso_extract_s42` | extract_compare | fixed | 290 (3.74%) | 0.8598 | 0.6829 | 0.4617 | +0.0282 | +235.8 | 0.93/0.37/0.98 | 0.61/0.87/0.57 | 0 |
| `random_meso_extract_s100` | extract_compare | fixed | 289 (3.62%) | 0.8614 | 0.6829 | 0.4613 | +0.0324 | +286.0 | 0.97/0.35/0.99 | 0.58/0.89/0.56 | 0 |
| `random_meso_extract_s200` | extract_compare | fixed | 290 (3.74%) | 0.8618 | 0.6865 | 0.4611 | +0.0332 | +256.3 | 0.97/0.36/0.99 | 0.59/0.87/0.52 | 0 |
| `random_meso_action_s42` | action_first | fixed | 347 (4.14%) | 0.8718 | 0.6811 | 0.4567 | +0.0531 | +323.3 | 0.80/0.61/0.91 | 0.66/0.79/0.70 | 0 |
| `random_meso_action_s100` | action_first | fixed | 349 (4.01%) | 0.8805 | 0.6892 | 0.4614 | +0.0648 | +478.9 | 0.87/0.64/0.93 | 0.65/0.79/0.69 | 0 |
| `random_meso_action_s200` | action_first | fixed | 350 (4.14%) | 0.8789 | 0.6877 | 0.4621 | +0.0616 | +473.5 | 0.88/0.69/0.94 | 0.65/0.80/0.65 | 0 |
| `eq5_no_leverage_extract` | extract_compare | fixed | 291 (3.39%) | 0.8869 | 0.6927 | 0.4618 | +0.0772 | +507.1 | 0.99/0.53/0.98 | 0.59/0.89/0.50 | 0 |
| `eq5_cosine_only_extract` | extract_compare | fixed | 292 (3.40%) | 0.8856 | 0.6925 | 0.4618 | +0.0757 | +479.0 | 0.99/0.52/0.98 | 0.59/0.89/0.50 | 0 |
| `eq6_no_conductance_extract` | extract_compare | fixed | 296 (3.44%) | 0.8862 | 0.7034 | 0.4890 | +0.0779 | +547.1 | 0.97/0.55/0.97 | 0.68/0.93/0.51 | 0 |
| `eq6_no_non_entity_extract` | extract_compare | fixed | 294 (3.24%) | 0.8858 | 0.6929 | 0.4621 | +0.0763 | +538.1 | 0.97/0.52/0.97 | 0.63/0.90/0.54 | 0 |
| `eq6_raw_tfidf_extract` | extract_compare | fixed | 294 (3.23%) | 0.8857 | 0.6929 | 0.4621 | +0.0762 | +533.7 | 0.97/0.52/0.97 | 0.62/0.90/0.52 | 0 |
| `eq5_no_leverage_action` | action_first | fixed | 354 (3.79%) | 0.9029 | 0.6947 | 0.4605 | +0.1032 | +655.1 | 0.92/0.81/0.90 | 0.65/0.82/0.66 | 0 |
| `eq5_cosine_only_action` | action_first | fixed | 353 (3.84%) | 0.9001 | 0.6943 | 0.4606 | +0.1004 | +639.8 | 0.92/0.80/0.90 | 0.65/0.82/0.66 | 0 |
| `eq6_no_conductance_action` | action_first | fixed | 358 (4.04%) | 0.9001 | 0.7023 | 0.4872 | +0.1003 | +636.9 | 0.90/0.78/0.86 | 0.67/0.81/0.64 | 0 |
| `eq6_no_non_entity_action` | action_first | fixed | 353 (3.78%) | 0.8976 | 0.6901 | 0.4602 | +0.0961 | +608.2 | 0.88/0.79/0.87 | 0.65/0.82/0.66 | 0 |
| `eq6_raw_tfidf_action` | action_first | fixed | 353 (3.78%) | 0.8977 | 0.6907 | 0.4602 | +0.0961 | +609.1 | 0.88/0.79/0.87 | 0.66/0.82/0.68 | 0 |
| `sens_ratio_80_20_action` | action_first | fixed | 352 (3.52%) | 0.8974 | 0.6904 | 0.4596 | +0.0958 | +607.1 | 0.88/0.77/0.88 | 0.72/0.82/0.64 | 0 |
| `sens_ratio_50_50_action` | action_first | fixed | 352 (3.84%) | 0.8976 | 0.6916 | 0.4630 | +0.0963 | +614.1 | 0.90/0.81/0.89 | 0.65/0.83/0.67 | 0 |
| `sens_cond_035_action` | action_first | fixed | 350 (3.36%) | 0.8954 | 0.6807 | 0.4405 | +0.0934 | +570.3 | 0.87/0.77/0.88 | 0.73/0.88/0.71 | 0 |
| `sens_cond_065_action` | action_first | fixed | 355 (4.03%) | 0.8997 | 0.7005 | 0.4806 | +0.0989 | +604.9 | 0.90/0.79/0.86 | 0.65/0.86/0.66 | 0 |
| `sens_ratio_80_20_extract` | extract_compare | fixed | 296 (3.06%) | 0.8865 | 0.6918 | 0.4597 | +0.0772 | +541.8 | 0.97/0.49/0.98 | 0.70/0.91/0.52 | 0 |
| `sens_ratio_50_50_extract` | extract_compare | fixed | 294 (3.34%) | 0.8853 | 0.6928 | 0.4631 | +0.0758 | +529.6 | 0.99/0.54/0.99 | 0.60/0.90/0.55 | 0 |
| `sens_cond_035_extract` | extract_compare | fixed | 295 (2.99%) | 0.8856 | 0.6856 | 0.4420 | +0.0764 | +536.9 | 0.97/0.50/0.98 | 0.61/0.92/0.46 | 0 |
| `sens_cond_065_extract` | extract_compare | fixed | 294 (3.41%) | 0.8858 | 0.6993 | 0.4818 | +0.0765 | +527.0 | 0.97/0.54/0.97 | 0.62/0.91/0.52 | 0 |

### Paired Wilcoxon vs `sug_meso_causal_fixed_action` (Key Comparators)
- vs `init_2dse`: ΔAMI=+0.0581 (p=3.05e-05, H=1.25e-03, dz=+1.76) | ΔMac=+0.0334 (p=6.10e-05, H=2.50e-03, dz=+1.65)
- vs `heur_meso_0tok`: ΔAMI=+0.0141 (p=4.43e-02, H=1.00e+00, dz=+0.50) | ΔMac=+0.0215 (p=4.27e-04, H=1.50e-02, dz=+1.17)
- vs `cequel_transitive`: ΔAMI=+0.0440 (p=2.14e-04, H=7.05e-03, dz=+1.31) | ΔMac=+0.0490 (p=6.10e-05, H=2.50e-03, dz=+1.72)
- vs `random_mean3`: ΔAMI=+0.0585 (p=3.05e-05, H=1.25e-03, dz=+1.78) | ΔMac=+0.0322 (p=9.16e-05, H=3.39e-03, dz=+1.55)
- vs `random_meso_action_mean3`: ΔAMI=+0.0206 (p=8.05e-04, H=2.58e-02, dz=+1.02) | ΔMac=+0.0041 (p=1.75e-01, H=1.00e+00, dz=+0.40)
- vs `random_meso_extract_mean3`: ΔAMI=+0.0366 (p=1.53e-04, H=5.19e-03, dz=+1.31) | ΔMac=+0.0060 (p=1.75e-01, H=1.00e+00, dz=+0.39)
- vs `eq5_no_leverage_action`: ΔAMI=-0.0053 (p=1.14e-01, H=1.00e+00, dz=-0.38) | ΔMac=-0.0046 (p=1.39e-01, H=1.00e+00, dz=-0.43)
- vs `eq5_cosine_only_action`: ΔAMI=-0.0025 (p=3.86e-01, H=1.00e+00, dz=-0.17) | ΔMac=-0.0042 (p=2.41e-01, H=1.00e+00, dz=-0.39)
- vs `eq6_no_conductance_action`: ΔAMI=-0.0025 (p=8.65e-01, H=1.00e+00, dz=-0.32) | ΔMac=-0.0122 (p=1.46e-02, H=4.52e-01, dz=-0.79)
- vs `eq6_no_non_entity_action`: ΔAMI=+0.0000 (p=1.00e+00, H=1.00e+00, dz=+0.00) | ΔMac=+0.0000 (p=1.00e+00, H=1.00e+00, dz=+0.00)
- vs `eq6_raw_tfidf_action`: ΔAMI=-0.0001 (p=3.17e-01, H=1.00e+00, dz=-0.25) | ΔMac=-0.0006 (p=3.17e-01, H=1.00e+00, dz=-0.25)
- vs `eq5_cosine_only_extract`: ΔAMI=+0.0120 (p=9.62e-02, H=1.00e+00, dz=+0.48) | ΔMac=-0.0024 (p=6.83e-01, H=1.00e+00, dz=-0.13)
- vs `eq6_no_conductance_extract`: ΔAMI=+0.0114 (p=2.11e-01, H=1.00e+00, dz=+0.46) | ΔMac=-0.0133 (p=7.63e-03, H=2.44e-01, dz=-0.71)
- vs `eq6_raw_tfidf_extract`: ΔAMI=+0.0119 (p=1.09e-01, H=1.00e+00, dz=+0.50) | ΔMac=-0.0028 (p=4.33e-01, H=1.00e+00, dz=-0.19)

### Paired Wilcoxon vs `sug_meso_causal_fixed_extract` (Key Comparators)
- vs `init_2dse`: ΔAMI=+0.0458 (p=3.05e-05, H=1.25e-03, dz=+1.26) | ΔMac=+0.0337 (p=3.05e-05, H=1.25e-03, dz=+1.72)
- vs `heur_meso_0tok`: ΔAMI=+0.0018 (p=4.04e-01, H=1.00e+00, dz=+0.09) | ΔMac=+0.0218 (p=2.14e-04, H=7.69e-03, dz=+1.52)
- vs `cequel_transitive`: ΔAMI=+0.0317 (p=7.63e-04, H=2.67e-02, dz=+1.15) | ΔMac=+0.0492 (p=3.05e-05, H=1.25e-03, dz=+1.72)
- vs `random_mean3`: ΔAMI=+0.0463 (p=3.05e-05, H=1.25e-03, dz=+1.27) | ΔMac=+0.0325 (p=3.05e-05, H=1.25e-03, dz=+1.58)
- vs `random_meso_action_mean3`: ΔAMI=+0.0083 (p=1.25e-01, H=1.00e+00, dz=+0.46) | ΔMac=+0.0044 (p=1.05e-01, H=1.00e+00, dz=+0.47)
- vs `random_meso_extract_mean3`: ΔAMI=+0.0244 (p=8.05e-04, H=2.74e-02, dz=+0.98) | ΔMac=+0.0063 (p=1.06e-02, H=3.28e-01, dz=+0.67)
- vs `eq5_no_leverage_action`: ΔAMI=-0.0175 (p=1.10e-02, H=3.30e-01, dz=-0.73) | ΔMac=-0.0043 (p=4.33e-01, H=1.00e+00, dz=-0.26)
- vs `eq5_cosine_only_action`: ΔAMI=-0.0147 (p=3.03e-02, H=8.19e-01, dz=-0.56) | ΔMac=-0.0039 (p=4.70e-01, H=1.00e+00, dz=-0.23)
- vs `eq6_no_conductance_action`: ΔAMI=-0.0148 (p=2.90e-02, H=8.12e-01, dz=-0.66) | ΔMac=-0.0119 (p=4.43e-02, H=1.00e+00, dz=-0.62)
- vs `eq6_no_non_entity_action`: ΔAMI=-0.0123 (p=1.09e-01, H=1.00e+00, dz=-0.51) | ΔMac=+0.0003 (p=9.25e-01, H=1.00e+00, dz=+0.02)
- vs `eq6_raw_tfidf_action`: ΔAMI=-0.0123 (p=1.09e-01, H=1.00e+00, dz=-0.51) | ΔMac=-0.0003 (p=9.25e-01, H=1.00e+00, dz=-0.02)
- vs `eq5_cosine_only_extract`: ΔAMI=-0.0003 (p=3.45e-01, H=1.00e+00, dz=-0.02) | ΔMac=-0.0021 (p=2.77e-02, H=7.76e-01, dz=-0.40)
- vs `eq6_no_conductance_extract`: ΔAMI=-0.0008 (p=1.73e-01, H=1.00e+00, dz=-0.33) | ΔMac=-0.0130 (p=3.77e-03, H=1.24e-01, dz=-1.00)
- vs `eq6_raw_tfidf_extract`: ΔAMI=-0.0003 (p=1.80e-01, H=1.00e+00, dz=-0.33) | ΔMac=-0.0025 (p=1.80e-01, H=1.00e+00, dz=-0.30)

### S2-3: Upgraded Cross-Day Tracking (`[0.65, 0.85)`, `tau_fb=0.85`, causal 6k/blk carryover)

#### Base Stream: `dev_calibrated_stream`
| Mode | Global AMI | Global Mac-B³ | Global Mic-B³ | Tail-B³ | Link Stats | Confusion |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| `no_link` | 0.9137 | 0.6913 | 0.8252 | 0.3555 | {'auto_linked': 0, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 136} | None |
| `proto_0.72` | 0.9407 | 0.7209 | 0.8998 | 0.3360 | {'auto_linked': 56, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 80} | None |
| `proto_0.80` | 0.9430 | 0.7302 | 0.9045 | 0.3368 | {'auto_linked': 45, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 91} | None |
| `proto_0.85` | 0.9429 | 0.7363 | 0.9044 | 0.3368 | {'auto_linked': 40, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 96} | None |
| `proto_0.90` | 0.9443 | 0.7421 | 0.9045 | 0.3431 | {'auto_linked': 34, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 102} | None |
| `oracle_065_085` | 0.9443 | 0.7442 | 0.9073 | 0.3366 | {'auto_linked': 39, 'llm_queried': 25, 'llm_linked': 5, 'llm_rejected': 20, 'spawned_new': 92} | None |
| `sug_cross_v2_extract` | 0.9425 | 0.7276 | 0.9035 | 0.3368 | {'auto_linked': 40, 'llm_queried': 24, 'llm_linked': 2, 'llm_rejected': 22, 'spawned_new': 94} | {'TP': 1, 'FP': 1, 'TN': 18, 'FN': 4, 'precision': 0.5, 'recall': 0.2, 'specificity': 0.9474} |
| `sug_cross_v2_action` | 0.9430 | 0.7302 | 0.9045 | 0.3368 | {'auto_linked': 40, 'llm_queried': 24, 'llm_linked': 5, 'llm_rejected': 19, 'spawned_new': 91} | {'TP': 2, 'FP': 3, 'TN': 16, 'FN': 3, 'precision': 0.4, 'recall': 0.4, 'specificity': 0.8421} |

#### Base Stream: `heldout_extract_stream`
| Mode | Global AMI | Global Mac-B³ | Global Mic-B³ | Tail-B³ | Link Stats | Confusion |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| `no_link` | 0.8794 | 0.6680 | 0.7218 | 0.4105 | {'auto_linked': 0, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 641} | None |
| `proto_0.72` | 0.8793 | 0.6425 | 0.7610 | 0.3464 | {'auto_linked': 297, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 344} | None |
| `proto_0.80` | 0.8878 | 0.6921 | 0.7760 | 0.4025 | {'auto_linked': 222, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 419} | None |
| `proto_0.85` | 0.8926 | 0.7051 | 0.7803 | 0.4048 | {'auto_linked': 188, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 453} | None |
| `proto_0.90` | 0.8913 | 0.7059 | 0.7726 | 0.4088 | {'auto_linked': 141, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 500} | None |
| `oracle_065_085` | 0.9008 | 0.7195 | 0.7957 | 0.4022 | {'auto_linked': 188, 'llm_queried': 171, 'llm_linked': 45, 'llm_rejected': 126, 'spawned_new': 408} | None |
| `sug_cross_v2_extract` | 0.8934 | 0.7023 | 0.7820 | 0.4046 | {'auto_linked': 189, 'llm_queried': 153, 'llm_linked': 20, 'llm_rejected': 133, 'spawned_new': 432} | {'TP': 12, 'FP': 8, 'TN': 105, 'FN': 28, 'precision': 0.6, 'recall': 0.3, 'specificity': 0.9292} |
| `sug_cross_v2_action` | 0.8914 | 0.6968 | 0.7816 | 0.3883 | {'auto_linked': 186, 'llm_queried': 158, 'llm_linked': 60, 'llm_rejected': 98, 'spawned_new': 395} | {'TP': 25, 'FP': 35, 'TN': 82, 'FN': 16, 'precision': 0.4167, 'recall': 0.6098, 'specificity': 0.7009} |

#### Base Stream: `heldout_action_stream`
| Mode | Global AMI | Global Mac-B³ | Global Mic-B³ | Tail-B³ | Link Stats | Confusion |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| `no_link` | 0.8870 | 0.6634 | 0.7458 | 0.4101 | {'auto_linked': 0, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 593} | None |
| `proto_0.72` | 0.8820 | 0.6359 | 0.7705 | 0.3476 | {'auto_linked': 275, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 318} | None |
| `proto_0.80` | 0.8942 | 0.6823 | 0.7928 | 0.4036 | {'auto_linked': 217, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 376} | None |
| `proto_0.85` | 0.8989 | 0.6940 | 0.7995 | 0.4050 | {'auto_linked': 186, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 407} | None |
| `proto_0.90` | 0.8972 | 0.7035 | 0.7897 | 0.4070 | {'auto_linked': 135, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 458} | None |
| `oracle_065_085` | 0.9026 | 0.7048 | 0.8071 | 0.4028 | {'auto_linked': 185, 'llm_queried': 140, 'llm_linked': 33, 'llm_rejected': 107, 'spawned_new': 375} | None |
| `sug_cross_v2_extract` | 0.8994 | 0.6970 | 0.8007 | 0.4050 | {'auto_linked': 186, 'llm_queried': 124, 'llm_linked': 14, 'llm_rejected': 110, 'spawned_new': 393} | {'TP': 11, 'FP': 3, 'TN': 94, 'FN': 16, 'precision': 0.7857, 'recall': 0.4074, 'specificity': 0.9691} |
| `sug_cross_v2_action` | 0.8970 | 0.6879 | 0.7945 | 0.3890 | {'auto_linked': 184, 'llm_queried': 129, 'llm_linked': 47, 'llm_rejected': 82, 'spawned_new': 362} | {'TP': 23, 'FP': 24, 'TN': 73, 'FN': 9, 'precision': 0.4894, 'recall': 0.7188, 'specificity': 0.7526} |

---

## Event2018
### S2-1A: Dev-Only Prompt Calibration (`M_02..M_05`, `CausalUniformBucket(10000)`)
- `dev_sug_meso_extract`: AMI=0.7222, Mac-B³=0.6919, Mic-B³=0.6872, Tail-B³=0.4205 | Merge={'TP': 14, 'FP': 0, 'TN': 14, 'FN': 14, 'precision': 1.0, 'recall': 0.5, 'specificity': 1.0} | Seam={'TP': 7, 'FP': 8, 'TN': 3, 'FN': 0, 'precision': 0.4667, 'recall': 1.0, 'specificity': 0.2727}
- `dev_sug_meso_action`: AMI=0.7549, Mac-B³=0.7089, Mic-B³=0.7618, Tail-B³=0.4200 | Merge={'TP': 28, 'FP': 4, 'TN': 12, 'FN': 4, 'precision': 0.875, 'recall': 0.875, 'specificity': 0.75} | Seam={'TP': 7, 'FP': 6, 'TN': 8, 'FN': 0, 'precision': 0.5385, 'recall': 1.0, 'specificity': 0.5714}
- **Selected Dev-Best Schema**: `action_first`

### S2-1B / S2-2 / S2-6: Causal Held-Out Results (`10k/blk`, `all_shown` uncontaminated $U_t$)
| Variant | Prompt | Ctrl | Calls (Shown%) | AMI | Mac-B³ | Tail-B³ | Unq ΔMic | UnqNet/blk | Merge P/R/S | Seam P/R/S | Err |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | ---: |
| `sug_meso_causal_fixed_extract` | extract_compare | fixed | 169 (1.86%) | 0.8100 | 0.7129 | 0.4276 | +0.0946 | +887.3 | 1.00/0.51/1.00 | 0.54/0.95/0.50 | 0 |
| `sug_meso_causal_fixed_action` | action_first | fixed | 196 (2.06%) | 0.8349 | 0.7258 | 0.4201 | +0.1453 | +1220.3 | 0.96/0.93/0.95 | 0.65/0.74/0.78 | 0 |
| `sug_meso_causal_carryover_extract` | extract_compare | carryover | 174 (1.92%) | 0.8102 | 0.7151 | 0.4276 | +0.0949 | +892.0 | 1.00/0.49/1.00 | 0.55/0.95/0.53 | 0 |
| `sug_meso_causal_carryover_action` | action_first | carryover | 203 (2.12%) | 0.8354 | 0.7261 | 0.4199 | +0.1461 | +1240.8 | 0.96/0.92/0.96 | 0.65/0.74/0.78 | 0 |
| `sug_meso_causal_sliding_action` | action_first | sliding | 162 (1.79%) | 0.8336 | 0.7256 | 0.4208 | +0.1416 | +1206.8 | 0.95/0.90/0.93 | 0.73/0.80/0.81 | 0 |
| `random_seed42` | extract_compare | fixed | 99 (2.05%) | 0.7633 | 0.6795 | 0.3510 | -0.0001 | +7.0 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0 |
| `random_seed100` | extract_compare | fixed | 101 (2.03%) | 0.7632 | 0.6791 | 0.3501 | -0.0001 | -60.5 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0 |
| `random_seed200` | extract_compare | fixed | 98 (2.10%) | 0.7632 | 0.6798 | 0.3526 | -0.0001 | -74.9 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 0 |
| `causal_merge_only_action` | action_first | fixed | 192 (1.47%) | 0.8378 | 0.7093 | 0.3450 | +0.1498 | +1210.4 | 0.96/0.88/0.97 | 0.00/0.00/0.00 | 0 |
| `causal_seam_only_action` | action_first | fixed | 71 (1.28%) | 0.7641 | 0.6966 | 0.4220 | -0.0015 | +35.1 | 0.00/0.00/0.00 | 0.65/0.74/0.81 | 0 |
| `causal_fiedler_seam_action` | action_first | fixed | 195 (1.91%) | 0.8365 | 0.7183 | 0.3669 | +0.1479 | +1239.8 | 0.96/0.89/0.96 | 0.69/0.73/0.79 | 0 |
| `causal_fiedler_seam_extract` | extract_compare | fixed | 168 (1.72%) | 0.8116 | 0.7086 | 0.3745 | +0.0977 | +891.5 | 0.97/0.48/0.98 | 0.58/0.93/0.57 | 0 |
| `random_meso_extract_s42` | extract_compare | fixed | 168 (2.23%) | 0.7675 | 0.6932 | 0.4153 | +0.0036 | +70.9 | 0.71/0.38/0.98 | 0.51/0.95/0.53 | 0 |
| `random_meso_extract_s100` | extract_compare | fixed | 165 (2.27%) | 0.7650 | 0.6958 | 0.4109 | -0.0011 | +124.3 | 1.00/0.40/1.00 | 0.52/1.00/0.56 | 0 |
| `random_meso_extract_s200` | extract_compare | fixed | 165 (2.17%) | 0.7653 | 0.6959 | 0.4227 | +0.0008 | +75.0 | 1.00/0.22/1.00 | 0.59/1.00/0.62 | 0 |
| `random_meso_action_s42` | action_first | fixed | 195 (2.47%) | 0.7740 | 0.6980 | 0.4094 | +0.0181 | +337.5 | 0.87/0.65/0.98 | 0.64/0.70/0.80 | 0 |
| `random_meso_action_s100` | action_first | fixed | 193 (2.52%) | 0.7722 | 0.6949 | 0.3987 | +0.0152 | +286.3 | 0.87/0.72/0.98 | 0.59/0.68/0.79 | 0 |
| `random_meso_action_s200` | action_first | fixed | 193 (2.43%) | 0.7757 | 0.6996 | 0.4166 | +0.0151 | +434.7 | 1.00/0.80/1.00 | 0.70/0.76/0.82 | 0 |
| `eq5_no_leverage_extract` | extract_compare | fixed | 167 (1.98%) | 0.8115 | 0.7156 | 0.4278 | +0.0922 | +916.9 | 1.00/0.51/1.00 | 0.56/0.95/0.52 | 0 |
| `eq5_cosine_only_extract` | extract_compare | fixed | 166 (1.96%) | 0.8007 | 0.7121 | 0.4278 | +0.0725 | +809.7 | 1.00/0.50/1.00 | 0.56/0.95/0.50 | 0 |
| `eq6_no_conductance_extract` | extract_compare | fixed | 170 (1.86%) | 0.8099 | 0.7282 | 0.4734 | +0.0954 | +903.5 | 1.00/0.51/1.00 | 0.60/1.00/0.41 | 0 |
| `eq6_no_non_entity_extract` | extract_compare | fixed | 169 (1.86%) | 0.8102 | 0.7133 | 0.4285 | +0.0947 | +887.8 | 1.00/0.51/1.00 | 0.54/0.95/0.50 | 0 |
| `eq6_raw_tfidf_extract` | extract_compare | fixed | 169 (1.86%) | 0.8101 | 0.7133 | 0.4285 | +0.0946 | +887.5 | 1.00/0.51/1.00 | 0.54/0.95/0.50 | 0 |
| `eq5_no_leverage_action` | action_first | fixed | 197 (2.23%) | 0.8437 | 0.7283 | 0.4201 | +0.1528 | +1317.0 | 0.97/0.87/0.91 | 0.65/0.74/0.77 | 0 |
| `eq5_cosine_only_action` | action_first | fixed | 196 (2.20%) | 0.8399 | 0.7264 | 0.4204 | +0.1436 | +1181.5 | 0.97/0.86/0.95 | 0.68/0.74/0.79 | 0 |
| `eq6_no_conductance_action` | action_first | fixed | 198 (2.07%) | 0.8345 | 0.7381 | 0.4635 | +0.1450 | +1239.2 | 0.96/0.93/0.95 | 0.71/0.86/0.76 | 0 |
| `eq6_no_non_entity_action` | action_first | fixed | 195 (2.04%) | 0.8350 | 0.7261 | 0.4209 | +0.1453 | +1216.0 | 0.96/0.93/0.95 | 0.64/0.73/0.78 | 0 |
| `eq6_raw_tfidf_action` | action_first | fixed | 195 (2.04%) | 0.8349 | 0.7262 | 0.4209 | +0.1453 | +1220.2 | 0.96/0.93/0.95 | 0.65/0.74/0.77 | 0 |
| `sens_ratio_80_20_action` | action_first | fixed | 193 (1.83%) | 0.8377 | 0.7266 | 0.4199 | +0.1489 | +1258.5 | 0.96/0.90/0.96 | 0.75/0.75/0.77 | 0 |
| `sens_ratio_50_50_action` | action_first | fixed | 197 (2.14%) | 0.8334 | 0.7248 | 0.4210 | +0.1424 | +1185.5 | 0.95/0.93/0.95 | 0.65/0.74/0.81 | 0 |
| `sens_cond_035_action` | action_first | fixed | 195 (1.95%) | 0.8352 | 0.7165 | 0.3610 | +0.1456 | +1222.8 | 0.96/0.89/0.96 | 0.60/0.80/0.76 | 0 |
| `sens_cond_065_action` | action_first | fixed | 197 (2.11%) | 0.8350 | 0.7284 | 0.4334 | +0.1454 | +1221.5 | 0.96/0.93/0.95 | 0.63/0.79/0.74 | 0 |
| `sens_ratio_80_20_extract` | extract_compare | fixed | 168 (1.61%) | 0.8119 | 0.7136 | 0.4037 | +0.0982 | +904.1 | 0.97/0.49/0.98 | 0.63/1.00/0.33 | 0 |
| `sens_ratio_50_50_extract` | extract_compare | fixed | 169 (2.02%) | 0.8078 | 0.7144 | 0.4279 | +0.0919 | +798.9 | 1.00/0.51/1.00 | 0.51/0.96/0.55 | 0 |
| `sens_cond_035_extract` | extract_compare | fixed | 168 (1.75%) | 0.8100 | 0.7073 | 0.3709 | +0.0945 | +857.3 | 0.97/0.49/0.98 | 0.50/1.00/0.52 | 0 |
| `sens_cond_065_extract` | extract_compare | fixed | 169 (1.89%) | 0.8103 | 0.7163 | 0.4432 | +0.0949 | +882.7 | 1.00/0.51/1.00 | 0.53/0.95/0.47 | 0 |

### Paired Wilcoxon vs `sug_meso_causal_fixed_action` (Key Comparators)
- vs `init_2dse`: ΔAMI=+0.0715 (p=9.77e-04, H=4.00e-02, dz=+1.76) | ΔMac=+0.0468 (p=9.77e-04, H=4.00e-02, dz=+2.95)
- vs `heur_meso_0tok`: ΔAMI=+0.0200 (p=5.37e-02, H=6.98e-01, dz=+0.59) | ΔMac=+0.0172 (p=4.20e-02, H=5.88e-01, dz=+0.81)
- vs `cequel_transitive`: ΔAMI=+0.0632 (p=9.77e-04, H=4.00e-02, dz=+1.50) | ΔMac=+0.0550 (p=9.77e-04, H=4.00e-02, dz=+1.93)
- vs `random_mean3`: ΔAMI=+0.0717 (p=9.77e-04, H=4.00e-02, dz=+1.76) | ΔMac=+0.0464 (p=9.77e-04, H=4.00e-02, dz=+2.97)
- vs `random_meso_action_mean3`: ΔAMI=+0.0609 (p=9.77e-04, H=4.00e-02, dz=+1.77) | ΔMac=+0.0283 (p=9.77e-04, H=4.00e-02, dz=+2.14)
- vs `random_meso_extract_mean3`: ΔAMI=+0.0690 (p=9.77e-04, H=4.00e-02, dz=+1.72) | ΔMac=+0.0309 (p=9.77e-04, H=4.00e-02, dz=+2.97)
- vs `eq5_no_leverage_action`: ΔAMI=-0.0088 (p=1.29e-01, H=1.00e+00, dz=-0.51) | ΔMac=-0.0025 (p=3.59e-01, H=1.00e+00, dz=-0.32)
- vs `eq5_cosine_only_action`: ΔAMI=-0.0050 (p=3.01e-01, H=1.00e+00, dz=-0.35) | ΔMac=-0.0006 (p=6.52e-01, H=1.00e+00, dz=-0.12)
- vs `eq6_no_conductance_action`: ΔAMI=+0.0004 (p=9.22e-01, H=1.00e+00, dz=+0.19) | ΔMac=-0.0122 (p=3.71e-02, H=5.80e-01, dz=-0.74)
- vs `eq6_no_non_entity_action`: ΔAMI=-0.0001 (p=2.50e-01, H=1.00e+00, dz=-0.49) | ΔMac=-0.0003 (p=2.50e-01, H=1.00e+00, dz=-0.33)
- vs `eq6_raw_tfidf_action`: ΔAMI=-0.0000 (p=6.25e-01, H=1.00e+00, dz=-0.15) | ΔMac=-0.0003 (p=3.75e-01, H=1.00e+00, dz=-0.35)
- vs `eq5_cosine_only_extract`: ΔAMI=+0.0342 (p=1.37e-02, H=2.32e-01, dz=+1.06) | ΔMac=+0.0137 (p=1.37e-02, H=3.28e-01, dz=+0.90)
- vs `eq6_no_conductance_extract`: ΔAMI=+0.0250 (p=4.88e-03, H=1.27e-01, dz=+1.14) | ΔMac=-0.0024 (p=5.20e-01, H=1.00e+00, dz=-0.15)
- vs `eq6_raw_tfidf_extract`: ΔAMI=+0.0248 (p=4.88e-03, H=1.27e-01, dz=+1.04) | ΔMac=+0.0126 (p=2.44e-02, H=5.13e-01, dz=+0.75)

### Paired Wilcoxon vs `sug_meso_causal_fixed_extract` (Key Comparators)
- vs `init_2dse`: ΔAMI=+0.0467 (p=9.77e-04, H=4.00e-02, dz=+1.24) | ΔMac=+0.0339 (p=9.77e-04, H=4.00e-02, dz=+1.57)
- vs `heur_meso_0tok`: ΔAMI=-0.0049 (p=3.65e-01, H=1.00e+00, dz=-0.11) | ΔMac=+0.0043 (p=8.31e-01, H=1.00e+00, dz=+0.19)
- vs `cequel_transitive`: ΔAMI=+0.0384 (p=9.77e-03, H=1.56e-01, dz=+0.87) | ΔMac=+0.0421 (p=2.93e-03, H=1.05e-01, dz=+1.20)
- vs `random_mean3`: ΔAMI=+0.0468 (p=9.77e-04, H=4.00e-02, dz=+1.24) | ΔMac=+0.0335 (p=9.77e-04, H=4.00e-02, dz=+1.57)
- vs `random_meso_action_mean3`: ΔAMI=+0.0361 (p=9.77e-04, H=4.00e-02, dz=+1.12) | ΔMac=+0.0154 (p=6.74e-02, H=1.00e+00, dz=+0.71)
- vs `random_meso_extract_mean3`: ΔAMI=+0.0441 (p=9.77e-04, H=4.00e-02, dz=+1.15) | ΔMac=+0.0180 (p=1.86e-02, H=5.20e-01, dz=+1.07)
- vs `eq5_no_leverage_action`: ΔAMI=-0.0337 (p=1.95e-03, H=6.05e-02, dz=-2.00) | ΔMac=-0.0154 (p=2.93e-03, H=1.05e-01, dz=-1.06)
- vs `eq5_cosine_only_action`: ΔAMI=-0.0299 (p=4.88e-03, H=1.27e-01, dz=-1.27) | ΔMac=-0.0135 (p=3.22e-02, H=6.77e-01, dz=-0.86)
- vs `eq6_no_conductance_action`: ΔAMI=-0.0245 (p=9.77e-03, H=1.56e-01, dz=-1.02) | ΔMac=-0.0251 (p=6.84e-03, H=2.26e-01, dz=-1.07)
- vs `eq6_no_non_entity_action`: ΔAMI=-0.0249 (p=4.88e-03, H=1.27e-01, dz=-1.05) | ΔMac=-0.0132 (p=2.44e-02, H=6.10e-01, dz=-0.82)
- vs `eq6_raw_tfidf_action`: ΔAMI=-0.0249 (p=4.88e-03, H=1.27e-01, dz=-1.05) | ΔMac=-0.0132 (p=2.44e-02, H=6.10e-01, dz=-0.82)
- vs `eq5_cosine_only_extract`: ΔAMI=+0.0093 (p=8.44e-01, H=1.00e+00, dz=+0.33) | ΔMac=+0.0009 (p=9.45e-01, H=1.00e+00, dz=+0.09)
- vs `eq6_no_conductance_extract`: ΔAMI=+0.0001 (p=7.65e-01, H=1.00e+00, dz=+0.05) | ΔMac=-0.0153 (p=1.86e-02, H=5.20e-01, dz=-0.86)
- vs `eq6_raw_tfidf_extract`: ΔAMI=-0.0000 (p=6.25e-01, H=1.00e+00, dz=-0.18) | ΔMac=-0.0003 (p=3.75e-01, H=1.00e+00, dz=-0.35)

### S2-3: Upgraded Cross-Day Tracking (`[0.65, 0.85)`, `tau_fb=0.85`, causal 6k/blk carryover)

#### Base Stream: `dev_calibrated_stream`
| Mode | Global AMI | Global Mac-B³ | Global Mic-B³ | Tail-B³ | Link Stats | Confusion |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| `no_link` | 0.6960 | 0.6598 | 0.5192 | 0.5502 | {'auto_linked': 0, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 111} | None |
| `proto_0.72` | 0.7904 | 0.6906 | 0.7648 | 0.5219 | {'auto_linked': 68, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 43} | None |
| `proto_0.80` | 0.7772 | 0.6985 | 0.7490 | 0.5474 | {'auto_linked': 53, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 58} | None |
| `proto_0.85` | 0.7720 | 0.7177 | 0.7420 | 0.5485 | {'auto_linked': 45, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 66} | None |
| `proto_0.90` | 0.7231 | 0.7109 | 0.6223 | 0.5501 | {'auto_linked': 34, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 77} | None |
| `oracle_065_085` | 0.7959 | 0.7255 | 0.7721 | 0.5232 | {'auto_linked': 43, 'llm_queried': 37, 'llm_linked': 15, 'llm_rejected': 22, 'spawned_new': 53} | None |
| `sug_cross_v2_extract` | 0.7743 | 0.7204 | 0.7453 | 0.5482 | {'auto_linked': 43, 'llm_queried': 34, 'llm_linked': 5, 'llm_rejected': 29, 'spawned_new': 63} | {'TP': 4, 'FP': 1, 'TN': 20, 'FN': 9, 'precision': 0.8, 'recall': 0.3077, 'specificity': 0.9524} |
| `sug_cross_v2_action` | 0.7894 | 0.7123 | 0.7645 | 0.5482 | {'auto_linked': 43, 'llm_queried': 38, 'llm_linked': 12, 'llm_rejected': 26, 'spawned_new': 56} | {'TP': 8, 'FP': 4, 'TN': 19, 'FN': 7, 'precision': 0.6667, 'recall': 0.5333, 'specificity': 0.8261} |

#### Base Stream: `heldout_extract_stream`
| Mode | Global AMI | Global Mac-B³ | Global Mic-B³ | Tail-B³ | Link Stats | Confusion |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| `no_link` | 0.7820 | 0.5908 | 0.5241 | 0.3420 | {'auto_linked': 0, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 362} | None |
| `proto_0.72` | 0.8222 | 0.6336 | 0.7204 | 0.2796 | {'auto_linked': 239, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 123} | None |
| `proto_0.80` | 0.8257 | 0.6746 | 0.7195 | 0.3211 | {'auto_linked': 195, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 167} | None |
| `proto_0.85` | 0.8307 | 0.6844 | 0.7189 | 0.2987 | {'auto_linked': 163, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 199} | None |
| `proto_0.90` | 0.8262 | 0.6882 | 0.7021 | 0.3061 | {'auto_linked': 137, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 225} | None |
| `oracle_065_085` | 0.8386 | 0.6916 | 0.7356 | 0.2987 | {'auto_linked': 160, 'llm_queried': 135, 'llm_linked': 31, 'llm_rejected': 104, 'spawned_new': 171} | None |
| `sug_cross_v2_extract` | 0.8340 | 0.6875 | 0.7300 | 0.2987 | {'auto_linked': 162, 'llm_queried': 96, 'llm_linked': 5, 'llm_rejected': 91, 'spawned_new': 195} | {'TP': 3, 'FP': 2, 'TN': 78, 'FN': 13, 'precision': 0.6, 'recall': 0.1875, 'specificity': 0.975} |
| `sug_cross_v2_action` | 0.8427 | 0.6864 | 0.7562 | 0.2987 | {'auto_linked': 158, 'llm_queried': 110, 'llm_linked': 27, 'llm_rejected': 83, 'spawned_new': 177} | {'TP': 17, 'FP': 10, 'TN': 77, 'FN': 6, 'precision': 0.6296, 'recall': 0.7391, 'specificity': 0.8851} |

#### Base Stream: `heldout_action_stream`
| Mode | Global AMI | Global Mac-B³ | Global Mic-B³ | Tail-B³ | Link Stats | Confusion |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| `no_link` | 0.7976 | 0.6001 | 0.5727 | 0.3239 | {'auto_linked': 0, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 329} | None |
| `proto_0.72` | 0.8365 | 0.6305 | 0.7457 | 0.2516 | {'auto_linked': 218, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 111} | None |
| `proto_0.80` | 0.8392 | 0.6788 | 0.7525 | 0.3121 | {'auto_linked': 173, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 156} | None |
| `proto_0.85` | 0.8416 | 0.6906 | 0.7493 | 0.2862 | {'auto_linked': 145, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 184} | None |
| `proto_0.90` | 0.8363 | 0.6926 | 0.7354 | 0.2881 | {'auto_linked': 129, 'llm_queried': 0, 'llm_linked': 0, 'llm_rejected': 0, 'spawned_new': 200} | None |
| `oracle_065_085` | 0.8480 | 0.6943 | 0.7607 | 0.2861 | {'auto_linked': 142, 'llm_queried': 126, 'llm_linked': 27, 'llm_rejected': 99, 'spawned_new': 160} | None |
| `sug_cross_v2_extract` | 0.8460 | 0.6925 | 0.7576 | 0.2862 | {'auto_linked': 144, 'llm_queried': 98, 'llm_linked': 10, 'llm_rejected': 88, 'spawned_new': 175} | {'TP': 9, 'FP': 1, 'TN': 80, 'FN': 8, 'precision': 0.9, 'recall': 0.5294, 'specificity': 0.9877} |
| `sug_cross_v2_action` | 0.8485 | 0.6868 | 0.7644 | 0.2861 | {'auto_linked': 142, 'llm_queried': 112, 'llm_linked': 26, 'llm_rejected': 86, 'spawned_new': 161} | {'TP': 17, 'FP': 9, 'TN': 81, 'FN': 5, 'precision': 0.6538, 'recall': 0.7727, 'specificity': 0.9} |
