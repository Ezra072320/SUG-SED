# IP&M Stage 1 Decisive Zero-Token Diagnostic Report (D1–D6)
- **Ledger Grand Audit**: `{'calls': 6420, 'live': 2440, 'cached': 3980, 'errors': 6, 'tokens': 3921749}`

---

## Event2012 (`n_blocks_eval=16`, `N_eval=32,695`)
### D1: Corrected Prompt-Shown Post Accounting & Uncontaminated $U_t$
| Strategy | Calls (%) | Primary (%) | All-Shown (%) | Unq ΔMic-B³ (All-Shown) | UnqNet/blk (Legacy→All-Shown) | UnqNet/Call | AMI | Mac-B³ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `random_direct` | 187 (0.57%) | 187 (0.57%) | 1149 (3.51%) | -0.0003 | -173.9 -> -172.5 | -14.8 | 0.8392 | 0.6573 |
| `rep_div_direct` | 183 (0.56%) | 183 (0.56%) | 1316 (4.03%) | -0.0001 | -52.1 -> -51.6 | -4.5 | 0.8394 | 0.6567 |
| `cequel_edge_direct` | 377 (1.15%) | 696 (2.13%) | 696 (2.13%) | -0.0007 | -169.9 -> -169.9 | -7.2 | 0.8382 | 0.6549 |
| `cequel_edge_repair` | 377 (1.15%) | 696 (2.13%) | 696 (2.13%) | -0.0000 | +21.1 -> +21.1 | +0.9 | 0.8395 | 0.6565 |
| `cequel_transitive` | 377 (1.15%) | 696 (2.13%) | 696 (2.13%) | +0.0253 | +132.1 -> +132.1 | -7.2 | 0.8537 | 0.6411 |
| `sug_router_repair` | 261 (0.80%) | 328 (1.00%) | 1210 (3.70%) | +0.0579 | +474.4 -> +466.1 | +28.6 | 0.8736 | 0.6806 |
| `sug_meso_only_direct` | 300 (0.92%) | 374 (1.14%) | 1096 (3.35%) | +0.0701 | +534.5 -> +521.9 | +27.8 | 0.8811 | 0.6888 |
| `sug_meso_only_repair` | 300 (0.92%) | 374 (1.14%) | 1096 (3.35%) | +0.0700 | +543.0 -> +531.0 | +28.3 | 0.8810 | 0.6837 |

### D2: Cross-Block `ProtoOnly` Threshold Sweep vs. `SUG-CrossLLM` & Oracle
| Variant (`SUG-Meso Direct` stream) | Global AMI | Global Mac-B³ | Global Mic-B³ | Tail-B³ | AutoLinked / LLM / Spawned |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `no_link` | 0.8785 | 0.6644 | 0.7206 | 0.4105 | 0 / 0 / 637 |
| `ProtoOnly(τ=0.55)` | 0.8422 | 0.4923 | 0.6883 | 0.2651 | 439 / 0 / 198 |
| `ProtoOnly(τ=0.65)` | 0.8698 | 0.5891 | 0.7422 | 0.3013 | 360 / 0 / 277 |
| `ProtoOnly(τ=0.68)` | 0.8713 | 0.6064 | 0.7463 | 0.3410 | 337 / 0 / 300 |
| `ProtoOnly(τ=0.72)` | 0.8773 | 0.6364 | 0.7575 | 0.3476 | 298 / 0 / 339 |
| `ProtoOnly(τ=0.76)` | 0.8825 | 0.6627 | 0.7660 | 0.3863 | 262 / 0 / 375 |
| `ProtoOnly(τ=0.78)` | 0.8840 | 0.6798 | 0.7715 | 0.3862 | 237 / 0 / 400 |
| `ProtoOnly(τ=0.80)` | 0.8860 | 0.6861 | 0.7734 | 0.4033 | 227 / 0 / 410 |
| `ProtoOnly(τ=0.82)` | 0.8859 | 0.6914 | 0.7714 | 0.4041 | 207 / 0 / 430 |
| `ProtoOnly(τ=0.85)` | 0.8915 | 0.7023 | 0.7770 | 0.4053 | 189 / 0 / 448 |
| `ProtoOnly(τ=0.90)` | 0.8904 | 0.7033 | 0.7708 | 0.4072 | 145 / 0 / 492 |
| **`SUG-CrossLLM(0.55,0.80)`** | **0.8846** | **0.6732** | **0.7725** | 0.3883 | 268 / 80 (L=3,R=77) / 366 |
| `Oracle-Band[0.55,0.80) (Same Budget)` | 0.8936 | 0.6793 | 0.7926 | 0.3883 | 268 / 76 (L=7,R=69) / 362 |
| `Oracle-Band[0.55,0.80) (Full)` | 0.8951 | 0.6969 | 0.7975 | 0.4016 | 230 / 208 (L=37,R=171) / 370 |
| `Oracle-Band[0.40,0.85)` | 0.9002 | 0.7163 | 0.8028 | 0.4026 | 188 / 372 (L=63,R=309) / 386 |

- **CrossLLM Confusion Matrix**: `TP=1, FP=2, TN=70, FN=7` (Prec=0.333, Rec=0.125, F1=0.182)
- **Held-Out Similarity Bins (`size>=3`) True Match Rate**: `<0.55`: 0/114 (0.0%), `[0.55,0.60)`: 5/44 (11.4%), `[0.60,0.65)`: 6/37 (16.2%), `[0.65,0.70)`: 10/40 (25.0%), `[0.70,0.75)`: 9/46 (19.6%), `[0.75,0.80)`: 6/44 (13.6%), `[0.80,0.85)`: 13/39 (33.3%), `[0.85,1.00]`: 123/179 (68.7%)

### D3: Zero-LLM & Dev-Tuned Baselines (Held-Out Mean ± SD)
| Method | AMI (Mean ± SD) | Mac-B³ (Mean ± SD) | Mic-B³ | Tail-B³ | p_Wilcoxon vs SUG-Meso (Dir) [AMI / Mac-B³] |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Init 2D-SE (Default) | 0.8395 ± 0.0726 | 0.6567 ± 0.0455 | 0.7708 | 0.4103 | p=3.05e-05 (H=7.02e-04, dz=1.38) / p=3.05e-05 (H=7.02e-04, dz=1.79) |
| Vanilla 2D-SE (w/o Node-Moves) | 0.8395 ± 0.0706 | 0.6535 ± 0.0449 | 0.7723 | 0.4090 | p=3.05e-05 (H=7.02e-04, dz=1.47) / p=3.05e-05 (H=7.02e-04, dz=1.84) |
| Semantic-Only 2D-SE (α=0) | 0.8416 ± 0.0733 | 0.6482 ± 0.0449 | 0.7763 | 0.3898 | p=3.05e-05 (H=7.02e-04, dz=1.59) / p=9.16e-05 (H=1.19e-03, dz=1.67) |
| Dev-Tuned 2D-SE {'knn_k': 15, 'tau_sim': 0.45, 'attr_alpha': 0.5} | 0.8417 ± 0.0733 | 0.7368 ± 0.0445 | 0.7700 | 0.6092 | p=3.05e-05 (H=7.02e-04, dz=1.41) / p=1.01e-03 (H=8.06e-03, dz=-1.22) |
| Louvain (res=1.2) | 0.8446 ± 0.0699 | 0.5471 ± 0.0499 | 0.7891 | 0.2319 | p=9.16e-05 (H=1.19e-03, dz=1.19) / p=3.05e-05 (H=7.02e-04, dz=3.83) |
| Agglom-Cosine (d=0.55) | 0.8562 ± 0.0907 | 0.7652 ± 0.0463 | 0.8283 | 0.7091 | p=5.07e-02 (H=3.55e-01, dz=0.62) / p=2.14e-04 (H=2.56e-03, dz=-1.26) |
| Meso-Heur MergeOnly (τ=0.8) | 0.8836 ± 0.0722 | 0.6686 ± 0.0483 | 0.8438 | 0.4085 | p=1.00e+00 (H=1.00e+00, dz=-0.11) / p=5.80e-04 (H=5.80e-03, dz=1.22) |
| Meso-Heur 0-Tok (τ=0.8,none) | 0.8836 ± 0.0722 | 0.6686 ± 0.0483 | 0.8438 | 0.4085 | p=1.00e+00 (H=1.00e+00, dz=-0.11) / p=5.80e-04 (H=5.80e-03, dz=1.22) |
| Cequel-Transitive (LLM) | 0.8537 ± 0.0618 | 0.6411 ± 0.0534 | 0.7963 | 0.4037 | p=1.01e-03 (H=1.11e-02, dz=1.17) / p=3.05e-05 (H=7.02e-04, dz=1.92) |
| **SUG-Meso (Direct, LLM)** | 0.8811 ± 0.0670 | 0.6888 ± 0.0511 | 0.8417 | 0.4639 | Anchor |

### D5: Paired Wilcoxon + Holm-Bonferroni Tests (`SUG-Meso Direct` vs. LLM Baselines, Repair, & Ablations)
| Comparator | AMI (Mean ± SD) | Mac-B³ (Mean ± SD) | ΔAMI vs Anchor (p / p_Holm / dz) | ΔMac-B³ vs Anchor (p / p_Holm / dz) |
| :--- | ---: | ---: | :--- | :--- |
| `random_direct` | 0.8392 ± 0.0726 | 0.6573 ± 0.0454 | +0.0419 (p=3.05e-05, H=7.02e-04, dz=+1.40) | +0.0315 (p=3.05e-05, H=7.02e-04, dz=+1.74) |
| `rep_div_direct` | 0.8394 ± 0.0727 | 0.6567 ± 0.0455 | +0.0417 (p=3.05e-05, H=7.02e-04, dz=+1.38) | +0.0321 (p=3.05e-05, H=7.02e-04, dz=+1.81) |
| `cequel_edge_repair` | 0.8395 ± 0.0727 | 0.6565 ± 0.0455 | +0.0415 (p=3.05e-05, H=7.02e-04, dz=+1.38) | +0.0323 (p=3.05e-05, H=7.02e-04, dz=+1.83) |
| `cequel_transitive` | 0.8537 ± 0.0618 | 0.6411 ± 0.0534 | +0.0274 (p=1.01e-03, H=1.11e-02, dz=+1.17) | +0.0477 (p=3.05e-05, H=7.02e-04, dz=+1.92) |
| `sug_router_repair` | 0.8736 ± 0.0657 | 0.6806 ± 0.0470 | +0.0075 (p=2.14e-02, H=2.14e-01, dz=+0.54) | +0.0082 (p=1.05e-01, H=4.18e-01, dz=+0.46) |
| `sug_meso_only_repair` | 0.8810 ± 0.0671 | 0.6837 ± 0.0488 | +0.0001 (p=8.75e-01, H=1.00e+00, dz=+0.12) | +0.0051 (p=1.69e-01, H=5.06e-01, dz=+0.51) |
| `abl_merge_only` | 0.8843 ± 0.0684 | 0.6683 ± 0.0471 | -0.0032 (p=1.44e-01, H=8.63e-01, dz=-0.39) | +0.0205 (p=3.05e-04, H=3.36e-03, dz=+1.18) |
| `abl_seam_only` | 0.8395 ± 0.0732 | 0.6799 ± 0.0505 | +0.0416 (p=6.55e-04, H=7.86e-03, dz=+1.33) | +0.0089 (p=2.68e-02, H=1.62e-01, dz=+0.62) |
| `abl_fiedler_only_seam` | 0.8806 ± 0.0659 | 0.6787 ± 0.0520 | +0.0005 (p=9.55e-01, H=1.00e+00, dz=+0.07) | +0.0101 (p=2.31e-02, H=1.62e-01, dz=+0.67) |
| `abl_no_extract_compare` | 0.8933 ± 0.0631 | 0.6880 ± 0.0549 | -0.0122 (p=3.56e-02, H=2.85e-01, dz=-0.65) | +0.0008 (p=8.20e-01, H=8.20e-01, dz=+0.06) |
| `abl_uniform_budget` | 0.8854 ± 0.0708 | 0.6904 ± 0.0551 | -0.0043 (p=4.41e-01, H=1.00e+00, dz=-0.31) | -0.0016 (p=3.74e-01, H=7.49e-01, dz=-0.16) |

### D4: Dev vs. Held-Out Initial Error Pathology & Cross-Day Recurrence
- **dev**: N=7,170, FS/FM=11.60× (blk 2.66×–17.93×), H_bnd=0=43.2%, Impure(|C|>=10, pur<0.90)=16/112 (14.3%), minority share median=0.168 [IQR 0.129–0.303] (68.8% in [0.10,0.25]), Recurrence=37.71%.
- **heldout**: N=32,695, FS/FM=44.64× (blk 1.76×–187.95×), H_bnd=0=36.2%, Impure(|C|>=10, pur<0.90)=101/479 (21.1%), minority share median=0.231 [IQR 0.174–0.389] (56.4% in [0.10,0.25]), Recurrence=55.67%.

---

## Event2018 (`n_blocks_eval=11`, `N_eval=33,161`)
### D1: Corrected Prompt-Shown Post Accounting & Uncontaminated $U_t$
| Strategy | Calls (%) | Primary (%) | All-Shown (%) | Unq ΔMic-B³ (All-Shown) | UnqNet/blk (Legacy→All-Shown) | UnqNet/Call | AMI | Mac-B³ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `random_direct` | 100 (0.30%) | 100 (0.30%) | 724 (2.18%) | -0.0001 | +69.3 -> +68.3 | +7.5 | 0.7633 | 0.6795 |
| `rep_div_direct` | 98 (0.30%) | 98 (0.30%) | 776 (2.34%) | -0.0001 | -78.3 -> -79.1 | -8.9 | 0.7634 | 0.6790 |
| `cequel_edge_direct` | 227 (0.68%) | 449 (1.35%) | 449 (1.35%) | +0.0001 | +145.4 -> +145.4 | +7.0 | 0.7635 | 0.6797 |
| `cequel_edge_repair` | 227 (0.68%) | 449 (1.35%) | 449 (1.35%) | -0.0001 | -73.9 -> -73.9 | -3.6 | 0.7635 | 0.6791 |
| `cequel_transitive` | 227 (0.68%) | 449 (1.35%) | 449 (1.35%) | +0.0305 | +485.1 -> +485.1 | +7.0 | 0.7717 | 0.6708 |
| `sug_router_repair` | 136 (0.41%) | 174 (0.52%) | 624 (1.88%) | +0.0537 | +855.1 -> +844.0 | +68.3 | 0.7877 | 0.7041 |
| `sug_meso_only_direct` | 165 (0.50%) | 199 (0.60%) | 593 (1.79%) | +0.0746 | +803.3 -> +794.8 | +53.0 | 0.7978 | 0.7127 |
| `sug_meso_only_repair` | 165 (0.50%) | 199 (0.60%) | 593 (1.79%) | +0.0735 | +819.8 -> +812.0 | +54.1 | 0.7968 | 0.7095 |

### D2: Cross-Block `ProtoOnly` Threshold Sweep vs. `SUG-CrossLLM` & Oracle
| Variant (`SUG-Meso Direct` stream) | Global AMI | Global Mac-B³ | Global Mic-B³ | Tail-B³ | AutoLinked / LLM / Spawned |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `no_link` | 0.7798 | 0.5884 | 0.5135 | 0.3443 | 0 / 0 / 365 |
| `ProtoOnly(τ=0.55)` | 0.7840 | 0.4567 | 0.6538 | 0.1780 | 344 / 0 / 21 |
| `ProtoOnly(τ=0.65)` | 0.8058 | 0.5510 | 0.6935 | 0.1888 | 293 / 0 / 72 |
| `ProtoOnly(τ=0.68)` | 0.8167 | 0.5961 | 0.7146 | 0.2537 | 277 / 0 / 88 |
| `ProtoOnly(τ=0.72)` | 0.8165 | 0.6303 | 0.7091 | 0.2818 | 241 / 0 / 124 |
| `ProtoOnly(τ=0.76)` | 0.8226 | 0.6593 | 0.7156 | 0.3121 | 216 / 0 / 149 |
| `ProtoOnly(τ=0.78)` | 0.8202 | 0.6680 | 0.7142 | 0.3138 | 203 / 0 / 162 |
| `ProtoOnly(τ=0.80)` | 0.8259 | 0.6712 | 0.7189 | 0.3232 | 197 / 0 / 168 |
| `ProtoOnly(τ=0.82)` | 0.8267 | 0.6719 | 0.7184 | 0.2972 | 182 / 0 / 183 |
| `ProtoOnly(τ=0.85)` | 0.8267 | 0.6793 | 0.7114 | 0.2994 | 165 / 0 / 200 |
| `ProtoOnly(τ=0.90)` | 0.8235 | 0.6829 | 0.6948 | 0.3085 | 139 / 0 / 226 |
| **`SUG-CrossLLM(0.55,0.80)`** | **0.8218** | **0.6499** | **0.7187** | 0.2727 | 225 / 45 (L=1,R=44) / 139 |
| `Oracle-Band[0.55,0.80) (Same Budget)` | 0.8218 | 0.6476 | 0.7158 | 0.2593 | 221 / 54 (L=3,R=51) / 141 |
| `Oracle-Band[0.55,0.80) (Full)` | 0.8239 | 0.6751 | 0.7180 | 0.3229 | 191 / 156 (L=21,R=135) / 153 |
| `Oracle-Band[0.40,0.85)` | 0.8347 | 0.6886 | 0.7281 | 0.2990 | 159 / 206 (L=33,R=173) / 173 |

- **CrossLLM Confusion Matrix**: `TP=0, FP=1, TN=42, FN=2` (Prec=0.000, Rec=0.000, F1=0.000)
- **Held-Out Similarity Bins (`size>=3`) True Match Rate**: `<0.55`: 0/17 (0.0%), `[0.55,0.60)`: 0/22 (0.0%), `[0.60,0.65)`: 3/28 (10.7%), `[0.65,0.70)`: 4/39 (10.3%), `[0.70,0.75)`: 6/33 (18.2%), `[0.75,0.80)`: 6/34 (17.6%), `[0.80,0.85)`: 13/38 (34.2%), `[0.85,1.00]`: 115/152 (75.7%)

### D3: Zero-LLM & Dev-Tuned Baselines (Held-Out Mean ± SD)
| Method | AMI (Mean ± SD) | Mac-B³ (Mean ± SD) | Mic-B³ | Tail-B³ | p_Wilcoxon vs SUG-Meso (Dir) [AMI / Mac-B³] |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Init 2D-SE (Default) | 0.7634 ± 0.0568 | 0.6790 ± 0.0916 | 0.6874 | 0.3500 | p=9.77e-04 (H=2.25e-02, dz=1.55) / p=1.95e-03 (H=4.10e-02, dz=1.41) |
| Vanilla 2D-SE (w/o Node-Moves) | 0.7671 ± 0.0575 | 0.6768 ± 0.0876 | 0.6989 | 0.3526 | p=9.77e-04 (H=2.25e-02, dz=1.44) / p=9.77e-04 (H=2.25e-02, dz=1.58) |
| Semantic-Only 2D-SE (α=0) | 0.7566 ± 0.0576 | 0.6286 ± 0.0627 | 0.6961 | 0.2176 | p=9.77e-04 (H=2.25e-02, dz=1.84) / p=1.95e-03 (H=4.10e-02, dz=1.61) |
| Dev-Tuned 2D-SE {'knn_k': 15, 'tau_sim': 0.45, 'attr_alpha': 0.8} | 0.7564 ± 0.0476 | 0.7000 ± 0.0819 | 0.6746 | 0.4376 | p=9.77e-04 (H=2.25e-02, dz=1.85) / p=3.20e-01 (H=1.00e+00, dz=0.31) |
| Louvain (res=0.5) | 0.8111 ± 0.0383 | 0.5361 ± 0.0646 | 0.8041 | 0.0789 | p=1.02e-01 (H=5.08e-01, dz=-0.60) / p=9.77e-04 (H=2.25e-02, dz=3.12) |
| Agglom-Cosine (d=0.55) | 0.6990 ± 0.0776 | 0.7301 ± 0.0651 | 0.6613 | 0.7518 | p=9.77e-04 (H=2.25e-02, dz=1.87) / p=4.13e-01 (H=1.00e+00, dz=-0.25) |
| Meso-Heur MergeOnly (τ=0.75) | 0.8147 ± 0.0805 | 0.6972 ± 0.0927 | 0.8090 | 0.3406 | p=2.40e-01 (H=9.61e-01, dz=-0.27) / p=1.75e-01 (H=1.00e+00, dz=0.48) |
| Meso-Heur 0-Tok (τ=0.75,loose) | 0.8149 ± 0.0805 | 0.7086 ± 0.0958 | 0.8097 | 0.3678 | p=2.40e-01 (H=9.61e-01, dz=-0.28) / p=6.38e-01 (H=1.00e+00, dz=0.14) |
| Cequel-Transitive (LLM) | 0.7717 ± 0.0575 | 0.6708 ± 0.0851 | 0.7188 | 0.3295 | p=1.86e-02 (H=1.67e-01, dz=0.95) / p=4.88e-03 (H=5.86e-02, dz=1.15) |
| **SUG-Meso (Direct, LLM)** | 0.7978 ± 0.0501 | 0.7127 ± 0.0853 | 0.7619 | 0.4308 | Anchor |

### D5: Paired Wilcoxon + Holm-Bonferroni Tests (`SUG-Meso Direct` vs. LLM Baselines, Repair, & Ablations)
| Comparator | AMI (Mean ± SD) | Mac-B³ (Mean ± SD) | ΔAMI vs Anchor (p / p_Holm / dz) | ΔMac-B³ vs Anchor (p / p_Holm / dz) |
| :--- | ---: | ---: | :--- | :--- |
| `random_direct` | 0.7633 ± 0.0567 | 0.6795 ± 0.0915 | +0.0344 (p=9.77e-04, H=2.25e-02, dz=+1.56) | +0.0332 (p=1.95e-03, H=4.10e-02, dz=+1.39) |
| `rep_div_direct` | 0.7634 ± 0.0568 | 0.6790 ± 0.0916 | +0.0344 (p=9.77e-04, H=2.25e-02, dz=+1.55) | +0.0337 (p=1.95e-03, H=4.10e-02, dz=+1.40) |
| `cequel_edge_repair` | 0.7635 ± 0.0569 | 0.6791 ± 0.0916 | +0.0343 (p=9.77e-04, H=2.25e-02, dz=+1.54) | +0.0336 (p=2.93e-03, H=4.10e-02, dz=+1.39) |
| `cequel_transitive` | 0.7717 ± 0.0575 | 0.6708 ± 0.0851 | +0.0261 (p=1.86e-02, H=1.67e-01, dz=+0.95) | +0.0419 (p=4.88e-03, H=5.86e-02, dz=+1.15) |
| `sug_router_repair` | 0.7877 ± 0.0529 | 0.7041 ± 0.0874 | +0.0101 (p=3.22e-02, H=2.58e-01, dz=+0.63) | +0.0086 (p=2.78e-01, H=1.00e+00, dz=+0.46) |
| `sug_meso_only_repair` | 0.7968 ± 0.0508 | 0.7095 ± 0.0870 | +0.0010 (p=6.74e-02, H=4.04e-01, dz=+0.59) | +0.0032 (p=4.65e-01, H=1.00e+00, dz=+0.28) |
| `abl_merge_only` | 0.8066 ± 0.0448 | 0.6980 ± 0.0858 | -0.0089 (p=3.22e-02, H=2.58e-01, dz=-0.57) | +0.0147 (p=6.74e-02, H=6.74e-01, dz=+0.59) |
| `abl_seam_only` | 0.7620 ± 0.0587 | 0.6959 ± 0.0900 | +0.0357 (p=1.95e-03, H=2.34e-02, dz=+1.45) | +0.0168 (p=1.95e-03, H=4.10e-02, dz=+1.57) |
| `abl_fiedler_only_seam` | 0.8001 ± 0.0482 | 0.7067 ± 0.0755 | -0.0024 (p=2.78e-01, H=9.61e-01, dz=-0.45) | +0.0060 (p=4.65e-01, H=1.00e+00, dz=+0.30) |
| `abl_no_extract_compare` | 0.8357 ± 0.0493 | 0.7262 ± 0.0868 | -0.0380 (p=9.77e-03, H=1.07e-01, dz=-0.95) | -0.0135 (p=1.37e-02, H=1.50e-01, dz=-0.87) |
| `abl_uniform_budget` | 0.8101 ± 0.0493 | 0.7130 ± 0.0880 | -0.0123 (p=3.12e-01, H=9.61e-01, dz=-0.35) | -0.0003 (p=8.44e-01, H=1.00e+00, dz=-0.03) |

### D4: Dev vs. Held-Out Initial Error Pathology & Cross-Day Recurrence
- **dev**: N=11,671, FS/FM=14.89× (blk 7.75×–25.80×), H_bnd=0=10.1%, Impure(|C|>=10, pur<0.90)=24/106 (22.6%), minority share median=0.305 [IQR 0.165–0.430] (41.7% in [0.10,0.25]), Recurrence=72.24%.
- **heldout**: N=33,161, FS/FM=11.59× (blk 2.22×–86.51×), H_bnd=0=16.2%, Impure(|C|>=10, pur<0.90)=83/313 (26.5%), minority share median=0.233 [IQR 0.150–0.366] (54.2% in [0.10,0.25]), Recurrence=62.05%.

---

## D6: Root-Cause Failure Audits
### Failure Audit 1: `Event2018 M_16` (`ΔAMI=+0.0224`, `ΔMac-B³=-0.0093`)
| Query | Type | Action | Verdict | Sizes | Top Events A vs B | Solo ΔMac-B³ | LOO Marginal ΔMac-B³ | LLM Event A / B |
| :--- | :--- | :--- | :--- | :--- | :--- | ---: | ---: | :--- |
| `merge_c1_c8` | merge_pair | SEPARATE | **FN** | [210, 151] | [[241, 208], [235, 1]] vs [[241, 150], [249, 1]] | +0.0000 | +0.0000 | `Norman's controversial comments on YouTube` // `Twitter users react to Squeezie's tweet on YouTube` |
| `seam_c11` | seam | SPLIT | **FP** | [22, 18] | [[96, 20], [238, 1]] vs [[96, 18]] | -0.0168 | -0.0168 | `France experiences intense heatwave on Monday and Tuesday` // `Europe faces heatwave with fires, pollution, and broken clocks` |
| `merge_c8_c14` | merge_pair | MERGE | **TP** | [151, 125] | [[241, 150], [249, 1]] vs [[241, 125]] | +0.0081 | +0.0078 | `Twitter users discuss #BalanceTonYoutubeur campaign` // `Public reacts to #BalanceTonYoutuber controversy` |
| `seam_c8` | seam | SPLIT | **FP** | [145, 6] | [[241, 144], [249, 1]] vs [[241, 6]] | -0.0004 | -0.0007 | `Users discuss #BalanceTonYoutubeur trend` // `User teases @xSqueeZie about age and minors` |

### Failure Audit 2: `Event2012` Dev Budget Curve `20k -> 40k` Macro-B³ Dip
- `M_02`: `20k dMac=+0.0316` (6q) -> `40k dMac=+0.0201` (14q) [diff=`-0.0114`]
- `M_03`: `20k dMac=+0.0251` (10q) -> `40k dMac=+0.0251` (18q) [diff=`+0.0000`]
- `M_04`: `20k dMac=+0.0071` (11q) -> `40k dMac=+0.0071` (22q) [diff=`+0.0000`]
- `M_05`: `20k dMac=+0.0054` (10q) -> `40k dMac=+0.0096` (19q) [diff=`+0.0042`]
  - Added `M_02 seam_c6` (KEEP_MERGED): `ΔMac over 20k=+0.0000`, `ΔAMI=+0.0000`, events=[[210, 7]] vs [[210, 7]] | `21-year-old Starbucks barista missing in Oregon` // `Search for missing Oregon woman underway`
  - Added `M_02 merge_c17_c25` (SEPARATE): `ΔMac over 20k=+0.0000`, `ΔAMI=+0.0000`, events=[[194, 145], [113, 1]] vs [[194, 13]] | `Google's stock drops due to early earnings release halt` // `Google stock crashes below $700 from early earnings leak`
  - Added `M_02 merge_c1_c12` (SEPARATE): `ΔMac over 20k=+0.0000`, `ΔAMI=+0.0000`, events=[[195, 82], [346, 1]] vs [[189, 29], [205, 10]] | `Anti-austerity protests in Greece cause economic standstill` // `Syrian airstrikes kill 20 in rebel towns`
  - Added `M_02 seam_c3` (SPLIT): `ΔMac over 20k=-0.0010`, `ΔAMI=-0.0013`, events=[[196, 128], [68, 10]] vs [[68, 6], [196, 2]] | `Boy Scouts release abuse files` // `BBC cancels Newsnight over Savile scandal`
  - Added `M_02 merge_c3_c17` (SEPARATE): `ΔMac over 20k=+0.0000`, `ΔAMI=+0.0000`, events=[[196, 130], [68, 16]] vs [[194, 145], [113, 1]] | `Boy Scouts release abuse files on sex offenders` // `Google stock drops due to mid-day earnings release`
  - Added `M_02 merge_c1_c7` (SEPARATE): `ΔMac over 20k=+0.0000`, `ΔAMI=+0.0000`, events=[[195, 82], [346, 1]] vs [[353, 17]] | `Anti-austerity protests in Greece cause economic standstill` // `UEFA charges Serbia and England over racial abuse incident`
  - Added `M_02 seam_c14` (SPLIT): `ΔMac over 20k=-0.0105`, `ΔAMI=-0.0020`, events=[[347, 11]] vs [[347, 10]] | `K-pop artists visit Okinawa for 2012 collection event` // `US military rape allegations in Okinawa reported`
  - Added `M_02 merge_c9_c12` (SEPARATE): `ΔMac over 20k=+0.0000`, `ΔAMI=+0.0000`, events=[[349, 25], [345, 1]] vs [[189, 29], [205, 10]] | `Cambodia mourns King Norodom Sihanouk's passing` // `Syrian airstrikes kill 20 people in rebel areas`

### Anonymized Qualitative Case Bank

#### Event2012
- **tp_merge** (`M_07 merge_c0_c17`, action=`MERGE`, sizes=`[352, 491]`, `ΔAMI=+0.0148`, `ΔMac-B³=+0.0011`, `NetFS=+171150`, `NetFM=-1682`): LLM: `Bob Schieffer comments on Romney during debate` // `Romney's remarks about being president during debate`
  - A medoids: ['bob schieffer is keeping these b*tches in their place. #debates', 'Bob Schieffer needs to take control here.']
  - B medoids: ['"When I\'m president of the United States" WOAH there Romney #debate', '"...when I am president of the united states..." Romney #holdyourhorses #debate']
- **tp_split** (`M_19 seam_c9`, action=`SPLIT`, sizes=`[51, 5]`, `ΔAMI=+0.0024`, `ΔMac-B³=+0.0315`, `NetFS=+0`, `NetFM=+255`): LLM: `Syria opposition meeting in Doha` // `Ukraine opposition protest rally over election count`
  - A medoids: ['Syria opposition to meet in Doha [URL]', 'Syria opposition to meet in Doha [URL]']
  - B medoids: ['Ukraine opposition calls protest rally over election count: KIEV (Reuters) - Ukraine opposition parties called a... [URL]', 'Ukraine opposition calls protest rally over election count: KIEV (Reuters) - Ukraine opposition parties called a... [URL]']
- **fp_error** (`M_17 merge_c0_c5`, action=`MERGE`, sizes=`[119, 159]`, `ΔAMI=-0.0141`, `ΔMac-B³=-0.0223`, `NetFS=+185`, `NetFM=-18736`): LLM: `CMA Awards ceremony for country music` // `Blake Shelton wins Male Vocalist of the Year at CMAs`
  - A medoids: ['Country Music Awards! 🎤😍 #CMAawards', 'CMA awards (: #love #country #music']
  - B medoids: ['male vocalist of the year goes to......BLAKE SHELTON congrats!!! @USER', 'Congratulations Blake Shelton ! Male vocalist of the year!']

#### Event2018
- **tp_merge** (`M_06 merge_c16_c19`, action=`MERGE`, sizes=`[601, 221]`, `ΔAMI=+0.0174`, `ΔMac-B³=+0.0079`, `NetFS=+125733`, `NetFM=-7088`): LLM: `Benalla interviewed on TF1's 20H amid investigation` // `Benalla interviewed on TF1's 20H as exclusive report`
  - A medoids: ['#tf1 #benalla il va nous faire presque pleurer avec sa petite voix ce doux nounours de Benalla...par contre pour casser du manifestant illégalement là c’est ...', '#Benalla au #20h de @USER. [URL]']
  - B medoids: ["EXCLUSIF - Alexandre Benalla au 20H de TF1 ce soir : découvrez un extrait de l'interview [URL]", "EXCLUSIF - Alexandre Benalla au 20H de TF1 ce soir : découvrez un extrait de l'interview : [URL]"]
- **tp_split** (`M_08 seam_c11`, action=`SPLIT`, sizes=`[172, 7]`, `ΔAMI=+0.0027`, `ΔMac-B³=+0.0371`, `NetFS=+0`, `NetFM=+1204`): LLM: `Social media users use #JeSuisCute hashtag` // `Ibrahim Maalouf comments on jazz classification`
  - A medoids: ['Je suis cuit. #JeSuisCute [URL]', 'Tant qu’on y est jvous apporte un peu de pureté sur ce # avec mon frère 🐣#JeSuisCute [URL]']
  - B medoids: ['Ibrahim Maalouf : "je me fiche de savoir si ce que je fais est du jazz" [URL] vía @USER', 'Ibrahim Maalouf : "je me fiche de savoir si ce que je fais est du jazz" [URL] via @USER']
- **fp_error** (`M_16 seam_c11`, action=`SPLIT`, sizes=`[22, 18]`, `ΔAMI=-0.0046`, `ΔMac-B³=-0.0168`, `NetFS=-360`, `NetFM=+36`): LLM: `France experiences intense heatwave on Monday and Tuesday` // `Europe faces heatwave with fires, pollution, and broken clocks`
  - A medoids: ['Canicule : lundi et mardi, «journées les plus chaudes» [URL] via @USER', "C'est bientôt fini !!! Après plusieurs jours de #canicule, la forte chaleur encore présente sur la France lundi et mardi devrait bientôt s'atténuer. #AFP [URL]"]
  - B medoids: ['#La canicule en Europe : incendies, pollution et horloge cassée [URL]', 'La canicule en Europe: incendies, pollution et horloge cassée [URL]']
