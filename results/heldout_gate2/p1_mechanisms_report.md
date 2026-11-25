# Track A Gate 2 P1 Mechanistic Attribution & Boundary Calibration Report

> Zero LLM API calls · 100% Dev-Locked · Reproducible via `python trackA_sug_sed/analyze_gate2_mechanisms.py`

## 1. Controlled 3-Factor Attribution (`Random` → `Random-Meso` → `Meso-Heur-Best` → `Full Pure-Star-Meso`)

- `ctrl_random_meso_*` runs `run_pure_star_meso(..., random_order=True)`: it inherits `Pure 2D-SE + AutoMerge (>=0.85) + GuardedUF (0.40) + Clean Seam Candidate Pool + Top-k Star Packing + Identical Prompts`, ablating **only** calibrated two-tier Bayes utility & seam score ordering.

| Decomposition Step | Scope | ΔAMI | ΔMac-$B^3$ | ΔMic-$B^3$ | ΔTail-$B^3$ | `p_raw / p_Holm` (`AMI`, `Mac-B³`, `Mic-B³`) |
|---|---|---:|---:|---:|---:|---|
| Factor 1: Mesoscopic Star-Pack Architecture (Random -> Random-Meso) | Event2012 | +5.99 | +1.61 | +10.12 | +0.38 | ami:3.05176e-05, macro:0.000427246, micro:3.05176e-05 |
| Factor 1: Mesoscopic Star-Pack Architecture (Random -> Random-Meso) | Event2018 | +5.53 | +2.35 | +11.11 | +2.04 | ami:0.000976562, macro:0.00292969, micro:0.000976562 |
| Factor 1: Mesoscopic Star-Pack Architecture (Random -> Random-Meso) | Pooled27 | +5.80 | +1.91 | +10.52 | +1.06 | ami:1.49012e-08, macro:8.19564e-07, micro:1.49012e-08 |
| Factor 2: Calibrated Priority Scheduling (Random-Meso -> Full) | Event2012 | +0.19 | +0.69 | +0.09 | +0.75 | ami:0.561401/0.561401, macro:0.0479126/0.143738, micro:0.454285/0.454285 |
| Factor 2: Calibrated Priority Scheduling (Random-Meso -> Full) | Event2018 | +1.47 | +1.17 | +2.63 | +1.71 | ami:0.174805/0.524414, macro:0.147461/0.442383, micro:0.0419922/0.125977 |
| Factor 2: Calibrated Priority Scheduling (Random-Meso -> Full) | Pooled27 | +0.71 | +0.89 | +1.12 | +1.14 | ami:0.089029/0.178058, macro:0.0176135/0.0352271, micro:0.0558532/0.05981 |
| Factor 3: LLM Semantic Gain over 0-Tok Cosine (Meso-Heur-Best -> Full) | Event2012 | +1.18 | +1.15 | +1.70 | +1.24 | ami:0.0934448/0.18689, macro:0.00515747/0.0206299, micro:0.0739136/0.151978 |
| Factor 3: LLM Semantic Gain over 0-Tok Cosine (Meso-Heur-Best -> Full) | Event2018 | +2.93 | +2.16 | +4.38 | +4.06 | ami:0.464844/0.524414, macro:0.0244141/0.0976562, micro:0.240234/0.480469 |
| Factor 3: LLM Semantic Gain over 0-Tok Cosine (Meso-Heur-Best -> Full) | Pooled27 | +1.90 | +1.56 | +2.79 | +2.39 | ami:0.0905648/0.178058, macro:0.000109777/0.000548884, micro:0.029905/0.05981 |
| Total LLM Routing Gain over Naive Random (Random -> Full) | Event2012 | +6.18 | +2.30 | +10.21 | +1.13 | ami:6.10352e-05, macro:0.000427246, micro:6.10352e-05 |
| Total LLM Routing Gain over Naive Random (Random -> Full) | Event2018 | +7.00 | +3.53 | +13.74 | +3.75 | ami:0.00488281, macro:0.00488281, micro:0.000976562 |
| Total LLM Routing Gain over Naive Random (Random -> Full) | Pooled27 | +6.52 | +2.80 | +11.65 | +2.20 | ami:2.08616e-07, macro:3.76999e-06, micro:2.98023e-08 |

## 2. 5-Pillar Ablation Paired Wilcoxon (`Full - Ablated`) & Structural Balance

| Pillar Variant | Role | Scope | ΔAMI (W/T/L, `p_raw / p_Holm`) | ΔMac-$B^3$ (W/T/L, `p_raw / p_Holm`) | ΔMic-$B^3$ (W/T/L, `p_raw / p_Holm`) | ΔTail-$B^3$ (W/T/L, `p_raw / p_Holm`) |
|---|---|---|---|---|---|---|
| w/o Star Packing (k=1 Pair) | Core Universal Engine | Event2012 | +0.79 (9/2/5, 0.216553 / 0.866212) | +0.06 (8/2/6, 0.6698 / 1.0) | +0.95 (9/2/5, 0.153076 / 0.612304) | -0.22 (3/7/6, 0.300781 / 1.0) |
| w/o Star Packing (k=1 Pair) | Core Universal Engine | Event2018 | +3.42 (8/0/3, 0.174805 / 0.69922) | +1.00 (8/0/3, 0.240234 / 0.960936) | +5.50 (8/0/3, 0.0537109 / 0.214844) | +0.10 (2/3/6, 0.460938 / 1.0) |
| w/o Star Packing (k=1 Pair) | Core Universal Engine | Pooled27 | +1.86 (17/2/8, 0.0451227 / 0.225614) | +0.45 (16/2/9, 0.156338 / 0.625352) | +2.80 (17/2/8, 0.0219083 / 0.109541) | -0.09 (5/10/12, 0.224686 / 0.674058) |
| w/o Seam Split | Core Universal Engine | Event2012 | -0.07 (6/1/9, 0.524475 / 1.0) | +0.67 (11/1/4, 0.0255737 / 0.127868) | -0.20 (6/1/9, 0.524475 / 1.0) | +1.29 (10/6/0, 0.00195312 / 0.0097656) |
| w/o Seam Split | Core Universal Engine | Event2018 | +0.41 (9/1/1, 0.0371094 / 0.185547) | +2.11 (10/1/0, 0.00195312 / 0.0097656) | +0.50 (9/1/1, 0.00585938 / 0.0292969) | +4.76 (6/4/1, 0.03125 / 0.15625) |
| w/o Seam Split | Core Universal Engine | Pooled27 | +0.12 (15/2/10, 0.395711 / 1.0) | +1.26 (21/2/4, 6.36578e-05 / 0.000318289) | +0.09 (15/2/10, 0.219986 / 0.879944) | +2.70 (16/10/1, 3.05176e-05 / 0.000152588) |
| w/o AutoMerge (>=0.85) | Efficiency & Risk Guardrail | Event2012 | +1.01 (6/9/1, 0.03125 / 0.15625) | +0.21 (6/9/1, 0.046875 / 0.1875) | +1.58 (6/9/1, 0.03125 / 0.15625) | -0.02 (0/14/2, 0.5 / 1.0) |
| w/o AutoMerge (>=0.85) | Efficiency & Risk Guardrail | Event2018 | -0.47 (3/3/5, 0.460938 / 1.0) | +0.28 (4/3/4, 0.945312 / 1.0) | -0.69 (2/3/6, 0.382812 / 1.0) | +1.25 (2/4/5, 0.375 / 1.0) |
| w/o AutoMerge (>=0.85) | Efficiency & Risk Guardrail | Pooled27 | +0.41 (9/12/6, 0.359131 / 1.0) | +0.24 (10/12/5, 0.359131 / 1.0) | +0.66 (8/12/7, 0.359131 / 0.893556) | +0.50 (2/18/7, 0.164062 / 0.656248) |
| w/o Bayes Band & AutoMerge | Efficiency & Risk Guardrail | Event2012 | +0.30 (7/2/7, 0.625732 / 1.0) | -0.06 (5/2/9, 1.0 / 1.0) | +0.57 (7/2/7, 0.501587 / 1.0) | -0.19 (4/6/6, 0.492188 / 1.0) |
| w/o Bayes Band & AutoMerge | Efficiency & Risk Guardrail | Event2018 | -0.40 (3/1/7, 0.492188 / 1.0) | +0.12 (5/1/5, 0.921875 / 1.0) | -0.69 (3/1/7, 0.492188 / 1.0) | +0.91 (3/3/5, 0.382812 / 1.0) |
| w/o Bayes Band & AutoMerge | Efficiency & Risk Guardrail | Pooled27 | +0.01 (10/3/14, 0.944107 / 1.0) | +0.01 (10/3/14, 0.855336 / 1.0) | +0.06 (10/3/14, 0.988813 / 0.988813) | +0.26 (7/9/11, 0.228752 / 0.674058) |
| w/o Guarded UF | Efficiency & Risk Guardrail | Event2012 | -0.57 (3/8/5, 0.25 / 0.866212) | +0.15 (4/8/4, 0.640625 / 1.0) | -0.96 (3/8/5, 0.3125 / 0.9375) | -0.04 (1/12/3, 0.25 / 1.0) |
| w/o Guarded UF | Efficiency & Risk Guardrail | Event2018 | +0.13 (2/3/6, 0.742188 / 1.0) | +0.14 (4/3/4, 0.84375 / 1.0) | +0.00 (3/3/5, 0.742188 / 1.0) | -0.00 (3/6/2, 0.8125 / 1.0) |
| w/o Guarded UF | Efficiency & Risk Guardrail | Pooled27 | -0.29 (5/11/11, 0.274445 / 1.0) | +0.15 (8/11/8, 0.705719 / 1.0) | -0.57 (6/11/10, 0.297852 / 0.893556) | -0.02 (4/18/5, 0.359375 / 0.674058) |

## 3. EN vs FR Encoder Cosine Calibration & `AutoMerge (>=0.85)` Anisotropy

| Dataset (`Embed Model`) | Diff-Event Cos `mean / p95 / p99` | Same-Event Cos `mean / p50` | `>=0.85` Pair Prec% (Size-Wtd%) | `[0.85,0.90)` Prec% | `[0.90,1.00]` Prec% | Ablation `Full-NoAM` ΔFM / ΔFS |
|---|---|---|---:|---:|---:|---|
| Event2012 (`sentence-transformers/all-MiniLM-L6-v2`) | 0.159 / 0.379 / 0.532 | 0.464 / 0.467 | 98.4% (99.1%) | 97.7% | 100.0% | +1,845 / -635,279 |
| Event2018 (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) | 0.342 / 0.604 / 0.745 | 0.637 / 0.652 | 87.1% (85.2%) | 84.6% | 100.0% | +749,776 / -201,358 |

## 4. `GuardedUnionFind` Transitive-Snowballing Risk Insurance

- **Event2012**: Prevents `+47,895` false-merge pairs (`+29.1%` vs Full; `post_llm_guard_blocks=2` plus pre-query `can_merge()` filtering).
- **Event2018**: Prevents `+1,446,601` false-merge pairs (`+68.1%` vs Full; `post_llm_guard_blocks=2` plus pre-query `can_merge()` filtering).

## 5. `Pred/True` Granularity & Cross-Day Link Acceptance Boundary

- **Cross-Day Event2012**: `sug_cross_star` G-AMI `90.62` vs `proto_attr_0.95` `90.75` and `oracle_band` `91.70` (gap `-1.08`); LLM link accept rate `72/181 = 39.8%` vs Oracle `82/199 = 41.2%`.
- **Cross-Day Event2018**: `sug_cross_star` G-AMI `84.76` vs `proto_attr_0.95` `84.57` and `oracle_band` `85.12` (gap `-0.36`); LLM link accept rate `16/91 = 17.6%` vs Oracle `29/114 = 25.4%`.
