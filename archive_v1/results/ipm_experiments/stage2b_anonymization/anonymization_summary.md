# IP&M Stage 2B: Entity/Hashtag Anonymization Anti-Memorization Report
- Generated: `2026-11-24 16:25:26` | Model: `Qwen/Qwen3-8B`
- Ledger Totals across Stage 2B: `{'calls': 1394, 'live': 17, 'tokens': 782145, 'errors': 0, 'all_journal_errors': 0}`

---

## Event2012
### 1. Leakage & Masking Coverage Audit (Executed Queries)
| Condition | Queries | Entity Mask | Hashtag Mask | Year Mask | URL Mask | PH Density | Action Agree vs Unmasked | Extr. Reconstr. Rate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: |
| `anon_pseudo_action` | 294 | 1815/1817 (99.89%) | 437/437 (100.00%) | 44/44 (100.00%) | 996/996 (100.00%) | 34.3% | 81.6% (M=82.2%, S=80.2%) | N/A |
| `anon_blind_action` | 322 | 2011/2017 (99.70%) | 484/484 (100.00%) | 45/45 (100.00%) | 1094/1094 (100.00%) | 33.7% | 71.7% (M=72.0%, S=71.1%) | N/A |
| `anon_pseudo_extract` | 249 | 1481/1482 (99.93%) | 365/365 (100.00%) | 37/37 (100.00%) | 818/818 (100.00%) | 34.5% | 75.1% (M=70.0%, S=86.1%) | 4.47% (11/246) |

### 2. Clustering Performance & Gain Retention (`CausalUniformFixed 10k/blk`, `all_shown` $U_t$)
| Variant | Prompt / Mask | AMI | Mac-B³ | Tail-B³ | UnqNet/blk | AMI Ret. | Mac Ret. | Merge P/R/S | Seam P/R/S | Err |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | ---: |
| `init_2dse` (Ref) | unmasked | 0.8395 | 0.6567 | 0.4103 | - | - | - | - | - | 0 |
| `random_mean3` (Ref) | unmasked | 0.8391 | 0.6579 | 0.4136 | - | - | - | - | - | 0 |
| `random_meso_action_mean3` (Ref) | unmasked | 0.8771 | 0.6860 | 0.4601 | - | - | - | - | - | 0 |
| `sug_meso_causal_fixed_action` (Ref) | unmasked | 0.8976 | 0.6901 | 0.4602 | - | - | - | - | - | 0 |
| `sug_meso_causal_fixed_extract` (Ref) | unmasked | 0.8854 | 0.6904 | 0.4618 | - | - | - | - | - | 0 |
| `anon_pseudo_action` | action_first / pseudonym | 0.8967 | 0.6855 | 0.4553 | +632.1 | 98.4% | 86.1% | 0.92/0.77/0.88 | 0.60/0.87/0.54 | 0 |
| `anon_blind_action` | action_first / blind | 0.8827 | 0.6609 | 0.4453 | +427.9 | 74.3% | 12.6% | 0.75/0.78/0.62 | 0.55/0.69/0.57 | 0 |
| `anon_pseudo_extract` | extract_compare / pseudonym | 0.8629 | 0.6839 | 0.4627 | +318.2 | 50.9% | 80.8% | 1.00/0.27/1.00 | 0.58/0.97/0.34 | 0 |

### 3. Paired Wilcoxon Tests (Anonymized vs. Baselines & Unmasked Reference)

#### `anon_pseudo_action`
- vs `init_2dse`: ΔAMI=+0.0572 (p=6.10e-05, H=4.88e-04, dz=+1.41) | ΔMac=+0.0288 (p=5.80e-04, H=4.06e-03, dz=+1.09)
- vs `random_mean3`: ΔAMI=+0.0576 (p=6.10e-05, H=4.88e-04, dz=+1.43) | ΔMac=+0.0276 (p=1.01e-03, H=6.04e-03, dz=+1.02)
- vs `random_meso_action_mean3`: ΔAMI=+0.0197 (p=3.09e-02, H=1.24e-01, dz=+0.62) | ΔMac=-0.0005 (p=8.21e-01, H=1.00e+00, dz=-0.02)
- vs `random_meso_extract_mean3`: ΔAMI=+0.0357 (p=5.80e-04, H=3.48e-03, dz=+1.00) | ΔMac=+0.0014 (p=7.06e-01, H=1.00e+00, dz=+0.07)
- vs `sug_meso_causal_fixed_action`: ΔAMI=-0.0009 (p=9.55e-01, H=9.55e-01, dz=-0.04) | ΔMac=-0.0046 (p=2.33e-01, H=9.32e-01, dz=-0.20)
- vs `sug_meso_causal_fixed_extract`: ΔAMI=+0.0114 (p=3.94e-01, H=7.88e-01, dz=+0.35) | ΔMac=-0.0049 (p=4.96e-01, H=1.00e+00, dz=-0.21)

#### `anon_blind_action`
- vs `init_2dse`: ΔAMI=+0.0432 (p=1.10e-02, H=7.69e-02, dz=+0.77) | ΔMac=+0.0042 (p=5.62e-01, H=1.00e+00, dz=+0.11)
- vs `random_mean3`: ΔAMI=+0.0436 (p=9.19e-03, H=7.35e-02, dz=+0.79) | ΔMac=+0.0030 (p=6.69e-01, H=1.00e+00, dz=+0.08)
- vs `random_meso_action_mean3`: ΔAMI=+0.0056 (p=8.60e-01, H=1.00e+00, dz=+0.13) | ΔMac=-0.0251 (p=6.29e-03, H=3.77e-02, dz=-0.81)
- vs `random_meso_extract_mean3`: ΔAMI=+0.0217 (p=9.34e-02, H=3.74e-01, dz=+0.43) | ΔMac=-0.0232 (p=3.35e-02, H=1.68e-01, dz=-0.72)
- vs `sug_meso_causal_fixed_action`: ΔAMI=-0.0149 (p=2.50e-02, H=1.50e-01, dz=-0.48) | ΔMac=-0.0292 (p=4.27e-04, H=3.42e-03, dz=-1.06)
- vs `sug_meso_causal_fixed_extract`: ΔAMI=-0.0027 (p=4.04e-01, H=1.00e+00, dz=-0.06) | ΔMac=-0.0295 (p=2.69e-03, H=1.88e-02, dz=-0.95)

#### `anon_pseudo_extract`
- vs `init_2dse`: ΔAMI=+0.0233 (p=1.01e-03, H=6.04e-03, dz=+0.88) | ΔMac=+0.0272 (p=3.05e-05, H=2.44e-04, dz=+1.74)
- vs `random_mean3`: ΔAMI=+0.0238 (p=7.63e-04, H=5.34e-03, dz=+0.90) | ΔMac=+0.0260 (p=3.05e-05, H=2.44e-04, dz=+1.55)
- vs `random_meso_action_mean3`: ΔAMI=-0.0142 (p=1.17e-01, H=3.50e-01, dz=-0.53) | ΔMac=-0.0021 (p=5.62e-01, H=1.00e+00, dz=-0.16)
- vs `random_meso_extract_mean3`: ΔAMI=+0.0019 (p=8.60e-01, H=8.60e-01, dz=+0.07) | ΔMac=-0.0002 (p=8.21e-01, H=1.00e+00, dz=-0.02)
- vs `sug_meso_causal_fixed_action`: ΔAMI=-0.0347 (p=2.14e-04, H=1.71e-03, dz=-1.23) | ΔMac=-0.0062 (p=1.93e-01, H=5.78e-01, dz=-0.31)
- vs `sug_meso_causal_fixed_extract`: ΔAMI=-0.0225 (p=2.69e-03, H=1.34e-02, dz=-0.76) | ΔMac=-0.0065 (p=7.39e-02, H=2.96e-01, dz=-0.51)

### 4. Sample Before / After Anonymized Prompts
- **[anon_pseudo_action | Event2012_M06 | merge_c16_c23]**
  - Raw: `Group A (size=90, keywords: lance armstrong, armstrong, france, tour, lance):   - Lance Armstrong Stripped of His 7 Tour de France Titles: The International Cycling Union on Monday ratified a re... http://t.co/8CPsc9CI   - Lance Armstrong Stripped of His 7 Tou`
  - Anon: `Group A (size=90, keywords: [ENT_3], [ENT_2], [ENT_4], [ENT_6], [ENT_2]):   - [ENT_2] of His 7 [ENT_4] Titles: [ENT_1] on Monday ratified a re... [URL]   - [ENT_2] of His 7 [ENT_4] Titles: [ENT_1] on Monday ratified a re... [URL]   - [ENT_2] of His 7 [ENT_4] T`
- **[anon_pseudo_action | Event2012_M07 | merge_c5_c17]**
  - Raw: `Group A (size=285, keywords: romney, obama, history, airbrush, #debate):   - Gov. Romney. you keep trying to airbrush history, here. --Obama #Debates   - "Gov Romney, you keep on trying to airbrush history..." #Obama2012   - Romney keeps trying to airbrush his`
  - Anon: `Group A (size=285, keywords: [ENT_3], [ENT_5], history, [ENT_2], [ENT_4]):   - Gov. [ENT_3]. you keep trying to [ENT_2] history, here. --[ENT_5] #[TAG_6]   - "Gov [ENT_3], you keep on trying to [ENT_2] history..." #[TAG_4]   - [ENT_3] keeps trying to [ENT_2] h`
- **[anon_blind_action | Event2012_M06 | merge_c16_c23]**
  - Raw: `Group A (size=90, keywords: lance armstrong, armstrong, france, tour, lance):   - Lance Armstrong Stripped of His 7 Tour de France Titles: The International Cycling Union on Monday ratified a re... http://t.co/8CPsc9CI   - Lance Armstrong Stripped of His 7 Tou`
  - Anon: `Group A (size=90, keywords: [ENTITY], [ENTITY], [ENTITY], [ENTITY], [ENTITY]):   - [ENTITY] of His 7 [ENTITY] Titles: [ENTITY] on Monday ratified a re... [URL]   - [ENTITY] of His 7 [ENTITY] Titles: [ENTITY] on Monday ratified a re... [URL]   - [ENTITY] of His`
- **[anon_blind_action | Event2012_M07 | merge_c5_c17]**
  - Raw: `Group A (size=285, keywords: romney, obama, history, airbrush, #debate):   - Gov. Romney. you keep trying to airbrush history, here. --Obama #Debates   - "Gov Romney, you keep on trying to airbrush history..." #Obama2012   - Romney keeps trying to airbrush his`
  - Anon: `Group A (size=285, keywords: [ENTITY], [ENTITY], history, [ENTITY], [ENTITY]):   - Gov. [ENTITY]. you keep trying to [ENTITY] history, here. --[ENTITY] [HASHTAG]   - "Gov [ENTITY], you keep on trying to [ENTITY] history..." [HASHTAG]   - [ENTITY] keeps trying `

---

## Event2018
### 1. Leakage & Masking Coverage Audit (Executed Queries)
| Condition | Queries | Entity Mask | Hashtag Mask | Year Mask | URL Mask | PH Density | Action Agree vs Unmasked | Extr. Reconstr. Rate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: |
| `anon_pseudo_action` | 182 | 1033/1033 (100.00%) | 657/657 (100.00%) | 19/19 (100.00%) | 849/849 (100.00%) | 29.8% | 83.5% (M=86.1%, S=78.3%) | N/A |
| `anon_blind_action` | 193 | 1124/1125 (99.91%) | 704/704 (100.00%) | 22/22 (100.00%) | 911/911 (100.00%) | 29.7% | 69.3% (M=67.9%, S=72.1%) | N/A |
| `anon_pseudo_extract` | 154 | 842/842 (100.00%) | 532/532 (100.00%) | 14/14 (100.00%) | 703/703 (100.00%) | 29.3% | 77.3% (M=74.5%, S=82.7%) | 0.00% (0/154) |

### 2. Clustering Performance & Gain Retention (`CausalUniformFixed 10k/blk`, `all_shown` $U_t$)
| Variant | Prompt / Mask | AMI | Mac-B³ | Tail-B³ | UnqNet/blk | AMI Ret. | Mac Ret. | Merge P/R/S | Seam P/R/S | Err |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | ---: |
| `init_2dse` (Ref) | unmasked | 0.7634 | 0.6790 | 0.3500 | - | - | - | - | - | 0 |
| `random_mean3` (Ref) | unmasked | 0.7632 | 0.6795 | 0.3512 | - | - | - | - | - | 0 |
| `random_meso_action_mean3` (Ref) | unmasked | 0.7740 | 0.6975 | 0.4082 | - | - | - | - | - | 0 |
| `sug_meso_causal_fixed_action` (Ref) | unmasked | 0.8349 | 0.7258 | 0.4201 | - | - | - | - | - | 0 |
| `sug_meso_causal_fixed_extract` (Ref) | unmasked | 0.8100 | 0.7129 | 0.4276 | - | - | - | - | - | 0 |
| `anon_pseudo_action` | action_first / pseudonym | 0.8191 | 0.7082 | 0.4056 | +1011.9 | 77.9% | 62.3% | 0.89/0.81/0.87 | 0.45/0.68/0.53 | 0 |
| `anon_blind_action` | action_first / blind | 0.7856 | 0.6897 | 0.4184 | +508.9 | 31.1% | 22.9% | 0.71/0.66/0.69 | 0.46/0.57/0.61 | 0 |
| `anon_pseudo_extract` | extract_compare / pseudonym | 0.7869 | 0.7050 | 0.4287 | +602.2 | 50.4% | 76.5% | 1.00/0.31/1.00 | 0.49/0.95/0.32 | 0 |

### 3. Paired Wilcoxon Tests (Anonymized vs. Baselines & Unmasked Reference)

#### `anon_pseudo_action`
- vs `init_2dse`: ΔAMI=+0.0557 (p=9.77e-04, H=7.81e-03, dz=+1.50) | ΔMac=+0.0292 (p=9.77e-04, H=7.81e-03, dz=+1.61)
- vs `random_mean3`: ΔAMI=+0.0559 (p=9.77e-04, H=7.81e-03, dz=+1.50) | ΔMac=+0.0287 (p=9.77e-04, H=7.81e-03, dz=+1.60)
- vs `random_meso_action_mean3`: ΔAMI=+0.0451 (p=9.77e-04, H=7.81e-03, dz=+1.43) | ΔMac=+0.0107 (p=3.22e-02, H=9.67e-02, dz=+0.91)
- vs `random_meso_extract_mean3`: ΔAMI=+0.0532 (p=9.77e-04, H=7.81e-03, dz=+1.46) | ΔMac=+0.0132 (p=4.88e-03, H=2.44e-02, dz=+1.27)
- vs `sug_meso_causal_fixed_action`: ΔAMI=-0.0158 (p=6.84e-03, H=2.05e-02, dz=-0.84) | ΔMac=-0.0176 (p=1.95e-03, H=1.17e-02, dz=-1.42)
- vs `sug_meso_causal_fixed_extract`: ΔAMI=+0.0091 (p=5.77e-01, H=9.30e-01, dz=+0.31) | ΔMac=-0.0048 (p=3.65e-01, H=7.30e-01, dz=-0.27)

#### `anon_blind_action`
- vs `init_2dse`: ΔAMI=+0.0223 (p=6.74e-02, H=3.37e-01, dz=+0.58) | ΔMac=+0.0107 (p=8.30e-02, H=3.32e-01, dz=+0.56)
- vs `random_mean3`: ΔAMI=+0.0224 (p=6.74e-02, H=3.37e-01, dz=+0.58) | ΔMac=+0.0102 (p=1.23e-01, H=3.69e-01, dz=+0.54)
- vs `random_meso_action_mean3`: ΔAMI=+0.0117 (p=3.65e-01, H=7.30e-01, dz=+0.33) | ΔMac=-0.0078 (p=1.47e-01, H=3.69e-01, dz=-0.44)
- vs `random_meso_extract_mean3`: ΔAMI=+0.0197 (p=1.02e-01, H=3.37e-01, dz=+0.50) | ΔMac=-0.0052 (p=4.13e-01, H=4.13e-01, dz=-0.30)
- vs `sug_meso_causal_fixed_action`: ΔAMI=-0.0492 (p=9.77e-04, H=7.81e-03, dz=-1.61) | ΔMac=-0.0361 (p=9.77e-04, H=7.81e-03, dz=-2.60)
- vs `sug_meso_causal_fixed_extract`: ΔAMI=-0.0244 (p=4.88e-03, H=3.42e-02, dz=-0.83) | ΔMac=-0.0232 (p=6.84e-03, H=4.79e-02, dz=-1.16)

#### `anon_pseudo_extract`
- vs `init_2dse`: ΔAMI=+0.0235 (p=9.77e-04, H=7.81e-03, dz=+1.43) | ΔMac=+0.0260 (p=4.88e-03, H=3.42e-02, dz=+1.16)
- vs `random_mean3`: ΔAMI=+0.0237 (p=9.77e-04, H=7.81e-03, dz=+1.45) | ΔMac=+0.0255 (p=4.88e-03, H=3.42e-02, dz=+1.16)
- vs `random_meso_action_mean3`: ΔAMI=+0.0129 (p=1.37e-02, H=5.47e-02, dz=+0.74) | ΔMac=+0.0074 (p=1.75e-01, H=3.50e-01, dz=+0.42)
- vs `random_meso_extract_mean3`: ΔAMI=+0.0210 (p=9.77e-04, H=7.81e-03, dz=+1.29) | ΔMac=+0.0100 (p=2.44e-02, H=9.77e-02, dz=+0.81)
- vs `sug_meso_causal_fixed_action`: ΔAMI=-0.0480 (p=9.77e-04, H=7.81e-03, dz=-1.29) | ΔMac=-0.0209 (p=1.95e-03, H=1.56e-02, dz=-1.37)
- vs `sug_meso_causal_fixed_extract`: ΔAMI=-0.0232 (p=4.20e-02, H=1.26e-01, dz=-0.60) | ΔMac=-0.0080 (p=8.30e-02, H=2.49e-01, dz=-0.63)

### 4. Sample Before / After Anonymized Prompts
- **[anon_pseudo_action | Event2018_M07 | merge_c0_c6]**
  - Raw: `Group A (size=265, keywords: cute, #jesuiscute, twitter, photos, hashtag):   - #JeSuisCute Voilà, la vraie plus cute du tweet! C'est notre petite lapine ! https://t.co/L2TKLBQufQ   - Ya de sacré tarpé quand même sur votre hashtag je suis cute   - Jspr mon gars`
  - Anon: `Group A (size=265, keywords: [ENT_7], #[TAG_1], [ENT_6], photos, hashtag):   - #[TAG_1] Voilà, la vraie plus [ENT_7] du tweet! C'est notre petite lapine ! [URL]   - [ENT_3] quand même sur votre hashtag je suis [ENT_7]   - Jspr mon gars il va pas sur le hashtag`
- **[anon_pseudo_action | Event2018_M08 | merge_c1_c11]**
  - Raw: `Group A (size=266, keywords: cute, twitter, #jesuiscute, corps, hashtag):   - @nevermindcelest C’est comme le hashtag je suis cute ou je sais pas quoi genre tout le monde s’insulte pour rien car chacun est libre d’expo   - Pas besoin d’être à moitié nude pour `
  - Anon: `Group A (size=266, keywords: [ENT_8], [ENT_6], #[TAG_2], corps, hashtag):   - [USER_2] C’est comme le hashtag je suis [ENT_8] ou je sais [ENT_11] quoi genre tout le monde s’insulte pour rien car chacun est libre   - [ENT_11] besoin d’être à moitié nude pour êt`
- **[anon_blind_action | Event2018_M06 | merge_c5_c6]**
  - Raw: `Group A (size=490, keywords: #lunareclipse, lune, nuages, éclipse, #eclipselunaire):   - l’éclipse lunaire qu’on attendait tous mais que la moitié ne verra pas à cause des ces ****** de nuages qui ont décidé de se pointer le soir   - Roh les nuages, c'est pas `
  - Anon: `Group A (size=490, keywords: [HASHTAG], [ENTITY], [ENTITY], [ENTITY], [HASHTAG]):   - l’[ENTITY] lunaire qu’on attendait tous mais que la moitié ne verra [ENTITY] à cause des ces ****** de [ENTITY] qui ont décidé de se pointer   - [ENTITY] les [ENTITY], c'est `
- **[anon_blind_action | Event2018_M07 | merge_c0_c6]**
  - Raw: `Group A (size=265, keywords: cute, #jesuiscute, twitter, photos, hashtag):   - #JeSuisCute Voilà, la vraie plus cute du tweet! C'est notre petite lapine ! https://t.co/L2TKLBQufQ   - Ya de sacré tarpé quand même sur votre hashtag je suis cute   - Jspr mon gars`
  - Anon: `Group A (size=265, keywords: [ENTITY], [HASHTAG], [ENTITY], photos, hashtag):   - [HASHTAG] Voilà, la vraie plus [ENTITY] du tweet! C'est notre petite lapine ! [URL]   - [ENTITY] quand même sur votre hashtag je suis [ENTITY]   - Jspr mon gars il va pas sur le `
