# Track A（SUG-SED）双语保留测试集核心消融实验报告（`Qwen/Qwen3-8B`）

- **评测范围**：与主实验完全一致的全部 27 个未见保留流式块（`Event2012 M_06..M_21` 16 块 + `Event2018 M_06..M_16` 11 块），同 `10k` 真实 Token/块严格累计预算，同 `Qwen/Qwen3-8B`（`enable_thinking=False`）。
- **消融可视化**：[ablation_bilingual_qwen3_8b.png](../figures/ablation_bilingual_qwen3_8b.png) · **LaTeX 表**：[ablation_table_qwen3_8b.tex](./ablation_table_qwen3_8b.tex)

## Event2012（英语 · M_06..M_21，16 块）（初始 2D-SE: AMI=0.8395, ARI=0.5862, 宏 B³=0.6567, 长尾 B³=0.4103）

| 变体 | 实耗 Token | AMI ($\Delta$) | ARI | 宏 $B^3$ ($\Delta$) | 长尾 $B^3$ | 未查询 $\Delta$微 $B^3$ | 净减 FM / 净减 FS | 胜率 (AMI / 宏 $B^3$) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| SUG-Meso (Full, Direct) | 155,275 | 0.8811 (+0.0416) | 0.7535 | 0.6888 (+0.0321) | 0.4639 | +0.0706 | -3,806 / +87,277 | 16/16 · 16/16 |
| SUG-Meso (Branch 2 Repair) | 155,275 | 0.8810 (+0.0414) | 0.7536 | 0.6837 (+0.0270) | 0.4522 | +0.0704 | -3,866 / +87,330 | 16/16 · 16/16 |
| SUG-Router (+Bridge/Seed, Repair) | 158,226 | 0.8736 (+0.0341) | 0.7330 | 0.6806 (+0.0239) | 0.4430 | +0.0574 | -443 / +81,660 | 16/16 · 15/16 |
| w/o SPLIT_SEAM (Merge-Only) | 156,358 | 0.8843 (+0.0447) | 0.7596 | 0.6683 (+0.0116) | 0.4081 | +0.0771 | -5,040 / +89,049 | 15/16 · 13/16 |
| w/o MERGE_PAIR (Seam-Only) | 54,263 | 0.8395 (-0.0000) | 0.5852 | 0.6799 (+0.0231) | 0.4668 | -0.0045 | +1,152 / -1,091 | 9/16 · 14/16 |
| w/o Minority-Pocket (Fiedler Seam) | 155,532 | 0.8806 (+0.0411) | 0.7494 | 0.6787 (+0.0220) | 0.4387 | +0.0702 | -4,606 / +87,193 | 16/16 · 13/16 |
| w/o Extract-Then-Compare (Action-First) | 155,551 | 0.8933 (+0.0537) | 0.7972 | 0.6880 (+0.0312) | 0.4622 | +0.0905 | -18,459 / +107,883 | 15/16 · 15/16 |
| w/o Adaptive Bucket (Uniform 10k/blk) | 154,103 | 0.8854 (+0.0458) | 0.7633 | 0.6904 (+0.0337) | 0.4618 | +0.0766 | -1,749 / +87,164 | 16/16 · 16/16 |

### 逐动作判决混淆矩阵与状态审计（消融变体）

- `merge_only` | statuses={'CACHED_REPLAY': 301} | `MERGE`={'TP': 72, 'FP': 4, 'TN': 135, 'FN': 90, 'precision': 0.9474, 'recall': 0.4444, 'specificity': 0.9712} | `SEAM`={'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0}
- `seam_only` | statuses={'CACHED_REPLAY': 102} | `MERGE`={'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0} | `SEAM`={'TP': 37, 'FP': 28, 'TN': 32, 'FN': 5, 'precision': 0.5692, 'recall': 0.881, 'specificity': 0.5333}
- `fiedler_only_seam` | statuses={'CACHED_REPLAY': 300, 'HTTP_ERROR:URLError': 2} | `MERGE`={'TP': 68, 'FP': 4, 'TN': 91, 'FN': 75, 'precision': 0.9444, 'recall': 0.4755, 'specificity': 0.9579} | `SEAM`={'TP': 21, 'FP': 20, 'TN': 21, 'FN': 2, 'precision': 0.5122, 'recall': 0.913, 'specificity': 0.5122}
- `no_extract_compare` | statuses={'CACHED_REPLAY': 359, 'HTTP_ERROR:URLError': 4, 'OK': 1} | `MERGE`={'TP': 111, 'FP': 19, 'TN': 102, 'FN': 41, 'precision': 0.8538, 'recall': 0.7303, 'specificity': 0.843} | `SEAM`={'TP': 33, 'FP': 18, 'TN': 33, 'FN': 7, 'precision': 0.6471, 'recall': 0.825, 'specificity': 0.6471}
- `uniform_budget` | statuses={'CACHED_REPLAY': 294} | `MERGE`={'TP': 68, 'FP': 2, 'TN': 76, 'FN': 62, 'precision': 0.9714, 'recall': 0.5231, 'specificity': 0.9744} | `SEAM`={'TP': 33, 'FP': 25, 'TN': 24, 'FN': 4, 'precision': 0.569, 'recall': 0.8919, 'specificity': 0.4898}

---

## Event2018（法语 · M_06..M_16，11 块）（初始 2D-SE: AMI=0.7634, ARI=0.5160, 宏 B³=0.6790, 长尾 B³=0.3500）

| 变体 | 实耗 Token | AMI ($\Delta$) | ARI | 宏 $B^3$ ($\Delta$) | 长尾 $B^3$ | 未查询 $\Delta$微 $B^3$ | 净减 FM / 净减 FS | 胜率 (AMI / 宏 $B^3$) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| SUG-Meso (Full, Direct) | 104,306 | 0.7977 (+0.0344) | 0.6677 | 0.7127 (+0.0337) | 0.4308 | +0.0745 | -15,306 / +166,729 | 11/11 · 10/11 |
| SUG-Meso (Branch 2 Repair) | 104,306 | 0.7968 (+0.0334) | 0.6661 | 0.7096 (+0.0306) | 0.4194 | +0.0734 | -15,481 / +166,337 | 11/11 · 10/11 |
| SUG-Router (+Bridge/Seed, Repair) | 104,985 | 0.7875 (+0.0241) | 0.6234 | 0.7038 (+0.0248) | 0.4084 | +0.0535 | -9,474 / +117,148 | 10/11 · 9/11 |
| w/o SPLIT_SEAM (Merge-Only) | 102,526 | 0.8066 (+0.0432) | 0.6908 | 0.6980 (+0.0190) | 0.3469 | +0.0906 | -19,555 / +180,473 | 10/11 · 10/11 |
| w/o MERGE_PAIR (Seam-Only) | 42,187 | 0.7620 (-0.0014) | 0.5133 | 0.6959 (+0.0169) | 0.4302 | -0.0061 | +3,034 / -3,769 | 3/11 · 8/11 |
| w/o Minority-Pocket (Fiedler Seam) | 103,609 | 0.8001 (+0.0367) | 0.6705 | 0.7067 (+0.0277) | 0.3777 | +0.0793 | -17,400 / +170,760 | 11/11 · 9/11 |
| w/o Extract-Then-Compare (Action-First) | 106,339 | 0.8357 (+0.0723) | 0.7996 | 0.7262 (+0.0472) | 0.4217 | +0.1442 | -88,504 / +298,077 | 11/11 · 11/11 |
| w/o Adaptive Bucket (Uniform 10k/blk) | 106,262 | 0.8101 (+0.0467) | 0.7123 | 0.7130 (+0.0340) | 0.4276 | +0.0943 | -14,424 / +172,895 | 11/11 · 11/11 |

### 逐动作判决混淆矩阵与状态审计（消融变体）

- `merge_only` | statuses={'CACHED_REPLAY': 159} | `MERGE`={'TP': 38, 'FP': 1, 'TN': 77, 'FN': 43, 'precision': 0.9744, 'recall': 0.4691, 'specificity': 0.9872} | `SEAM`={'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0}
- `seam_only` | statuses={'CACHED_REPLAY': 68} | `MERGE`={'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0} | `SEAM`={'TP': 22, 'FP': 21, 'TN': 25, 'FN': 0, 'precision': 0.5116, 'recall': 1.0, 'specificity': 0.5435}
- `fiedler_only_seam` | statuses={'CACHED_REPLAY': 163} | `MERGE`={'TP': 31, 'FP': 1, 'TN': 57, 'FN': 39, 'precision': 0.9688, 'recall': 0.4429, 'specificity': 0.9828} | `SEAM`={'TP': 13, 'FP': 10, 'TN': 11, 'FN': 1, 'precision': 0.5652, 'recall': 0.9286, 'specificity': 0.5238}
- `no_extract_compare` | statuses={'CACHED_REPLAY': 194, 'OK': 2} | `MERGE`={'TP': 64, 'FP': 3, 'TN': 63, 'FN': 10, 'precision': 0.9552, 'recall': 0.8649, 'specificity': 0.9545} | `SEAM`={'TP': 17, 'FP': 6, 'TN': 29, 'FN': 4, 'precision': 0.7391, 'recall': 0.8095, 'specificity': 0.8286}
- `uniform_budget` | statuses={'CACHED_REPLAY': 168, 'OK': 1} | `MERGE`={'TP': 33, 'FP': 0, 'TN': 49, 'FN': 32, 'precision': 1.0, 'recall': 0.5077, 'specificity': 1.0} | `SEAM`={'TP': 21, 'FP': 16, 'TN': 17, 'FN': 1, 'precision': 0.5676, 'recall': 0.9545, 'specificity': 0.5152}

---

