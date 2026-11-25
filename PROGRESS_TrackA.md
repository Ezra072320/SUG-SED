# Track A 工作最新进度与核心结论归档（`SUG-SED v2` · `Pure-Star-Meso` + `SUG-CrossStar`）

> **定位与投稿目标**：面向流式社交事件检测的预算受限轻量开源 LLM 方法（目标期刊 *Information Processing & Management*）。  
> **最新状态**：v2 架构重构、单元测试、Gate 1 Dev 双千问闭环、Gate 2 全部 `27` 留出块（`65,856` 帖 · `Qwen/Qwen3-8B`）实证套件、P1 机制归因定界、第一优先级审计封箱、第二优先级基线公平性加固与独立方法代码包拷贝均已 **100% 完成**（全 `0` 解析错误）。

---

## 一、当前最新实证进度快照

1. **协议冻结与独立方法包**：
   * Dev 超参全参锁定于 [protocol_v2.json](./trackA_sug_sed/protocol_v2.json)（`SUG-SED-v2.0-frozen`，`20/20` 源码 SHA-256 一致，`10k` 前瞻桶最大末次溢出 `+0.35%`）；
   * 已按独立拷贝方式从 [core_shared/](./core_shared/data_loader.py) 与 [trackA_sug_sed/](./trackA_sug_sed/star_meso.py) 提取五阶段主体方法流程至独立包 `sug_sed_method/`（零外部基线依赖，原仓文件未作改动）。
2. **Gate 1 Dev（`M_02..M_05`，`Qwen3-8B` & `Qwen2.5-7B-Instruct`，全 `0` 错误）**：
   * `Qwen3-8B`：`Event2012` 达 **`AMI 94.06 / Mac-B³ 81.31 / Mic-B³ 92.13 / Tail-B³ 63.73 / UnqNet +649.0`**；`Event2018` 达 **`78.80 / 79.05 / 81.01 / 68.70 / +1358.2`**。
   * `Qwen2.5-7B-Instruct`：`Event2012` 达 **`94.08 / 81.40 / 92.28 / 63.41 / +723.2`**；`Event2018` 达 **`75.61 / 76.29 / 76.17 / 60.51 / +1436.8`**。
   * Dev 扩网确认零 LLM 基线真内点最优（`EventX` EN `2.0` / FR `1.5`；`HiSEvent Dev-Tuned` EN `(0.50,0.5)` / FR `(0.55,1.0)`；`HyperSED` EN `0.26` / FR `0.14` 伴 `n_anchors <= 360`；`SBERT-Agglom` `0.60`）。
3. **Gate 2 全部 `27` 个留出块主实验（`Qwen/Qwen3-8B`，vs 10 篇已发表论文，全 `0` 错误）**：
   * **`Event2012`（16 块，`32,695` 帖）**：`Pure-Star-Meso` 达 **`AMI 90.15 / Mac-B³ 76.56 / Mic-B³ 86.91 / Tail-B³ 64.06 / UnqNet +687.0`**（`272 calls`, `137,201 tok`, `2.92%` 展示，`Router CPU 22.1s / E2E CPU 35.3s / API 948.1s`）。
   * **`Event2018`（11 块，`33,161` 帖）**：`Pure-Star-Meso` 达 **`AMI 82.69 / Mac-B³ 75.27 / Mic-B³ 81.74 / Tail-B³ 57.58 / UnqNet +1179.5`**（`142 calls`, `105,173 tok`, `1.84%` 展示，`Router CPU 18.9s / E2E CPU 50.7s / API 331.9s`）；叠在新内点 `HiSEvent Dev-Tuned (0.55, 1.0)` 上净提 `+8.37 AMI / +4.05 Mac-B³`，达 **`84.43 / 79.57 / 84.52 / 68.11 / +1493.8`**。
   * **公平修复后的 `ClusterLLM-NoFT (EMNLP'23)`**：改用抗负偏斜平衡准确率 `(bal_acc, F1, acc)`、动态分量 `CANNOT-LINK` 与 `SBERT-Agglom (0.60)` 底座 `UnqNet` 后，EN 恢复至 `84.21 / 75.44 / 81.46`（`UnqNet -979.6`），FR 恢复至 `64.96 / 72.22 / 60.71`（`UnqNet -1695.0`）。
   * **`Pooled27` 配对双侧 Wilcoxon + Holm-Bonferroni**：对**全部 10 篇论文**的 `AMI`（$p_{\text{Holm}} \le 2.51\times 10^{-4}$）与 $\mathrm{Mic}\text{-}B^3$（$p_{\text{Holm}} \le 4.17\times 10^{-4}$）均显著胜出；$\mathrm{Mac}\text{-}B^3$ 均值领先全部 10 篇且对 9/10 篇显著（$p_{\text{Holm}} \le 0.0267$；对 `ClusterLLM-NoFT` 达 `+1.91 Mac-B³`, `17/0/10`, $p_{\text{Holm}}=0.1286$）；相对同池 `Meso-Heur-Best`（`+1.56 Mac-B³ p_Holm=5.49e-4`）与 3-seed `Random-Meso`（`+0.89 Mac-B³ p_Holm=0.0352`）稳健占优。
4. **P1 机制归因、Cross-Day、预算曲线与端到端脱敏**：
   * **三因子与五支柱分层**：`Star Packing`（省调用 `1.93–2.70×`，FR 提 `3.42 AMI`）与 `Clean Seam Split`（`Mac p_Holm=3.18e-4`, `Tail p_Holm=1.53e-4`）为核心通用引擎；`AutoMerge (>=0.85)`（EN 精度 `98.4%` vs FR `87.1%`）与 `GuardedUF`（FR 阻断 `+1,446,601` 误合并对）作为效率与尾部防雪崩守卫；
   * **Cross-Day（`M_06+`）**：FR `sug_cross_star` 达 **`84.76 G-AMI / 72.72 G-MacB³ / 75.73 G-MicB³`**（击败全部原型阈值），EN 达 `90.62 / 73.77`（与 `proto_0.90` 持平）；
   * **因果预算曲线 & E2E 全脱敏**：`2k`/块（`0.52–0.75 tok/帖`）超越 `20k` 单帖/单边基线（`9.7–12.8×` 效率）；分层 8 块 `e2e_pseudonym` 达 **`69.95 / 56.49 AMI`**（`LeakQ=3/57` EN / `1/43` FR），击败同全脱敏底座 `e2e_pseudonym_meso_heur_0tok`（`67.94 / 54.62`）`+2.01 / +1.87 AMI` 与 `+355.8 / +1076.8 UnqNet`。

---

## 二、旧工作精简备忘（`v1 → v2` 关键经验，完整历史已归档）

* v1 初版中 `Log-Leverage`、硬电导阈值、复杂 Eq. (5)/(6) 边权在消融中呈负向/冗余贡献，且单日特征未做端到端编码前脱敏；v2 已全部剥离并重构为纯净 5 大支柱与无截断端到端 `e2e_pseudonym`。
* v1 完整代码、草稿及旧日志已封存于 [archive_v1/](./archive_v1/Proposal_TrackA_SUG-SED.md)，不参与当前评测。

---

## 三、下一步可选工作与索引

* **待办**：
  1. （可选 · 第三优先级）补跑 `Qwen/Qwen2.5-7B-Instruct` 全部 27 个留出块跨模型矩阵（`python trackA_sug_sed/run_heldout_gate2.py --stage qwen25_full`）；
  2. 基于 `results/heldout_gate2/` 生成 IP&M 出版级矢量图表并完成正式 LaTeX 论文成稿。
* **核心文件索引**：
  * 方案总纲：[Proposal_TrackA_SUG-SED.md](./Proposal_TrackA_SUG-SED.md) · 冻结协议：[protocol_v2.json](./trackA_sug_sed/protocol_v2.json)
  * 核心方法与底层模块：[core_shared/](./core_shared/data_loader.py) · [trackA_sug_sed/](./trackA_sug_sed/star_meso.py) · [tests/test_sug_sed.py](./tests/test_sug_sed.py)
  * 最新实证报告：[dev_gate1_report.md](./results/dev_gate1/dev_gate1_report.md) · [gate2_full_report.md](./results/heldout_gate2/gate2_full_report.md) · [p1_mechanisms_report.md](./results/heldout_gate2/p1_mechanisms_report.md)
