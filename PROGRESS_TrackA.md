# Track A 工作最新进度与核心结论归档（`SUG-SED v2.1` · `Pure-Star-Meso` + `SUG-CrossStar`）

> **定位与投稿目标**：面向流式社交事件检测的预算受限轻量开源 LLM 方法（目标期刊 *Information Processing & Management*）。  
> **最新状态**：`v2.1` 五阶段主流程审查重构（先拆后合 + 统一 `GuardedUnionFind` + 无监督 $p_{95}$ 自校准 + 滚动代表文本与同块合流双守卫）、基线公平性加固、全链路单元测试（[test_sug_sed.py](./tests/test_sug_sed.py) 全 PASS）、工作区精简以及 **双语（`Event2012` & `Event2018`）Dev 7 基线公平评测**（[event2012_v21_7baselines.md](./results/dev_gate1/event2012_v21_7baselines.md) · [event2018_v21_7baselines.md](./results/dev_gate1/event2018_v21_7baselines.md)）均已完成；当前源码相对 `v2.0` 冻结快照（[protocol_v2.json](./trackA_sug_sed/protocol_v2.json)）已有 `17/20` 文件演进（`3/20` 未变），待 `v2.1` 全量实验重跑后重冻结。

---

## 一、`SUG-SED v2.1` 最新实证进度（`Event2012` & `Event2018` Dev 7 基线公平评测）

对应报告：[event2012_v21_7baselines.md](./results/dev_gate1/event2012_v21_7baselines.md) · [event2018_v21_7baselines.md](./results/dev_gate1/event2018_v21_7baselines.md)（`M_02..M_05`，`Qwen/Qwen3-8B`，`10,000` tokens/块，双数据集全 `0` 解析错误）：

1. **`Event2012` Dev（`M_02..M_05`）**：
   * **`Pure-Star-Meso v2.1`**：仅耗 **`64 calls / 33,104 tok`**（`3.49%` 展示，`CPU 1.41s`），取得 **`AMI 93.13 / Mac-B³ 81.00 / Mic-B³ 90.16 / Tail-B³ 63.73 / UnqNet +634.8`**。
   * **对比 0-LLM 参照与 7 篇基线**：`Pure 2D-SE Base (0.50, 0.5)` `88.88 / 78.17 / 82.18`、`AutoMerge-Only (0-Tok)` `91.19 / 79.28 / 87.18`、`SBERT-Agglom (EMNLP'19, d=0.6)` `90.51 / 77.65 / 87.74`、`HiSEvent Default` `88.53 / 69.17 / 82.10`、`HiSEvent Dev-Tuned` `88.88 / 78.17 / 82.18`、`HyperSED` `90.13 / 73.38 / 85.17`、`ClusterLLM-NoFT` `86.28 / 74.68 / 81.33`（`UnqNet -726.8`）、`Few-Shot LLM-Post` `88.85 / 78.24 / 82.15`（`+78.5`）、`LSED` `88.90 / 77.43 / 82.36`（`-16.2`）、`RagSEDE` `88.85 / 78.17 / 82.12`（`-35.0`）。
2. **`Event2018` Dev（`M_02..M_05`）**：
   * **`Pure-Star-Meso v2.1`**：仅耗 **`48 calls / 38,318 tok`**（`1.94%` 展示，`CPU 2.41s`），取得 **`AMI 78.14 / Mac-B³ 78.37 / Mic-B³ 78.64 / Tail-B³ 66.92 / UnqNet +1426.0`**（无监督 $p_{95}$ 自校准在 `M_02..M_04` 抬升 `eff_tau_auto=0.90`、在 `M_05` 抬升至 `0.8704`，Seam 拆分 `9/12` 次）。
   * **对比 0-LLM 参照与 7 篇基线**：`Pure 2D-SE Base (0.50, 0.5)` `68.68 / 71.24 / 59.69`、`AutoMerge-Only (0-Tok)` `69.77 / 72.11 / 62.00`、`SBERT-Agglom (d=0.6)` `67.52 / 71.24 / 68.56`、`HiSEvent Default` `68.06 / 64.02 / 59.12`、`HiSEvent Dev-Tuned (0.60, 1.0)` `67.36 / 76.12 / 57.80`（`Tail 77.99`）、`HyperSED (ar=0.26)` `66.83 / 65.69 / 57.79`、`ClusterLLM-NoFT` `60.35 / 72.81 / 53.51`（`Tail 82.04`, `UnqNet -1517.5`）、`Few-Shot LLM-Post` `68.84 / 71.32 / 59.78`（`+540.0`）、`LSED` `68.37 / 70.77 / 58.97`（`+264.8`）、`RagSEDE` `68.72 / 71.24 / 59.70`（`+83.0`）。

---

## 二、`SUG-SED v2.0-frozen` 完整实证档案（待 `v2.1` 全量重跑刷新）

> 以下结果源自 `SUG-SED-v2.0-frozen` 完整评测快照（[dev_gate1_report.md](./results/dev_gate1/dev_gate1_report.md)、[gate2_full_report.md](./results/heldout_gate2/gate2_full_report.md)、[p1_mechanisms_report.md](./results/heldout_gate2/p1_mechanisms_report.md)），作为 `v2.1` 全量重跑前的参考基线：

1. **协议冻结**：`SUG-SED-v2.0-frozen` 在 `10k` 前瞻桶下实测最大末次溢出仅 `+0.35%`。
2. **Gate 1 Dev（`M_02..M_05`，`Qwen3-8B` & `Qwen2.5-7B-Instruct`，全 `0` 错误）**：
   * `Qwen3-8B`：`Event2012` 达 **`AMI 94.06 / Mac-B³ 81.31 / Mic-B³ 92.13 / Tail-B³ 63.73 / UnqNet +649.0`**；`Event2018` 达 **`78.80 / 79.05 / 81.01 / 68.70 / +1358.2`**。
   * `Qwen2.5-7B-Instruct`：`Event2012` 达 **`94.08 / 81.40 / 92.28 / 63.41 / +723.2`**；`Event2018` 达 **`75.61 / 76.29 / 76.17 / 60.51 / +1436.8`**。
   * Dev 扩网确认零 LLM 基线真内点最优（`EventX` EN `2.0` / FR `1.5`；`HiSEvent Dev-Tuned` EN `(0.50,0.5)` / FR `(0.55,1.0)`；`HyperSED` EN `0.26` / FR `0.14`；`SBERT-Agglom` `0.60`）。
3. **Gate 2 全部 `27` 个留出块主实验（`Qwen/Qwen3-8B`，vs 10 篇已发表论文，全 `0` 错误）**：
   * **`Event2012`（16 块，`32,695` 帖）**：`Pure-Star-Meso` 达 **`AMI 90.15 / Mac-B³ 76.56 / Mic-B³ 86.91 / Tail-B³ 64.06 / UnqNet +687.0`**（`272 calls`, `137,201 tok`, `2.92%` 展示，`Router CPU 22.1s / E2E CPU 35.3s / API 948.1s`）。
   * **`Event2018`（11 块，`33,161` 帖）**：`Pure-Star-Meso` 达 **`AMI 82.69 / Mac-B³ 75.27 / Mic-B³ 81.74 / Tail-B³ 57.58 / UnqNet +1179.5`**（`142 calls`, `105,173 tok`, `1.84%` 展示，`Router CPU 18.9s / E2E CPU 50.7s / API 331.9s`）；叠在 `HiSEvent Dev-Tuned (0.55, 1.0)` 上达 **`84.43 / 79.57 / 84.52 / 68.11 / +1493.8`**。
   * **公平修复后的 `ClusterLLM-NoFT (EMNLP'23)`**：采用抗负偏斜平衡准确率 `(bal_acc, F1, acc)`、动态分量 `CANNOT-LINK` 与 `SBERT-Agglom (0.60)` 底座 `UnqNet` 后，EN 达 `84.21 / 75.44 / 81.46`（`UnqNet -979.6`），FR 达 `64.96 / 72.22 / 60.71`（`UnqNet -1695.0`）。
   * **`Pooled27` 配对双侧 Wilcoxon + Holm-Bonferroni**：对**全部 10 篇论文**的 `AMI`（$p_{\text{Holm}} \le 2.51\times 10^{-4}$）与 $\mathrm{Mic}\text{-}B^3$（$p_{\text{Holm}} \le 4.17\times 10^{-4}$）均显著胜出；$\mathrm{Mac}\text{-}B^3$ 均值领先全部 10 篇且对 9/10 篇显著（$p_{\text{Holm}} \le 0.0267$；对 `ClusterLLM-NoFT` 达 `+1.91 Mac-B³`, $p_{\text{Holm}}=0.1286$）；相对 `Meso-Heur-Best`（`+1.56 Mac-B³ p_Holm=5.49e-4`）与 3-seed `Random-Meso`（`+0.89 Mac-B³ p_Holm=0.0352`）稳健占优。
4. **P1 机制归因、Cross-Day、预算曲线与端到端脱敏**：
   * **三因子与五支柱分层**：`Star Packing` 与 `Clean Seam Split`（`Mac p_Holm=3.18e-4`, `Tail p_Holm=1.53e-4`）为核心通用引擎；`AutoMerge (>=0.85)`（EN 精度 `98.4%` vs FR `87.1%`）与 `GuardedUF`（FR 阻断 `+1,446,601` 误合并对）作为效率与尾部防雪崩守卫。
   * **Cross-Day（`M_06+`）**：FR `sug_cross_star` 达 **`84.76 G-AMI / 72.72 G-MacB³ / 75.73 G-MicB³`**（击败全部原型阈值），EN 达 `90.62 / 73.77`。
   * **因果预算曲线 & E2E 全脱敏**：`2k`/块（`0.52–0.75 tok/帖`）超越 `20k` 单帖/单边基线；分层 8 块 `e2e_pseudonym` 达 **`69.95 / 56.49 AMI`**（`LeakQ=3/57` EN / `1/43` FR），击败同全脱敏底座 `e2e_pseudonym_meso_heur_0tok`（`67.94 / 54.62`）。

---

## 三、演进备忘（`v1 → v2.0 → v2.1`）

* **`v1 → v2.0`**：剥离在消融中呈负向/冗余贡献的 `Log-Leverage`、硬电导阈值及复杂 Eq. (5)/(6) 边权，确立高纯度 `Pure 2D-SE` + `AutoMerge` + `GuardedUF` + `Clean Seam Split` + `Star-Merge` + `SUG-CrossStar` 纯净架构与无截断端到端 `e2e_pseudonym`。
* **`v2.0 → v2.1`**：
  1. **五阶段主流程定型（先拆后合）**：在 Stage 1 `Pure 2D-SE` 后先执行 Stage 2 `Clean Seam Split` 预清洗混杂大簇并注入硬 `CANNOT-LINK`，再以真实成员向量和 `comp_vecs` 初始化贯穿 Stage 3–4 的**唯一** `GuardedUnionFind`；
  2. **编码器各向异性自校准 & 并发/错误处理**：Stage 3 引入无监督 $p_{95}$ 背景分位数自校准 `eff_tau_auto`；Stage 4 隔离 API/越界解析错误至 `failed_root_pairs`，`llm_client` 统一 `3.4` 字符/token 预检并兼容历史 3 字段缓存键；
  3. **Cross-Day 双守卫与基线公平加固**：Stage 5 引入滚动锚定代表文本及同块 `CANNOT-LINK` + 余弦下界 `0.40` 双守卫；`HyperSED` 放宽 `max_anchors=800` 并去除互近邻重复计权，统一基线 $L_2$ 归一化、`e2e_local_sec` 计时、成对等价规则与 `select_display_medoids`。

---

## 四、下一步待办与核心文件索引

* **待办**：
  1. 补跑 `v2.1` 全量 Gate 1（含内部对照、消融与 `Qwen/Qwen2.5-7B-Instruct` 跨模型）及 Gate 2（`27` 留出块，`Qwen/Qwen3-8B` + 可选 `Qwen/Qwen2.5-7B-Instruct`）；
  2. 运行 `python trackA_sug_sed/freeze_protocol_v2.py` 重新生成与 `v2.1` 源码 `20/20` SHA-256 一致的 `protocol_v2.json`；
  3. 基于刷新后的 `results/heldout_gate2/` 数据绘制 IP&M 出版级矢量图表并完成正式 LaTeX 论文成稿。
* **核心文件索引**：
  * 索引与规格：[README.md](./README.md) · [TrackA_SED_Method_Spec.md](./TrackA_SED_Method_Spec.md) · [protocol_v2.json](./trackA_sug_sed/protocol_v2.json)
  * 核心方法与底层模块：[core_shared/](./core_shared/data_loader.py) · [trackA_sug_sed/](./trackA_sug_sed/star_meso.py) · [tests/test_sug_sed.py](./tests/test_sug_sed.py)
  * 实证报告：[event2012_v21_7baselines.md](./results/dev_gate1/event2012_v21_7baselines.md) · [event2018_v21_7baselines.md](./results/dev_gate1/event2018_v21_7baselines.md)（v2.1 Dev）· [dev_gate1_report.md](./results/dev_gate1/dev_gate1_report.md) · [gate2_full_report.md](./results/heldout_gate2/gate2_full_report.md) · [p1_mechanisms_report.md](./results/heldout_gate2/p1_mechanisms_report.md)（v2.0 快照）
