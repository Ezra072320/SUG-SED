# SED（社交事件检测）双轨研究规划与门控裁决总览（v5 · Track A v2 `Pure-Star-Meso` 实证更新版）

> **核心研究规范**（贯穿全部实验与文档）：  
> 1. **严禁非证据性预期表述**：不作任何“保底 / 零风险 / 多篇顶会”承诺，所有结论均以**冻结协议、可复核真实 Token 账本、未见保留块实测结果**为准。  
> 2. **先讲透影响整条事件流的系统性失效，再设计并定界技术**：通过受控机制对照和单因素消融诚实剥离有效组件与失效假设，绝不保留与数据相悖的过度主张。  
> 3. **主次分明**：**Track A（`SUG-SED v2` · `Pure-Star-Meso`）已全面完成算法主体重构、Gate 1 Dev 双千问闭环验证与 Gate 2 全部 27 个留出块（`Event2012 M_06..M_21` + `Event2018 M_06..M_16`，`65,856` 帖）在 `Qwen/Qwen3-8B` 上的全量实证套件**（含 10 篇已发表论文基线 + 配对 Wilcoxon/Holm + 5 支柱消融 + 全量 Cross-Day + 5 档因果预算曲线 + 4 模式含端到端全脱敏 `e2e_pseudonym` 反记忆审计；v1 旧工作已完整归档至 [archive_v1/](./archive_v1/Proposal_TrackA_SUG-SED.md)），当前仅待按需补齐 `Qwen/Qwen2.5-7B-Instruct` 留出集跨模型验证后即可撰写 IP&M 正式文稿；**Track B（`Evid-MSED`）已完成 Gate B 诊断以及 `Validation → Holdout` 端到端 Phase 2 冻结评测（`Qwen/Qwen3-VL-8B-Instruct`，24 保留事件 / `11,955` 帖，`360/360 OK`）**。

---

## 一、双轨当前进展总览

| 维度 | Track A：`SUG-SED v2`（当前主线 · **27 块 `Qwen3-8B` 全套实证已完成**） | Track B：`Evid-MSED`（多模态主线 · **Gate B & Phase 2 已完成**） |
| :--- | :--- | :--- |
| **详细方案书** | [Proposal_TrackA_SUG-SED.md](./Proposal_TrackA_SUG-SED.md) *(v2 `Pure-Star-Meso` 实证定稿总纲)* · [protocol_v2.json](./trackA_sug_sed/protocol_v2.json) · [v1 归档](./archive_v1/Proposal_TrackA_SUG-SED.md) | [Proposal_TrackB_Evid-MSED.md](./Proposal_TrackB_Evid-MSED.md) *(v2 门控草案)* · [protocol_trackB_v1.json](./trackB_evid_msed/protocol_trackB_v1.json) |
| **核心科学问题** | **在严格零前瞻因果真实 Token 预算（`10k/blk` ≈ `3.17–4.20 tok/帖`，`all_shown` 展示率 `1.84%–2.92%`）下，为何单帖/单边 LLM 基线陷入 *Single-Post Fallacy*；纯净 `Pure 2D-SE + AutoMerge + GuardedUF + Clean Seam Split + Star-Merge` 如何在 10 篇基线论文与端到端脱敏条件下释放跨簇净改善与统计显著性？** | **在剔除时间捷径的同期同主题竞争事件中，图片究竟挽回了多少文本漏连（$N_{\text{rescue}}$）、制造了多少误合并（$N_{\text{mislead}}$），能否可靠分离二者？** |
| **关键实证状态** | **Gate 1 Dev（双千问）+ Gate 2 全部 27 留出块（`Qwen/Qwen3-8B`）+ P1 机制定界 + Priority 2 基线公平性加固完成**：<br>1. **Gate 1 Dev（`M_02..M_05` 双千问，10 篇论文对照）**：`Qwen3-8B` 达 `EN 94.06/81.31`、`FR 78.80/79.05`；`Qwen2.5-7B-Instruct` 达 `EN 94.08/81.40`、`FR 75.61/76.29`（全 0 错误；Dev 扩展网格确认 `EventX/HiSEvent/HyperSED` 内点最优，`ClusterLLM-NoFT` 升级为抗负偏斜 `bal_acc` 粒度选择与动态 `CANNOT-LINK`，全参冻结至 [protocol_v2.json](./trackA_sug_sed/protocol_v2.json)）；<br>2. **Gate 2 27 块全量主对比（`Qwen3-8B`，vs 10 篇已发表论文 + 控制组，全 0 错误）**：`Event2012`（16 块，`32,695` 帖）`Pure-Star-Meso` 达 **`AMI 90.15 / Mac-B³ 76.56 / Mic-B³ 86.91 / UnqNet +687.0`**（`err=0/272`, `Router CPU 22.1s / E2E CPU 35.3s / API 948.1s`）；`Event2018`（11 块，`33,161` 帖）达 **`AMI 82.69 / Mac-B³ 75.27 / Mic-B³ 81.74 / UnqNet +1179.5`**（`err=0/142`, `Router CPU 18.9s / E2E CPU 50.7s / API 331.9s`；叠在 Dev 内点最优 `HiSEvent Dev-Tuned (0.55, 1.0)` 上达 **`84.43 / 79.57 / 84.52 / UnqNet +1493.8`**）；Pooled 27 块对**全部 10 篇论文**的 `AMI`（$p_{\text{Holm}} \le 2.51\times 10^{-4}$）与 $\mathrm{Mic}\text{-}B^3$（$p_{\text{Holm}} \le 4.17\times 10^{-4}$）均显著胜出，且在 $\mathrm{Mac}\text{-}B^3$ 上显著胜出 9/10 篇（$p_{\text{Holm}} \le 0.0267$；修正负偏斜后的 `ClusterLLM-NoFT` 为 `+1.91 Mac-B³`, $p_{\text{Holm}}=0.1286$）；<br>3. **Gate 2 P1 机制归因 / 消融 / 跨天 / 预算 / 端到端反记忆（全 0 错误）**：<br>  - **受控三因子拆解 & 核心引擎 vs 守卫定界**：`Random → Random-Meso (随机调度顺序 + Top-4 星型打包) → Full` 证实**介观星型结构诱导先验**贡献 `+5.80 AMI / +10.52 Mic-B³ / +1.91 Mac-B³`，校准调度再添 `+0.89 Mac-B³`（`p_Holm=0.0352`，相对同池 0-Tok `Meso-Heur-Best` 达 `+1.56 Mac-B³ p_Holm=5.49e-4`）；5 支柱 Wilcoxon 确证 **`Star Packing`**（`AMI p_raw=0.0451/p_Holm=0.226`, `Mic p_raw=0.0219/p_Holm=0.110`，省调用 `1.93–2.70×`）与 **`Seam Split`**（`Mac p_raw=6.37e-5/p_Holm=3.18e-4`, `Tail p_raw=3.05e-5/p_Holm=1.53e-4`）为核心通用引擎，而 **`AutoMerge` / `GuardedUF`** 的英法不对称收益归因于 `size>=2` 编码器余弦校准差异（EN `>=0.85` 精度 `98.4%` vs FR `87.1%`）与传递链式误合并防爆（`GuardedUF` 在 FR 阻断 `+1,446,601 FM`）；<br>  - **全量 Cross-Day（`M_06+`）**：FR `sug_cross_star` (`84.76 / 72.72 / 75.73`, Q=91) 击败全部原型阈值；EN (`90.62 / 73.77`, Q=181) 与 `proto_0.90` (`90.72 / 73.81`) 持平；<br>  - **27 块预算曲线（`B=0..20k`）**：`Pure-Star-Meso (2k, 0.52–0.75 tok/帖)` 即超越 `FewShotPost / Cequel-Trans (20k)`（**9.7–12.8× Token 效率**）；<br>  - **分层 8 块 4 模式反记忆**：新增**无截断端到端全脱敏 `e2e_pseudonym`**（构图编码前映射 `4,140/7,886` 实体，块级实体遮蔽 `100.0%` EN / `99.47%–99.81%` FR；`medoids[:2]` 原文本独立审计 `EntMask 98.2% / 97.6%`、`TagMask 100.0%`、`LeakQ=3/57` 与 `1/43`，即 `94.7%–97.7%` 查询零泄漏），相对 `e2e_pseudonym_automerge_0tok`（`67.22 / 53.47`）及新增 0-API `e2e_pseudonym_meso_heur_0tok`（`67.94 / 54.62`）仍取得 **`69.95 / 56.49 AMI`**（净提 **`+2.01 / +1.87` $\mathrm{AMI}$**、**`+1.10 / +1.97` $\mathrm{Mac}\text{-}B^3$** 与 **`+355.8 / +1076.8 UnqNet`**，全 `0` 错误）。 | **Phase 1 Gate B + Phase 2 NED `Validation → Holdout` 全部完成（`Qwen/Qwen3-VL-8B-Instruct`）**：<br>1. **Gate B（Dev 8 事件）**：`Visual-IDF MV-2DSE` 宏 $B^3$ `+7.39 pp`，`vlm_self_gate` 将原图 Acc 推至 **`85.00%`（`0` 误导，`77→5` 错配翻转）**，证伪 `cntd_raw`；<br>2. **Stage 1（Val/Holdout Clean + Stress-30%）**：Holdout Clean 较纯文本提 `AMI +3.84 pp` / 宏 $B^3$ `+6.38 pp` / 微 $B^3$ `+8.02 pp`；Stress-30% 下 `naive_mv` 跌破纯文本（`-8.11..-13.35 pp`），而 `Visual-IDF` + §3.3 软插补（5/5 提升，Holdout Stress `+3.27 pp`）牢守优势；<br>3. **Stage 2（`360/360 OK` · `277,557 tok`）**：Validation Clean `evid_self_gate_inline` 达 `AMI 0.5748 / 宏B³ 0.5841 (Fix/Q 19.45)`，Stress-30% 由 `clean_fallback` 恢复（`18.85 Fix/Q`）；Holdout `K=60/180` 达成 `Micro-B³ 0.4988–0.5060`、`ARI 0.4898–0.4913` 及 **`5.6×` 于盲信多模态的未查询杠杆（`24.63 Fix/Q @ K=60`）**。 |
| **实证报告与图表索引** | - **v2 最新完整实证报告**：[Gate 1 Dev 双千问报告](./results/dev_gate1/dev_gate1_report.md) · [Gate 2 Held-Out 27 块完整报告](./results/heldout_gate2/gate2_full_report.md) · [P1 机制归因与定界报告](./results/heldout_gate2/p1_mechanisms_report.md)<br>- **Gate 2 结构化数据**：[stage_0tok.json](./results/heldout_gate2/stage_0tok.json) · [stage_qwen3_main.json](./results/heldout_gate2/stage_qwen3_main.json) · [stage_qwen3_abl.json](./results/heldout_gate2/stage_qwen3_abl.json) · [stage_qwen3_crossday.json](./results/heldout_gate2/stage_qwen3_crossday.json) · [stage_budget_anon.json](./results/heldout_gate2/stage_budget_anon.json) · [stage_p1_mechanisms.json](./results/heldout_gate2/stage_p1_mechanisms.json)<br>- v1 历史归档：[archive_v1/](./archive_v1/Proposal_TrackA_SUG-SED.md) | - Gate B 执行裁决：[GateB_Qwen3_VL_8B_执行裁决.md](./results/trackB_gateB/GateB_Qwen3_VL_8B_执行裁决.md) · [图](./results/figures/gateB_ned_dev_qwen3_vl_8b.png)<br>- Phase 2 完整报告：[Phase2_NED_Val_Holdout_EvidMSED_Report.md](./results/trackB_phase2/Phase2_NED_Val_Holdout_EvidMSED_Report.md)<br>- Phase 2 图表与 LaTeX：[主图](./results/figures/trackB_phase2_ned_val_holdout.png) · [Holdout 抗噪与分层诊断图](./results/figures/trackB_phase2_holdout_stress_and_strata.png) · [paper_table_trackB_ned.tex](./results/trackB_phase2/paper_table_trackB_ned.tex)<br>- **完整 ACM 论文源码**：[paper_trackB/main.tex](./paper_trackB/main.tex) · [编译指南](./paper_trackB/README_COMPILE.md) |

---

## 二、分阶段路线图状态（Gate-Driven Roadmap）

| 阶段 | 核心目标 | 当前状态 |
| :--- | :--- | :--- |
| **Phase 0：底座与 v2 单测** | `SocialED` 流式读取、精确全图体积更新 2D-SE、$O(N)$ 指标审计、因果 Token 桶、分区缓存、RPM 节流器、匿名化与 10 篇基线适配器 | **已完成**（[test_sug_sed.py](./tests/test_sug_sed.py) 全量单测 PASS；含 `pseudonym/blind` 匿名化及 Cross-Day 冒烟） |
| **Phase 1：Gate 1 Dev 双千问闭环** | 仅限 Dev 块 `M_02..M_05`：锁定 `Pure-Star-Meso` 与 4 篇 0-LLM 超参，完成 `Qwen/Qwen3-8B` + `Qwen/Qwen2.5-7B-Instruct` 对照 10 篇基线论文验证 | **已完成（PASS · 全 0 错误）**（[dev_gate1_report.md](./results/dev_gate1/dev_gate1_report.md)） |
| **Phase 2：Gate 2 27 块留出套件 & P1 归因审计（`Qwen3-8B`）** | 严格零留出调参跑完 10 篇基线对比（含 Wilcoxon/Holm）、5 支柱消融、全量 Cross-Day、5 档预算曲线、端到端脱敏及 P1 机制归因定界 | **已完成**（[gate2_full_report.md](./results/heldout_gate2/gate2_full_report.md) · [p1_mechanisms_report.md](./results/heldout_gate2/p1_mechanisms_report.md)） |
| **Phase 2.5：第二模型留出扩展（可选）** | 在全部 27 个留出块补跑 **`Qwen/Qwen2.5-7B-Instruct`**（`--stage qwen25_full` 或仅主方法+Cross-Day），补齐双模型留出集跨模型泛化矩阵 | **待用户指令启动**（Dev 集 `Qwen2.5` 闭环已先行 PASS） |
| **Phase 3：IP&M 论文重写与矢量图成稿** | 基于 [Proposal_TrackA_SUG-SED.md](./Proposal_TrackA_SUG-SED.md) 与 `results/heldout_gate2/` 生成出版级图表并重写 *Information Processing & Management* (IP&M) 正式 LaTeX 论文 | **准备就绪（v1 旧手稿已归档于 [archive_v1/paper/](./archive_v1/paper/README_COMPILE.md)）** |
| **Track B Phase 1：Gate B 裁决** | 仅限 NED `development`（8 并发事件，移除 72h 捷径）：Tier 1 全量因果 + MV-2DSE，Tier 2 `Qwen/Qwen3-VL-8B-Instruct` 三条件配对审计 | **已完成（CONDITIONAL PASS）**（`360/360 OK`，Stage 1 采纳 `Visual-IDF`，Stage 2 采纳 `vlm_self_gate`/`comp_evid_gate`，[执行裁决](./results/trackB_gateB/GateB_Qwen3_VL_8B_执行裁决.md)） |
| **Track B Phase 2：NED `Val → Holdout`** | Clean + Stress-30% 验证，冻结 [protocol_trackB_v1.json](./trackB_evid_msed/protocol_trackB_v1.json) 后一次性解锁 24 保留事件（`11,955` 帖）+ 头中尾归因诊断 | **已完成**（`Holdout Clean 360/360 OK` + `Stress 119/120 OK`，[Phase 2 报告](./results/trackB_phase2/Phase2_NED_Val_Holdout_EvidMSED_Report.md)） |
| **Track B Phase 3：LaTeX 论文成稿** | 独立成稿 ACM `sigconf` 完整手稿、BibTeX 与静态语法/数值一致性审计 | **已完成（`verify_latex.py` 0 错误通过）**（[paper_trackB/](./paper_trackB/main.tex)） |

---

## 三、当前仓库实际代码结构

```text
gemini-paper/
├── README.md                              # 双轨研究总览（v5 · Track A v2 更新）
├── Proposal_TrackA_SUG-SED.md             # Track A v2 Pure-Star-Meso 实证定稿总纲
├── Proposal_TrackB_Evid-MSED.md           # Track B 候选方案（v2 门控草案）
├── archive_v1/                            # Track A v1 历史代码、手稿、单测与旧版实证全量归档
├── core_shared/
│   ├── data_loader.py                     # Event2012 / Event2012_100 / Event2018 时序分块加载器（支持 SED_DATASET_DIR）
│   ├── graph_builder.py                   # 稀疏异构消息图 + 非实体 TF-IDF + GPU 嵌入提取
│   ├── se_solver.py                       # 精确二维结构熵最小堆凝聚与不动点精化求解器
│   ├── partition_cache.py                 # 2D-SE 初始分区持久化缓存器（cache/partitions/*.npz）
│   ├── metrics_audit.py                   # O(N) AMI/NMI/ARI/宏微 B³、FM/FS 跃迁与未查询帖归因
│   ├── token_ledger.py                    # 线程安全只追加 Token 账本 + CausalUniform / CausalSliding / Cumulative 桶
│   └── key_loader.py                      # 内存安全 API 密钥解析器（支持 SED_API_KEY_FILE，永不打印落盘）
├── trackA_sug_sed/                        # Track A v2 Pure-Star-Meso 主体流水线与 10 篇论文基线适配
│   ├── star_meso.py                       # Pure-Star-Meso 主算法（AutoMerge + GuardedUF + Clean Seam Split + Star-Merge）
│   ├── cross_day.py                       # 流式全局事件记忆库 GlobalEventMemory 与 SUG-CrossStar 跨天评估
│   ├── llm_client.py                      # OpenAI 兼容真实计费客户端 + 跨线程自适应 RPM 节流器 + JSON 提示构建器
│   ├── anonymizer.py                      # 查询作用域伪名/致盲匿名化器（QueryAnonymizer）与重建泄漏审计
│   ├── embed_cache.py                     # 冻结 HF 句向量编码与分块持久化嵌入缓存
│   ├── baselines_sed.py                   # 4 篇零 LLM 已发表 SED 论文（EventX / SBERT-Agglom / HiSEvent / HyperSED）
│   ├── baselines_llm.py                   # 6 篇同预算 LLM 论文（ClusterLLM / FewShot / Cequel / LSED / RagSEDE / LHG）+ 内部控制组
│   ├── run_dev_gate1.py                   # Gate 1 开发集双千问闭环评测与 Dev 超参锁定
│   ├── run_heldout_gate2.py               # Gate 2 27 块留出集全量评测器（主对比 / Wilcoxon-Holm / 消融 / Cross-Day / 预算 / 端到端脱敏）
│   ├── analyze_gate2_mechanisms.py        # 0-API Gate 2 P1 机制归因、编码器余弦校准与 GuardedUF 防爆审计脚本
│   ├── freeze_protocol_v2.py              # Track A v2 SHA-256 协议冻结脚本
│   └── protocol_v2.json                   # Track A v2 不可变协议快照（SUG-SED-v2.0-frozen）
├── trackB_evid_msed/
│   ├── ned_splits.json                    # NED 泄漏隔离划分（8 dev / 8 val / 24 holdout）
│   ├── ned_loader.py                      # 相对时间并发对齐、无 PIL 图像解码与 dHash64 计算
│   ├── extract_ned_features.py            # 本地 GTX 1650 全离线 MiniLM / CLIP 特征提取
│   ├── mv_2dse_pipeline.py                # 分块 Stage-1 Visual-IDF MV-2DSE 与 §3.3 软补全 / Stress-30% 引擎
│   ├── vlm_client.py                      # OpenAI 兼容多模态客户端（SiliconFlow Qwen3-VL-8B + 双图缝隙裁决）
│   ├── mm_seam_adjudicator.py             # Stage-2 介观双模态缝隙路由与 5 种冲突门控策略
│   ├── run_phase2_ned.py                  # Validation → Holdout 端到端 Phase 2 评测流水线
│   ├── run_holdout_stress_and_diag.py     # Holdout Stress-30% 端到端 + 24 事件头中尾分层/动作混淆诊断
│   ├── freeze_trackB_protocol.py          # Track B SHA-256 协议冻结脚本
│   ├── protocol_trackB_v1.json            # 冻结不可变协议快照（解锁 holdout 前固化）
│   ├── plot_phase2_results.py             # Phase 2 三联出版图表生成器
│   ├── gateB_paired_audit.py              # Tier 1 全量 + Tier 2 VLM 三条件成对因果审计与三联图绘制
│   └── inspect_gateB_cases.py             # VLM 真实挽救 / 自然误导 / 对抗错配案例核查脚本
├── paper_trackB/                          # Track B 完整匿名 ACM sigconf LaTeX 论文（main.tex / sections / figures / verify_latex.py）
├── results/
│   ├── dev_gate1/                         # Track A Gate 1 Dev 双千问评测 JSON、报告与共享缓存
│   ├── heldout_gate2/                     # Track A Gate 2 27 块留出集全量 JSON、Markdown 总报告、预测 npz 与账本
│   ├── trackB_gateB/                      # Track B：Gate B Tier 1/2 因果审计 JSON、Token 账本与执行裁决
│   └── trackB_phase2/                     # Track B：Validation/Holdout 完整 JSON、Phase 2 报告与 LaTeX 表
└── tests/                                 # test_sug_sed.py 全链路单测与各轻量 8B 端点连通性脚本
```
