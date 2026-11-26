# SUG-SED v2.1：面向流式社交事件检测的预算受限轻量开源 LLM 方法

> **研究定位**：面向流式社交事件检测（Streaming Social Event Detection, SED）的介观星型打包裁决与跨块原型演化框架（`Pure-Star-Meso` + `SUG-CrossStar`，目标期刊 *Information Processing & Management*）。

---

## 一、核心文档索引

- **主流程权威规格书**：[**TrackA_SED_Method_Spec.md**](./TrackA_SED_Method_Spec.md)（v2.1 五阶段主流程、数学定义、锁定超参、防雪崩与因果预算不变量及源码映射）
- **最新进度与实证归档**：[**PROGRESS_TrackA.md**](./PROGRESS_TrackA.md)（`v2.1` 最新 Dev 进展与 `v2.0` 27 留出块完整实证对照档案）
- **最新 v2.1 Dev 公平评测**：[**event2012_v21_7baselines.md**](./results/dev_gate1/event2012_v21_7baselines.md) · [**event2018_v21_7baselines.md**](./results/dev_gate1/event2018_v21_7baselines.md)（`Event2012` & `Event2018` `M_02..M_05`，7 篇外部基线 + `Pure-Star-Meso v2.1`）
- **v2.0 留出集全量实证参考**：[**gate2_full_report.md**](./results/heldout_gate2/gate2_full_report.md) · [**p1_mechanisms_report.md**](./results/heldout_gate2/p1_mechanisms_report.md)（`27` 个留出块全量评测与机制归因，待 `v2.1` 全量重跑刷新）

---

## 二、核心创新点

1. **介观两阶段星型打包裁决（Two-Tier Size-Weighted Star-Merge）**：突破小预算下“单帖/单边纠错陷阱”，利用 Stage 1 纯度优先 2D-SE 划分的 $\text{FS} \gg \text{FM}$ 结构非对称性，在两阶段原型相似度带（Tier 1 $[0.55, \tau_{\text{auto}}^{\text{eff}})$ 优先多帖分量 → Tier 2 $[0.45, \tau_{\text{auto}}^{\text{eff}})$ 补全长尾与单帖）上按规模加权预期收益效用调度，单次调用对比裁决 1 个锚分量与至多 `k_nb=4` 个活跃邻分量，并执行在线分量刷新与错误隔离（`failed_root_pairs`）。
2. **“先拆后合”的干净介观缝隙预清洗（Clean Seam Split）**：在任何合并前先对 $|C| \ge 8$ 的潜在混杂大簇并行执行确定性最远点球面 $k$-means 与 Fiedler 谱二分，以语义分离度（$\text{sem\_sep} \ge 0.12$）及精准转发团伙豁免筛选候选缝隙，确认拆分后立即向统一并查集注入硬 `CANNOT-LINK` 约束，防止被后续阶段静默合回并有效提升长尾与宏平均质量（`Mac-B³` / `Tail-B³`）。
3. **无监督编码器各向异性自校准 + 统一质心守卫并查集（Self-Calibrated AutoMerge & Unified `GuardedUnionFind`）**：针对多语编码器高分端余弦抬升问题，以多帖微簇上三角余弦 $p_{95}$ 分位数实现零监督阈值托底 $\tau_{\text{auto}}^{\text{eff}} = \max(\tau_{\text{auto}}, \min(0.90, p_{95}(\mathbf{S}_{\text{upper}}) + 0.28))$（`Event2012` 维持 `0.85`，`Event2018` 自适应抬升至 `0.87–0.90`）；并以真实成员向量和 `comp_vecs` 初始化贯穿 Stage 3–4 的唯一 `GuardedUnionFind`，强制执行 `CANNOT-LINK`、质心余弦下界（`guard_sim=0.40`）与实时合并阈值三重守卫，阻断 2-hop 链式漂移与传递闭包雪崩。
4. **滚动锚定原型记忆与同块合流双守卫跨块演化（`SUG-CrossStar`）**：14 块滑动窗口主动淘汰维持 $O(|E_{\text{active}}|\cdot d)$ 复杂度；保留 Day-0 起源锚定文本与最新跨日演化文本缓解话题漂移；以“日内 `CANNOT-LINK` 硬拦截 + 同块质心余弦 $\ge 0.40$”双守卫替代一刀切块内唯一性假设，在支持同事件残余碎片合法合流的同时实现高置信自动续连（$\ge 0.90$）与不确定带 Top-3 LLM 裁决（$[0.65, 0.90)$）。
5. **严格零前瞻因果预算与可审计评测协议**：统一 `CausalUniformBucket`（块内 `10k` 不结转 / 跨块 `6k` 有限结转）、`chars/3.4` 前瞻门控 + API 真实计费账本（`TokenLedger`）、扣除展示样本的 `UnqNet` 未查询净改善审计与端到端编码前脱敏对照（`e2e_pseudonym`），保证与外部基线在公平防泄漏口径下的可复现对比。

---

## 三、核心研究规范

1. **严禁非证据性预期表述**：全部结论以冻结协议（[protocol_v2.json](./trackA_sug_sed/protocol_v2.json) SHA-256 校验）、真实 Token 账本及未见保留块实测结果为准。
2. **严格泄漏隔离与消融归因**：超参均于 Dev 阶段（`M_02..M_05`）锁定后再解锁 Holdout（`M_06+`）；忠实披露各模块边界与失效案例。
3. **严格零前瞻因果预算**：块内细化受控于 `CausalUniformBucket(tokens_per_block=10000, allow_carryover=False)`，跨块演化链接受控于 `CausalUniformBucket(tokens_per_block=6000, allow_carryover=True, max_carryover_mult=1.5)`，均以字符前瞻门控并按 API 真实用量扣减。

---

## 四、五阶段主流程概览

```text
Stage 1 (Pure 2D-SE Base)
  ──► Stage 2 (Clean Seam Split 预清洗)
  ──► Stage 3 (AutoMerge + 统一 GuardedUnionFind)
  ──► Stage 4 (Two-Tier Star-Merge)
  ──► Stage 5 (SUG-CrossStar 跨块流式原型演化)
```

---

## 五、当前工作区目录结构

```text
gemini-paper/
├── README.md                              # 工作区总索引（本文件）
├── TrackA_SED_Method_Spec.md              # v2.1 五阶段主流程权威方法规格书
├── PROGRESS_TrackA.md                     # 最新实证进度与核心结论归档
├── .gitignore                             # Git 忽略规则
├── core_shared/                           # 底层数据、构图、2D-SE、账本与指标模块
│   ├── data_loader.py                     # Event2012 / Event2018 时序分块加载与按天切分
│   ├── graph_builder.py                   # 异构语义 k-NN + 属性共现消息图构建
│   ├── se_solver.py                       # 精确二维结构熵最小堆凝聚与节点迁移精化求解器
│   ├── partition_cache.py                 # Stage-1 2D-SE 初始分区持久化缓存（含图指纹校验）
│   ├── token_ledger.py                    # 线程安全 JSONL Token 账本与零前瞻因果预算桶
│   ├── metrics_audit.py                   # AMI / B³ / Tail-B³ / FS-FM / UnqNet 指标审计
│   └── key_loader.py                      # 本地 API 密钥安全读取器
├── trackA_sug_sed/                        # 核心算法、10 篇对比基线与 Gate 1/2 评测套件
│   ├── star_meso.py                       # Pure-Star-Meso 主算法（Seam / AutoMerge / GuardedUF / Star-Merge）
│   ├── cross_day.py                       # GlobalEventMemory 与 SUG-CrossStar 跨块演化
│   ├── llm_client.py                      # OpenAI 兼容预算受限客户端、RPM 节流器与 Prompts
│   ├── anonymizer.py                      # QueryAnonymizer（pseudonym / blind）与泄漏审计
│   ├── embed_cache.py                     # 冻结 HuggingFace 句向量编码与分块缓存
│   ├── baselines_sed.py                   # 4 篇 0-LLM 对比基线（EventX / SBERT-Agglom / HiSEvent / HyperSED）
│   ├── baselines_llm.py                   # 6 篇同预算 LLM 对比基线及内部对照
│   ├── run_dev_gate1.py                   # Gate 1 Dev（M_02..M_05）闭环验证与 baselines7 评测入口
│   ├── run_heldout_gate2.py               # Gate 2 Held-Out（27 留出块）全量评测入口
│   ├── analyze_gate2_mechanisms.py        # P1 机制归因零额外 API 审计脚本
│   ├── freeze_protocol_v2.py              # 不可变协议冻结脚本（生成 SHA-256 摘要）
│   └── protocol_v2.json                   # 当前为 v2.0 冻结快照（待 v2.1 全量重跑后重冻结）
├── results/                               # 实证报告、统计数据、预测结果与可复用账本
│   ├── dev_gate1/                         # Gate 1 Dev 报告（含 v2.0 报告与 v2.1 event2012/event2018_v21_7baselines.md）
│   └── heldout_gate2/                     # Gate 2 27 块留出评测、P1 归因报告、预测与账本（v2.0 快照）
├── cache/                                 # 预处理缓存
│   ├── embeddings/                        # 冻结句向量分块缓存（39 个 .npy：Event2012 + Event2018）
│   └── partitions/                        # Stage-1 2D-SE 初始分区缓存（62 个含图指纹 .npz）
└── tests/
    └── test_sug_sed.py                    # 单元测试与端到端链路校验套件
```

---

## 六、常用运行命令

```powershell
# 1. 运行单元测试与链路校验
python tests/test_sug_sed.py

# 2. 运行 v2.1 Dev 7 基线公平评测（M_02..M_05）
python trackA_sug_sed/run_dev_gate1.py --stage baselines7 --datasets Event2012,Event2018

# 3. 运行 Gate 1 Dev 全阶段验证（M_02..M_05）
python trackA_sug_sed/run_dev_gate1.py

# 4. 运行 Gate 2 Held-Out 全量评测（M_06+）
python trackA_sug_sed/run_heldout_gate2.py

# 5. 运行 P1 机制归因审计
python trackA_sug_sed/analyze_gate2_mechanisms.py

# 6. 刷新不可变协议快照与 SHA-256 校验
python trackA_sug_sed/freeze_protocol_v2.py
```
