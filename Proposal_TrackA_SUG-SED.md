# Track A 论文研究方案与实证定稿总纲（`SUG-SED v2` · `Pure-Star-Meso`）

> **目标期刊**：*Information Processing & Management* (IP&M，CCF-B / 中科院一区 Top)  
> **英文拟题**：*From Single-Post Fallacy to Mesoscopic Star-Merge Adjudication: Budgeted Open-Source LLM Event Clustering for Streaming Social Media*  
> **中文拟题**：从单帖纠错陷阱到介观星型打包裁决：面向流式社交事件检测的预算受限轻量开源 LLM 方法（`SUG-SED v2`）  
> **当前状态**：v1 旧工作（初版路由器、Eq. (5)/(6) 复杂权重及初版草稿）已完整归档至 [archive_v1/](./archive_v1/Proposal_TrackA_SUG-SED.md)；当前 **v2 主体架构（`Pure-Star-Meso` + `SUG-CrossStar`）已全面通过单元测试、Gate 1 Dev 双模型闭环验证（`Qwen/Qwen3-8B` + `Qwen/Qwen2.5-7B-Instruct`）以及 Gate 2 全部 27 个留出块（`65,856` 帖）在 `Qwen/Qwen3-8B` 上的完整实证套件**。`Qwen/Qwen2.5-7B-Instruct` 留出集跨模型扩展与正式论文重写为下一步待办。

---

## 一、针对 v1 审稿五大硬伤的重构与诚实口径定界

v2 严格对照早期审稿总评提出的五个关键断裂进行逐项结构性重构，绝不在论文中保留与实验数据冲突或无法站稳的主张：

| v1 审稿痛点 | v2 架构与实验重构方案 | 当前实证检验结论 |
|---|---|---|
| **1. 消融反噬核心设计**（Log-Leverage、硬电导上限、非实体割散度无贡献或负贡献） | 彻底剥离冗余启发项，提炼出**纯净 5 大支柱并区分功能层级**：**核心通用引擎**（两阶段多候选星型打包 `Star-Merge` `k_nb=4` + 干净语义分离 `Clean Seam Split`）协同**效率与尾部风险守卫**（`Pure 2D-SE` 起步 → `AutoMerge (>=0.85)` → 质心守卫 `GuardedUnionFind`）。 | 27 块留出集全量 0 错误配对 Wilcoxon 消融证实（见 §5.3 / §5.7）：仅**核心通用引擎**呈现跨语种一致方向收益——`Star Packing`（Pooled `AMI +1.86 p_raw=0.0451 / p_Holm=0.226`, `Mic-B3 +2.80 p_raw=0.0219 / p_Holm=0.110`, 调用降 `1.93×/2.70×`, FR 升 `3.42 AMI`）与 `Clean Seam Split`（Pooled `Mac-B3 +1.26 p_raw=6.37e-5 / p_Holm=0.000318`, `Tail-B3 +2.70 p_raw=3.05e-5 / p_Holm=0.000153`）；而 `AutoMerge` / `GuardedUF` / `Bayes Band` 作为效率与尾部风险守卫呈现双语非对称权衡，根因已由 `size>=2` 编码器余弦校准（EN `98.4%` vs FR `87.1% @ >=0.85`）与 FM 防爆（FR 阻断 `+1,446,601 FM`）定量解释。 |
| **2. 零 LLM 强基线反超与单一基线范围偏窄** | 在同一 Dev 冻结协议下构建 **10 篇已发表论文完整基线组**（4 篇零 LLM 论文含扩展内点寻优后的 `Dev-Tuned HiSEvent` 与 `HyperSED AAAI'25`，6 篇同预算 LLM 论文含修正负类偏斜坍塌后的 `ClusterLLM-NoFT` / `Few-Shot` / `Cequel` / `LSED` / `RagSEDE` / `LHG`）及即插即用基座对比（`HiSEvent Default` 与 `Dev-Tuned`）。 | `Pure-Star-Meso (Qwen3-8B)` 在全部 27 个留出块上于 `AMI`、`Mic-B3` 全面显著超越**全部 10 篇已发表论文**（Pooled 27 块 `AMI` 均 $p_{\text{Holm}} \le 2.51\times 10^{-4}$，`Mic-B3` 均 $p_{\text{Holm}} \le 4.17\times 10^{-4}$），并在 $\mathrm{Mac}\text{-}B^3$ 上显著超越其中 9 篇（$p_{\text{Holm}} \le 0.0267$，修正负偏斜后的 `ClusterLLM-NoFT` 仍以 `+10.74 AMI / +1.91 Mac-B3` 落后于本方法）；在 `Event2018` 更强的 `Dev-Tuned (0.55, 1.0)` 上叠加本方法仍净拉升 **`+8.37 AMI / +4.05 Mac-B3 / +16.85 Mic-B3`**；同时验证 `Llama-3.1` 暂不纳入、先行聚焦双千问家族泛化。 |
| **3. 理论与主算法脱节** | 将主算法统一为严格单一执行路径 [star_meso.py](./trackA_sug_sed/star_meso.py)，`GuardedUnionFind` 直接作用于主流程全部 `AutoMerge` 与 `Star-Merge` 合并操作；并同时报告直接叠加 `HiSEvent Default` 与 `Pure 2D-SE` 的性能。 | 保证“伪代码—实现—理论守卫范围”三者一致。 |
| **4. 统计与跨天宣称与表格冲突 / 回流风险** | 所有超参 **100% 锁定于 Dev 块（`M_02..M_05`）**（扩宽 `EventX`/`HiSEvent`/`HyperSED` Dev 网格确认真内点）并冻结于 [protocol_v2.json](./trackA_sug_sed/protocol_v2.json)；全部显著性均报告双数据集及合并 27 块（`Pooled27`）的双侧配对 Wilcoxon 精确 `p_raw` 与 Holm-Bonferroni 校正 `p_holm`；通过 `Random → Random-Meso (Star-Meso Rand-Order) → Meso-Heur-Best → Full` 进行三因子受控归因；Cross-Day 扫描完整展示 `ProtoOnly 0.60..0.95` 与 `Proto+Attr 0.80..0.95`。 | 正文将严格区分并诚实陈述：`Pooled27` 对全部 10 篇论文的 `AMI`/`Mic-B3` 均 $p_{\text{Holm}} \le 4.17\times 10^{-4}$、9/10 论文的 $\mathrm{Mac}\text{-}B^3$ 均 $p_{\text{Holm}} \le 0.0267$；`Random-Meso` 即 `Pure-Star-Meso (random_order=True)`（随机化锚点与 Seam 调度顺序，保留 Top-4 余弦近邻打包），其相对 `Random` 的跃迁（`+5.80 AMI / +10.52 Mic-B3`）量化**介观星型打包结构诱导先验**，而 `Full` 在此基础上提供校准调度增量（`Mac-B3 +0.89 p_Holm=0.0352`）及相对同池 0-Tok 启发式的语义增量（`Mac-B3 +1.56 p_Holm=5.49e-4`, `+1.90 AMI`）；Cross-Day 在 FR 全面反超全部原型阈值（`84.76`），在 EN 与 `proto_0.90` 持平并指出距 `oracle_band` 接受率/误连边界（§5.7）。 |
| **5. 反记忆测试未脱敏上游特征与可复现审计缺失** | 在保留 3 种查询级匿名条件（`raw`、`pseudonym`、`extract_blind`）之外，新增**端到端无截断全脱敏对照组 `e2e_pseudonym`**：在送入句向量编码器（剥离占位符括号防表征塌陷）与异构消息图（不透明 `ent_<k>/tag_<k>` ID）前即对原始文本全部实体/话题标签/`@mention` 替换为一致伪名并重算 2D-SE，同时增设 `e2e_pseudonym_automerge_0tok`、`e2e_pseudonym_meso_heur_0tok` 对照组与基于未脱敏原文本的独立审计；全量公开提示词模板、真实 Token 账本、响应缓存及 CPU/E2E CPU/API 分离耗时审计。 | 分层 8 块端到端全脱敏下（无截断块级实体遮蔽率 `100.0%` EN / `99.47%–99.81%` FR，`Tag=100.0%`，查询独立审计 `EntMask% = 98.2%` EN `LeakQ=3/57 (5.3%)` / `97.6%` FR `LeakQ=1/43 (2.3%)`，即 `94.7%–97.7%` 查询零实体泄漏），相对 `e2e_pseudonym_base_2dse`（`67.08 / 53.17`）、`e2e_pseudonym_automerge_0tok`（`67.22 / 53.47`）与 `e2e_pseudonym_meso_heur_0tok`（`67.94 / 54.62`），`e2e_pseudonym` 在全 0 错误下仍分别实现 `69.95 / 56.49 AMI`（相对 `AutoMerge` 净提 **`+2.73 / +3.02` $\mathrm{AMI}$**，相对 `Meso-Heur-Best` 净提 **`+2.01 / +1.87` $\mathrm{AMI}$**、**`+1.10 / +1.97` $\mathrm{Mac}\text{-}B^3$** 与 **`+355.8 / +1076.8 UnqNet`**），克制而可靠地支撑“结构语义共指贡献”结论。 |

---

## 二、核心方法与系统架构（`100% Dev Locked`）

### 2.1 模块划分与文件索引
* **共享底层基础设施（`core_shared/`）**：
  * [data_loader.py](./core_shared/data_loader.py)：`Event2012`（EN）与 `Event2018`（FR）时序流式按天/最小块切分加载器。
  * [graph_builder.py](./core_shared/graph_builder.py) & [se_solver.py](./core_shared/se_solver.py) & [partition_cache.py](./core_shared/partition_cache.py)：异构语义 $k$-NN + 属性共现消息图、精确二维结构熵最小堆凝聚与节点迁移精化求解器及持久化缓存。
  * [token_ledger.py](./core_shared/token_ledger.py)：线程安全 JSONL 账本（`TokenLedger`）与零前瞻因果预算桶（`CausalUniformBucket(tokens_per_block=10000, allow_carryover=False)`，采用 `chars/3.4 + est_comp` 前瞻预算门控放行并按真实计费账本扣减；在全部 27 块 `10k` 实验上末次调用实测最大溢出仅 `+0.35%`，即单块最高 `10,035` tokens）。
  * [metrics_audit.py](./core_shared/metrics_audit.py)：$O(N)$ `AMI`、`Macro-B3`、`Micro-B3`、`Tail-B3`（长尾事件分层 `Macro-B3`）、`ARI`、假分裂/假合并配对计数（FS/FM）与严格扣除 `all_shown` 后的未查询净改善审计（`UnqNet`）。
* **Track A v2 核心流水线（`trackA_sug_sed/`）**：
  * [star_meso.py](./trackA_sug_sed/star_meso.py)：`Pure-Star-Meso` 核心算法（`GuardedUnionFind`、`detect_clean_seams`、`run_pure_star_meso`）。
  * [cross_day.py](./trackA_sug_sed/cross_day.py)：流式全局事件记忆库 `GlobalEventMemory` 与 `SUG-CrossStar` 跨天评估。
  * [llm_client.py](./trackA_sug_sed/llm_client.py)：OpenAI 兼容真实计费客户端 `BudgetedLLMClient`、`build_star_merge_prompt` / `build_seam_split_prompt` / `build_cross_star_prompt`、内置 `text_override` 脱敏覆写，以及**进程级跨线程自适应 RPM 节流器**（`_pace_acquire` / `_pace_on_rate_limit` / `_pace_on_success`，`max_retries=10` + 带随机抖动指数退避，成功将并发请求 HTTP 429 错误率压制在 `<0.5%`）。
  * [anonymizer.py](./trackA_sug_sed/anonymizer.py)：`QueryAnonymizer(mode="pseudonym"|"blind")`、`AnonymizationAuditStore` 与 `check_summary_reconstruction_leakage`。
  * [embed_cache.py](./trackA_sug_sed/embed_cache.py)：冻结 HuggingFace 编码缓存器（EN：`sentence-transformers/all-MiniLM-L6-v2`；FR：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`）。
  * [baselines_sed.py](./trackA_sug_sed/baselines_sed.py) & [baselines_llm.py](./trackA_sug_sed/baselines_llm.py)：4 篇 0-LLM 基线、6 篇同预算 LLM 基线及全部内部对照。
  * [run_dev_gate1.py](./trackA_sug_sed/run_dev_gate1.py) & [run_heldout_gate2.py](./trackA_sug_sed/run_heldout_gate2.py) & [analyze_gate2_mechanisms.py](./trackA_sug_sed/analyze_gate2_mechanisms.py)：Dev 闭环寻优锁定、27 块留出集全自动断点评测器与 P1 零额外 API 机制审计器。
  * [freeze_protocol_v2.py](./trackA_sug_sed/freeze_protocol_v2.py) & [protocol_v2.json](./trackA_sug_sed/protocol_v2.json)：Track A v2 不可变协议冻结脚本与快照（`SUG-SED-v2.0-frozen`，固化全套 Dev 锁定超参、预算控制器规格及 20 份 `core_shared/*.py` / `trackA_sug_sed/*.py` SHA-256 摘要）。
  * [test_sug_sed.py](./tests/test_sug_sed.py)：完整单元测试与全链路冒烟校验（全部 PASS）。

### 2.2 `Pure-Star-Meso` 算法规格（Dev 锁定超参，零留出集调整）
1. **高纯度 2D-SE 起步（Pure 2D-SE Base · 底座）**：针对 2D-SE 初始划分中 FS $\gg$ FM 的非对称错分特征（`Event2012` 留出集 Pure FS/FM=`63.47×`、`Event2018` Pure FS/FM=`12.23×`），采用高语义门槛 `knn_k=15, knn_sim_threshold=0.50, attr_alpha=0.5` 优先保证微簇纯度（同时在 `use_purity_base=False` 即 `HiSEvent Default (0.35, 0.5)` 及 `Dev-Tuned` 上验证即插即用能力）。
2. **零成本 `AutoMerge` + 质心守卫并查集（`GuardedUnionFind` · `[效率与尾部风险守卫]`）**：
   * `AutoMerge`（`tau_auto=0.85`）：按原型余弦降序合并高置信相同事件微簇（作为 0-Token 减载器；其有效性依赖句向量高分端校准，详见 §5.7.3）；
   * `GuardedUnionFind(guard_sim=0.40, min_cl_votes=1)`：实时维护合并后分量质心并拒绝质心余弦低于 `0.40` 或违背 LLM `CANNOT_LINK` 投票的合并，从根源阻断传递闭包链式误合并崩溃（在 LLM 执行前经 `uf.can_merge()` 预先拦截并在执行后阻断残余冲突，详见 §5.7.4）。
3. **干净介观缝隙拆分（`detect_clean_seams` · `[核心通用引擎]`）**：
   * 对大小 $\ge 8$ 的簇并行运行确定性最远点球面 $k$-means（$k=\min(6,\max(3,|C|//12))$）与 Fiedler 谱二分，仅保留口袋大小 $\ge 3$ 且 $\le 0.48|C|$、语义分离度 `sem_sep = intra_pock - cross_sim >= 0.12`（不含 `use_seam_gate` 电导硬上限及冗余权重）的候选缝隙，按 `sem_sep * log(1 + |pocket|)` 排序后，至多取前 `max_seam_calls=3` 个送交平衡对比 Prompt 裁决。
4. **两阶段星型打包合并（`Star-Merge` · `[核心通用引擎]`）**：
   * 第一阶段覆盖模糊核心区 `[0.55, 0.85)`（目标簇大小 $\ge 2$），第二阶段扩展至 `[0.45, 0.85)` 含孤立微簇（大小 $\ge 1$），总上限 `max_star_calls=24` 次，每次打包 1 个锚簇与至多 `k_nb=4` 个最模糊邻簇并发做一次对比选择，将单次调用决策杠杆从单帖/单边放大数倍（双阶段调度仅作为排序引导，核心跨语种通用引擎是 `k_nb=4` 星型打包本身）。
5. **跨天流式链接（`SUG-CrossStar`）**：
   * 热身块 `M_00..M_05`（直接以 0-Token `Pure 2D-SE Base` 初始化历史记忆），评测块 `M_06+`；高阈值 `tau_hi=0.90` 自动续连，回退阈值 `tau_fb=0.90`，中间模糊带 `[0.65, 0.90)`（且日簇大小 $\ge 3$）在 `CausalUniformBucket(budget_per_block=6000, allow_carryover=True, max_carryover_mult=1.5)` 因果结转配额内调用星型跨块匹配 Prompt（滑动窗口 `window_blocks=14`，`top_k_cand=3`）。

---

## 三、10 篇已发表基线论文评测组与小预算适配协议

所有方法严格统一：**冻结句向量编码器**、**统一 `CausalUniformBucket(10000, allow_carryover=False)`**（0-LLM 为 `0` token）、**`temperature=0.0, max_tokens=384` JSON 模式**，以及 **`all_shown` 未查询掩码**。

| 类别与论文 | 文献来源 | 小预算 / 超参公平设计说明（[baselines_sed.py](./trackA_sug_sed/baselines_sed.py) · [baselines_llm.py](./trackA_sug_sed/baselines_llm.py)） | Dev (`M_02..M_05`) 锁定参数 / 对照作用 |
|---|---|---|---|
| **1. EventX** | Liu et al., ACM TKDD 2020 | 实体/话题（权 `1.4`）与词元（权 `0.7`）倒排 + IDF 时间窗构图（`half_win=6`）+ Louvain 社区发现 | EN `resolution=2.0`；FR `1.5`（Dev `[0.6..3.0]` 内点最优选） |
| **2. SBERT-Agglom** | Reimers & Gurevych, EMNLP 2019 | 同款冻结 MiniLM 句向量 + 平均链接余弦层次凝聚聚类 | EN/FR `distance_threshold=0.60`（内点最优） |
| **3a/3b. HiSEvent** | Cao et al., AAAI 2024 | 异构语义+属性消息图 2D-SE（含默认 `Default (0.35, 0.5)` 与针对审稿意见专门设定的 `Dev-Tuned`） | EN `(0.50, 0.5)`；FR `(0.55, 1.0)`（Dev 扩展网格内点最优） |
| **4. HyperSED** | Yu et al., AAAI 2025 | 确定性球面锚点图 SAMG + PyTorch 庞加莱双曲空间优化（`60 epochs, dim=32, n_anchors<=360`）+ 双曲衰减 2D-SE | EN `anchor_ratio=0.26`；FR `0.14` |
| **5. ClusterLLM-NoFT** | Zhang et al., EMNLP 2023 | 层次聚类树 5 档切分 `[0.45..0.65]` + 跨切分分歧熵优先对 medoid 对做成对查询直至 10k 预算耗尽 + **均衡准确率 (`bal_acc`) 抗负类偏斜粒度校准** + 动态 `CANNOT-LINK` 守卫；`UnqNet` 对齐其 `SBERT-Agglom (0.60)` 底座 | 零训练小预算适配版（明确标注省略编码器微调阶段，防止成对负样本偏斜导致树底 `d=0.45` 退化坍塌） |
| **6. Few-Shot LLM-Post** | Viswanathan et al., TACL 2024 | 从 `Pure 2D-SE` 起步，按 `margin = s_cur - max_alt` 升序优先对低置信边界帖做 4 候选单帖多选重分配直至预算耗尽 | 检验经典单帖纠错陷阱 |
| **7a/7b. Cequel** | Wang et al., arXiv 2025 | 对跨簇高不确定边/medoid 对按 `~0.72` 模糊度排序（每簇对限查 2 边），衍生**直接边修改 `Cequel-Edge`** 与**受限簇级传递闭包 `Cequel-Transitive`** | 检验单边查询与无质心守卫传递闭包的差异 |
| **8. LSED-Budgeted** | Qiu et al., ACL 2025 | 在 10k 预算内按 `(1 - margin) + 0.3*(1-len/80) + 0.2*(1-0.5*ent)` 优先选短/模糊帖做 LLM 事实重写，再按 `0.7*new + 0.3*old` 重编码与重解 2D-SE | 检验高预算重写法迁移至小预算的效果 |
| **9. RagSEDE-Budgeted** | Liu et al., WWW 2026 | MMR 多样性/代表性迭代选样 + Top-3 稠密 RAG 邻帖增强分配 + 图边加权修复与热启动 2D-SE | 检验基于 RAG 的单帖图修复性能 |
| **10. LHG-Budgeted** | Qiu et al., WWW 2026 | 按异构交互图 Hub 分数挑选中心帖 + Top-3 邻帖生成规范伪标签 → MPLSS 筛选三元组 → 庞加莱空间微调重解 2D-SE | 检验伪标签双曲微调在小预算下的稀疏三元组瓶颈 |
| **内部 0-Tok / 随机对照** | — | `Pure 2D-SE Base`、`AutoMerge-Only (0.85)`、`Meso-Heur-Best (0.80, 0.99)`（同介观池零成本余弦基线）、3-seed `Random` 与 3-seed `Random-Meso`（即 [star_meso.py](./trackA_sug_sed/star_meso.py#L428) `run_pure_star_meso(..., random_order=True)`，随机打乱锚点簇与 Seam 候选调度顺序，选定锚点后仍保留 Top-4 余弦近邻星型打包、`AutoMerge`、`GuardedUF` 与同款 Prompt） | **三因子受控归因**：严格拆解介观星型结构先验（`Random→Random-Meso`）、校准调度增益（`Random-Meso→Full`）与 LLM 语义增益（`Meso-Heur-Best→Full`） |

---

## 四、Gate 1 开发集双千问闭环验证摘要（`M_02..M_05`，`10,000` tokens/块）

全部 Dev 开发结果保存于 [dev_gate1_report.md](./results/dev_gate1/dev_gate1_report.md)（零留出集接触）：

* **`Qwen/Qwen3-8B` Dev（`M_02..M_05`）**：
  * **`Event2012`**：`Pure-Star-Meso` 达 **AMI `94.06` / Mac-$B^3$ `81.31` / Mic-$B^3$ `92.13` / Tail-$B^3$ `63.73` / UnqNet `+649.0`**（仅 32,864 tokens, 3.36% 展示），超越最强 0-LLM `Meso-Heur-Best`（`92.37 / 79.90`）、`HyperSED`（`90.24 / 74.39`）、`SBERT-Agglom`（`90.51 / 77.65`）、最强 LLM 基线 `Cequel-Transitive`（`92.11 / 79.12`）、修复后的 `ClusterLLM-NoFT`（`88.37 / 77.10`）与 3-seed `Random-Meso`（`92.71 / 79.50`）。
  * **`Event2018`**：`Pure-Star-Meso` 达 **AMI `78.80` / Mac-$B^3$ `79.05` / Mic-$B^3$ `81.01` / Tail-$B^3$ `68.70` / UnqNet `+1358.2`**（38,297 tokens, 1.92% 展示），大幅领先 `Meso-Heur-Best`（`74.44 / 73.86`）、`HiSEvent Dev-Tuned (0.55, 1.0)`（`68.57 / 75.01`）、`LHG-Budgeted`（`68.66 / 72.48`）、`ClusterLLM-NoFT`（`57.10 / 71.20`）与 `Random-Meso`（`75.85 / 75.99`）。
* **`Qwen/Qwen2.5-7B-Instruct` Dev 跨模型验证（全部 `err=0`）**：
  * `Event2012`：`Pure-Star-Meso` 达 **`94.08 / 81.40 / 92.28 / 63.41 / UnqNet=+723.2`**（领先第二名 `Random-Meso 93.49/79.95`、`Cequel-Trans 91.57/79.10` 与 `ClusterLLM-NoFT 87.83/75.90`）；
  * `Event2018`：`Pure-Star-Meso` 达 **`75.61 / 76.29 / 76.17 / 60.51 / UnqNet=+1436.8`**（领先所有 6 篇 LLM 基线含 `ClusterLLM-NoFT 58.20/72.42` 与 `Random-Meso 71.54/73.39`）。
* **Dev Cross-Day 校准与超参锁死**：双语 `SUG-CrossStar` 均达到 Dev 最优区间（EN `94.71 / 81.52`，FR `80.39 / 80.15`），由此将全部超参严格锁定于 `DEV_LOCKED_0TOK` 并进入一次性盲测的 Gate 2 留出评测。

---

## 五、Gate 2 留出集 27 块全量实证结果（`Qwen/Qwen3-8B`，零留出调参）

全部汇总及分块结果文件：[gate2_full_report.md](./results/heldout_gate2/gate2_full_report.md)、[stage_0tok.json](./results/heldout_gate2/stage_0tok.json)、[stage_qwen3_main.json](./results/heldout_gate2/stage_qwen3_main.json)、[stage_qwen3_abl.json](./results/heldout_gate2/stage_qwen3_abl.json)、[stage_qwen3_crossday.json](./results/heldout_gate2/stage_qwen3_crossday.json)、[stage_budget_anon.json](./results/heldout_gate2/stage_budget_anon.json)。

### 5.1 27 个留出块主实验全量对比

#### (a) `Event2012` Held-Out（`M_06..M_21`，16 块，32,695 帖 | Pure 2D-SE FS/FM=`63.47×`, Def FS/FM=`44.64×`）

| 分组 | 方法 | Calls | Tok | Show% | Err | AMI | Mac-$B^3$ | Mic-$B^3$ | Tail-$B^3$ | UnqNet | CPU(s) | E2E CPU(s) | API(s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-LLM Paper 1 | EventX (TKDD'20) | 0 | 0 | 0.00% | 0 | 81.59 | 55.30 | 76.85 | 23.54 | +0.0 | 20.3 | 20.3 | 0.0 |
| 0-LLM Paper 2 | SBERT-Agglom (EMNLP'19) | 0 | 0 | 0.00% | 0 | 87.66 | 75.72 | 85.29 | 68.91 | +0.0 | 18.7 | 18.7 | 0.0 |
| 0-LLM Paper 3a | HiSEvent Default (AAAI'24) | 0 | 0 | 0.00% | 0 | 83.95 | 65.67 | 77.08 | 41.03 | +0.0 | 16.3 | 16.3 | 0.0 |
| 0-LLM Paper 3b | HiSEvent Dev-Tuned (AAAI'24) | 0 | 0 | 0.00% | 0 | 83.98 | 74.26 | 76.71 | 62.91 | +0.0 | 13.2 | 13.2 | 0.0 |
| 0-LLM Paper 4 | HyperSED (AAAI'25) | 0 | 0 | 0.00% | 0 | 85.35 | 69.03 | 79.96 | 53.60 | +0.0 | 19.3 | 19.3 | 0.0 |
| Internal 0-Tok | Pure 2D-SE Base (0.50) | 0 | 0 | 0.00% | 0 | 83.98 | 74.26 | 76.71 | 62.91 | +0.0 | 13.2 | 13.2 | 0.0 |
| Internal 0-Tok | AutoMerge-Only (0.85) | 0 | 0 | 0.00% | 0 | 88.03 | 75.49 | 83.83 | 62.85 | +0.0 | 0.2 | 13.4 | 0.0 |
| Internal 0-Tok | Meso-Heur-Best (Pure) | 0 | 0 | 0.00% | 0 | 88.96 | 75.41 | 85.21 | 62.82 | +0.0 | 1.9 | 15.1 | 0.0 |
| LLM Paper 5 | ClusterLLM-NoFT (EMNLP'23) | 803 | 158502 | 1.96% | 0 | 84.21 | 75.44 | 81.46 | **70.58** | -979.6 | 74.6 | 74.6 | 1886.5 |
| LLM Paper 6 | Few-Shot LLM-Post (TACL'24) | 287 | 154406 | 4.81% | 0 | 83.98 | 74.35 | 76.72 | 63.17 | -8.1 | 1.3 | 14.5 | 826.2 |
| LLM Paper 7a | Cequel-Edge (arXiv'25) | 748 | 154118 | 3.73% | 0 | 83.90 | 74.23 | 76.63 | 62.97 | -157.6 | 4.5 | 17.7 | 1954.0 |
| LLM Paper 7b | Cequel-Transitive (arXiv'25) | 748 | 154118 | 3.73% | 0 | 86.86 | 74.37 | 81.71 | 62.97 | +206.2 | 4.5 | 17.7 | 1954.0 |
| LLM Paper 8 | LSED-Budgeted (ACL'25) | 1256 | 158314 | 3.84% | 0 | 84.09 | 74.52 | 76.84 | 63.39 | -73.1 | 212.6 | 225.8 | 2503.4 |
| LLM Paper 9 | RagSEDE-Budgeted (WWW'26) | 247 | 153549 | 5.69% | 0 | 83.99 | 74.57 | 76.71 | 63.64 | -53.4 | 6.5 | 19.7 | 521.2 |
| LLM Paper 10 | LHG-Budgeted (WWW'26) | 699 | 156811 | 4.10% | 0 | 83.56 | 74.37 | 75.88 | 63.31 | -56.1 | 138.9 | 152.1 | 1083.9 |
| Internal Ctrl | Random (3-seed mean) | 274.33 | 154586 | 4.76% | 0 | 83.96 | 74.26 | 76.70 | 62.93 | -35.7 | 1.0 | 14.2 | 520.6 |
| Internal Ctrl | Random-Meso (3-seed mean) | 275.0 | 135966 | 3.03% | 0 | 89.95 | 75.87 | 86.82 | 63.31 | +617.1 | 23.1 | 36.3 | 591.2 |
| **OUR METHOD** | **Pure-Star-Meso (Qwen3-8B)** | **272** | **137201** | **2.92%** | **0** | **90.15** | **76.56** | **86.91** | 64.06 | **+687.0** | 22.1 | 35.3 | 948.1 |
| Plug-and-Play | Star-Meso on HiSEvent Default | 222 | 110313 | 2.47% | 0 | 90.06 | 69.58 | 86.75 | 47.13 | +680.2 | 18.3 | 34.6 | 611.4 |
| Plug-and-Play | Star-Meso on HiSEvent Dev-Tuned** | 272 | 137201 | 2.92% | 0 | 90.15 | 76.56 | 86.91 | 64.06 | +687.0 | 22.1 | 35.3 | 948.1 |

#### (b) `Event2018` Held-Out（`M_06..M_16`，11 块，33,161 帖 | Pure 2D-SE FS/FM=`12.23×`, Def FS/FM=`11.59×`）

| 分组 | 方法 | Calls | Tok | Show% | Err | AMI | Mac-$B^3$ | Mic-$B^3$ | Tail-$B^3$ | UnqNet | CPU(s) | E2E CPU(s) | API(s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-LLM Paper 1 | EventX (TKDD'20) | 0 | 0 | 0.00% | 0 | 79.12 | 66.11 | 77.66 | 27.79 | +0.0 | 25.4 | 25.4 | 0.0 |
| 0-LLM Paper 2 | SBERT-Agglom (EMNLP'19) | 0 | 0 | 0.00% | 0 | 73.47 | 67.27 | 71.80 | 60.21 | +0.0 | 24.8 | 24.8 | 0.0 |
| 0-LLM Paper 3a | HiSEvent Default (AAAI'24) | 0 | 0 | 0.00% | 0 | 76.34 | 67.90 | 68.74 | 35.00 | +0.0 | 36.2 | 36.2 | 0.0 |
| 0-LLM Paper 3b | HiSEvent Dev-Tuned (AAAI'24) | 0 | 0 | 0.00% | 0 | 76.06 | 75.52 | 67.67 | 65.64 | +0.0 | 33.1 | 33.1 | 0.0 |
| 0-LLM Paper 4 | HyperSED (AAAI'25) | 0 | 0 | 0.00% | 0 | 75.17 | 58.08 | 70.77 | 24.06 | +0.0 | 12.3 | 12.3 | 0.0 |
| Internal 0-Tok | Pure 2D-SE Base (0.50) | 0 | 0 | 0.00% | 0 | 75.68 | 71.74 | 67.99 | 53.85 | +0.0 | 31.8 | 31.8 | 0.0 |
| Internal 0-Tok | AutoMerge-Only (0.85) | 0 | 0 | 0.00% | 0 | 78.06 | 72.93 | 73.56 | 53.70 | +0.0 | 0.2 | 32.0 | 0.0 |
| Internal 0-Tok | Meso-Heur-Best (Pure) | 0 | 0 | 0.00% | 0 | 79.76 | 73.11 | 77.36 | 53.51 | +0.0 | 1.8 | 33.6 | 0.0 |
| LLM Paper 5 | ClusterLLM-NoFT (EMNLP'23) | 452 | 108514 | 1.13% | 0 | 64.96 | 72.22 | 60.71 | **79.75** | -1695.0 | 114.7 | 114.7 | 1124.0 |
| LLM Paper 6 | Few-Shot LLM-Post (TACL'24) | 134 | 105403 | 2.29% | 0 | 75.78 | 71.82 | 68.08 | 53.85 | +432.0 | 1.8 | 33.6 | 299.6 |
| LLM Paper 7a | Cequel-Edge (arXiv'25) | 437 | 108504 | 2.26% | 0 | 75.67 | 71.75 | 68.01 | 53.85 | +173.2 | 13.0 | 44.8 | 989.1 |
| LLM Paper 7b | Cequel-Transitive (arXiv'25) | 437 | 108504 | 2.26% | 0 | 75.57 | 70.79 | 70.00 | 53.74 | +114.5 | 13.0 | 44.8 | 989.1 |
| LLM Paper 8 | LSED-Budgeted (ACL'25) | 763 | 108875 | 2.30% | 0 | 75.81 | 73.16 | 67.97 | 57.73 | +360.8 | 247.0 | 278.9 | 1304.3 |
| LLM Paper 9 | RagSEDE-Budgeted (WWW'26) | 114 | 106712 | 3.28% | 0 | 75.69 | 71.74 | 68.00 | 53.82 | -43.5 | 6.7 | 38.6 | 241.8 |
| LLM Paper 10 | LHG-Budgeted (WWW'26) | 369 | 108665 | 2.50% | 0 | 75.17 | 72.01 | 67.39 | 54.71 | -306.8 | 214.8 | 246.6 | 610.2 |
| Internal Ctrl | Random (3-seed mean) | 131.0 | 106145 | 2.60% | 0 | 75.69 | 71.74 | 68.00 | 53.83 | +26.5 | 1.3 | 33.1 | 273.9 |
| Internal Ctrl | Random-Meso (3-seed mean) | 143.67 | 105353 | 1.91% | 0 | 81.22 | 74.10 | 79.11 | 55.87 | +1023.9 | 22.6 | 54.4 | 325.8 |
| **OUR METHOD** | **Pure-Star-Meso (Qwen3-8B)** | **142** | **105173** | **1.84%** | **0** | **82.69** | **75.27** | **81.74** | 57.58 | **+1179.5** | 18.9 | 50.7 | 331.9 |
| Plug-and-Play | Star-Meso on HiSEvent Default | 143 | 105240 | 1.78% | 0 | 82.76 | 71.64 | 81.71 | 42.68 | +1273.4 | 17.8 | 54.0 | 343.8 |
| Plug-and-Play | Star-Meso on HiSEvent Dev-Tuned** | 136 | 103660 | 1.89% | 0 | **84.43** | **79.57** | **84.52** | 68.11 | **+1493.8** | 5.1 | 38.2 | 370.3 |

*\*注：`CPU(s)` 为方法/路由纯本地计算耗时（严格剔除 API 往返、`RPM` 节流休眠与 HTTP 重试退避等待），`E2E CPU(s)` 为叠加上 Stage-1 2D-SE 构图与求解耗时后的端到端本地耗时；`API(s)` 为各分块账本及共享缓存恢复的累计真实网络调用耗时；`10k` 前瞻预算门控下全 27 块实测末次调用最大溢出不超过 `+0.35%`（最高单块 `10,035` tokens）；\*\*`Event2012` 上 `HiSEvent Dev-Tuned` 锁定超参 `(0.50, 0.5)` 即为 Pure Base，`Event2018` 上 `(0.55, 1.0)` 叠加上本方法仍净涨 `+8.37 AMI / +4.05 Mac-B3 / +16.85 Mic-B3`。`ClusterLLM-NoFT (EMNLP'23)` 在零训练预算评测下省略对比微调，采用抗负偏斜均衡准确率粒度选择及动态 `CANNOT-LINK` 守卫，且 `UnqNet` 对齐其 `SBERT-Agglom (0.60)` 底座；`HyperSED` 锚点上限为 `n_anchors <= 360`。*

### 5.2 配对双侧 Wilcoxon + Holm-Bonferroni 显著性检验（`Pooled27` 全量 27 块）

| 对比基线 | $\Delta\mathrm{AMI}$ (`W/T/L`, `p_raw / p_Holm`) | $\Delta\mathrm{Mac}\text{-}B^3$ (`W/T/L`, `p_raw / p_Holm`) | $\Delta\mathrm{Mic}\text{-}B^3$ (`W/T/L`, `p_raw / p_Holm`) | $\Delta\mathrm{Tail}\text{-}B^3$ (`W/T/L`, `p_raw / p_Holm`) |
|---|---|---|---|---|
| EventX (TKDD'20) | +6.52 (21/0/6, 8.37e-05 / **2.51e-04**) | +16.33 (26/0/1, 7.45e-08 / **8.94e-07**) | +7.63 (21/0/6, 1.25e-04 / **3.76e-04**) | +36.15 (27/0/0, 1.49e-08 / **2.09e-07**) |
| SBERT-Agglom (EMNLP'19) | +5.23 (25/0/2, 3.77e-06 / **1.89e-05**) | +3.76 (18/0/9, 0.00355 / **0.0142**) | +5.01 (23/0/4, 8.37e-05 / **3.35e-04**) | -3.95 (10/0/17, 0.1167 / 0.4193) |
| HiSEvent Default (AAAI'24) | +6.26 (24/0/3, 8.20e-07 / **4.92e-06**) | +9.45 (26/0/1, 2.98e-08 / **4.17e-07**) | +11.12 (26/0/1, 7.45e-08 / **4.47e-07**) | +22.84 (26/0/1, 1.04e-07 / **1.36e-06**) |
| HiSEvent Dev-Tuned (AAAI'24) | +6.36 (25/0/2, 2.09e-07 / **2.50e-06**) | +1.26 (22/0/5, 0.00889 / **0.02667**) | +11.77 (26/0/1, 2.98e-08 / **4.17e-07**) | -2.60 (14/3/10, 0.8115 / 0.8115) |
| HyperSED (AAAI'25) | +5.91 (26/0/1, 2.98e-08 / **3.87e-07**) | +11.47 (26/0/1, 4.47e-08 / **5.81e-07**) | +8.59 (26/0/1, 2.98e-08 / **4.17e-07**) | +19.85 (24/0/3, 4.92e-07 / **5.90e-06**) |
| ClusterLLM-NoFT (EMNLP'23) | +10.74 (27/0/0, 1.49e-08 / **2.09e-07**) | +1.91 (17/0/10, 0.1286 / 0.1286) | +11.80 (26/0/1, 2.98e-08 / **4.17e-07**) | -12.90 (4/0/23, 7.99e-06 / **8.79e-05**) |
| Few-Shot LLM-Post (TACL'24) | +6.47 (24/0/3, 2.09e-07 / **2.50e-06**) | +2.72 (24/0/3, 4.57e-06 / **4.57e-05**) | +11.60 (26/0/1, 2.98e-08 / **4.17e-07**) | +2.04 (15/2/10, 0.01355 / 0.09488) |
| Cequel-Edge (arXiv'25) | +6.56 (24/0/3, 2.09e-07 / **2.50e-06**) | +2.81 (24/0/3, 3.08e-06 / **3.39e-05**) | +11.69 (26/0/1, 2.98e-08 / **4.17e-07**) | +2.16 (16/3/8, 0.003143 / **0.02514**) |
| Cequel-Trans (arXiv'25) | +4.85 (24/0/3, 3.77e-06 / **1.89e-05**) | +3.13 (24/0/3, 1.14e-05 / **9.08e-05**) | +7.87 (25/0/2, 1.64e-06 / **8.20e-06**) | +2.21 (17/3/7, 0.002001 / **0.01801**) |
| LSED-Bud (ACL'25) | +6.39 (24/0/3, 2.83e-07 / **2.50e-06**) | +2.07 (23/0/4, 4.09e-05 / **2.46e-04**) | +11.58 (26/0/1, 4.47e-08 / **4.17e-07**) | +0.33 (15/1/11, 0.3153 / 0.6305) |
| RagSEDE-Bud (WWW'26) | +6.50 (24/0/3, 2.09e-07 / **2.50e-06**) | +2.62 (23/0/4, 9.54e-06 / **8.58e-05**) | +11.64 (26/0/1, 2.98e-08 / **4.17e-07**) | +1.78 (15/2/10, 0.04826 / 0.2549) |
| LHG-Bud (WWW'26) | +6.96 (24/0/3, 2.83e-07 / **2.50e-06**) | +2.63 (24/0/3, 1.88e-05 / **1.32e-04**) | +12.38 (26/0/1, 2.98e-08 / **4.17e-07**) | +1.61 (14/1/12, 0.1048 / 0.4193) |
| Meso-Heur-Best (0-Tok) | +1.90 (16/0/11, 0.09056 / 0.1781) | +1.56 (21/0/6, 1.10e-04 / **5.49e-04**) | +2.79 (19/0/8, 0.02991 / 0.05981) | +2.39 (17/3/7, 9.63e-04 / **0.00963**) |
| Random-Meso (3-seed) | +0.71 (16/1/10, 0.08903 / 0.1781) | +0.89 (18/1/8, 0.01761 / **0.03523**) | +1.12 (17/1/9, 0.05585 / 0.05981) | +1.14 (16/3/8, 0.04249 / 0.2549) |

* **边界说明**：`SBERT-Agglom` 与 `ClusterLLM-NoFT` 因过度碎片化在 `Tail-B3` 上更高（EN `68.91 / 70.58`, FR `60.21 / 79.75` vs 本文 `64.06 / 57.58`），代价是 `AMI`/`Mic-B3` 明显受损（修复负偏斜坍塌后，`ClusterLLM-NoFT` 仍比本文低 `10.74 AMI` 与 `11.80 Mic-B3`，且 `Mac-B3` 均值仍低于本文 `1.91` 分，`p_Holm=0.1286`）；`Pure-Star-Meso` 相对同候选池 `Meso-Heur-Best`/`Random-Meso` 的核心差异化统计显著性体现为全局 **$\mathrm{Macro}\text{-}B^3$ 提升**（$p_{\text{Holm}}=5.49\times 10^{-4}$ 与 $0.0352$）及更高的未查询净增益（单数据集细节中，EN vs `SBERT-Agglom` 的 `Mac/Mic` 及 FR vs `EventX` 的 `AMI/Mic` 单独未达 0.05，完整双向检验表见 [gate2_full_report.md](./results/heldout_gate2/gate2_full_report.md#L61-L119)）。

### 5.3 27 块 5 大核心支柱消融（全部 0 错误）

| 数据集 | 消融变体 | Calls (Err) | Tok | AMI | Mac-$B^3$ | Mic-$B^3$ | Tail-$B^3$ | UnqNet |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`Event2012`** | **Full Pure-Star-Meso** | 272 (0) | 137201 | 90.15 | 76.56 | 86.91 | 64.06 | **+687.0** |
| | w/o AutoMerge (`>=0.85`) | 275 (0) | 142220 | 89.13 | 76.35 | 85.32 | 64.08 | +619.9 |
| | w/o Bayes Band & AutoMerge | 255 (0) | 147743 | 89.85 | **76.62** | 86.34 | 64.25 | +680.2 |
| | w/o Star Packing ($k=1$ Pair) | 526 (0) | 120131 | 89.35 | 76.50 | 85.96 | **64.29** | +629.8 |
| | w/o Guarded UF | 281 (0) | 141802 | **90.72** | 76.41 | **87.87** | 64.10 | +677.3 |
| | w/o Seam Split | 220 (0) | 113261 | 90.22 | 75.89 | 87.10 | 62.77 | +673.8 |
| **`Event2018`** | **Full Pure-Star-Meso** | 142 (0) | 105173 | 82.69 | **75.27** | 81.74 | **57.58** | +1179.5 |
| | w/o AutoMerge (`>=0.85`) | 141 (0) | 104993 | **83.16** | 74.99 | **82.43** | 56.32 | +1246.5 |
| | w/o Bayes Band & AutoMerge | 132 (0) | 104695 | 83.09 | 75.14 | **82.43** | 56.67 | **+1276.8** |
| | w/o Star Packing ($k=1$ Pair) | 383 (0) | 108846 | 79.27 | 74.26 | 76.25 | 57.48 | +737.0 |
| | w/o Guarded UF | 138 (0) | 104987 | 82.56 | 75.13 | 81.74 | **57.58** | +1066.5 |
| | w/o Seam Split | 134 (0) | 104644 | 82.28 | 73.16 | 81.24 | 52.82 | +1078.9 |

### 5.4 全量时序 Cross-Day Stream（热身 `M_00..M_05` 0-Tok Pure 2D-SE，评测 `M_06+`；`CausalUniformBucket(6000, allow_carryover=True, max_carryover_mult=1.5)`）

| 数据集 | 关键模式 | G-AMI | G-Mac$B^3$ | G-Mic$B^3$ | G-Tail$B^3$ | `auto` / `llm_m` / `Q` | Tok |
|---|---|---:|---:|---:|---:|---|---:|
| **`Event2012`** | `no_link` / `proto_0.85` / `proto_0.90` / `proto_attr_0.95` | 89.52 / 90.64 / 90.72 / **90.75** | 69.53 / 73.84 / 73.81 / **74.07** | 75.12 / 80.18 / 79.94 / 80.17 | 59.12 / 61.02 / 59.49 / 58.98 | `0-208 / 0 / 0` | 0 |
| | **`sug_cross_star`**（`[0.65, 0.90)`） | 90.62 | 73.77 | 80.03 | 59.76 | `153 / 72 / 181` (err=0) | 92,870 |
| | `oracle_band`（参考上界） | 91.70 | 76.60 | 82.20 | 59.92 | `155 / 82 / 199` | 0 |
| **`Event2018`** | `no_link` / `proto_0.88` / `proto_0.90` / `proto_attr_0.95` | 79.50 / 84.35 / 84.13 / 84.57 | 61.70 / 71.99 / 72.28 / 72.34 | 57.58 / 74.38 / 74.06 / 74.37 | 45.03 / 44.31 / 44.31 / 44.31 | `0-127 / 0 / 0` | 0 |
| | **`sug_cross_star`**（`[0.65, 0.90)`） | **84.76** | **72.72** | **75.73** | 44.24 | `119 / 16 / 91` (err=0) | 62,224 |
| | `oracle_band`（参考上界） | 85.12 | 73.23 | 76.26 | 45.31 | `119 / 29 / 114` | 0 |

*(注：完整 `proto_0.60..0.95` 与 `proto_attr_0.80..0.95` 扫描见 [gate2_full_report.md](./results/heldout_gate2/gate2_full_report.md#L144-L191)。Cross-Day 使用 `allow_carryover=True`（最高单块 `1.5×` 弹性额度）并严守累计因果上界 $\le 6000 \cdot t$。如实定界：`Event2018` 上跨天 LLM 全面胜出所有原型阈值；`Event2012` 上与 `proto_0.90` 持平、略低于事后峰值 `proto_attr_0.95`。)*

### 5.5 27 块因果预算扩展曲线（`B = 0, 2k, 5k, 10k, 20k`）

* **`Event2012`**：`Pure-Star-Meso` 仅用 `2k`/块（**`24,601 tokens`**, `0.75 tok/帖`）即达 **`88.49 AMI / 76.06 Mac-B3 / +500.4 Unq`**，超过 `FewShotPost`（`20k` 耗 `314,537 tokens`: `84.04 / 74.80`，**12.8× 差距**）与 `Cequel-Trans`（`20k` 耗 `239,511 tokens`: `86.66 / 74.11`，**9.7× 差距**）；`5k`/块即达 **`90.11 / 76.60`** 近似饱和；`20k` 稳定至 **`90.17 / 76.58 / 86.92 / +711.6`**（`err=0`）。
* **`Event2018`**：`Pure-Star-Meso` 在 `2k`（**`17,218 tokens`**, `0.52 tok/帖`）达 **`78.07 / 74.20`**（压倒 `20k` 耗 `215k–218k` 的 `FewShotPost 75.86/72.12` 与 `Cequel-Trans 74.40/69.25`）；`5k`（`50,436 tokens`）即达 **`82.31 / 75.15 / +1084.6`**，在 $\mathrm{Mac}\text{-}B^3$（`75.15 vs 74.69`）与 $\mathrm{Tail}\text{-}B^3$（`57.58 vs 54.66`）上反超耗资 4 倍的 `Random-Meso (20k, 204,987 tokens)`；`20k`（`200,053 tokens`, `err=0`）达 **`82.83 / 75.47 / 81.91 / +1253.2`**。

### 5.6 分层 8 块 4 模式反记忆与端到端全脱敏审计（`Event2012 [6,9,14,19]` & `Event2018 [6,9,12,15]`，全部 `err=0`）

| 数据集 | 条件 | Calls (Err) | Tok | AMI | Mac-$B^3$ | Mic-$B^3$ | Tail-$B^3$ | UnqNet | EntMask% / TagMask% | LeakQ / LeakT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| **`Event2012`** | `base_2dse_0tok` / `raw` | 0 / 64 (0) | 0 / 32328 | 88.31 / **92.75** | 75.17 / **77.13** | 80.97 / **89.56** | 64.89 / 65.10 | +0.0 / +492.2 | - | 0 / 0 |
| | `pseudonym`（查询伪名） | 62 (0) | 34819 | 90.92 | 75.82 | 86.37 | **65.44** | +255.2 | 100.0% / 100.0% | 0 / 0 |
| | `extract_blind`（严格致盲） | 65 (0) | 32989 | 91.78 | 74.58 | 88.09 | 64.25 | +347.5 | 99.2% / 100.0% | 1 / 1 |
| | `e2e_pseudonym_base_2dse` | 0 (0) | 0 | 67.08 | 58.17 | 52.47 | 58.01 | +0.0 | - | 0 / 0 |
| | `e2e_pseudonym_automerge_0tok` | 0 (0) | 0 | 67.22 | 58.28 | 52.88 | 58.01 | +0.0 | - | 0 / 0 |
| | `e2e_pseudonym_meso_heur_0tok` | 0 (0) | 0 | 67.94 | 58.59 | 55.20 | 58.01 | +0.0 | - | 0 / 0 |
| | **`e2e_pseudonym`**（端到端伪名） | **57 (0)** | 38644 | **69.95** | **59.69** | **58.27** | 57.90 | **+355.8** | 98.2% / 100.0%* | 3 / 3 |
| **`Event2018`** | `base_2dse_0tok` / `raw` | 0 / 52 (0) | 0 / 38602 | 77.02 / **84.43** | 68.98 / **73.56** | 68.08 / **83.66** | 51.75 / 53.42 | +0.0 / +1704.2 | - | 0 / 0 |
| | `pseudonym`（查询伪名） | 48 (0) | 37687 | 83.43 | 72.99 | 81.78 | **53.44** | +1602.0 | 100.0% / 100.0% | 1 / 1 |
| | `extract_blind`（严格致盲） | 55 (0) | 38226 | 78.73 | 67.77 | 77.82 | 52.84 | +1137.0 | 99.1% / 100.0% | 1 / 1 |
| | `e2e_pseudonym_base_2dse` | 0 (0) | 0 | 53.17 | 44.68 | 42.67 | 29.50 | +0.0 | - | 0 / 0 |
| | `e2e_pseudonym_automerge_0tok` | 0 (0) | 0 | 53.47 | 45.10 | 43.99 | 29.50 | +0.0 | - | 0 / 0 |
| | `e2e_pseudonym_meso_heur_0tok` | 0 (0) | 0 | 54.62 | 45.68 | 46.89 | 29.12 | +0.0 | - | 0 / 0 |
| | **`e2e_pseudonym`**（端到端伪名） | **43 (0)** | 38134 | **56.49** | **47.65** | **52.79** | **32.52** | **+1076.8** | 97.6% / 100.0%* | 1 / 1 |

*\*`e2e_pseudonym` 在编码器与图谱构建之前即对全量原始文本执行**无截断**实体/标签/`@mention` 一致伪名映射（共映射 `4,140 / 7,886` 实体、`1,279 / 3,334` 话题标签、`1,055 / 2,917` `@mention`；块级实体遮蔽率 EN **`9,480/9,480 = 100.0%`**，FR **`18,966/19,024 = 99.47%–99.81%`**，块级残余主为 HTML/标点碎片如 `«&nbsp;` 及 1 例重音姓氏 `Piechaczyk`），仅将去方括号的谓词语义送入冻结 MiniLM、并以不透明 `ent_<k>/tag_<k>` ID 构建异构图（防止占位符污染表征塌陷）；查询侧直接对照未脱敏原文本独立审计（仅统计严格展示给 LLM 的 `medoids[:2]`）：`EntMask% = 98.2%`（`324/330`, `LeakQ = 3/57 = 5.3%`）与 `97.6%`（`326/334`, `LeakQ = 1/43 = 2.3%`），`TagMask/Year/URL = 100.0%`——其中 `94.7%–97.7%` 的 E2E 查询完全无实体泄漏与重建，残留的 `3` 条 EN 与 `1` 条 FR 查询来自罕见跨帖小写实体未替换或 LLM 依据局部赛事语境的实体重建（如 `Arsenal`、`Heat vs Celtics`、`Bears vs Titans`、`Paros`）。即便在同一全脱敏底座上叠加 0-Token 启发式 `e2e_pseudonym_meso_heur_0tok`，`e2e_pseudonym` 仍实现 **`+2.01 / +1.87` $\mathrm{AMI}$**、**`+1.10 / +1.97` $\mathrm{Mac}\text{-}B^3$** 与 **`+355.8 / +1076.8 UnqNet`** 净提升。*

### 5.7 机制归因、编码器校准与诚实方法定界（P1 深度审计 · 零额外 API）

本节完全基于既有 Gate 2 产物与缓存，通过确定性复现脚本 [analyze_gate2_mechanisms.py](./trackA_sug_sed/analyze_gate2_mechanisms.py) 生成（完整 JSON 与表格报告见 [stage_p1_mechanisms.json](./results/heldout_gate2/stage_p1_mechanisms.json) · [p1_mechanisms_report.md](./results/heldout_gate2/p1_mechanisms_report.md)）。

#### 1. 受控三因子归因：`Random → Random-Meso → Meso-Heur-Best → Full Pure-Star-Meso`

在源码 [star_meso.py](./trackA_sug_sed/star_meso.py#L428) 中，`ctrl_random_meso_*` 本质上是 `run_pure_star_meso(..., random_order=True)`——随机打乱锚点簇与 Seam 候选调度顺序，选定锚点后仍保留 `Pure 2D-SE` 起步、`AutoMerge (>=0.85)`、`GuardedUF (0.40)`、干净 Seam 候选池、Top-4 余弦近邻星型打包及同款 Prompt，**仅消融“两层贝叶斯效用与 Seam 优先级排序”**。由此可严格拆解三项独立贡献：

| 归因因子与对照跃迁 | 范围 | $\Delta\mathrm{AMI}$ | $\Delta\mathrm{Mac}\text{-}B^3$ | $\Delta\mathrm{Mic}\text{-}B^3$ | $\Delta\mathrm{Tail}\text{-}B^3$ | 统计显著性与结构机理释义 |
|---|---|---:|---:|---:|---:|---|
| **因子 1：介观星型打包结构先验**<br>（`Random → Random-Meso`） | `Event2012`<br>`Event2018`<br>**`Pooled27`** | +5.99<br>+5.53<br>**+5.80** | +1.61<br>+2.35<br>**+1.91** | +10.12<br>+11.11<br>**+10.52** | +0.38<br>+2.04<br>**+1.06** | `Pooled` 极显著（`AMI/Mic p_raw=1.49e-8`, `Mac p_raw=8.20e-7`）；`UnqNet` 由 `-35.7/+26.5` 跃至 `+617.1/+1023.9`，量化**微簇对比决策域与星型候选池**的主体结构先验 |
| **因子 2：校准贝叶斯优先级调度**<br>（`Random-Meso → Full`） | `Event2012`<br>`Event2018`<br>**`Pooled27`** | +0.19<br>+1.47<br>**+0.71** | +0.69<br>+1.17<br>**+0.89** | +0.09<br>+2.63<br>**+1.12** | +0.75<br>+1.71<br>**+1.14** | 固定同一介观候选池，仅恢复校准排序即在 `Pooled27` 取得 **`Mac-B3 +0.89`（`p_Holm=0.0352`, `18/1/8` 胜）** 与 FR `Mic-B3 +2.63`（`p_raw=0.0420`） |
| **因子 3：LLM 语义裁决 vs 同池 0-Tok 余弦**<br>（`Meso-Heur-Best → Full`） | `Event2012`<br>`Event2018`<br>**`Pooled27`** | +1.18<br>+2.93<br>**+1.90** | +1.15<br>+2.16<br>**+1.56** | +1.70<br>+4.38<br>**+2.79** | +1.24<br>+4.06<br>**+2.39** | 同微簇池下较最优 0-Tok 余弦启发显著拉升 **`Mac-B3 +1.56`（`p_Holm=5.49e-04`）** 与 **`Tail-B3 +2.39`（`p_Holm=0.00963`）**，并在 FR 将总误合并对从 `3,283,538` 压降 `35.3%` 至 `2,125,067` |
| **总路由增量（`Random → Full`）** | **`Pooled27`** | **+6.52** | **+2.80** | **+11.65** | **+2.20** | `p_raw <= 3.77e-6`（全部指标） |

#### 2. 5 大支柱功能分层与配对 Wilcoxon 检验（`Full - Ablated`，含 5 变体 Holm-Bonferroni 校正）

| 消融变体 | 功能定位 | EN (`16` 块) 配对差异 (`p_raw`) | FR (`11` 块) 配对差异 (`p_raw`) | **`Pooled27` 配对 Wilcoxon (`W/T/L`, `p_raw / p_Holm`)** & 运行计数 |
|---|---|---|---|---|
| `w/o Star Packing (k=1)` | **核心通用引擎** | AMI +0.79 (`0.217`), Mic +0.95 (`0.153`)<br>Calls `272→526` (**1.93×**), StarMerges `71→39` | AMI **+3.42** (`0.175`), Mic +5.50 (`0.0537`)<br>Calls `142→383` (**2.70×**), StarMerges `70→30` | `AMI +1.86 (17/2/8, 0.0451 / 0.226)`, `Mic-B3 +2.80 (17/2/8, 0.0219 / 0.110)` |
| `w/o Seam Split` | **核心通用引擎** | `Mac +0.67 (0.0256)`, `Tail +1.29 (0.00195 / 0.00977)` | `AMI +0.41 (0.0371)`, `Mac +2.11 (0.00195 / 0.00977)`, `Mic +0.50 (0.00586 / 0.0293)`, `Tail +4.76 (0.0313)` | **`Mac-B3 +1.26 (21/2/4, 6.37e-5 / 3.18e-4)`**, **`Tail-B3 +2.70 (16/10/1, 3.05e-5 / 1.53e-4)`**（EN/FR 共触发 `31 / 19` 次真拆分） |
| `w/o AutoMerge (>=0.85)` | **效率与风险守卫** | **`AMI +1.01 (0.0313)`, `Mic +1.58 (0.0313)`**<br>`Full-NoAM`: `ΔFM +1,845 / ΔFS -635,279` | `AMI -0.47 (0.461)`, `Mac +0.28 (0.945)`<br>`Full-NoAM`: **`ΔFM +749,776`** / `ΔFS -201,358` | `AMI +0.41 (0.359 / 1.0)`, `Mac +0.24 (0.359 / 1.0)`（FR 异常集中于 `M_13/M_14`） |
| `w/o Bayes Band & AutoMerge` | **效率与风险守卫** | `AMI +0.30 (0.626)`, CheckedEdges `658→451` | `AMI -0.40 (0.492)`, `Mac +0.12 (0.922)`, `Tail +0.91` (`56.67→57.58`) | `AMI +0.01 (0.944 / 1.0)`, `Mac +0.01 (0.855 / 1.0)`；主要将待检边数压减 `-31.5%`（EN）并守稳 FR 长尾 |
| `w/o Guarded UF` | **效率与风险守卫** | `AMI -0.57 (0.250)`, 但 **`NoG_dFM +47,895` (+29.1%)** | `AMI +0.13 (0.742)`, 但 **`NoG_dFM +1,446,601` (+68.1%)** | `Mac +0.15 (0.706 / 1.0)`；在少数大簇上牺牲微弱保守度（EN `M_07`）换取全流零雪崩 |

#### 3. 英/法双语编码器余弦校准差异与 `AutoMerge (>=0.85)` 机理诊断（`size >= 2` 实域）

严格限定于 `AutoMerge` 实际作用的多帖微簇域（`size >= 2`），全量统计 27 块 `Pure 2D-SE` 初始微簇对真实余弦校准表明，`AutoMerge` 在英法两语上的不对称表现并非实现缺陷，而是由两个冻结编码器的背景各向异性差异直接决定：

| 数据集与冻结句向量模型 | 异事件微簇对余弦<br>`mean / p95 / p99` | 同事件微簇对余弦<br>`mean / p50` | `[0.90, 1.00]`<br>对精度 | `[0.85, 0.90)`<br>对精度 | **`>=0.85` 综合精度**<br>（规模加权精度） | `AutoMerge` 消融<br>`Full - NoAM` ($\Delta\mathrm{FM} / \Delta\mathrm{FS}$) |
|---|---|---|---|---|---|---|
| **`Event2012`** (`all-MiniLM-L6-v2`) | `0.159 / 0.379 / 0.532` | `0.464 / 0.467` | **100.0%** (`20/20`) | **97.7%** (`43/44`) | **98.4%** (`63/64`, **99.1%**) | `+1,845 / -635,279`（高纯度净获益 `+1.01 AMI`） |
| **`Event2018`** (`paraphrase-multilingual-MiniLM-L12-v2`) | `0.342 / 0.604 / 0.745` | `0.637 / 0.652` | **100.0%** (`5/5`) | 84.6% (`22/26`) | 87.1% (`27/31`, 85.2%) | **`+749,776 / -201,358`**（`M_13/M_14` 占比 **96.9%**） |

* **定位结论**：单语 `all-MiniLM-L6-v2` 背景相似度极低（异事件 `p99=0.532`），故通用 `tau_auto=0.85` 精度高达 `98.4%`；多语 `paraphrase-multilingual-MiniLM-L12-v2` 余弦基准整体上移（异事件 `mean=0.342, p99=0.745`），使 `[0.85, 0.90)` 区间降至 `84.6%`，并在 `Event2018` `M_13`（`+566,697 FM`）与 `M_14`（`+159,633 FM`）两个超大簇块触发大额误合并，而 `[0.90, 1.00]` 在双语均维持 **100.0%**（论文正文中将在保持 Dev-locked 公平的前提下把多语编码器自适应阈值校准列为结构性发现与后续工程启示）。

#### 4. `GuardedUnionFind` 传递链式防爆保险机制

* 在默认 Dev 锁定设置下，全流程显式记录的 LLM 后置拦截为 EN `2` 次 / FR `2` 次（其余多组冲突在发送 LLM 查询前已被 `uf.can_merge()` 质心守卫预先过滤）。
* **反事实防爆价值**：移除 `GuardedUF` 后，`Event2018` 总误合并对由 `2,125,067` 激增至 `3,571,668`（**暴涨 `+1,446,601` 对，即 `+68.1%`**），并在 **`M_09`（`-7.09 AMI, +938,270 FM`）** 与 **`M_10`（`-5.39 AMI, +372,863 FM`）** 直接引发严重传递闭合塌陷；`Event2012` 亦拦截 `+47,895 FM`（**`+29.1%`**），代价仅是 `Event2012 M_07` 上因 1 次保守质心拦截使单一超大事件未合并而损失 `-5.30 AMI`——验证了 `GuardedUF` 作为**高置信尾部雪崩保险器**而非均值放大器的真实定位。

#### 5. `Tail-B3` 过碎片化效应与 `Cross-Day` 接受率边界

* **过碎片化 `Pred/True` 归因**：按全量真实事件数归一化（EN `611`、FR `290`），修复负偏斜坍塌并启用动态 `CANNOT-LINK` 守卫后的 `ClusterLLM-NoFT` 仍输出高达 **`2,315`（`3.76×`）** 与 **`2,851`（`10.36×`）** 个预测簇（原坍塌前为 `5.13× / 15.13×`），`SBERT-Agglom` 为 **`2.33× / 3.51×`**，而 `Pure-Star-Meso` 仅 **`1.62× / 1.20×`**。极端切分在 `Tail-B3` 指标计算下虽易把少量孤立长尾帖切进超高纯度微簇，却伴随全局 `AMI` 降至 `84.21 / 64.96`（`Mic-B3` `81.46 / 60.71`）且 `FS` 剧增。
* **Cross-Day 精确率/召回率权衡**：`sug_cross_star` 在 `Event2012` 仅接受 `72/181 = 39.8%` 的跨日 LLM 候选（接近 `oracle_band` `82/199 = 41.2%`，距上界仅 `-1.08` G-AMI）；在语言更多样的 `Event2018` 上保守拒连 `75/91`（仅接受 **`16/91 = 17.6%`** vs Oracle `29/114 = 25.4%`），由此解释为何 FR 上高精准率助其反超全部静态原型阈值（`84.76` vs `84.57`），而距 `oracle_band` 仅余 `-0.36` G-AMI 召回缺口。

---

## 六、后续待办与投稿路线

1. **已完成（第一 & 第二优先级全量收口）**：
   - **第一优先级（全面审查与账本审计封箱）**：核心代码/审计口径修复、5 处残余瞬态 HTTP 错误清零、基于共享缓存全量刷新 `stage_qwen3_main` / `stage_qwen3_abl` / `stage_qwen3_crossday` / `stage_budget_anon` 与 P1 审计报告、新增 `e2e_pseudonym_meso_heur_0tok`、生成不可变冻结协议 [protocol_v2.json](./trackA_sug_sed/protocol_v2.json)；
   - **第二优先级（基线公平性加固）**：将 `ClusterLLM-NoFT (EMNLP'23)` 粒度选择改为抗负类偏斜均衡准确率（`bal_acc`）+ 动态 `CANNOT-LINK` 守卫并将 `UnqNet` 对齐至 `SBERT-Agglom (0.60)`（100% 命中 Gate 2 本地缓存）；扩展 Dev（`M_02..M_05`）零 LLM 超参网格确认真内点（`EventX` EN `2.0` / FR `1.5`、`HiSEvent Dev-Tuned` FR `(0.55, 1.0)`、`HyperSED` EN `0.26` 伴 `n_anchors <= 360`），全 0 错误刷新 Gate 1/Gate 2 及 `Event2018 Star-Meso on Dev-Tuned`（`84.43 AMI / 79.57 Mac-B3`），并重冻结 `protocol_v2.json`。
2. **后续可选扩展（第三优先级，待确认需求后启动）**：
   - **多模型留出泛化**：在全部 27 个留出块上运行 `python trackA_sug_sed/run_heldout_gate2.py --stage qwen25_full` 补齐 `Qwen/Qwen2.5-7B-Instruct` 留出集跨模型验证矩阵（视预算可扩展非 Qwen 8B 家族对照）。
3. **IP&M 学术图表与正式 LaTeX 成稿**：基于 `results/heldout_gate2/` 全量 JSON（含 [gate2_full_report.json](./results/heldout_gate2/gate2_full_report.json) 与 [stage_p1_mechanisms.json](./results/heldout_gate2/stage_p1_mechanisms.json)）绘制出版级矢量图并撰写完整 *Information Processing & Management* (IP&M) 正式文稿。
