# Track A 流式社交事件检测主流程详细定稿（`SUG-SED v2.1` · `Pure-Star-Meso` + `SUG-CrossStar`）

> **文档定位**：本文档为审查并重构后的 **Track A 社交事件检测（SED）主流程权威方法规格书**，仅描述当前确定并落地的五阶段流式 SED 主体算法、数学定义、全部锁定超参、因果预算/防雪崩不变量与源码一一对应关系。外部对比基线、实验重跑及论文行文/图表细节暂列于文末待办（TODO）。

---

## 0. 问题形式化与系统输入/输出

考虑按时间顺序到达的离散社交文本流块序列 $\{M_0, M_1, \dots, M_T\}$。对于任意消息块 $M_t$（含 $N_t = |M_t|$ 条按时间戳升序排列的帖子），系统输入为：
1. **原始帖子与元属性**：每条帖子 $i \in \{0, \dots, N_t - 1\}$ 的文本 $x_i$、发布时间戳 $\tau_i \in \mathbb{R} \cup \{\mathrm{NaN}\}$、归一化实体集合 $E_i$、话题标签集合 $H_i$、作者用户 ID $u_i$、提及用户集合 $U_i$ 以及词形还原词元表 $L_i$；
2. **冻结句向量表示**：由冻结编码器提取并严格单位 $L_2$ 归一化的稠密向量矩阵 $\mathbf{E}_t \in \mathbb{R}^{N_t \times d}$（$\|\mathbf{e}_i\|_2 = 1$）；
3. **严格零前瞻因果预算约束**：
   * 块内细化（Stage 2–4）受控于 `CausalUniformBucket(tokens_per_block=10000, allow_carryover=False)`；
   * 跨块演化链接（Stage 5）受控于 `CausalUniformBucket(tokens_per_block=6000, allow_carryover=True, max_carryover_mult=1.5)`；
   * 每次调用前通过统一字符估算器 `estimate_tokens(sys_msg, user_msg, est_completion) = int((len(sys_msg) + len(user_msg)) / 3.4) + est_completion` 执行门控检查，调用后按 API 真实计费（缺失 `usage` 时按 `3.4` 字符/token 兜底并显式标记 `USAGE_MISSING`）精确扣减，绝不窥探未来块长度或数量；[llm_client.py](./trackA_sug_sed/llm_client.py) 在标准默认配置（`max_tokens=384, use_json_mode=True, enable_thinking=default`）下保留对历史 3 字段缓存键（`model`, `temperature`, `messages`）的只读兼容回退。

**系统输出**：
* 块内聚类划分标签 $\hat{\mathbf{y}}_t \in \mathbb{Z}_{\ge 0}^{N_t}$；
* 严格展示掩码 `all_shown` $\in \{\text{False}, \text{True}\}^{N_t}$（标记所有曾作为锚点代表、候选代表、Seam 子组代表或参考上下文送入 LLM Prompt 的帖子索引，供未查询净改善 `UnqNet` 审计）；
* 块内确认不可合并互斥簇对列表 `intra_cannot_link_pairs`（传递至 Stage 5）；
* 跨块全局事件演进映射 $\hat{\mathbf{g}}_t \in \mathbb{Z}_{\ge 0}^{N_t}$。

---

## 1. 总体五阶段处理流程与核心不变量

审查后定型的 Track A 主流程将阶段顺序统一固化为：

$$\text{Stage 1 (Pure 2D-SE Base)} \longrightarrow \text{Stage 2 (Clean Seam Split 预清洗)} \longrightarrow \text{Stage 3 (AutoMerge + 统一 GuardedUF)} \longrightarrow \text{Stage 4 (Two-Tier Star-Merge)} \longrightarrow \text{Stage 5 (SUG-CrossStar)}$$

```text
  [流式块 M_t]
       │
       ▼
  Stage 1: 异构语义 k-NN + 属性共现构图 -> 精确 2D-SE 最小堆凝聚 + 节点迁移精化 (高纯度微簇，FS >> FM)
       │
       ▼
  Stage 2: Clean Seam Split 预清洗 (先拆混杂大簇 |C|>=8，记录 Seam CANNOT-LINK，Seam 预算 <= 35%)
       │
       ▼  ──► 一次性初始化统一 GuardedUnionFind (真实成员向量和 comp_vecs + 注入全部 Seam CANNOT-LINK)
  Stage 3: Zero-Token AutoMerge (size>=2, 无监督背景 p95 自校准 eff_tau_auto，实时质心 + 互斥守卫防漂移)
       │
       ▼  ──► 沿用同一个 GuardedUnionFind 与实时刷新分量状态
  Stage 4: Two-Tier Size-Weighted Star-Merge (Tier1 [0.55, tau_auto) -> Tier2 [0.45, tau_auto), k_nb=4)
       │
       ├──────────────────────────────────────────────► 输出块内标签 y_t, all_shown, intra_cannot_link_pairs
       ▼
  Stage 5: SUG-CrossStar (滑动 14 块全局记忆 + 滚动代表文本 + cos>=0.90 自动续连 / [0.65,0.90) LLM 裁决
                          + 同块 CANNOT-LINK 与质心余弦下界 0.40 守卫)
```

### 核心不变量（Invariants）
1. **先拆后合与全流程统一 `GuardedUnionFind`**：Stage 2（`Clean Seam Split`）在任何零成本合并前清洗 Stage 1 潜在的少数混杂大簇；其确认拆分的簇对立即作为硬 `CANNOT_LINK` 注入贯穿 Stage 3 与 Stage 4 的**唯一** `GuardedUnionFind` 实例中，从根源上杜绝“刚被 Seam 拆开、随即被 `AutoMerge` 或 `Star-Merge` 静默合回”。
2. **真实分量质心与严格互斥守卫**：`GuardedUnionFind` 直接以各初始簇真实成员嵌入向量和 $\mathbf{v}_r = \sum_{i \in C_r} \mathbf{e}_i$ 初始化，合并时精确累加 $\mathbf{v}_{r_1} \leftarrow \mathbf{v}_{r_1} + \mathbf{v}_{r_2}$，使单位化质心 $\boldsymbol{\mu}_r = \mathbf{v}_r / \|\mathbf{v}_r\|_2$ 与真实簇原型零偏差；同时实时维护分量间互斥计数，在 `min_cl_votes=1` 与 `guard_sim=0.40` 下以 $O(1)$ 复杂度阻断传递闭包雪崩。
3. **术语与公式诚实一致性**：完全去除“贝叶斯效用 / Bayes Band”等过度数学命名，所有启发式排序规则（两阶段相似度分层 `Two-Tier Similarity Band`、规模加权预期收益效用 `Size-Weighted Expected-Yield Utility`）均严格按代码真实表达式定义。

---

## 2. Stage 1 · 异构消息图构建与高纯度 2D-SE 起步

对应源码：[data_loader.py](./core_shared/data_loader.py)、[embed_cache.py](./trackA_sug_sed/embed_cache.py)、[graph_builder.py](./core_shared/graph_builder.py)、[se_solver.py](./core_shared/se_solver.py)、[partition_cache.py](./core_shared/partition_cache.py)。

1. **时间戳安全加载与向量单位化**：
   * `load_socialed_npy` 采用时区/分辨率无关的 `(ts - UTC_EPOCH).total_seconds()` 提取 Unix 秒级时间戳（`NaT` 转为 `np.nan`），按 `(timestamps, tweet_ids)` 严格时序排序；
   * 输入嵌入经过 `_ensure_l2_normalized` 校验并归一化为单位向量（$\|\mathbf{e}_i\|_2 = 1$）。
2. **稀疏语义 $k$-NN 图（`sem_adj`）**：
   * 分批计算余弦相似度 $\mathbf{e}_i^\top \mathbf{e}_j$，排除自环，每节点保留相似度最高的 `knn_k=15` 个邻居中满足 $\mathbf{e}_i^\top \mathbf{e}_j \ge \tau_{\text{sim}} = 0.50$ 的边；
   * 通过 $\max(\mathbf{A}_{\text{sem}}, \mathbf{A}_{\text{sem}}^\top)$ 对称化（若启用 `time_decay_per_day > 0` 则乘以 $\exp(-\lambda |\Delta \tau_{ij}|_{\text{day}})$，默认 `0.0`）。
3. **稀疏属性共现图（`attr_adj`）**：
   * 为每条帖子建立 `("ent", e)`、`("tag", h)`、`("usr", user_id / mentioned_user)` 倒排桶，类型基础权重为 $w_{\text{ent}}=1.0,\ w_{\text{tag}}=1.2,\ w_{\text{usr}}=0.6$；
   * 桶内按时间戳升序（`NaN` 置后）排列，单属性阻尼权重 $w_{\text{base}} = w_{\text{type}} / \log_2(2 + m)$（$m$ 为桶大小）：
     * 常规桶（$m \le 300$）：时间半窗 `eff_half_win = 6`（即 `max_neighbors_per_attr // 2`），语义下限 `eff_min_sim = 0.05`；
     * 大热点桶（$m > 300$）：不丢弃事件桶，而是收紧时间半窗至 `eff_half_win = min(half_win, 4)` 且提高语义下限至 `eff_min_sim = max(0.05, 0.35)`；
   * 仅当 $\mathbf{e}_u^\top \mathbf{e}_v \ge \text{eff\_min\_sim}$ 时连边，边权累加 $w_{\text{base}} \cdot (\mathbf{e}_u^\top \mathbf{e}_v)$。
4. **异构合图与精确 2D-SE 求解**：
   * 合成图邻接矩阵 $\mathbf{A} = \mathbf{A}_{\text{sem}} + \alpha \mathbf{A}_{\text{attr}}$（默认 $\alpha = 0.5$，去除冗余非实体 TF-IDF 计算）；
   * `solve_2d_se(adj, refine_moves=True)` 交替执行精确体积更新最小堆凝聚 `minimize_2d_se_agglomerative` 与严格单调节点迁移精化 `refine_2d_se_node_moves` 直至收敛，得到高纯度初始微簇划分（假分裂远多于假合并）。

---

## 3. Stage 2 · 干净介观缝隙拆分（Clean Seam Split 预清洗）

对应源码：[star_meso.py](./trackA_sug_sed/star_meso.py)（`_spherical_kmeans_deterministic`、`_fiedler_bipartition`、`detect_clean_seams`、`run_pure_star_meso`）与 [llm_client.py](./trackA_sug_sed/llm_client.py)（`build_seam_split_prompt`）。

在任何合并发生之前，对 Stage 1 输出划分中的混杂大簇执行局部缝隙检测与清洗：

1. **摘要构建**：对当前所有簇调用 `build_cluster_summaries`，计算：
   * 单位化原型 $\boldsymbol{\mu}_C = \frac{\sum_{i \in C} \mathbf{e}_i}{\|\sum_{i \in C} \mathbf{e}_i\|_2}$；
   * 按与 $\boldsymbol{\mu}_C$ 余弦降序的最多 `3` 条中心代表帖 `medoids`；
   * 簇级 TF-IDF（$(1 + \ln \mathrm{tf}) \cdot (\ln \frac{1 + K}{1 + \mathrm{df}} + 1)$，实体与 `#hashtag` 计 2 次权重，长度 $\ge 3$ 词元计 1 次权重）排序的最多 `5` 个关键词。
2. **候选缝隙生成（`detect_clean_seams`）**：遍历所有大小 $|C| \ge 8$（`min_cluster_size=8`）的簇：
   * **确定性球面 $k$-means 提议**：$k_{\text{sub}} = \min(6, \max(3, |C| // 12))$；当 $|C| < 18$ 时同时运行 $k=2$（兼顾如 $4+4$ 的小簇均分）与 $k=k_{\text{sub}}$，否则运行 $k=k_{\text{sub}}$。初始中心采用全线性 $O(md)$ 确定性最远点规则（首中心取 $\arg\max_i \mathbf{e}_i^\top \sum_j \mathbf{e}_j$，后续按 $(1 - \max_c \mathbf{e}_i^\top \boldsymbol{\mu}_c)^2$ 最大值选取），迭代至多 15 步。每个簇 $p$ 产生一个 `(主部 r_loc, 少数口袋 p_loc)` 二分候选；
   * **Fiedler 谱二分提议**：对诱导子图 $\mathbf{A}_{C,C}$ 若存在 $\ge 2$ 个连通分量则取最大连通分量 vs 其余；否则计算归一化邻接矩阵次大特征向量（$|C| \le 80$ 用精确 `eigh`，$|C| > 80$ 用 `eigsh`）并在 30 个分位切点中选择电导最小切分；不可解时回退至中心化嵌入第一主成分 SVD 中位数二分。将较小一侧作为口袋 `fg2`；
   * **结构与语义过滤**：
     1. 大小门控：$3 \le |P| \le |C| - |P|$（且主部 $\ge 3$，即口袋占比 $\le 50\%$）；
     2. 计算主部原型 $\boldsymbol{\mu}_{\text{main}}$ 与口袋原型 $\boldsymbol{\mu}_{\text{pock}}$、口袋内聚度 $\mathrm{intra\_pock} = \frac{1}{|P|}\sum_{i \in P} \mathbf{e}_i^\top \boldsymbol{\mu}_{\text{pock}}$、跨部余弦 $\mathrm{cross\_sim} = \boldsymbol{\mu}_{\text{main}}^\top \boldsymbol{\mu}_{\text{pock}}$；
     3. **精准转发小团伙豁免**：若 $|P| < 6$ 且 $\mathrm{intra\_pock} > 0.96$ 且 $\mathrm{cross\_sim} > 0.65$，视为同事件完全转发簇，直接跳过以免浪费 Seam 配额；
     4. **语义分离门控**：定义 $\mathrm{sem\_sep} = \max(0, \mathrm{intra\_pock} - \mathrm{cross\_sim})$，保留 $\mathrm{sem\_sep} \ge 0.12$（`min_sem_sep=0.12`；`use_seam_gate=False` 不启用跨部硬上限）。
   * **候选排序**：每个混杂簇保留得分 $\mathrm{score} = \mathrm{sem\_sep} \cdot \ln(1 + |P|)$ 最高的单个最优口袋；所有簇的候选按 `score` 降序排列（若 `random_order=True` 则按 `seed` 随机打乱）。
3. **预算保护与平衡对比裁决**：
   * 最多查询前 `max_seam_calls = 3` 个候选，且单块 Seam 累计消耗严格不得超过块预算的 `max_seam_budget_frac = 0.35`（防止小预算如 `2k` 时耗尽后续 `Star-Merge` 额度）；
   * 向 LLM 发送 `build_seam_split_prompt`（各自展示主部与口袋最多 3 条代表帖与大小，要求先摘要 `event_a`、`event_b` 再返回 `different_events`），将展示的至多 6 条代表帖标记为 `all_shown = True`；
   * 仅当响应无错且 `_parse_bool(different_events)` 为 `True` 时，为口袋 `sub2` 分配新簇 ID，并将代表节点对 `(sub1[0], sub2[0])` 记入 `seam_split_node_pairs`。

---

## 4. Stage 3 · 零成本高置信合并与统一质心守卫并查集（`AutoMerge` + `GuardedUnionFind`）

对应源码：[star_meso.py](./trackA_sug_sed/star_meso.py)（`GuardedUnionFind`、`run_pure_star_meso`）。

1. **统一 `GuardedUnionFind` 初始化与 Seam 硬约束注入**：
   * 在 Stage 2 清洗后的划分上重算簇摘要 `summs`、排序簇 ID `all_cids`、初始原型矩阵 $\mathbf{P}$ 与全矩阵相似度 $\mathbf{S} = \mathbf{P}\mathbf{P}^\top$；
   * 按各簇真实成员向量和 $\mathbf{v}_c = \sum_{i \in C_c} \mathbf{e}_i$ 初始化**唯一**的 `GuardedUnionFind(all_cids, S, guard_sim=0.40, enforce_guards=use_guarded_uf, protos=P, sizes=sizes, comp_vecs=comp_vecs, min_cl_votes=1)`；
   * 立即将 Stage 2 确认拆分的全部 `seam_split_node_pairs` 映射为当前簇 ID 并调用 `uf.add_cannot_link(c_a, c_b)`，同时维护 $O(1)$ 分量级互斥索引 `comp_cl_counts`；
   * 同步维护活根集合 `active_roots`、实时分量摘要字典 `comp_summs` 与动态相似度矩阵 `sim_mat`。
2. **零成本高置信 `AutoMerge`（含无监督编码器各向异性自校准）**：
   * 在多帖微簇域（$|C| \ge 2$）上提取初始余弦子矩阵 $\mathbf{S}_{\ge 2}$；
   * **无监督背景分位数自校准（默认开启 `auto_calibrate_tau=True`）**：当多帖微簇数量 $K_{\ge 2} \ge 10$ 时，计算 $\mathbf{S}_{\ge 2}$ 严格上三角全部微簇对余弦的 95% 分位数 $p_{95}(\mathbf{S}_{\text{upper}})$（在多事件块中稳定刻画异事件背景余弦上尾，规避 $p_{99}/p_{99.5}$ 被少数同事件真合并污染），自适应托底有效阈值：
     $$\tau_{\text{auto}}^{\text{eff}} = \max\bigl(\tau_{\text{auto}},\ \min(0.90,\ p_{95}(\mathbf{S}_{\text{upper}}) + 0.28)\bigr)$$
     在各向同性单语编码器（`Event2012`，$p_{95} \approx 0.35\text{--}0.48$）下自动维持 $\tau_{\text{auto}}^{\text{eff}} = \tau_{\text{auto}} = 0.85$ 零副作用；在背景余弦整体抬升的多语编码器（`Event2018`，$p_{95} \approx 0.60\text{--}0.76$）下自动上调至 $0.88\text{--}0.90$，精准拦截 $[0.85, 0.90)$ 假合并并将其转交 Stage 4 LLM 裁决（实际生效阈值记录于 `stats["eff_tau_auto"]`）；
   * 在多帖微簇域（`|C| >= 2`）中抽取满足 $S_{ij} \ge \tau_{\text{auto}}^{\text{eff}}$ 的所有初始簇对并按相似度降序遍历；
   * 对每对 `(c1, c2)` 取实时根 `(r1, r2)`，执行**三重守卫**：
     1. 若 `r1 == r2` 则跳过；
     2. 校验实时分量质心余弦 $\boldsymbol{\mu}_{r_1}^\top \boldsymbol{\mu}_{r_2} \ge \tau_{\text{auto}}^{\text{eff}}$（开启 `use_guarded_uf` 时生效，阻断 2-hop 传递链式阈值漂移）；
     3. 在 `_apply_merged_root_refresh` 中调用 `uf.merge(r1, r2)`（校验 `comp_cl_counts[r1][r2] < min_cl_votes` 且 $\boldsymbol{\mu}_{r_1}^\top \boldsymbol{\mu}_{r_2} \ge \text{guard\_sim}=0.40$）；
   * 合并成功后，实时合并成员列表、更新 `comp_summs[new_r]`、剔除旧根 `old_r`、刷新 `sim_mat` 对应行列，并递增 `auto_merge_ops`。

---

## 5. Stage 4 · 两阶段多候选星型打包裁决（Two-Tier Size-Weighted Star-Merge）

对应源码：[star_meso.py](./trackA_sug_sed/star_meso.py)（`run_pure_star_meso`）与 [llm_client.py](./trackA_sug_sed/llm_client.py)（`build_star_merge_prompt`）。

直接沿用 Stage 3 的统一 `uf`、`active_roots`、`comp_summs` 与 `sim_mat`：

1. **两阶段相似度分层带（Two-Tier Similarity Band）与阶段分工**：
   * **图结构与原型语义带的职责分离**：Stage 1 构图与 Stage 2 Fiedler 谱切分已充分利用 `attr_adj` 异构属性共现结构；而 Dev 实证表明在低余弦区间 $[0.35, 0.45)$ 内包含 `attr_adj` 跨簇边的簇对中约 `90%–95%` 属于共享泛化实体的不同事件，若直接将低余弦属性桥接边拉入 LLM 候选池会严重稀释小预算。因此 Stage 3 与 Stage 4 刻意保持纯粹的原型语义相似度候选带；
   * **Tier 1（核心模糊多帖带）**：相似度区间 $[\tau_{\text{lo}}, \tau_{\text{auto}}^{\text{eff}}) = [0.55, \tau_{\text{auto}}^{\text{eff}})$，最小分量规模 `min_sz = 2`（优先合并高价值多帖片段）；
   * **Tier 2（长尾与单帖补全带）**：相似度区间 $[\tau_{\text{tier2}}, \tau_{\text{auto}}^{\text{eff}}) = [0.45, \tau_{\text{auto}}^{\text{eff}})$，最小分量规模 `min_sz = 1`；但对**双单帖对**（$|C_{r_1}| = 1 \land |C_{r_2}| = 1$）强制保留额外下界 `sim >= 0.55`，防止单帖噪声泛滥；
   * 注：`use_bayes_band=False`（对应消融 `abl_no_bayes_band`）时合并为单层 $[0.35, \tau_{\text{auto}}^{\text{eff}}), \text{min\_sz}=1$；`random_order=True` 时仍严格保留两阶段分层顺序，仅在每层内部随机选取候选锚点。
2. **邻簇星型打包与规模加权预期收益效用排序**：
   * 在当前层内，对每个满足 $|C_{r_1}| \ge \text{min\_sz}$ 的活跃根分量 $r_1$，筛选同层内相似度满足条件的所有活跃邻分量，按实时余弦相似度**从高到低**排序，依次过滤：
     * 非单帖对单帖 `< 0.55`；
     * 未在 `queried_root_nbrs[r1]` 中（未曾成功查询过）；
     * 不在 `failed_root_pairs` 中（本次同根状态尚未因解析/API 错误失败过）；
     * 通过 `uf.can_merge(r1, r2)`（无 `CANNOT_LINK` 互斥且实时质心余弦 $\ge 0.40$）；
   * 选取前至多 `k_nb = 4` 个合法邻分量构成星型包 $\mathcal{P}(r_1) = \{(s_j, r_{2,j})\}_{j=1}^{m}$（$1 \le m \le 4$；当 `use_star_packing=False` 时 `k_nb=1` 且上限放宽为 `max(80, 3 * max_star_calls)`）；
   * 计算规模加权预期收益效用：
     $$U(r_1, \mathcal{P}) = \sum_{(s, r_2) \in \mathcal{P}} s^2 \cdot \bigl(1 + 0.25 \ln(1 + \min(|C_{r_1}|, |C_{r_2}|))\bigr)$$
   * 选取效用最高的 `(anchor_r, pack)`（若 `random_order=True` 则均匀随机抽取）。
3. **对比裁决、严格边界校验与在线分量刷新**：
   * 在预算桶 `CausalUniformBucket(10000, allow_carryover=False)` 与总调用上限 `max_star_calls = 24` 内调用 `build_star_merge_prompt`（向 LLM 展示锚簇及至多 4 个候选簇各 `top_keywords[:5]` 与 `medoids[:2]`），将锚簇与全部候选展示的 `medoids[:2]` 全部记为 `all_shown = True`；
   * 若返回 HTTP/JSON 错误或 `matches` 字段非列表或存在越界/非正整数 ID（`id < 1` 或 `id > len(cand_roots)`），则记 `parse_errors += 1`、将该批次根对记入临时 `failed_root_pairs` 以便在结构发生变化前跳过，**绝不污染 `queried_root_nbrs`，也绝不写入 `CANNOT_LINK`**；
   * 若响应合法：
     1. 将 `(anchor_r, r2)` 双向记入 `queried_root_nbrs`；
     2. 先对所有**未匹配**候选调用 `uf.add_cannot_link(anchor_r, r2)`；
     3. 再依次对所有**匹配**候选执行 `_apply_merged_root_refresh(cur_anchor, cur_r2)`：通过 `uf.merge`（含质心守卫与互斥校验）合并后，立即重建新根分量的 `ClusterSummary`（新成员表、单位原型、最新 `medoids[:3]`、最新 `top_keywords[:5]`）、合并 `queried_root_nbrs` 邻接、清除涉及合并双方旧根的 `failed_root_pairs` 记录，并刷新 `sim_mat` 对应行列。
4. **最终导出**：循环结束后汇总 `blocked_by_guard = uf.blocked_count`，通过 `uf.apply_to_labels(labels)` 生成连续重编号分区标签 $\hat{\mathbf{y}}_t$，并将所有最终不同簇之间的 `seam_split_node_pairs` 与 `uf.cannot_pairs` 归一化为 `intra_cannot_link_pairs` 输出。

---

## 6. Stage 5 · 跨块流式原型记忆与不确定带裁决（`SUG-CrossStar`）

对应源码：[cross_day.py](./trackA_sug_sed/cross_day.py)（`GlobalEventMemory`、`evaluate_cross_day_stream`）与 [llm_client.py](./trackA_sug_sed/llm_client.py)（`build_cross_star_prompt`）。

在连续时间块流 $\{M_0, \dots, M_T\}$ 上运行（前 `M_00..M_05` 为预热记忆建立期，`M_06+` 为在线评测期）：

1. **滑动窗口全局记忆（`GlobalEventMemory`）**：
   * 每到达新块 $M_t$，先调用 `evict_expired(t)` 主动清除 `t - ev.last_block > window_blocks`（`window_blocks = 14`）的过期历史事件，保持扫描复杂度严格为 $O(|E_{\text{active}}| \cdot d)$；
   * 为新块 $M_t$ 构建日内簇摘要，按簇大小 $|C|$ 从大到小遍历。
2. **记忆初始化与滚动演化更新**：
   * `spawn_event`：创建新全局事件 `gid`，存储原型、总帖数、首/末出现块号、`rep_texts`（至多 3 条代表文本）与 `top_keywords`（至多 5 词）；
   * `update_event`：
     * **带上限 EMA 原型更新**：取旧权重 $w_{\text{old}} = \min(N_{\text{hist}}, 200)$、新权重 $w_{\text{new}} = |C|$，更新单位化原型 $\boldsymbol{\mu}_{\text{gid}} \leftarrow \mathrm{norm}(w_{\text{old}} \boldsymbol{\mu}_{\text{gid}} + w_{\text{new}} \boldsymbol{\mu}_C)$；
     * **滚动锚定代表文本与关键词更新**：保留首条 Day-0 锚定代表文本 `ev.rep_texts[0]`，并将最新演化文本 `new_txt` 置于 `ev.rep_texts[1]`（`[ev.rep_texts[0], new_txt, ...][:3]`），确保 `build_cross_star_prompt` 展示的 `rep_texts[:2]` 同时呈现事件起源与最新进展；关键词保留历史前 2 词与当前最新前 3 词去重截断至 5 词。
3. **同块合流守卫（替代一刀切唯一性假定）**：
   * 若外部传入日内互斥表 `block_cannot_links`（且未显式强制 `enforce_block_uniqueness=True`），则允许同一块内的多个尚未完全合并的同事件碎片合流到同一历史事件 `gid`，但必须同时满足两项守卫：
     1. **日内 `CANNOT_LINK` 硬拦截**：待合流日簇 $c$ 与已链接至同一 `gid` 的同块日簇 $c'$ 之间不得存在日内 `CANNOT_LINK` 约束（命中记入 `blocked_cannot_link_collisions`）；
     2. **同块质心余弦下界守卫**：$\boldsymbol{\mu}_c^\top \boldsymbol{\mu}_{c'} \ge \text{same\_block\_guard\_sim} = 0.40$（低于下界拦截并记入 `blocked_same_block_collisions`）。
   * 若未提供 `block_cannot_links`（或显式 `enforce_block_uniqueness=True`），则退化为每块至多一个日簇链接至同一 `gid`。
4. **三级决策与因果预算降级**：
   * 计算日簇 $c$ 与所有合法活跃历史事件的余弦相似度，取最高匹配 `best_gid` 及 `best_cos`：
     * **高置信自动续连**：若 `best_cos >= tau_hi`（`0.90`），零消耗直接链接至 `best_gid`；
     * **不确定带 LLM 对比裁决**：若 `0.65 <= best_cos < 0.90` 且 $|C| \ge \text{min\_cross\_size}=3$（且处于评测块），提取模糊带内余弦最高的至多 `top_k_cand = 3` 个活跃历史事件，在 `CausalUniformBucket(6000, allow_carryover=True, max_carryover_mult=1.5)` 内调用 `build_cross_star_prompt`；
       * 若合法返回 `matched_id in [1..k]` 且再次通过 `_gid_allowed`，链接并更新该历史事件；
       * 若模型显式返回 `matched_id = null`，记 `llm_rejected += 1` 并新建全局事件；
       * 若预算耗尽（`res is None`），记 `budget_skipped += 1` 并按回退阈值 `tau_fb = 0.90` 决策（保守新建）；
       * 若发生 API/JSON 解析错误或 `matched_id` 越界，仅记 `parse_errors += 1` 并保守新建，**不计入 `llm_rejected`**；
     * **低置信新建**：若 `best_cos < 0.65`（或 $|C| < 3$ 且 `< tau_fb`），直接新建全局事件。

---

## 7. 锁定超参总表与代码模块映射

| 阶段 | 参数名 | 锁定值 | 代码位置 |
|---|---|---|---|
| **Stage 1** | `knn_k` / `knn_sim_threshold` / `attr_alpha` | `15` / `0.50` / `0.5` | [graph_builder.py](./core_shared/graph_builder.py) · [star_meso.py](./trackA_sug_sed/star_meso.py) |
| **Stage 1** | `max_attr_bucket_size` / `max_neighbors_per_attr` | `300` / `12`（大桶 `<=4, sim>=0.35`） | [graph_builder.py](./core_shared/graph_builder.py) |
| **Stage 1** | `refine_moves` / `max_refine_iters` | `True` / `10` | [se_solver.py](./core_shared/se_solver.py) |
| **Stage 2** | `min_cluster_size` / `min_pocket_size` / `min_sem_sep` | `8` / `3` / `0.12` | [star_meso.py](./trackA_sug_sed/star_meso.py) |
| **Stage 2** | `max_seam_calls` / `max_seam_budget_frac` / `use_seam_gate` | `3` / `0.35` / `False` | [star_meso.py](./trackA_sug_sed/star_meso.py) |
| **Stage 3** | `tau_auto` / `auto_calibrate_tau` / `margin` / `cap` | `0.85` / `True` / `0.28` / `0.90`（记录 `stats["eff_tau_auto"]`） | [star_meso.py](./trackA_sug_sed/star_meso.py) |
| **Stage 3** | `guard_sim` / `min_cl_votes` | `0.40` / `1` | [star_meso.py](./trackA_sug_sed/star_meso.py) |
| **Stage 4** | `tau_lo` / `tier2_lo` / `k_nb` / `max_star_calls` | `0.55` / `0.45` / `4` / `24` | [star_meso.py](./trackA_sug_sed/star_meso.py) |
| **块内预算** | `tokens_per_block` / `allow_carryover` / `EST_CHARS_PER_TOKEN` | `10000` / `False` / `3.4` | [token_ledger.py](./core_shared/token_ledger.py) · [llm_client.py](./trackA_sug_sed/llm_client.py) |
| **Stage 5** | `window_blocks` / `tau_lo` / `tau_hi` / `tau_fb` | `14` / `0.65` / `0.90` / `0.90` | [cross_day.py](./trackA_sug_sed/cross_day.py) |
| **Stage 5** | `min_cross_size` / `top_k_cand` / `same_block_guard_sim` | `3` / `3` / `0.40` | [cross_day.py](./trackA_sug_sed/cross_day.py) |
| **跨天预算** | `budget_per_block` / `allow_carryover` / `max_carryover_mult` | `6000` / `True` / `1.5` | [cross_day.py](./trackA_sug_sed/cross_day.py) · [token_ledger.py](./core_shared/token_ledger.py) |

* **权威源码与测试覆盖**：上述五阶段完整主流程统一维护于 [core_shared/](./core_shared/data_loader.py) 与 [trackA_sug_sed/](./trackA_sug_sed/star_meso.py)，并由 [tests/test_sug_sed.py](./tests/test_sug_sed.py) 提供全链路单元测试覆盖。

---

## 8. 待办事项（TODO · 暂缓处理）

1. **实验产物刷新与协议重冻结**：
   * `Event2012` 与 `Event2018` Dev 7 基线公平验证均已完成（`python trackA_sug_sed/run_dev_gate1.py --stage baselines7 --datasets Event2012,Event2018`，产出 [event2012_v21_7baselines.md](./results/dev_gate1/event2012_v21_7baselines.md) 与 [event2018_v21_7baselines.md](./results/dev_gate1/event2018_v21_7baselines.md)）；
   * 后续需重跑受阶段顺序与缓存修复影响的 Gate 1（Dev 全阶段）与 Gate 2（Held-Out）全量阶段（包括 `stage_0tok`、`stage_qwen3_main`、`stage_qwen3_abl`、`stage_qwen3_crossday`、`stage_budget_anon` 与 `analyze_gate2_mechanisms.py`，以及可选 `qwen25_full`）；
   * 当前 [protocol_v2.json](./trackA_sug_sed/protocol_v2.json) 仍为 `SUG-SED-v2.0-frozen` 快照（与 `v2.1` 源码有 `17/20` 文件差异），待上述实验定稿后运行 `python trackA_sug_sed/freeze_protocol_v2.py` 重新生成与 `v2.1` 源码 `20/20` SHA-256 一致的冻结快照。
2. **论文文稿与图表细节完善**：
   * 基于刷新后的 `results/heldout_gate2/` 实证数据更新 IP&M 论文中的伪代码、方法章节公式/命名（去除“贝叶斯效用/Bayes Band”表述）、实验表格、统计检验与矢量图表。
