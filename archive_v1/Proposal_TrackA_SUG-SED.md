# Track A 论文研究方案与实证升级定稿总纲（`SUG-SED` · v4 IP&M 版）

> **目标期刊**：*Information Processing & Management* (IP&M，CCF-B / 中科院一区 Top)  
> **英文拟题**：*From Single-Post Fallacy to Mesoscopic Seam Adjudication: Budgeted Open-Source LLM Event Clustering for Streaming Social Media*  
> **中文拟题**：从单帖纠错陷阱到介观缝隙裁决：面向流式社交事件检测的预算受限轻量开源 LLM 方法（`SUG-SED`）  
> **当前状态**：**Stage 1 零 Token 口径审计与强基线（D1–D6）+ Stage 2A `Qwen/Qwen3-8B` 全量严格因果实验（38 组×2 数据集 + `SUG-CrossLLM-v2`）+ Stage 2B 全部 27 个保留块（`65,856` 帖）实体/标签/年份匿名化防记忆压力测试已全部完成（零密钥泄露 · 11/11 单测通过）**；当前仅待补齐 **跨模型 27 块泛化验证**，随后统一撰写 IP&M 正式 LaTeX 论文与矢量图表（旧版 ACM 手稿见 [paper/README_COMPILE.md](./paper/README_COMPILE.md)，尚未改稿）。

---

## 1. 问题重新定义、审计澄清与已验证的科学贡献

### 1.1 我们真正解决的核心 SED 问题与诚实口径披露
流式社交事件检测（Streaming SED）中单纯降低 LLM API 调用次数本身不是独立科学贡献。**`SUG-SED` 建立在经过双语多轮反例与大规模流式因果实验双重检验的核心发现之上**：

> **在严格受限且零未来前瞻的因果真实 Token 预算（<1% 帖子触发 API 调用；考虑 Prompt 中每个查询展示最多 6 个 medoid 帖后，实际曝光帖占比仅 `1.79%–3.87%`）下，单帖级或点对边级查询几乎无法改善全局事件划分；唯有转向*介观结构缝隙（Mesoscopic Topology Seams）*并结合*高纯度跨块模糊带历史裁决*，才能依靠实体共指与谓词语义结构（而非历史百科记忆）以单次 8B 轻量 LLM 裁决净改善数十条完全未进入 Prompt 的帖子（$U_t$，已彻底剔除全部展示 medoid 帖）。**

来自本仓库 Stage 1 与 Stage 2A/2B 的四重直接审计证据：
1. **单帖/点对纠错的天花板**：在严格消除展示污染后的 $U_t$（`all_shown` 掩码）上，同等 10k/blk 预算下随机采样（`Random` 3-seed 均值）在 `Event2012 / Event2018` 的 $\Delta\text{AMI}$ 仅 **`-0.0004 / -0.0002`**，MMR 多样性采样（`Rep-Div`）`UnqNet/blk` 为 **`-51.6 / -79.1`**，边级主动作答 `Cequel-Edge (Repair)` 仅 **`+21.1 / -73.9`**；即便赋予簇级传递闭包（`Cequel-Transitive`），虽然微 $B^3$ 上升，却因链式误合并把宏观事件 $B^3$ 拉低至初始 2D-SE 以下（`0.6411 < 0.6567` 英，`0.6708 < 0.6790` 法）。
2. **错误结构非对称与零边界熵盲区（Stage 1 D4）**：在时序块初期二维结构熵（2D-SE）聚类中，`Event2012` 保留集（16 块，`N=32,695`）假分裂/假合并比（FS/FM）高达 **`44.64×`**（Dev 块 `11.60×`），`Event2018`（11 块，`N=33,161`）FS/FM 为 **`11.59×`**（Dev 块 `14.89×`）；同时 `36.2%`（英）/ `16.2%`（法）的真实事件跨簇边界熵 $H_{\text{bnd}}=0$，而混杂簇（`|C|>=10, pur<0.90` 占比 `21.1%` 英 / `26.5%` 法）的少数派事件中位占比仅为 **`0.231`（英）/ `0.233`（法）**（过半落入 `[0.10, 0.25]` 区间），导致经典均衡谱二分（Fiedler）误撕裂主事件。
3. **跨日原型链接的双重校准发现（Stage 1 D2 → Stage 2A S2-3）**：后续块帖子中有 **`55.67%`（英）/ `62.05%`（法）** 复现历史事件。Stage 1 D2 全阈值扫描客观揭示：初版 `SUG-CrossLLM-v1`（`[0.55, 0.80)`, `tau_fb=0.72`）在双语上均被零 LLM 原型阈值 `ProtoOnly(0.80)` 反超（原因在于 `0.72` 回退偏低、低区间真匹配率低而漏掉了真匹配率达 `33.3%–34.2%` 的 `[0.80, 0.85)` 模糊带，且旧版提取提示召回过低 `TP=1` 英 / `0` 法）。据此升级的 **`SUG-CrossLLM-v2`（`[0.65, 0.85)`, `tau_fb=0.85`, 因果 `6k/blk carryover`）** 在 Stage 2A 中于双语全指标反超最优 `ProtoOnly(0.85)`。
4. **Prompt 曝光率透明核算（`AUD-01` 修复）**：[topology_surgery.py](./trackA_sug_sed/topology_surgery.py) 现默认采用 `mask_mode="all_shown"`，严格把全部出现在 LLM Prompt 中的中心 medoid 节点从未查询集 $U_t$ 中排除，同时平行报告 API 调用率（`<1%`）、主代表占比及全展示帖占比（`1.79%–3.87%`），消除任何口径争议。

### 1.2 针对审稿人核心质疑的精确回答（v4 实证对齐）

| 对比工作 / 审稿质疑 | 既有机制 | 本文经 Stage 1 + 2A + 2B 实测后的精确实证回答 |
| :--- | :--- | :--- |
| **RagSEDE** (`arXiv:2601.12035`) | 代表性+多样性（MMR）单帖采样 | 在同块、同 2D-SE、同 `Qwen3-8B`、严格零前瞻 10k/blk 预算及 `all_shown` $U_t$ 下，`Rep-Div` 的 `Unq ΔMic-B³` 仅 `-0.0001`，`SUG-Meso (Action-First)` 稳定获得 **`+0.0961`（英）/ `+0.1453`（法）** 未查询微 $B^3$ 与 **`+608.3 / +1220.3`** 未查询净改善帖/块。 |
| **Cequel** (`arXiv:2504.15640`) | 固定预算主动选择文本对做 Must/Cannot-Link | `Cequel-Edge` 与升级后的簇级传递闭包 **`Cequel-Transitive`** 均在双语上被 `SUG-Meso` 经 Holm-Bonferroni 校正显著击败（AMI/Mac-B³ 均 $p_{\text{Holm}} \le 0.04$；且传递闭包会因误连崩溃将宏 $B^3$ 压低至初始 `Init 2D-SE` 之下）。 |
| **零 LLM 启发式与经典聚类** | `Louvain`、`Agglom-Cosine`、`Dev-Tuned 2D-SE`、`Meso-Heur 0-Tok` | Stage 1 D3 表明：`Louvain`（宏 $B^3$ 仅 `0.5471/0.5361`）与 `Agglom-Cosine`（FR AMI 仅 `0.6990`）严重失衡；即便是共享同样介观候选结构、直接用余弦阈值代替 LLM 裁决的 `Meso-Heur 0-Tok`，在 Stage 2A 下亦显著弱于 `SUG-Meso`（EN 宏 $B^3$ `+2.15 pp, H=0.015`，FR AMI `+2.00 pp`, 宏 $B^3$ `+1.72 pp`）。 |
| **随机介观基线 (`Random-Meso`)** | 同等 65/35 介观操作但**随机排序**选点 | Stage 2A 3-seed `Random-Meso (Action-First)` 仅达 AMI `0.8771`（EN）/ `0.7740`（FR），`SUG-Meso` 凭借 Eq. (5)/(6) 排序对其保留 **`+2.06 pp`（EN, $p_{\text{Holm}}=0.0258$）** 与 **`+6.09 pp` AMI / `+2.83 pp` 宏 $B^3$（FR, $p_{\text{Holm}}=0.040$）** 显著优势。 |
| **预训练实体记忆泄露质疑** | 担忧 8B 模型依靠预训练记忆作答 2012/2018 历史事件 | Stage 2B 在全部 27 块将 `99.7%–100%` 实体、100% `#hashtag` 与年份脱敏为 `[ENT_k]` 伪名后，依然保留 **`98.4%`（EN）/ `77.9%`（FR）** 的 AMI 增益且真历史实体重建率 **`0%`**，因果证实模型依赖的是局部共指与动作语义。 |

---

## 2. 核心方法体系与机制定界（`SUG-Meso` + `SUG-CrossLLM-v2`）

### 2.1 Stage 1：轻量异构消息图与持久化 2D-SE 底座（`core_shared/`）
* **嵌入、构图与分区缓存**（[graph_builder.py](./core_shared/graph_builder.py) · [se_solver.py](./core_shared/se_solver.py) · [partition_cache.py](./core_shared/partition_cache.py)）：冻结 `all-MiniLM-L6-v2`（英）/ `paraphrase-multilingual-MiniLM-L12-v2`（法），融合语义 $k$-NN（$k=15, \tau_{\text{sim}}=0.35$）与 IDF 阻尼的实体/标签/用户共现边（$\alpha=0.5$），抽取**剥离显式实体后的非实体 TF-IDF 矩阵**；全部 39 个块通过懒惰最小堆凝聚 + 局部节点迁移不动点收敛求得 2D-SE 初始划分并持久化于 `cache/partitions/*.npz`。

### 2.2 Stage 2：对称介观结构缝隙路由器与简化洞见（`trackA_sug_sed/high_leverage_router.py`）
以比例交错队列（默认 `65% MERGE_PAIR + 35% SPLIT_SEAM`）对齐任意实时因果预算边界：
1. **Eq. (5) 簇间碎片合并评分（`MERGE_PAIR`）**：
   $$U_{\text{merge}}(C_a, C_b) = \big(\max(0, \cos(\mu_a, \mu_b)) + 0.15\,\text{kw\_ov} + 0.15\,\mathbb{I}[w_{ab}>0]\big) \cdot \big(0.5 + 0.15\ln(1+\min(|C_a|,|C_b|))\ln(1+|C_a|+|C_b|)\big)$$
   每次抽取各 3 条中心 medoid 与 Top-5 关键词询问 LLM。
2. **Eq. (6) 偏态少数口袋内分割评分（`SPLIT_SEAM`）**：
   对 $|C_k| \ge 10$ 的簇在球面空间同时执行**确定性最远点球面 k-means**（deterministic farthest-point spherical k-means，$k \in [3, 6]$）与 **Fiedler 谱二分**，筛出 $|S_p| \in [4, 0.48|C_k|]$ 且语义分离 $\text{intra}(S_p) - \cos(\mu_{\text{main}}, \mu_p) \ge 0.08$ 的可疑少数口袋，按导电度阻尼、非实体内容正交度 $(1 - \phi_{\text{non-ent}})$ 与尺度对数评分排序（已完成稠密/稀疏矩阵向量化优化，单块 `Event2012 M_07` 耗时从 `7.38s` 降至 `0.41s`）。
   * **Eq. (5)/(6) 精简实证新发现（Stage 2A S2-2 / S2-6）**：对比 10 组逐项消融证明：`MERGE+SEAM` 双缺陷定位与球面切分不可或缺（缺 `SPLIT_SEAM` 致长尾跌 `5.4–7.5 pp`，缺 `MERGE_PAIR` 致 AMI 收益归零，用 Fiedler 替代球面切分致长尾跌 `2.4–5.3 pp`）；同时发现默认 `max_conductance <= 0.50` 硬阈值在真实流中过滤过严——**移除该导电度截断（`eq6_no_conductance_action`）能在 `Event2012` 与 `Event2018` 上额外分别提升 `+1.22 / +1.23 pp` 宏 $B^3$（`0.7023 / 0.7381`）与 `+2.70 / +4.34 pp` 长尾 $B^3$（`0.4872 / 0.4635`）**，且在 `extract_compare`（FR）下 $\phi_{\text{non-ent}}$ 与杠杆项显式保护 AMI 不塌缩。

### 2.3 Stage 3：严格因果预算控制、Dev 无泄露提示标定与 2D-SE 理论（`token_ledger.py` · `llm_client.py` · `topology_surgery.py`）
* **彻底消除未来序列泄露的因果桶（[token_ledger.py](./core_shared/token_ledger.py)）**：Stage 2A/2B 全面弃用依赖全局未来帖数 $N_{\text{eval}}$ 的历史分配，严格采用 **`CausalUniformBucket`（固定 `10k/blk` 与 `1.5×` 结转）** 和仅基于已观测历史滑窗均值的 **`CausalSlidingBucket`**。
* **基于候选基率（Base-Rate）的无泄露提示选择原则（Stage 2A S2-1A + S2-3）**：
  * **高先验块内精修（`MERGE_PAIR` + `SPLIT_SEAM`）**：在独立开发块 `M_02..M_05` 上标定，`action_first` 显著胜过 `extract_compare`（EN Dev AMI `0.9367 vs 0.9165`，FR `0.7549 vs 0.7222`，且大幅改善合并召回 `0.60/0.50 → 0.89/0.88`），因而在保留集释放极高净改善；同时在较弱指令遵循模型（如早期 Dev `Llama-3.1-8B`）上需警惕 `action_first` 的自回归 Yes-Bias，这构成了跨模型 27 块实验的核心观察维度。
  * **低先验跨日链接（`SUG-CrossLLM-v2`）**：跨块模糊带真匹配率显著较低，`extract_compare` 充当高精度安全阀（EN 精确率 `0.7857`/特异度 `0.9691`，FR `0.9000`/`0.9877`），与 `action_first` 形成互补。
* **Branch 1 (`Direct`) vs. Branch 2 (`Repair`) 诚实定界 & 精确 2D-SE 剪枝定理**：
  Stage 2A 在 `all_shown` $U_t$ 下复核再次确认 **Direct 全面优于或打平 Repair**（EN `0.8976/0.6901` vs `0.8976/0.6863`；FR `0.8349/0.7258` vs `0.8338/0.7226`）。同时保留已验证的**命题 1（完全剪边防误合并定理）**与 Astra 3-簇反例精确增量 **`0.164160 bit`**。

---

## 3. 完整已审计实验证据总览（Stage 1 + Stage 2A + Stage 2B）

### 3.1 Gate A 开发块双语多档帕累托曲线（v3 归档 · `M_02..M_05`）
* 报告：[Budget_Curve_Qwen3_8B_Report.md](./results/budget_curves/Budget_Curve_Qwen3_8B_Report.md) · 附图：[budget_curve_bilingual_qwen3_8b.png](./results/figures/budget_curve_bilingual_qwen3_8b.png)
* 核心结论：**`SUG-Meso (Direct)` 在 `20k` Token 即大幅超越 `Random / Rep-Div / Cequel-Edge` 在 `80k`（4× 预算）的表现**（EN `20k: +2.88 pp AMI`，FR `20k: +2.66 pp AMI` → `80k: +5.28 pp AMI`）。

### 3.2 Stage 1：零 Token 全面诊断（D1–D6，[stage1_summary.md](./results/ipm_audit/stage1_summary.md)）
1. **D1 真实展示率与无污染 $U_t$（以旧版 `sug_meso_only_direct` 为基准）**：
   * `Event2012`（16 块，`32,695` 帖）：`Calls=300 (0.92%)` / `Primary=374 (1.14%)` / **`All-Shown=1,096 (3.35%)`**，严格剔除全部展示帖后的 `Unq ΔMic-B³=+0.0701`、`UnqNet/blk=+521.9`（原口径 `+534.5`），稳超 `Rep-Div (-51.6)`、`Cequel-Edge(Rep) (+21.1)` 与 `Cequel-Transitive (+132.1，但宏 B³ 跌至 0.6411 < Init 0.6567)`。
   * `Event2018`（11 块，`33,161` 帖）：`Calls=165 (0.50%)` / `Primary=199 (0.60%)` / **`All-Shown=593 (1.79%)`**，`Unq ΔMic-B³=+0.0746`、`UnqNet/blk=+794.8`（原口径 `+803.3`），优于 `Rep-Div (-79.1)`、`Cequel-Edge(Dir) (+145.4)` 与 `Cequel-Transitive (+485.1，宏 B³ 跌至 0.6708 < Init 0.6790)`。
2. **D3 零 LLM 与 Dev 调参强基线对比**：
   * `Event2012`：`Init 2D-SE` (`0.8395 / 0.6567`)、`Dev-Tuned 2D-SE` (`0.8417 / 0.7368`)、`Louvain` (`0.8446 / 0.5471`)、`Agglom-Cosine` (`0.8562 / 0.7652`)、`Meso-Heur 0-Tok` (`0.8836 / 0.6686`)。
   * `Event2018`：`Init 2D-SE` (`0.7634 / 0.6790`)、`Dev-Tuned 2D-SE` (`0.7564 / 0.7000`)、`Louvain` (`0.8111 / 0.5361`)、`Agglom-Cosine` (`0.6990 / 0.7301`)、`Meso-Heur 0-Tok` (`0.8149 / 0.7086`)。
3. **D6 失败案例根因审计**：精确定位 `Event2018 M_16` 宏 $B^3$ 微跌（`-0.0093`）系单条 `seam_c11` 对热浪事件 `96` 过拆分所致（LOO `ΔMac=-0.0168`）；`Event2012` Dev `20k→40k` 抖动系 `M_02 seam_c14` 对冲绳事件 `347` 过拆分所致（`ΔMac=-0.0105`）。

### 3.3 Stage 2A：严格因果主协议与多维对照（`Qwen/Qwen3-8B`，[stage2a_summary.md](./results/ipm_experiments/stage2a_qwen3/stage2a_summary.md)）
*(共 18,588 次调用，`38` 组×2 数据集组内实验全部 **`errors=0`**；严格 `CausalUniform/Sliding` 零前瞻预算 + `all_shown` $U_t$)*

| 数据集与基准 | 变体（Prompt / 预算器） | 调用数 (展示%) | AMI ($\Delta$) | 宏 $B^3$ ($\Delta$) | 长尾 $B^3$ | 未查询 $\Delta$微 | UnqNet/blk | vs 主变体配对 Wilcoxon $p_{\text{Holm}}$ ($d_z$) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **`Event2012`** (Init 0.8395/0.6567) | `random_mean3` (3-seed) | ~186 (3.69%) | 0.8391 (-0.0004) | 0.6579 (+0.0012) | 0.4136 | -0.0003 | -62.2 | AMI `H=1.25e-3 (+1.78)` / Mac `H=3.39e-3 (+1.55)` |
| `Event2012` | `random_meso_action_mean3` | ~349 (4.10%) | 0.8771 (+0.0376) | 0.6860 (+0.0293) | 0.4601 | +0.0598 | +425.2 | AMI `H=2.58e-2 (+1.02)` / Mac `p=0.175` |
| `Event2012` | `heur_meso_0tok` (0 tok) | 0 (0.00%) | 0.8836 (+0.0441) | 0.6686 (+0.0119) | 0.4085 | — | — | AMI `p=4.43e-2 (+0.50)` / Mac **`H=1.50e-2 (+1.17)`** |
| `Event2012` | `sug_meso_causal_fixed_extract` | 294 (3.24%) | 0.8854 (+0.0458) | 0.6904 (+0.0337) | 0.4618 | +0.0760 | +522.1 | 高精度安全阀（Merge `0.97/0.52/0.97`） |
| **`Event2012`** | **`sug_meso_causal_fixed_action`** | 353 (3.78%) | **0.8976 (+0.0581)** | **0.6901 (+0.0334)** | 0.4602 | **+0.0961** | **+608.3** | **主锚点**（Merge `0.88/0.79/0.87`，Repair `0.8976/0.6863`） |
| `Event2012` | `sug_meso_causal_carryover_action` | 366 (3.87%) | **0.8986 (+0.0590)** | **0.6907 (+0.0340)** | 0.4612 | **+0.0970** | **+616.5** | 因果余额结转再微升 |
| `Event2012` | `sug_meso_causal_sliding_action` | 310 (3.44%) | 0.8926 (+0.0530) | 0.6880 (+0.0312) | 0.4605 | +0.0885 | +584.2 | 纯历史滑动规模桶同样稳健 |
| `Event2012` | `eq6_no_conductance_action` | 358 (4.04%) | **0.9001 (+0.0606)** | **0.7023 (+0.0456)** | **0.4872** | +0.1003 | +636.9 | 取消 `cond<=0.50` 硬截断解锁高尾增益（`p=1.46e-2` Mac） |
| **`Event2018`** (Init 0.7634/0.6790) | `random_mean3` (3-seed) | ~99 (2.06%) | 0.7632 (-0.0002) | 0.6795 (+0.0005) | 0.3512 | -0.0001 | -42.8 | AMI `H=4.00e-2 (+1.76)` / Mac `H=4.00e-2 (+2.97)` |
| `Event2018` | `random_meso_action_mean3` | ~194 (2.47%) | 0.7740 (+0.0106) | 0.6975 (+0.0185) | 0.4082 | +0.0161 | +352.8 | AMI **`H=4.00e-2 (+1.77)`** / Mac **`H=4.00e-2 (+2.14)`** |
| `Event2018` | `heur_meso_0tok` (0 tok) | 0 (0.00%) | 0.8149 (+0.0515) | 0.7086 (+0.0296) | 0.3678 | — | — | AMI `p=5.37e-2 (+0.59)` / Mac `p=4.20e-2 (+0.81)` |
| `Event2018` | `sug_meso_causal_fixed_extract` | 169 (1.86%) | 0.8100 (+0.0467) | 0.7129 (+0.0339) | 0.4276 | +0.0946 | +887.3 | 高精度安全阀（Merge `1.00/0.51/1.00`） |
| **`Event2018`** | **`sug_meso_causal_fixed_action`** | 196 (2.06%) | **0.8349 (+0.0715)** | **0.7258 (+0.0468)** | 0.4201 | **+0.1453** | **+1220.3** | **主锚点**（Merge `0.96/0.93/0.95`，Repair `0.8338/0.7226`） |
| `Event2018` | `sug_meso_causal_carryover_action` | 203 (2.12%) | **0.8354 (+0.0720)** | **0.7261 (+0.0471)** | 0.4199 | **+0.1461** | **+1240.8** | 因果结转稳健一致 |
| `Event2018` | `sug_meso_causal_sliding_action` | 162 (1.79%) | 0.8336 (+0.0702) | 0.7256 (+0.0466) | 0.4208 | +0.1416 | +1206.8 | 显著省调用同时不损性能 |
| `Event2018` | `eq6_no_conductance_action` | 198 (2.07%) | 0.8345 (+0.0711) | **0.7381 (+0.0591)** | **0.4635** | +0.1450 | +1239.2 | 取消 `cond<=0.50` 硬截断再涨长尾 `+4.34 pp`（`p=3.71e-2` Mac） |

*(注：除上表外，`stage2a_summary.md` 已完整收录 `causal_merge_only`、`causal_seam_only`、`causal_fiedler_seam`、`eq5_no_leverage`、`eq5_cosine_only`、`eq6_no_non_entity`、`eq6_raw_tfidf`，及比例 `80/20 · 50/50` 与导电度 `0.35 · 0.65` 全套双向敏感性测试；其中 `MERGE/SEAM` 比例与高导电度区间均呈平缓宽平台鲁棒性。)*

### 3.4 Stage 2A 升级版跨块纵向追踪 `SUG-CrossLLM-v2`（`[0.65, 0.85), tau_fb=0.85, 6k/blk carryover`）
在 **`heldout_action_stream`** 27 块全量串联上：
* **`Event2012`**（`32,695` 帖串联）：`no_link` (`AMI=0.8870 / MacB3=0.6634 / MicB3=0.7458`) → 最优无 LLM 基线 `proto_0.85` (`0.8989 / 0.6940 / 0.7995`) → **`sug_cross_v2_extract`**（124 次 LLM，P=`0.7857`, Spec=`0.9691`）达 **`0.8994 / 0.6970 / 0.8007`**，三项核心指标首次同时超越 `proto_0.85`（`Oracle[0.65, 0.85)`: `0.9026 / 0.7048 / 0.8071`）。
* **`Event2018`**（`33,161` 帖串联）：`no_link` (`0.7976 / 0.6001 / 0.5727`) → 最优无 LLM 基线 `proto_0.85` (`0.8416 / 0.6906 / 0.7493`) → **`sug_cross_v2_extract`**（98 次 LLM，P=`0.9000`, Spec=`0.9877`）达 **`0.8460 / 0.6925 / 0.7576`**；**`sug_cross_v2_action`**（112 次 LLM，Recall=`0.7727`）更达 **`0.8485 / 0.6868 / 0.7644`**（在 AMI 与 Mic-B³ 上持平并略超 `Oracle 0.8480 / 0.7607`）。

### 3.5 Stage 2B：27 个保留块实体/标签/年份匿名化防记忆测试（[anonymization_summary.md](./results/ipm_experiments/stage2b_anonymization/anonymization_summary.md)）
*(共 1,394 次调用，**`errors=0, all_journal_errors=0`**；严格同图、同初划分、同 `10k/blk` 因果桶)*

| 数据集 | 脱敏条件 | 实体 / 标签 / 年份 / URL 掩蔽率 | 决策一致率 | 摘要历史实体重建 | AMI ($\Delta$) | 宏 $B^3$ ($\Delta$) | UnqNet/blk | AMI / 宏 $B^3$ 增益保留率 |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Event2012` | `anon_pseudo_action` (伪名共指) | `99.89% / 100% / 100% / 100%` | **81.6%** | — | **0.8967 (+5.72 pp)** | **0.6855 (+2.88 pp)** | **+632.1** | **98.4% / 86.1%** *(vs 未脱敏 `p=0.955` 无差异)* |
| `Event2012` | `anon_blind_action` (无编号盲删) | `99.70% / 100% / 100% / 100%` | 71.7% | — | 0.8827 (+4.32 pp) | 0.6609 (+0.42 pp) | +427.9 | 74.3% / 12.6% |
| `Event2012` | `anon_pseudo_extract` | `99.93% / 100% / 100% / 100%` | 75.1% | 4.47% 词面*(0 例真历史实体)* | 0.8629 (+2.33 pp) | 0.6839 (+2.72 pp) | +318.2 | 50.9% / 80.8% |
| `Event2018` | `anon_pseudo_action` (伪名共指) | **`100% / 100% / 100% / 100%`** | **83.5%** | — | **0.8191 (+5.57 pp)** | **0.7082 (+2.92 pp)** | **+1011.9** | **77.9% / 62.3%** *(vs `Random-Meso` `H=7.81e-3`)* |
| `Event2018` | `anon_blind_action` (无编号盲删) | `99.91% / 100% / 100% / 100%` | 69.3% | — | 0.7856 (+2.23 pp) | 0.6897 (+1.07 pp) | +508.9 | 31.1% / 22.9% |
| `Event2018` | `anon_pseudo_extract` | **`100% / 100% / 100% / 100%`** | 77.3% | **0.00% (0/154)** | 0.7869 (+2.35 pp) | 0.7050 (+2.60 pp) | +602.2 | 50.4% / 76.5% |

---

## 4. 面向 IP&M 的剩余实验与投稿重构规划

### 4.1 投稿前仅剩的实验步骤
1. **跨模型 27 块全量泛化验证**（下一步）：在全部 27 个 Held-out 块运行 **`Qwen/Qwen2.5-7B-Instruct`**（硅基流动 API，`extract_compare` 与 `action_first` 双 Prompt 模式及跨日验证；已探活可用）及可选跨架构 **`@cf/meta/llama-3.1-8b-instruct-fast`**（Cloudflare，已探活可用），系统验证高基率（块内）vs 低基率（跨日）提示词选择律的跨模型/跨家族泛化性。
2. **Stage 3：IP&M 论文全文重写与矢量图输出**：在跨模型实验验收后，统一绘制高分辨率出版矢量图（`.pdf/.svg`）与汇总统计表，将 `paper/*.tex` 按 *Information Processing & Management* (Elsevier `elsarticle`) 规范全面重写（纳入 Stage 1/2A/2B 与跨模型全部新发现）。

### 4.2 IP&M 论文推荐叙事结构
1. **Introduction**：以“单帖纠错无效 + 3–45× 假分裂主导 + 零边界熵少数口袋盲区 + 跨日原型精度-召回困境”立题；强调严格零前瞻因果预算与无污染 $U_t$。
2. **Problem Formulation & Theoretical Analysis**：因果流式预算器数学定义 + 精确全图体积收缩 2D-SE 剪枝定理（命题 1 与 `0.164160 bit` 解析反例）。
3. **Methodology (`SUG-Meso` + `SUG-CrossLLM-v2`)**：对称介观路由 Eq. (5)/(6)、确定性最远点球面 k-means、高/低基率对称提示工程、Direct vs Repair 机制定界、跨日高精准模糊带追踪 `[0.65, 0.85)`。
4. **Experiments & Deep Diagnostics**：Stage 1 零 LLM 强基线对照 → Stage 2A 全量因果与 `Random-Meso` / Eq. (5)/(6) 逐项消融（含无 `cond<=0.50` 精简洞见） → `SUG-CrossLLM-v2` 跨周追踪 → **跨模型 27 块泛化** → **Stage 2B 27 块匿名化防记忆机制验证** → D6 失败案例显微归因。
