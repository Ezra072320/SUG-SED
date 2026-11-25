# `sug_sed_method` · `SUG-SED v2` (`Pure-Star-Meso` + `SUG-CrossStar`) 核心方法独立代码包

本目录是从主研究仓库**独立拷贝并自洽重构**的 Track A 论文主体方法处理流程代码包，仅包含论文提出的 5 阶段核心算法、流式数据/构图/结构熵基础设施、零前瞻因果预算账本、防雪崩并查集与跨天演化模块（不含 10 篇外部对比基线与 Gate 1/2 评测脚手架）。原仓库中的 [core_shared/](../core_shared/)、[trackA_sug_sed/](../trackA_sug_sed/) 及 [protocol_v2.json](../trackA_sug_sed/protocol_v2.json) 均保持原样未改动。

---

## 一、五阶段方法主体流程

1. **Stage 1 · 高纯度异构消息图与二维结构熵底座（Pure 2D-SE Base）**  
   由 [data_loader.py](./core/data_loader.py)、[embed_cache.py](./method/embed_cache.py)、[graph_builder.py](./core/graph_builder.py) 与 [se_solver.py](./core/se_solver.py)（配 [partition_cache.py](./core/partition_cache.py)）构成：在冻结句向量（`float32` 缓存一致精度）与高语义门槛 `knn_k=15, knn_sim_threshold=0.50, attr_alpha=0.5` 上构建稀疏语义 $k$-NN + 属性共现异构图（大属性桶按时间窗局部近邻连接，不丢弃头条事件桶），采用全图精确体积更新最小堆凝聚求解二维结构熵并做节点迁移精化，优先抑制初始假合并，获得高纯度介观微簇。
2. **Stage 2 · 零成本高置信合并与质心防雪崩并查集（`AutoMerge` + `GuardedUnionFind`）**  
   位于 [star_meso.py](./method/star_meso.py)：在多帖微簇域（`size >= 2`）按原型余弦降序自动合并高置信微簇（`tau_auto=0.85`）；全部 `AutoMerge` 与后续 LLM 合并操作统一受控于 `GuardedUnionFind(guard_sim=0.40, min_cl_votes=1)`，阻断传递闭包链式误合并。
3. **Stage 3 · 干净介观缝隙拆分（`Clean Seam Split` · 核心通用引擎）**  
   位于 [star_meso.py](./method/star_meso.py)（`detect_clean_seams`）：对大小 `>= 8` 的混杂簇并行执行确定性最远点球面 $k$-means（含小簇 `k=2` 均衡二分）与 Fiedler 谱二分，提取口袋大小 `>= 3`（`<= |C|//2`）、语义分离度 `sem_sep >= 0.12` 且排除高交叉相似度精准转发团伙（`len < 6, intra_pock > 0.96, cross_sim > 0.65`）的候选缝隙，按效用排序取前 `max_seam_calls=3`（且单块 Seam 支出不超过块预算 `35%`，防止低预算下饿死 Stage 4）送交 [llm_client.py](./method/llm_client.py) 平衡对比裁决。
4. **Stage 4 · 两阶段多候选星型打包裁决（`Star-Merge` · 核心通用引擎）**  
   位于 [star_meso.py](./method/star_meso.py)（`run_pure_star_meso`）：第 1 阶段聚焦模糊核心带 `[0.55, 0.85)`（目标大小 `>= 2`），第 2 阶段扩充至 `[0.45, 0.85)` 含孤立微簇；每次合并后实时重建根分量的质心、`size`、`medoids`、簇级 TF-IDF 关键词及相似度矩阵行，并将 1 个锚点根分量与至多 `k_nb=4` 个最模糊邻分量打包并发查询（结构安全上限 `max_star_calls=24`，在 `10,000` tokens/block 桶下实际通常受 Token 预算约束运行 `7–18` 次；全流程受 [token_ledger.py](./core/token_ledger.py) `CausalUniformBucket(10000, allow_carryover=False)` 与 [anonymizer.py](./method/anonymizer.py) 可选脱敏审计约束）。
5. **Stage 5 · 跨天流式原型记忆与不确定带裁决（`SUG-CrossStar`）**  
   位于 [cross_day.py](./method/cross_day.py)（`GlobalEventMemory` & `evaluate_cross_day_stream`）：在 14 块滑动窗口内主动淘汰过期事件以维持 $O(|E_{\text{active}}|\cdot d)$ 复杂度，并施加同块唯一性及日内 `CANNOT_LINK`/Seam 拆分互斥约束防止静默二次合并；执行 `tau_hi=0.90` 自动续连，对中间模糊带 `[0.65, 0.90)`（且日簇大小 `>= 3`）在因果结转预算桶（`budget=6000, max_carryover_mult=1.5`）内调用跨天星型匹配 Prompt 合并跨日演化事件（`tau_fb=0.90` 即预算耗尽时保守新建事件）。

---

## 二、目录结构与模块清单

```text
sug_sed_method/
├── README.md                  # 方法流程与运行说明（本文件）
├── __init__.py                # 顶层统一 API 导出
├── pipeline.py                # 独立端到端运行入口（synthetic / dry-run-oracle / live API）
├── core/
│   ├── __init__.py
│   ├── data_loader.py         # SocialED 时序分块加载与按天切分
│   ├── graph_builder.py       # 异构语义 k-NN + 属性共现消息图构建
│   ├── se_solver.py           # 精确二维结构熵最小堆凝聚与节点迁移精化求解器
│   ├── partition_cache.py     # Stage-1 2D-SE 初始分区持久化缓存
│   ├── token_ledger.py        # 线程安全 JSONL Token 账本与零前瞻因果预算桶
│   ├── metrics_audit.py       # AMI / Mac-B3 / Mic-B3 / Tail-B3 / FS-FM / UnqNet 审计
│   └── key_loader.py          # 本地 API 密钥安全读取器
└── method/
    ├── __init__.py
    ├── embed_cache.py         # 冻结 HuggingFace 句向量提取与分块缓存
    ├── llm_client.py          # OpenAI 兼容预算受限 LLM 客户端 + 跨线程 RPM 节流器 + Prompts
    ├── anonymizer.py          # QueryAnonymizer（pseudonym / blind）与重建泄漏审计
    ├── star_meso.py           # Pure-Star-Meso 主算法（AutoMerge / GuardedUF / Seam Split / Star-Merge）
    └── cross_day.py           # GlobalEventMemory 与 SUG-CrossStar 跨天流式链接
```

---

## 三、Dev 锁定核心超参（对应 `SUG-SED-v2.0-frozen`）

* **Stage-1 构图与 2D-SE**：`knn_k=15`, `knn_sim_threshold=0.50`, `attr_alpha=0.5`, `refine_moves=True`
* **守卫与预算阈值**：`tau_auto=0.85`, `guard_sim=0.40`, `min_cl_votes=1`, 块内预算 `CausalUniformBucket(10000, allow_carryover=False)`
* **核心引擎**：`tau_lo=0.55`, `tier2_lo=0.45`, `k_nb=4`, `max_star_calls=24`, `max_seam_calls=3`, `min_cluster_size_seam=8`, `min_pocket_size_seam=3`, `min_sem_sep=0.12`
* **Cross-Day**：`window_blocks=14`, `tau_lo=0.65`, `tau_hi=0.90`, `tau_fb=0.90`, `min_cross_size=3`, `top_k_cand=3`, `budget_per_block=6000`（`allow_carryover=True`, `max_carryover_mult=1.5`）

---

## 四、快速运行示例

```powershell
# 1. 全离线合成流 5 阶段端到端自检（0 API 消耗）
python sug_sed_method/pipeline.py --mode synthetic

# 2. 在真实 Event2012 块上运行全离线 Oracle 模拟闭环（自动复用仓库缓存，0 API 消耗）
python sug_sed_method/pipeline.py --mode dataset --dataset Event2012 --blocks 2,3 --dry-run-oracle --cross-day

# 3. 连接开源 LLM API 实测（例：SiliconFlow Qwen/Qwen3-8B，含可选伪名脱敏）
python sug_sed_method/pipeline.py --mode dataset --dataset Event2012 --blocks 2,3 --provider siliconflow --model Qwen/Qwen3-8B --cross-day --anonymize pseudonym
```
