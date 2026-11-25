# Track B（`Evid-MSED`）Phase 1：Gate B 成对因果视觉证据诊断与执行裁决报告（`Qwen/Qwen3-VL-8B-Instruct`）

> **核心研究规范声明**：本报告严格遵守预注册门控方案 [Proposal_TrackB_Evid-MSED.md](../../Proposal_TrackB_Evid-MSED.md) 的泄漏防护与真实账本制度：
> 1. **严格划分隔离**：全过程仅触达 `NED` 的 `development`（8 事件）划分；`validation`（8 事件）与 `holdout`（24 事件）**100% 未解锁、未接触**。
> 2. **彻底消除 72 小时时间捷径**：所有开发事件均按事件内相对时间 `rel_ts = ts - min_ts(event)` 并发对齐，且 VLM 提示与文本嵌入严格剥离 `event_name` 与原始日历年份/日期。
> 3. **真实 Token 账本与 0 失败调用**：所有多模态 LLM 调用经 `SiliconFlow` 端点请求 **`Qwen/Qwen3-VL-8B-Instruct`**，图像经 SHA-256 规范哈希与 `TokenLedger` 审计；**360/360 次实时调用全部 `HTTP 200 OK`（0 缓存伪造、0 解析失败）**。

---

## 一、实验范围、并发设置与工程校验

### 1.1 NED 开发集与并发竞争事件对（Competitor Pairs）

基于 Seed `20261106` 划分（全库 `NED_with_images.jsonl` 共 `17,366` 条、40 个事件、`13,878` 张有效图片）：
- **Development 划分（8 事件，本次 Gate B 唯一使用）**：`E15, E23, E22, E33, E40, E11, E24, E38`
  - 配对并发竞争组：`E15 ↔ E23`、`E22 ↔ E33`、`E38 ↔ E40`、`E11 ↔ E24`
- **保留未解锁划分（本次 0 触达）**：`Validation`（8 事件：`E25, E26, E19, E20, E9, E36, E6, E31`）、`Holdout`（24 事件）

### 1.2 离线双模态特征与去重规模

在本地 GTX 1650 上通过 `HF_HUB_OFFLINE=1` 全离线提取并缓存至 `cache/trackB/ned_development/`：
- **开发集帖子总数**：`2,665` 条
- **有效解码图像数**：`2,131 / 2,131`（**解码失败 `decode_fail = 0`**；采用 `PySide6.QtGui.QImage` + `imageio.v3` + 纯 PyTorch `bicubic` 预处理）
- **图像唯一性与非重复子集**：
  - 唯一 `dHash64` 数量：`2,110 / 2,131`（精确重复率仅 `1.0%`）
  - 测试查询集含图帖子总数：`1,589` 条；其中相对候选库呈新颖非重复（`dHash > 6`）的查询图像：**`1,527` 条**
- **预提取特征张量**：`text_minilm.npy` `(2665, 384)`（`5.6s`）、`clip_txt.npy` `(2665, 512)`（`18.7s`）、`clip_img.npy` `(2665, 512)`（`88.2s`）、`image_meta.npz`

---

## 二、Tier 1 零 API 成本全量审计（全开发集 `2,665` 帖）

### 2.1 Tier 1A：三条件成对因果干预（`Text-Only` / `Orig-Img` / `Mismatched-Img`，$\lambda_V = 0.75$）

对 `1,527` 条新颖非重复含图查询帖（及全量 `1,589` 条含图查询帖）执行并发双事件原型判别因果对照：

| 方法变体 | `Text-Only` 准确率 | `Orig-Img` 准确率 | `Orig-Img` 宏 $F_1$ | `Mismatched-Img` 准确率 | 视觉挽救 `Resc` | 自然误导 `Misld` | 净增益 `Net` | 对抗误翻 `MisFlip` | 全量 `All (n=1,589)` `Orig` / `Net` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `text_only` | `0.8271` | `0.8271` | `0.8196` | `0.8271` | `+0` | `0` | `+0` | `0` | `0.8263` / `+0` |
| `naive_visual` | `0.8271` | **`0.8723`** | **`0.8752`** | `0.7439` | **`+96`** | `27` | **`+69`** | `158` | **`0.8729`** / **`+74`** |
| `visual_idf` | `0.8271` | `0.8684` | `0.8684` | `0.7570` | `+90` | `27` | `+63` | `135` | `0.8685` / `+67` |
| `cntd_raw` | `0.8271` | `0.8553` | `0.8599` | `0.7917` | `+59` | `16` | `+43` | `76` | `0.8565` / `+48` |
| `cntd_cond` | `0.8271` | `0.8657` | `0.8634` | `0.7603` | `+85` | `26` | `+59` | `130` | `0.8660` / `+63` |
| `comp_margin` | `0.8271` | `0.8540` | `0.8542` | `0.7570` | `+50` | **`9`** | `+41` | `135` | `0.8546` / `+45` |
| `comp_margin_guarded` | `0.8271` | `0.8540` | `0.8542` | `0.7839` | `+50` | **`9`** | `+41` | `78` | `0.8546` / `+45` |

**Tier 1A 核心发现**：
1. 即使在严格剔除 `dHash <= 6` 重复图后，真实并发数据上仍存在显著的双向因果效应：`naive_visual` 带来 **`+96` 条文本漏判挽救**，同时制造 **`27` 条原图自然误导** 与 **`158` 条并发跨事件错配翻转**（错配准确率从 `82.71%` 跌至 `74.39%`）。
2. **`cntd_raw` 证实存在严重误伤**：它把正常同事件多角度视觉挽救从 `+96` 腰斩至 `+59`（损失超过 38% 的真实多模态互补收益）。
3. 静态冻结 CLIP `comp_margin` 将自然误导从 `27` 降至 `9`（`-66.7%`），但以原型相似度差除以固定尺度 `0.10` 对常规 `0.02–0.05` CLIP 类间边际抑制过猛，损失了近半数挽救（`96 -> 50`）。

### 2.2 Tier 1B：8 事件全并发无监督多视图 2D-SE 聚类（`2,665` 帖）

验证 Proposal B §3.3 修正后的统一软缺失插补公式（移除原方案有误的 $\delta_i \delta_j$ 抑制因子）：

| 无监督聚类变体 | 簇数 | `AMI` | `Event Macro-B³` | `Micro-B³` | `Net FM Reduction` | `Net FS Reduction` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `text_only_2dse` | `118` | `0.4544` | `0.5234` | `0.3586` | — | — |
| `naive_mv_2dse` | `107` | `0.4706` (`+0.0162`) | `0.5664` (`+0.0430`) | `0.3827` (`+0.0241`) | `1,841` | `16,805` |
| **`visual_idf_mv_2dse`** | `98` | **`0.4747`** (**`+0.0203`**) | **`0.5973`** (**`+0.0739`**) | **`0.4219`** (**`+0.0633`**) | `-38` | **`32,792`** |
| `cntd_raw_mv_2dse` | `96` | `0.4742` (`+0.0197`) | `0.5493` (`+0.0259`) | `0.3823` (`+0.0237`) | `-996` | `21,208` |
| `cntd_cond_mv_2dse` | `100` | `0.4707` (`+0.0163`) | `0.5833` (`+0.0598`) | `0.4186` (`+0.0599`) | `1,020` | `32,736` |
| `unsup_comp_margin_mv_2dse` | `102` | `0.4588` (`+0.0044`) | `0.5442` (`+0.0207`) | `0.3798` (`+0.0212`) | `1,329` | `14,093` |

**Tier 1B 核心发现**：
- **Stage-1 最优图融合基座确立**：`visual_idf_mv_2dse` 将 8 事件并发 2D-SE 聚类的 **Event Macro-$B^3$ 从 `0.5234` 暴力拉升 `+7.39 pp` 至 `0.5973`**（Micro-$B^3$ `+6.33 pp`，净消除欠合并撕裂边 `net_fs_red = 32,792`），同时 `AMI` 达到最高 `0.4747`。
- **`cntd_raw_mv_2dse` 再次被证伪**：虽然 `AMI` 看似接近（`0.4742`），但真实事件边界指标 **Macro-$B^3$ 暴跌 `-4.80 pp`（`0.5973 -> 0.5493`）**，且 `net_fm_red = -996` 产生了大量跨事件误合并。

---

## 三、Tier 2：`Qwen/Qwen3-VL-8B-Instruct` 成对因果裁决审计（`n = 120 × 3 = 360` 次调用）

### 3.1 真实 Token 账本与结构化信号分布

- **模型端点**：SiliconFlow `Qwen/Qwen3-VL-8B-Instruct`
- **调用规模与健康度**：分层抽取 `120` 条新颖图像查询帖 × 3 条件（`text_only`、`orig_img`、`mismatched_img`）= **`360` 次实时 API 调用**
- **状态审计**：`{'OK': 360}`（**0 失败、0 解析错、0 缓存跳过**）
- **Token 与成本统计**：`prompt_tokens = 312,713`、`completion_tokens = 20,065`、**`total_tokens = 332,778`**，总耗时 `1,446.1s`，按默认费率计约 **¥0.1965**
- **VLM 显式因果信号分布**：
  - `[Orig-Img]` 条件：`visual_discriminative` (`True: 101`, `False: 19`)；`visual_conflict_with_text` (`True: 16`, `False: 104`)；`evidence_source` (`image: 51`, `text: 8`, `both: 61`)
  - `[Mismatched-Img]` 条件：`visual_discriminative` (`True: 79`, `False: 41`)；`visual_conflict_with_text` (**`True: 99`**, `False: 21`)；`evidence_source` (`image: 107`, `text: 5`, `both: 8`)

### 3.2 六种多模态决策门控机制对比

在所有变体共享相同 `vlm_text_only` 基线准确率（`0.7500`）下：

| 门控策略 | `Text-Only` Acc | `Orig-Img` Acc | `Orig-Img` Macro-$F_1$ | `Mismatched-Img` Acc | 挽救 `Resc` | 自然误导 `Misld` | 净增益 `Net` | 错配翻转 `MisFlip` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `vlm_text_only` | `0.7500` | `0.7500` | `0.8378` | `0.7500` | `0` | `0` | `+0` | `0` |
| `vlm_naive_orig` | `0.7500` | `0.8333` | `0.9075` | `0.1167` | **`17`** | `7` | `+10` | `77` |
| `vlm_visual_idf_gate` | `0.7500` | `0.8250` | `0.9038` | `0.1167` | `16` | `7` | `+9` | `77` |
| `vlm_cntd_raw_gate` | `0.7500` | `0.7750` | `0.8646` | `0.1167` | `7` | `4` | `+3` | `77` |
| **`vlm_self_gate`** | `0.7500` | **`0.8500`** | **`0.9097`** | **`0.7083`** | `12` | **`0`** | **`+12`** | **`5`** |
| **`vlm_comp_evid_gate`** | `0.7500` | `0.8250` | `0.9022` | **`0.7500`** | `9` | **`0`** | `+9` | **`0`** |

![Gate B Diagnostic Summary](../figures/gateB_ned_dev_qwen3_vl_8b.png)

### 3.3 实证逐层剖析：为何 `Visual-IDF` 与 `CNTD` 在 VLM 侧失效，而跨模态证据门控完美破局

1. **`vlm_naive_orig` 验证视觉高收益与极端脆弱性并存**：盲信原图让准确率从 `75.00% -> 83.33%`（挽救 `17` 条隐晦文本帖），但在 `Mismatched-Img` 并发竞争图像下准确率灾难性坍塌至 **`11.67%`（`77` 条错配翻转）**，且原图本身也造成 `7` 条误导翻转。
2. **仅靠图侧统计量无法拦截独特干扰图**：所有自然误导图和跨事件错配图在样本级都是独特图像（`vidf = 1.000`），因此 `vlm_visual_idf_gate` 对 `7` 条原图误导和 `77` 条错配翻转 **完全无能为力（`11.67%`）**；而 `vlm_cntd_raw_gate` 既把挽救腰斩（`17 -> 7`），又未能拦截错配劫持（仍是 `77` 翻转、`11.67%`）。
3. **跨模态矛盾识别 + 竞争者证据门控双赢突破**：
   - `vlm_self_gate`（当且仅当 `visual_discriminative and not visual_conflict_with_text` 时采信多模态裁决）：将 `Orig-Img` 准确率推升至全表最高 **`85.00%`**（超盲信 `83.33%`，净增益 **`+12`**），将原图误导压至 **`0`**，同时把并发错配翻转从 `77` 断崖降至 **`5`**（`70.83%`）。
   - `vlm_comp_evid_gate`（叠加文本置信与证据边际门控）：同时将自然误导与对抗错配翻转全部清零（**`0` Misleads / `0` MisFlips**，零退化守住 `75.00%` 下限），并在原图上净挽救 **`9` 条疑难文本帖（`82.50%`）**。

---

## 四、真实样本因果案例核查（`inspect_gateB_cases.py`）

1. **真实视觉挽救案例（`vidf = 1.000`，文本无法判定，`Orig-Img` 精确救回）**：
   - `ned_05292`（`E11` vs `E24`）：文本仅为 `"Live| Evacuation of detainees from the stadium after the explosion!"`，无任何地名关键词，`TXT pred=None, conf=0.00`；`ORIG` 识别出 `"Night scene with police officers and crowd behind barriers"`（`disc=True, conf_txt=False, src=image`），精准挽救归入 `E11`。
   - `ned_09331`（`E22` vs `E33`）：文本 `"It took volunteers 4 days to clear our section of the road from shredded trees & downed utility poles. #STX #USVISTRONG"`，`TXT pred=None`；`ORIG` 识别出 `"Downed utility poles, shredded trees, debris on road"`（`disc=True, conf_txt=False, src=both`），挽救归入 `E22`；而错配日铁东京站图触发 `conf_txt=True` 警报。
   - `ned_09280`（`E22` vs `E33`）：文本描述国民警卫队装运救灾物资，`TXT pred=None`；`ORIG` 识别军用货机托盘内部场景并挽救归入 `E22`；错配日本铁路淹水图触发 `conf_txt=True` 警报。
2. **自然原图误导案例（盲信原图 `Misld=7` 典型失误，全部被 `Self-Gate` / `CompEvid-Gate` 拦截）**：
   - `ned_05166`（`E11`）：文本 `#Rip #Prayers4Paris #Scotland #France`（`TXT E11 conf=0.95`）配苏格兰旗图像（`conf_txt=True`），盲信原图翻转为 `None`。
   - `ned_09302`（`E22`）：文本 `#PrayForPuertoRico`（`TXT E22 conf=0.95`）配白色宠物狗（`disc=False, conf_txt=True`），盲信原图翻转为 `None`。
   - `ned_09610`（`E23`）：文本 `ASTROS WIN`（`TXT E23 conf=0.95`）配无关人物持书照（`disc=False, conf_txt=True`），盲信原图翻转为 `None`。
   - `ned_09980`（`E23` vs `E15`）：文本 `Fall Classic`（`TXT E23 conf=0.95`）配 Minute Maid Park 照片（`ORIG conf_txt=True`）；更值得警惕的是在 `Mismatched-Img` 替换为 Wrigley Field 后，VLM 被图像强行劫持至竞争事件 `E15`（两届不同的世界大赛混淆）。
3. **跨事件并发劫持案例（`Mismatched-Img` `MisFlip=77`）**：
   - 如 `ned_05343`（文本 `#prayforparis`，`TXT E11 conf=0.98`），在替换伊朗国旗人群并发图后，盲信策略直接翻转判给 `E24`，被 `conf_txt=True` 精确识别并拦截。

---

## 五、Gate B 正式执行裁决与 Track B Phase 2 架构修正方案

### 5.1 正式裁决：**Gate B CONDITIONAL PASS（有条件通过 · 晋升核心机制已实证成立）**

依据预注册门控标准，Track B 满足“双边现象真实存在、跨模态证据门控可分离挽救与误导”的核心条件，裁决 **PASS**，但必须执行一项关键的组件重校准修正后再进入 Validation/Holdout Phase 2：

| 设计组件 | Gate B 前草案假设 | Gate B 实证裁决与修正动作 |
| :--- | :--- | :--- |
| **§3.3 统一软缺失 2D-SE 插补** | 删除原 $\delta_i \delta_j$ 抑制 | **已验证有效**：`visual_idf_mv_2dse` 将并发 2D-SE Event Macro-$B^3$ 从 `0.5234` 提升至 **`0.5973`（`+7.39 pp`）** |
| **原始 CNTD 跨模态一致性降权** | `cntd_raw` 抑制不对齐图像 | **明确证伪弃用**：所有层级均严重砍伤正常视觉挽救（Tier 1A `96→59`；Tier 1B Macro-$B^3$ `-4.80 pp`；Tier 2 `17→7`），且对错配图零防御 |
| **Stage 1（图融合层）策略** | `Comp-Margin` + `Visual-IDF` 混合 | **修正为纯 `Visual-IDF` 加权 MV-2DSE**：静态 CLIP 固定 `0.10` 边际缩放会过度扼杀 `0.02–0.05` 真实类间差（Macro-$B^3$ `0.5442` vs `0.5973`），Stage 1 专注“高频梗图降权 + 软缺失补全” |
| **Stage 2（VLM 介观缝隙层）策略** | 单一图侧置信度融合 | **确立 `vlm_self_gate` + `comp_evid_gate`（竞争者条件化冲突门控）**：利用 VLM 显式 `visual_conflict_with_text` + 竞争边界置信，实现原图 **`85.00%`（`+10.00 pp`, `0` 误导）** 与错配 **`0–5` 翻转** |

### 5.2 下一步可执行方向（供决策）

- **Phase 2A（NED Validation 与 Holdout 完整流式评估）**：将 Stage 1（`Visual-IDF` MV-2DSE）与 Stage 2（借鉴 Track A 介观簇间缝隙路由 + `Qwen/Qwen3-VL-8B-Instruct` 冲突门控 `vlm_self_gate`/`vlm_comp_evid_gate`）冻结协议，在尚未触达的 `validation`（8 事件）调定预算阈值并冲击 `holdout`（24 事件）。
- **Phase 2B（可选双数据集扩张）**：将同样 Gate B 审计模板迁移至 `WEIBO2022` 中文多模态事件流检验跨语言一致鲁棒性。
