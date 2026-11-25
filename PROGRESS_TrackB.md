# Track B 工作最新进度与核心结论归档（`Evid-MSED`）

> **定位与稿件状态**：竞争事件场景下的条件视觉证据与冲突门控多模态社交事件检测（ACM `sigconf` 匿名完整手稿已成稿）。  
> **最新状态**：**Phase 1（Gate B）+ Phase 2（`NED Validation → Holdout` Clean & Stress-30% + 24 保留事件分层归因）+ Phase 3（完整 LaTeX 论文成稿与静态校验）全部完成**。

---

## 一、当前最新实证与稿件进度快照

1. **协议冻结与零泄漏保障**：
   * 仅基于 `development`（8 事件 / `2,665` 帖）完成 Gate B 诊断与 `validation`（8 事件 / `2,746` 帖）校准后，固化 [protocol_trackB_v1.json](./trackB_evid_msed/protocol_trackB_v1.json)（锁定 9 个核心文件 SHA-256）；
   * 全部划分采用相对时间 `rel_ts = ts - min_ts(event)` 并发对齐（`time_decay_per_day = 0.0`），彻底剔除时间捷径并向编码器/VLM 屏蔽事件名与原始日期；全库有效图像零失败离线解码（`13,878` 张）。
2. **Phase 1 · Gate B 成对因果诊断（`development`，`Qwen/Qwen3-VL-8B-Instruct`）**：
   * 实时 VLM 三条件审计 `360/360 OK`（`0` 缓存伪造、`0` 错），实证盲信图像在原图条件下带来挽救但也引发误导，而在同期竞争错配图下 VLM 准确率从 `75.00%` 跌至 `11.67%`（`77` 次错配翻转；冻结 CLIP `n=1,527` 亦出现 `96` 挽救 vs `27` 原图误导 / `158` 错配翻转）；
   * `Visual-IDF MV-2DSE` 无监督聚类较纯文本提升 `Event Macro-B³ +0.0739`（`0.5234 -> 0.5973`），`vlm_self_gate` 将原图 Acc 推高至 **`85.00%`**（`0` 原图误导，错配翻转由 `77` 降至 `5`）。
3. **Phase 2 · `NED Validation → Holdout`（24 保留事件 / `11,955` 帖 / `9,311` 图）**：
   * **Stage 1（`Visual-IDF MV-2DSE` + 修正版 §3.3 软插补）**：
     * `Holdout Clean` 较纯文本 `text_only_2dse` 提升 **`AMI +3.84 pp`（`0.5466 -> 0.5850`）**、**`Macro-B³ +6.38 pp`（`0.3485 -> 0.4124`）**、**`Micro-B³ +8.02 pp`（`0.3830 -> 0.4632`）**；
     * 修正版缺失模态软插补在全部 5 组测试中 `5/5` 提升 `Macro-B³`（`+0.68` 至 `+3.27 pp`），并在 `Stress-30%`（`2,793` 错配图）下守住 `+2.88 pp` 净增益，彻底避免 `naive_mv_2dse` 的崩盘（Val `-13.35 pp` / Holdout `-8.11 pp` 跌穿纯文本）。
   * **Stage 2（`Qwen/Qwen3-VL-8B-Instruct` 冲突门控缝隙路由）**：
     * `Holdout Clean` `360/360 OK`（`277,557` tokens）：`evid_self_gate_inline @ K=60` 达 **`AMI 0.5877 / Macro-B³ 0.4136 / Micro-B³ 0.4988`**，单次调用净修复未查询帖 **`24.63` 帖**（对比盲信 `naive_vlm` `12.77`）；`K=180` 拒绝对冲 `27.22%` 冲突边，保持 **`4.82–4.99 Fix/Q`（`5.6×` 于盲信多模态 `0.86`）**。
     * `Holdout Stress-30%`（`119/120 OK`，`91,540` tokens）：`evid_self_gate_inline` 取得最高 **`AMI 0.5605 / Macro-B³ 0.3778 / ARI 0.4174`**，合并精准率稳守 **`100% (1 TP / 0 FP)`**；`Validation Stress-30%` 下 `clean_fallback` 将被行内锚定污染的分数恢复至 `0.5678`（`18.85 Fix/Q`）。
   * **24 事件分层诊断**：`Head (>=600 帖)` B³ 从 `0.3703` 跃升至 `0.4887–0.5008`（`+11.84..+13.05 pp`）；精确定位全局 `Macro-B³` 饱和系 `2` 例跨尺寸 `Small-vs-Head` 不可逆硬并查集误合并所致。
4. **Phase 3 · LaTeX 论文成稿**：
   * 完整匿名 ACM `sigconf` 论文 [paper_trackB/main.tex](./paper_trackB/main.tex) 及全部图表已生成并通过 `verify_latex.py` 静态与数值一致性校验（`0` 错误）。

---

## 二、旧工作精简备忘（初版方案关键修订记录）

* **剔除时间捷径**：早期 `NED` 全量流上仅按 `72h` 切簇即达 `B³ F1 = 0.9900` 会掩盖视觉真实作用，现行流程统一采用相对时间并发窗口对齐。
* **证伪 `cntd_raw`**：原方案按邻居文本分散度直接惩罚视觉边的 `cntd_raw` 在 Gate B 与 Holdout 中均被证实重创正常多角度事件挽救（CLIP 挽救 `96 -> 59`、VLM `17 -> 7`），已由 `Visual-IDF` 与冲突门控取代。
* **理论与数学公式修正**：撤下无目标变量条件下不严格的 PID 分解主张；移除原 §3.4 中使无图帖软补全失效的 $\delta_i\delta_j$ 乘积矛盾。

---

## 三、下一步工作与索引

* **待办**：
  1. 最终提交前 `pdflatex` / `bibtex` PDF 编译检查与投稿格式微调；
  2. 后续工程可扩展方向：在大小悬殊簇对合并处以柔性图边加权聚集替代硬并查集合并，消除尾部误合并饱和。
* **核心文件索引**：
  * 方案草案：[Proposal_TrackB_Evid-MSED.md](./Proposal_TrackB_Evid-MSED.md) · 冻结协议：[protocol_trackB_v1.json](./trackB_evid_msed/protocol_trackB_v1.json)
  * 主体代码：[trackB_evid_msed/](./trackB_evid_msed/gateB_paired_audit.py)
  * 实证报告：[Gate B 执行裁决](./results/trackB_gateB/GateB_Qwen3_VL_8B_执行裁决.md) · [Phase 2 完整报告](./results/trackB_phase2/Phase2_NED_Val_Holdout_EvidMSED_Report.md)
  * 论文手稿与校验：[paper_trackB/main.tex](./paper_trackB/main.tex) · [paper_trackB/verify_latex.py](./paper_trackB/verify_latex.py)
