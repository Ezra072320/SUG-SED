# SED（社交事件检测）双轨研究工作区总索引

> **进度分轨管理说明**：两条研究线的工作进度已独立拆分存档，各自仅保留最新实证进度与精简历史备忘：
> - **Track A（`SUG-SED v2` · 预算受限文本/图事件聚类，IP&M 方向）**：详见 [**PROGRESS_TrackA.md**](./PROGRESS_TrackA.md)
> - **Track B（`Evid-MSED` · 竞争事件条件视觉证据与多模态门控，ACM `sigconf` 方向）**：详见 [**PROGRESS_TrackB.md**](./PROGRESS_TrackB.md)

---

## 一、核心规范

1. **严禁非证据性预期表述**：全部结论以冻结协议（SHA-256 校验）、真实 Token 账本及未见保留块实测结果为准。
2. **严格的泄漏隔离与消融归因**：超参均于 Dev / Validation 阶段锁定后再解锁 Holdout；忠实披露各模块边界与失效案例。

---

## 二、工作区目录索引

```text
gemini-paper/
├── README.md                          # 工作区总索引（本文件）
├── PROGRESS_TrackA.md                 # Track A 最新独立进度文档（SUG-SED v2）
├── PROGRESS_TrackB.md                 # Track B 最新独立进度文档（Evid-MSED）
├── Proposal_TrackA_SUG-SED.md         # Track A v2 详细研究方案与实证定稿总纲
├── Proposal_TrackB_Evid-MSED.md       # Track B Evid-MSED 修订版研究方案
├── core_shared/                       # Track A 共享底层数据、构图、2D-SE、账本与指标模块
├── trackA_sug_sed/                    # Track A v2 主体、10 篇对比基线与 Gate 1/2 完整评测套件
├── trackB_evid_msed/                  # Track B Evid-MSED Gate B 与 Phase 2 评测流水线
├── paper_trackB/                      # Track B 完整匿名 ACM sigconf LaTeX 论文源码
├── results/
│   ├── dev_gate1/                     # Track A Gate 1 Dev 评测结果
│   ├── heldout_gate2/                 # Track A Gate 2 27 块留出评测与 P1 归因结果
│   ├── trackB_gateB/                  # Track B Gate B 诊断结果
│   └── trackB_phase2/                 # Track B Validation → Holdout 完整实证报告与图表
├── tests/                             # 单元测试与连通性检查脚本
└── archive_v1/                        # Track A v1 历史代码与手稿归档（仅供追溯）
```
