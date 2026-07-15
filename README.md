# 影子层构思 / Shadow Layer Concept

**作者 / Author：sisicat77**  
**建议引用格式 / Suggested Citation：sisicat77，《影子层构思》，2026**  
**协议 / License：MIT**

---

## 关于作者 / About the Author

本构思以 MIT 协议开源，允许自由使用、修改、分发。
This concept is open-sourced under the MIT license — free to use, modify, and distribute.

虽然 MIT 协议不强制署名，但作为方案原创者，我希望：
While the MIT license does not require attribution, as the original author I request:

- 在论文、文章、正式报告中引用本构思时，注明作者及出处
- When citing this concept in papers, articles, or formal reports, please credit the author and source
- 在项目中采纳本构思时，在致谢或文档中提及 **"sisicat77"**
- When adopting this concept in a project, please mention **"sisicat77"** in acknowledgments or documentation

这是对我持续创作的鼓励，也是开源社区互相尊重的体现。
This encourages ongoing creation and reflects the mutual respect that sustains the open-source community.

---

## 核心理念 / Core Idea

影子层是一个不修改 AI 模型权重的外部引导系统。
The Shadow Layer is an external guidance system that does not modify AI model weights.

它在模型内部的高维语义空间中，为"部分需要特殊提示"的文字设置影子点。
It places shadow points in the model's internal high-dimensional semantic space, only for text that needs special guidance.

影子点本身不存内容，只存位置。
A shadow point stores only a position — no content.

影子层与主空间分离：影子空间是独立的内存区域，与模型的 Embedding / 残差流空间一一对应，但不是同一个空间。
The shadow space is a separate memory region — it maps one-to-one to the model's embedding/residual stream space, but is not the same space.

空隙即位置：高维连续空间中，任意两个坐标之间有无限中间位置。影子向量占据这些空位，由向量差（"线"）连接。
Voids are positions: in a continuous high-dimensional space, there are infinite intermediate positions between any two coordinates. Shadow vectors occupy these voids, connected by vector differences ("lines").

无影子点处不受影响：大多数文字完全没有影子点，保持原有输出，零干扰。
Where there is no shadow point, nothing changes — output remains original, zero interference.

---

## 架构设计 / Architecture

```
+----------------------------------------------------------+
|  模型内部 / Inside the Model                                |
|                                                              |
|  主路径 / Main Path                                          |
|  输入 → Embedding → Transformer Layer 0 → Layer 1 → … → 输出 |
|                        │            │                         |
|  影子路径（并行）/ Shadow Path (parallel)                      |
|                        ▼            ▼                         |
|                   ┌──────────────────────┐                   |
|                   │  影子注入点            │                   |
|                   │  Shadow Injection    │                   |
|                   └─────────┬────────────┘                   |
|                             │                                 |
|                             ▼                                 |
|                   ┌──────────────────────┐                   |
|                   │  影子提示表            │                   |
|                   │  Shadow Prompt Table  │                   |
|                   │  {trigger → 向量对}    │                   |
|                   └──────────────────────┘                   |
+--------------------------------------------------------------+
```

影子层是模型内部的原生并行模块，存在于另一个内存区域的高维向量空间中。
The Shadow Layer is a native parallel module inside the model, living in a high-dimensional vector space in a separate memory region.

它与主路径并行运行，在每一层（或指定层）的 Transformer 输入处注入引导向量。
It runs in parallel with the main path, injecting guidance vectors at the input of each (or specified) Transformer layer.

整个过程不经过 API 接口、不经过文本 Prompt，完全是模型内部数据流。
No API calls, no text prompts — it is purely an internal data flow within the model.

---

## 组件详解 / Components

### 影子提示表 / Shadow Prompt Table

与模型的 Embedding Table 同级的内置组件，属于模型本身的配置数据，而非外部数据库。
It is a built-in component at the same level as the model's Embedding Table — configuration data, not an external database.

```
影子表条目 / Table Entry = {
  触发词 / Trigger: "遗憾" / "regret",
  硬提示向量 / Hard Prompt Vector: [0.32, -0.87, 0.14, ...],
  软提示向量 / Soft Prompt Vector: [0.05, 0.02, -0.01, ...],
  目标层号 / Target Layer: 12,
  强度 / Strength: 0.3
}
```

### 映射函数 / Mapping Function

负责在主空间的 token 表示与影子空间坐标之间建立一一对应关系。
Establishes a one-to-one correspondence between token representations in the main space and coordinates in the shadow space.

大多数位置是空的（无影子点），被标记的 token 在影子空间中有对应的影子点。
Most positions are empty (no shadow point) — only marked tokens have corresponding shadow points.

### 硬提示（向量注入）/ Hard Prompt (Vector Injection)

硬提示不是文字 Prompt，而是高维语义方向向量。
A hard prompt is not text — it is a high-dimensional semantic direction vector.

实现方式：在影子点被命中时，直接在当前层 Transformer 的残差流输入上加一个方向向量。
Implementation: when a shadow point is hit, a direction vector is added directly to the residual stream input of the current Transformer layer.

```
x_layer_n = x_layer_n + 硬提示向量 × 强度系数
x_layer_n = x_layer_n + hard_prompt_vector × strength
```

核心原理：以"语气坚定"为例——通过对比实验（加引导 / 不加引导）得出方向向量 v = β - α。
Core principle: using "firm tone" as an example — direction vector v = β - α is derived through contrastive experiments (guided vs. unguided).

在推理时将此向量注入残差流，模型就会向"语气坚定"方向偏移。
During inference, injecting this vector into the residual stream shifts the model's output toward a "firm tone".

### 软提示（浮标机制）/ Soft Prompt (Buoy Mechanism)

软提示在影子空间中作为可检索的参考标记存在，不施加力。
A soft prompt exists as a retrievable reference marker in the shadow space — it exerts no force.

类比：
Analogy:

- 硬提示："往前走三步，必须"
- Hard prompt: "Take three steps forward — you must."
- 软提示：地上画了一个圈，里面写着"这里可能有帮助"，模型可以自己走进圈子看看，也可以不理
- Soft prompt: A circle drawn on the ground that says "this might help." The model can step inside to look, or ignore it.

在工程上，软提示向量以额外的 attention key-value 形式挂载在模型层上，模型可以选择 attend 到它或忽略它。
In engineering terms, the soft prompt vector is mounted as an extra attention key-value pair — the model can choose to attend to it or ignore it.

这类似于 RAG 的思路，但发生在向量层而非文本层。
This is similar to RAG in spirit, but operates at the vector level rather than the text level.

---

## 执行流程 / Execution Flow

```
输入文字 / Input text
    │
    ├── 主路径：正常编码 → Transformer 层处理
    │   Main path: normal encoding → Transformer layers
    │
    └── 影子路径（并行）/ Shadow path (parallel):
            │
            ▼
       检查映射函数 → 当前 token 在影子空间是否有对应影子点？
       Check mapping → does this token have a shadow point?
            │
       ┌────┴────┐
       ▼         ▼
       否 / No   是 / Yes
       │          │
       │          ├── 检索影子提示表 / Look up shadow prompt table
       │          │
       │          ├── 下一层 Transformer 输入：
       │          │   残差流 + 硬提示向量（强制）
       │          │   Next layer input: residual + hard vector (forced)
       │          │
       │          └── 同层/额外层：
       │              软提示向量挂载为 attention key（可选）
       │              Soft vector as attention key (optional)
       │
       └── 什么都不做，保持原输出
           Nothing happens, output unchanged
            │
            ▼
        继续 / Continue
```

## 动态更新机制 / Dynamic Updates

影子提示表可以动态更新，分为不同权限等级：
The shadow prompt table supports dynamic updates with tiered permissions:

| 操作 / Operation | 执行者 / Who | 说明 / Notes |
|---|---|---|
| 初始建表 / Initial build | 工程师 / Engineer | 离线脚本生成 / Offline script |
| 增/删/改 / CRUD | 工程师 / Engineer | 有限访问 + 审核 / Access-controlled + review |
| AI 建议新影子点 / AI suggests | AI → 待审核区 → 工程师审核 / Pending → Engineer review | AI 检测到需要引导的模式 / AI detects patterns needing guidance |
| AI 自动更新 / Auto-update | 需合规审批 / Requires compliance approval | 根据具体法规决定 / Depends on regulations |

---

## 跨模型迁移：AI 教 AI / Cross-Model Transfer: AI Teaching AI

影子层是与特定模型绑定的配置数据。不同模型的残差流空间不同，一套影子表不能直接复用。
The Shadow Layer is bound to a specific model — different models have different residual stream spaces, so shadow tables are not directly reusable.

但可以通过知识蒸馏方式迁移：
However, transfer is possible through knowledge distillation:

1. 源模型（带影子层）在触发场景下运行，产生引导后的输出
   Source model (with shadow layer) runs on trigger scenarios, producing guided outputs
2. 同样输入在目标模型（无影子层）上运行，产生原始输出
   Same inputs run on target model (without shadow layer), producing original outputs
3. 对比两组残差流激活值差异 → 为目标模型计算新的影子方向向量
   Compare activation differences → compute new shadow direction vectors for the target model

这是让 AI 教 AI 的过程，不需要人工重新标记。
This is AI teaching AI — no manual re-labeling needed.

---

## 工程可行性说明 / Feasibility Assessment

| 构想 / Idea | 可行性 / Feasibility |
|---|---|
| 影子层是内部并行路径 / Shadow layer as internal parallel path | 成立 / Feasible |
| 影子点是高维坐标锚点 / Shadow points as coordinate anchors | 成立，连续向量空间的基本性质 / Feasible — property of continuous vector space |
| 硬提示 = 方向向量注入残差流 / Hard prompt = direction vector injection | 成立 / Feasible |
| 软提示 = 浮标可检索但不强制 / Soft prompt = retrievable buoy, not forced | 成立 / Feasible |
| 映射函数一一对应 / One-to-one mapping function | 成立 / Feasible |
| 动态更新 / Dynamic updates | 可做，权限分层即可 / Feasible with tiered permissions |
| 跨模型不可直接复用 / Not directly reusable across models | 先天约束，需重新计算或知识蒸馏 / Inherent constraint — needs recomputation or distillation |
| 多个影子点同时命中 / Multiple shadow points hit simultaneously | 需设计叠加规则 / Needs conflict resolution rules |
| 影子向量生成方式 / Shadow vector generation | 需通过对比对计算方向差 / Needs contrastive pairs to compute direction differences |
| 推理性能影响 / Inference performance impact | 影子表是稀疏查询，成本几乎可忽略 / Sparse lookup — cost is negligible |

---

## 适用场景 / Use Cases

- 客服对话：控制语气，避免冷漠或过度热情 / Customer service — control tone
- 安全风控：对敏感话题强制拒绝 / Safety — enforce refusal on sensitive topics
- 角色扮演：同一模型扮演不同人物 / Role-playing — same model, different personas
- 教育解释：对难词自动补充通俗说明 / Education — auto-explain difficult terms

---

*by sisicat77 / shadow-layer / MIT*
