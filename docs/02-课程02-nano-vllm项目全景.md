# 课程 02：nano-vllm 项目全景

> 用一张「四层架构」把仓库读薄：从 `LLM` 入口到 `Scheduler`/`BlockManager`，再到 `ModelRunner` 与 `layers`，最后对照 vLLM 说清裁剪边界与面试话术。

## 本课目标

1. **熟记** nano-vllm 的 **目录结构** 与每个顶层目录的职责。
2. 能画出 **接口层 → 引擎层 → 调度层 → 执行层** 的调用关系。
3. 用一段话描述 **用户输入到 token 输出** 的 **数据流**（可与面试官白板同步）。
4. 回答 **nano-vllm 与 vLLM** 在 **代码量、功能覆盖** 上的对比，不说空话。

## 核心概念

### 完整目录结构（与仓库一致）

```
nanovllm/
├── __init__.py          # 导出 LLM, SamplingParams
├── config.py            # Config dataclass
├── llm.py               # LLM(LLMEngine) 空子类或薄封装
├── sampling_params.py   # SamplingParams dataclass
├── engine/
│   ├── llm_engine.py    # 引擎主循环
│   ├── scheduler.py     # Prefill/Decode 调度
│   ├── sequence.py      # 序列状态管理
│   ├── block_manager.py # KV Cache 块管理
│   └── model_runner.py  # GPU 模型执行
├── layers/
│   ├── attention.py     # FlashAttention + Triton KV 存储
│   ├── linear.py        # 列/行/QKV 并行 Linear
│   ├── embed_head.py    # 词表并行 Embedding + LMHead
│   ├── layernorm.py     # RMSNorm
│   ├── activation.py    # SiluAndMul (SwiGLU)
│   ├── rotary_embedding.py
│   └── sampler.py       # 温度采样
├── models/
│   └── qwen3.py         # Qwen3 模型实现
└── utils/
    ├── context.py       # 全局推理上下文
    └── loader.py        # safetensors 权重加载
```

### 四层架构分析

**第一层：接口层（`llm.py` + `sampling_params.py` + `config.py`）**

- 用户只接触 **`LLM`** 与 **`SamplingParams`**。
- **`Config`** 聚合 **显存、batch、TP、KV 块** 等全局约束，在引擎初始化时一次性生效。

**第二层：引擎层（`engine/llm_engine.py`）**

- **主循环**：从 `Scheduler` 取可运行批次，调用 **`ModelRunner`**，再 **采样**、更新 **序列状态**。
- 是 **「业务逻辑心脏」**：连接调度与模型执行。

**第三层：调度层（`scheduler.py` + `sequence.py` + `block_manager.py`）**

- **`Scheduler`**：决定本轮是 **prefill** 还是 **decode**，拼 **batch**，受 `max_num_batched_tokens` 等约束。
- **`Sequence`**：跟踪每条请求 **token、步数、是否结束**。
- **`BlockManager`**：为 KV 分配 **物理块**（Paged 思想），与 `Attention` 里的 `block_table` 呼应。

**第四层：执行层（`engine/model_runner.py` + `models/qwen3.py` + `layers/*`）**

- **`ModelRunner`**：准备 **输入张量**、设置 **全局 `context`**（`is_prefill`、`cu_seqlens_q` 等），调用 **`Qwen3`**。
- **`layers`**：算子级实现；**`attention.py`** 同时承担 **写 KV** 与 **FlashAttention 前向**。

### 各模块职责速查表

| 模块 | 一句话职责 |
|------|------------|
| `config.py` | 超参与 HF 配置对齐，约束 max length 与 batch |
| `llm.py` | 对外 `generate`，内部转 `LLMEngine` |
| `llm_engine.py` | 推理循环：调度 → 前向 → 采样 → 更新 |
| `scheduler.py` | 选序列、组 batch、prefill/decode 策略 |
| `sequence.py` | 单请求状态机 |
| `block_manager.py` | KV 块分配与回收 |
| `model_runner.py` | CUDA 侧执行与 context 注入 |
| `attention.py` | Triton 存 KV + FlashAttention 两路径 |
| `embed_head.py` | 词表并行嵌入与 LMHead |
| `qwen3.py` | 整体 Transformer 堆叠 |

## 源码解析（带完整源码和逐行注释）

下面用 **极简示意代码** 说明四层如何串起来（真实实现以仓库为准，此处突出 **调用方向**）：

```python
# 概念串联：接口 -> 引擎 -> 调度 -> 执行（伪代码）

class LLM:
    def __init__(self, model_path, **kwargs):
        self.engine = LLMEngine(Config(model=model_path, **kwargs))

    def generate(self, prompts, sampling_params):
        return self.engine.generate(prompts, sampling_params)


class LLMEngine:
    def generate(self, prompts, sampling_params):
        # 1) tokenize（可能在 engine 外或内，示例常在外部）
        # 2) 加入 scheduler 管理的序列
        while not all_finished():
            batch = self.scheduler.schedule()   # 调度层：谁跑 prefill / decode
            logits = self.model_runner.run(batch)  # 执行层：前向
            self.sample_and_update(logits, sampling_params)
        return decoded_texts
```

逐行说明：

1. **`LLM`**：构造 **`LLMEngine`**，注入 **`Config`**。
2. **`generate`**：引擎负责 **直到所有序列结束** 的循环。
3. **`scheduler.schedule()`**：返回 **本轮可执行子集** 及 **阶段标记**（prefill/decode）。
4. **`model_runner.run`**：设置 **`get_context()`** 所需字段，跑 **`Qwen3`**。

## 图解（用文字/ASCII 描述）

**数据流（用户输入 → 输出）**：

```
用户字符串
    |
    v
Tokenizer（HF AutoTokenizer，通常在业务脚本）
    |
    v
token id 序列  -->  LLM.generate
    |
    v
LLMEngine: 注册 Sequence，分配 KV blocks（BlockManager）
    |
    v
Scheduler: 组装本轮 prefill 或 decode 的 batch
    |
    v
ModelRunner: 填充 context（is_prefill, cu_seqlens_q, block_tables, ...）
    |
    v
Qwen3 forward -> Attention（写 KV + flash_attn_*）
    |
    v
ParallelLMHead -> logits -> Sampler（温度）
    |
    v
下一 token 写回 Sequence；若未结束则进入下一轮 decode
    |
    v
拼接输出 token -> 字符串
```

**四层折叠图**：

```
 [ LLM.generate ]          <-- 接口层
       |
 [ LLMEngine 循环 ]        <-- 引擎层
       |
 [ Scheduler + Seq + BM ]  <-- 调度层
       |
 [ ModelRunner + Qwen3 ]   <-- 执行层
```

## 与 vLLM 对比（代码量、功能覆盖）

| 维度 | nano-vllm | vLLM |
|------|-----------|------|
| **代码量** | ~1200 行量级，单仓库可读 | 数万行 + 多模块 |
| **核心思想** | Paged KV、连续批处理、FlashAttention、TP | 同类 + 更多生产特性 |
| **功能覆盖** | Qwen3 路径 + 教学向最小闭环 | 多模型、量化、API、分布式等 |
| **学习价值** | 快速建立 **可讲解的源码地图** | 对标线上，但阅读成本高 |

面试表述建议：**nano-vllm 不是用来替代 vLLM，而是把 vLLM 的主干算法与工程路径压缩到可读范围**。

## 面试考点

- **四层架构** 能否在 1 分钟内画完。
- **`Scheduler` 与 `ModelRunner` 的分工**：谁决定「跑谁」，谁决定「怎么张量化执行」。
- **`BlockManager` 与 `attention` 中 `block_table` 的关系**。
- **与 vLLM 对比时强调「裁剪」而非「落后」**。

## 常见面试题

1. **如果让你加一个功能，你会改哪个文件？**  
   答：先归类——调度策略改 `scheduler`；新算子改 `layers`；新模型结构改 `models`；入口参数改 `config`/`llm`。

2. **KV 存在哪里？谁分配？**  
   答：物理张量在 **`Attention` 的 `k_cache`/`v_cache`**（或 runner 初始化绑定）；**逻辑块映射**在 **`BlockManager`**，通过 **`block_table`** 传给 FlashAttention。

3. **连续批处理体现在哪一层？**  
   答：主要在 **`Scheduler` + `LLMEngine` 循环**：动态增删序列、每步可能不同的 batch 组成。

## 小结

nano-vllm 用 **清晰的四层划分** 把推理引擎拆开：**接口** 极简、**引擎** 管循环、**调度** 管 batch 与 KV、**执行** 管 GPU 张量与算子。对照 **vLLM**，它是 **小代码量、主干全覆盖** 的学习样本。

**读后任务（建议手写）**：

1. 画出 `LLMEngine` 一次循环里调用的 **3～5 个关键函数名**（可空着，学到 18 课再填）。
2. 用一句话解释：**为什么 `BlockManager` 属于调度层而不是 layers**（提示：管资源分配与生命周期，不直接做矩阵乘）。

## 下一课预告

下一课 **《03-课程03-配置与入口》** 将逐字段拆解 **`Config` / `SamplingParams`**，并结合 **`example.py`** 走通从本地模型路径到 **`llm.generate`** 的第一次「可运行认知」。
