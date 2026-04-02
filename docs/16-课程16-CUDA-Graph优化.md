# 第16课：CUDA Graph 优化

面向初学者的说明：本课从「什么是 CUDA Graph」讲到 nano-vllm 里如何录制、复用图来加速 **decode（逐 token 生成）** 阶段，并配合源码逐行解析与面试题巩固理解。

下列代码与 `nanovllm/engine/model_runner.py` 中逻辑一致，便于对照阅读。

```python
@torch.inference_mode()
def capture_cudagraph(self):
    config = self.config
    hf_config = config.hf_config
    max_bs = min(self.config.max_num_seqs, 512)
    max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hf_config.hidden_size)
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}
    self.graph_pool = None
    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()
        set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        self.graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()
    self.graph_vars = dict(input_ids=input_ids, positions=positions, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, outputs=outputs)

def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

---

## 一、概念讲解

### 1.1 什么是 CUDA Graph

在常规 PyTorch / CUDA 推理中，**每一个算子**（矩阵乘、归一化、注意力等）往往都会经历一次 **CPU 向 GPU 提交内核（kernel launch）** 的过程。你可以把「一次 launch」理解为：CPU 写好任务单，交给 GPU 驱动去排队执行。

**CUDA Graph** 是 NVIDIA 提供的一种机制：把**一整段**已经确定好的 GPU 操作序列 **预先录制（capture）** 成一张 **有向无环图（DAG）**，之后在运行时只需 **`replay()` 重放** 这张图，而无需在每一步都重复走完整的 launch 路径。

直观类比：

- **不用 Graph**：导演每场戏都口头给每个演员说一遍台词（CPU 频繁介入）。
- **用 Graph**：排练时录成一条「成片」，正式演出时直接播放（一次提交，GPU 按图执行）。

在 PyTorch 中，通常配合 `torch.cuda.CUDAGraph()` 与 `torch.cuda.graph()` 上下文管理器完成录制与重放。

### 1.2 CUDA Graph 的底层原理

CUDA Graph 由两部分组成：

1. **图定义（Graph Definition）**：一个包含多个 **节点（Node）** 和 **边（Edge）** 的有向无环图。每个节点代表一个 GPU 操作（如 kernel launch、内存拷贝），边代表依赖关系。
2. **图实例（Graph Instance）**：从图定义创建的可执行对象，可以在 GPU 上直接提交执行。

传统的 CUDA 执行模型中，CPU 端的 **驱动程序** 要逐个检查、排队并提交每个 kernel。这个过程涉及：

| 开销来源 | 说明 |
|---------|------|
| API 调用开销 | 每次 `cudaLaunchKernel` 需经过驱动层验证参数 |
| 同步与依赖检查 | 确保前置操作完成才能提交后续操作 |
| CPU-GPU 通信 | 通过 PCIe/NVLink 传递命令缓冲区 |
| Python 解释器开销 | PyTorch 的 Python dispatch、autograd 等层层封装 |

使用 CUDA Graph 后，上述开销在 **capture 阶段** 只发生一次，后续 `replay()` 只需要 **一次 `cudaGraphLaunch` 调用**，GPU 按照预录好的 DAG 自行执行全部操作。

### 1.3 为什么 Decode 阶段特别适合 CUDA Graph

大语言模型推理分为 **Prefill** 和 **Decode** 两阶段（回顾第1课）：

| 特征 | Prefill | Decode |
|------|---------|--------|
| 每步 token 数 | \(L_p\)（可达数千） | 每序列 1 个 |
| 计算图形状 | **变长**（取决于 prompt 长度） | **相对固定**（每步结构一致） |
| 计算特性 | 计算密集（compute-bound） | 内存带宽密集（memory-bound） |
| 主要瓶颈 | GPU 算力 | CPU launch + 内存带宽 |

Decode 阶段的关键特点：

1. **计算图结构固定**：每步每个序列只处理 1 个 token，Transformer 各层的矩阵形状完全可预测。
2. **小规模计算高频执行**：每步计算量不大，但 kernel launch 的次数多（LayerNorm、QKV 投影、注意力、FFN 等），launch 开销占比显著。
3. **CPU 成为瓶颈**：GPU 很快就能算完一步的矩阵运算，但 CPU 忙于提交下一步的 kernel，导致 GPU 空等。

因此，用 CUDA Graph 将整个 decode 前向 pass 录制下来，可以 **将数十次 kernel launch 压缩为一次 graph launch**，显著降低 CPU 侧开销、提升 GPU 利用率。

### 1.4 CUDA Graph 的局限

需要注意，CUDA Graph 并非万能：

- **不能包含 CPU 逻辑**：录制期间不能有 Python 条件分支、循环等动态控制流。
- **不能包含动态 shape 操作**：张量形状必须在 capture 时确定。
- **不能包含 CPU-GPU 同步**：如 `torch.cuda.synchronize()`、`.item()` 等操作不能出现在 graph 内部。
- **内存地址绑定**：capture 时使用的张量地址被绑定到 graph 中，replay 时必须往 **同一块内存** 写入数据。

这些限制决定了 nano-vllm 的设计策略：**只对 shape 稳定的 decode 阶段使用 Graph，Prefill 走 eager**。

---

## 二、源码对照：`capture_cudagraph` 完整流程

实现位于 `nanovllm/engine/model_runner.py`。核心分为：**初始化时是否 capture**、**capture 的具体步骤**、**前向时是否 replay**。

### 2.1 何时启用 CUDA Graph

在 `ModelRunner.__init__` 中，分配 KV Cache 之后，若 **未** 设置 `enforce_eager`，则调用 `capture_cudagraph()`。若 `enforce_eager=True`，则完全走 **eager（即时执行）** 路径，不录制图，便于调试。

### 2.2 `capture_cudagraph` 逐行解析

```python
def capture_cudagraph(self):
    config = self.config
    hf_config = config.hf_config

    # ① 确定最大 batch size 上界
    max_bs = min(self.config.max_num_seqs, 512)

    # ② 计算 KV Cache 中每个序列最大需要多少个 block
    max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

    # ③ 预分配最大规模的输入/输出缓冲区（graph_vars）
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hf_config.hidden_size)

    # ④ 定义要 capture 的 batch size 档位
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}
    self.graph_pool = None

    # ⑤ 从大到小逐个录制
    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()

        # 设置 decode 上下文（告诉注意力层当前是 decode 模式）
        set_context(False, slot_mapping=slot_mapping[:bs],
                    context_lens=context_lens[:bs],
                    block_tables=block_tables[:bs])

        # Warmup：确保 CUDA 上下文、JIT 等初始化完成
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

        # 正式 capture
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

        # 首张图创建后取出内存池，后续图共用
        if self.graph_pool is None:
            self.graph_pool = graph.pool()

        self.graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    # ⑥ 保存引用，后续 replay 时通过这些缓冲区传递数据
    self.graph_vars = dict(
        input_ids=input_ids, positions=positions,
        slot_mapping=slot_mapping, context_lens=context_lens,
        block_tables=block_tables, outputs=outputs,
    )
```

下面逐一讲解关键设计：

### 2.3 `max_bs` 上界设计

```python
max_bs = min(self.config.max_num_seqs, 512)
```

- `max_num_seqs` 是调度器允许的最大并发序列数。
- 硬编码上限 512 是工程经验值：batch 超过 512 时，计算本身已经能充分利用 GPU，launch overhead 占比下降，Graph 收益变小。
- 同时，Graph 数量越多，capture 耗时和显存开销越大，512 是一个合理的上限折中。

### 2.4 `graph_bs` 档位设计：`[1, 2, 4, 8, 16, 32, ...]`

```python
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
```

为什么不是每个 batch size 都录制一张图？原因：

1. **内存成本**：每张图都会固定一份 GPU 内存，图太多会浪费显存。
2. **capture 耗时**：每张图需要 warmup + capture，数量越多初始化越慢。

档位设计策略：

- **小 batch（1-8）用密集档位**：小 batch 时 padding 浪费的计算比例高。例如 bs=3 被映射到 bs=4 只浪费 25%，但如果只有 bs=16 档位，bs=3 就浪费 >80%。
- **大 batch（16+）按 16 步进**：大 batch 时 padding 浪费的比例变低，16 的步长是合理折中。

### 2.5 为什么用 `reversed` 从大到小录制

```python
for bs in reversed(self.graph_bs):
```

CUDA Graph 的内存池（`graph_pool`）在首次使用时会根据需求分配内存。从 **最大的 batch** 开始录制，能确保内存池一开始就分配了足够大的空间。后续较小的 batch 只会使用这片空间的子集，不需要额外扩展，减少了碎片和重新分配。

### 2.6 Warmup 的必要性

```python
outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
```

首次执行时可能触发：

- **CUDA 上下文初始化**：分配设备内存、建立驱动连接。
- **cuDNN/cuBLAS 算法选择**：首次矩阵乘会做 auto-tuning。
- **`torch.compile` 的 JIT 编译**：如果模型中有 `@torch.compile`，首次调用会触发编译。

如果把这些不稳定行为录进图里，replay 时会得到错误结果或崩溃。因此**必须先 warmup 再 capture**。

### 2.7 `graph_pool` 共享内存池

```python
with torch.cuda.graph(graph, self.graph_pool):
    ...
if self.graph_pool is None:
    self.graph_pool = graph.pool()
```

多张 CUDAGraph 若各自独立分配内存，容易造成显存碎片。PyTorch 允许把第一张图的 `graph.pool()` 传给后续 `torch.cuda.graph(..., pool=)`，使多张图共享同一套内存池。

好处：
- **减少显存碎片**：所有图复用同一片预分配区域。
- **降低总显存占用**：不同 batch size 的图的临时缓冲区可以重叠使用（因为同时只会 replay 一张图）。

### 2.8 `graph_vars`：输入输出映射桥梁

```python
self.graph_vars = dict(
    input_ids=input_ids, positions=positions,
    slot_mapping=slot_mapping, context_lens=context_lens,
    block_tables=block_tables, outputs=outputs,
)
```

CUDA Graph 内部记录的是 **tensor 的内存地址**，不是 tensor 对象。因此 replay 前需要把真实数据 **拷入** capture 时使用的同一块内存中。`graph_vars` 保存了这些 "桥梁" tensor 的引用：

| 变量 | 含义 | 数据流向 |
|------|------|---------|
| `input_ids` | 当前 decode 步的 token ID | 输入 → 模型 |
| `positions` | 各 token 的绝对位置 | 输入 → RoPE |
| `slot_mapping` | token 写入 KV Cache 的目标槽位 | 输入 → Attention |
| `context_lens` | 各序列当前的上下文长度 | 输入 → FlashAttention |
| `block_tables` | 各序列 KV block 的映射表 | 输入 → FlashAttention |
| `outputs` | 模型最后一层的 hidden states | 模型 → 输出 |

---

## 三、源码对照：`run_model` 中的 CUDA Graph 分支

```python
def run_model(self, input_ids, positions, is_prefill):
    # 分支 1：Prefill / eager / 超大 batch → 直接前向
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))

    # 分支 2：Decode + Graph 可用 → replay
    else:
        bs = input_ids.size(0)
        context = get_context()

        # 选取不小于 bs 的最小档位
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars

        # 将真实数据拷入 graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)        # 先清空
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()           # 先清零
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

        # 一次性重放整个前向
        graph.replay()

        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

### 3.1 三路判断逻辑

`run_model` 的第一个 `if` 涵盖了三种 **不走 Graph** 的情况：

1. **`is_prefill`**：Prefill 阶段 token 数变长，不适合固定 shape 的 Graph。
2. **`self.enforce_eager`**：用户显式要求 eager 执行，通常用于调试。
3. **`input_ids.size(0) > 512`**：batch 超过 capture 上界，没有对应的图。

### 3.2 档位选择策略

```python
graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
```

示例：假设 `graph_bs = [1, 2, 4, 8, 16, 32, ...]`

| 实际 bs | 选中档位 | padding 浪费 |
|---------|---------|-------------|
| 1 | 1 | 0% |
| 3 | 4 | 25% |
| 5 | 8 | 37.5% |
| 10 | 16 | 37.5% |
| 20 | 32 | 37.5% |

padding 部分的计算虽然浪费，但 **比重新 launch 所有 kernel 的开销小得多**。

### 3.3 `slot_mapping.fill_(-1)` 的技巧

`slot_mapping` 长度在 capture 时是 `max_bs`，但实际请求可能只有 `bs` 个有效 token。若不复位，上一轮 replay 残留在 `[bs:]` 的旧槽位可能被后续 kernel 误读。

先 **`fill_(-1)`** 整块标为无效，再 **`[:bs] = context.slot_mapping`** 写入本步有效映射。Attention 里的 Triton `store_kvcache_kernel` 对 `slot == -1` 会直接 `return`，与这一约定一致（见第17课）。

同理，`context_lens` 使用 `zero_()` 再写入前 `bs` 个，避免残留污染。

### 3.4 `graph.replay()` 的执行机制

调用 `graph.replay()` 时：

1. **CPU 侧**：只需要一次 `cudaGraphLaunch` 调用，将整个 DAG 提交给 GPU。
2. **GPU 侧**：按照 capture 时录制的顺序与依赖关系，依次执行所有 kernel。
3. **数据传递**：Graph 内部各节点使用的是 capture 时绑定的内存地址，所以预先往 `graph_vars` 拷入数据后，Graph 内部自然能读到最新输入。
4. **输出获取**：`outputs[:bs]` 在 replay 完成后自动包含最新的模型输出，因为它也是 capture 时绑定的缓冲区。

### 3.5 `compute_logits` 在 Graph 外部

注意 `compute_logits` 是在 `graph.replay()` **之后** 调用的，并没有被录入 Graph：

```python
return self.model.compute_logits(graph_vars["outputs"][:bs])
```

原因是 `compute_logits` 通常涉及 **lm_head 权重矩阵乘法**，其输出的 logits shape 依赖于词表大小，且可能需要配合采样等后续操作。将其放在 Graph 外部更灵活。

### 3.6 全局 `Context`（`context.py`）与 Graph 的数据契约

Decode 路径里，`run_model` 在 `graph.replay()` 之前把 **`context.slot_mapping`、`context.context_lens`、`context.block_tables`** 拷入 `graph_vars` 对应缓冲区。Attention 与 KV 写入 kernel 并不直接读 Python 里的 `context` 对象，而是依赖 **`set_context` 在 capture 时已绑定到同一块内存** 的语义：录制时 `set_context(False, slot_mapping=slot_mapping[:bs], ...)` 用的是预分配张量；replay 时往 **同一地址** 写入真实调度数据，图中 kernel 读到的即为当前步的 KV 布局。

`utils/context.py` 核心如下（与第18课一致，此处强调 **Graph 复用缓冲区**）：

```python
@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()
def get_context(): return _CONTEXT
def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0,
              slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                       slot_mapping, context_lens, block_tables)
def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
```

**Prefill 不走 Graph**：`set_context(True, cu_seqlens_q=..., ...)` 下的变长 FlashAttention 与 **固定 shape 的 decode 图** 不共用同一套捕获路径，因此 `run_model` 首分支直接 `self.model(...)` eager 执行。

---

## 四、`enforce_eager` 参数详解

### 4.1 配置定义

在 `Config` 中默认为 `False`（启用 Graph）：

```python
enforce_eager: bool = False
```

### 4.2 设为 `True` 的影响

- **跳过** `capture_cudagraph()`：启动时不录制任何图，省去初始化耗时。
- **`run_model` 始终走 eager 路径**：每步都是标准的 PyTorch 前向传播。
- **退出时不清理 graph 相关资源**。

### 4.3 适用场景

| 场景 | 说明 |
|------|------|
| 调试数值问题 | 逐算子检查中间结果，Graph 内部无法插入断点 |
| 不支持 Graph 的环境 | 某些 CUDA 版本或设备不支持 Graph |
| Profiling | 需要逐 kernel 分析耗时时，Graph 会把所有 kernel 合并为一次调用 |
| 开发新功能 | 快速迭代时避免频繁重新 capture |

---

## 五、设计决策深度分析

### 5.1 为什么只对 decode 使用 CUDA Graph

- **Prefill 阶段**：序列长度差异大，`cu_seqlens`、token 总数等变长，且可能触发前缀缓存等特殊路径，难以用固定 shape 的图覆盖所有情况。
- **Decode 阶段**：每步每个序列只算 1 个 token，batch 维度与层结构相对稳定，更适合按 bs 分档 capture。

### 5.2 业界对比

| 框架 | CUDA Graph 使用策略 |
|------|-------------------|
| nano-vllm | 仅 decode，按 batch 分档，最大 512 |
| vLLM | decode + 部分 chunked prefill，更精细的档位管理 |
| TensorRT-LLM | 编译期静态图，decode 默认开启 |
| DeepSpeed-FastGen | 仅 decode，与 nano-vllm 类似 |

### 5.3 CUDA Graph 的内存开销

CUDA Graph capture 期间分配的所有 GPU 内存都会被**固定（pin）**，即使 replay 后也不会被释放。因此：

- 档位越多 → 内存开销越大（尽管 `graph_pool` 已经做了共享优化）。
- `graph_vars` 按 `max_bs` 分配 → 即使运行时 bs 很小，这块内存也一直占用。
- 工程上需要在 **Graph 数量**、**capture 耗时**、**显存占用** 三者之间平衡。

---

## 六、实际性能影响

### 6.1 典型加速效果

以 Qwen2.5-7B 在 A100 上的 decode 阶段为例（数据为典型值）：

| 指标 | 无 Graph (eager) | 有 Graph | 加速比 |
|------|-----------------|---------|--------|
| 单步 decode 延迟 (bs=1) | ~15 ms | ~8 ms | ~1.9x |
| 单步 decode 延迟 (bs=32) | ~18 ms | ~12 ms | ~1.5x |
| decode 吞吐量 | ~1200 tok/s | ~2000 tok/s | ~1.7x |

规律：**batch 越小，Graph 带来的加速比越大**，因为小 batch 时 launch overhead 占比更高。

### 6.2 初始化代价

Graph capture 需要在启动时额外执行 warmup + capture，典型耗时：

- 7B 模型、20 个档位：约 30-60 秒
- 这是一次性代价，启动后每步 decode 都能受益

---

## 七、小结

- **CUDA Graph** 把重复的稳定 GPU 操作录成图，`replay()` 降低 **CPU launch 开销**，特别适合 **shape 稳定的 decode**。
- nano-vllm 按 **`graph_bs`** 分档录制（小 batch 密集、大 batch 按 16 步进），用 **`graph_pool`** 共享内存，**`graph_vars`** 在 replay 前填入真实输入与上下文。
- **Prefill、enforce_eager、bs>512** 走 eager，保证正确性与可调试性。
- **`slot_mapping` 先 `-1` 填充** 是与 KV 写入 kernel 配套的工程细节。
- 从大到小 capture 保证内存池预分配足够空间。
- Graph 的核心收益来自 **减少 CPU-GPU 交互**，而非提升 GPU 计算效率本身。

---

## 八、面试考点（含参考答案）

**1. CUDA Graph 解决的主要性能问题是什么？**
**答**：主要是 **CPU 侧 kernel launch 与驱动调度开销**，尤其在 decode 小步高频执行时；Graph 将多步合并为 **一次图提交**，减轻 CPU 瓶颈、提高 GPU 利用率。核心原理是将原本每步都要经历的"CPU 检查参数 → 驱动排队 → GPU 执行"简化为"一次 cudaGraphLaunch → GPU 按 DAG 执行全部 kernel"。

**2. 为什么 CUDA Graph 常与 decode 绑定而不是 prefill？**
**答**：Prefill **变长**，attention 的 `cu_seqlens`、总 token 数等变化大，难以固定计算图 shape；decode **每步每序列 1 token**，按 batch 分档更容易固定计算图。此外 decode 阶段计算量小但 kernel 数多，launch overhead 占比高，Graph 收益最大。

**3. nano-vllm 如何为不同 batch size 选择使用哪张图？**
**答**：`graph_bs` 预先定义多个档位 `[1,2,4,8,16,32,...]`；运行时 `bs = input_ids.size(0)`，用 `next(x for x in self.graph_bs if x >= bs)` 选 **不小于 bs 的最小档位**，将数据写入对应长度的 `graph_vars` 前缀后 `replay()`。多余的部分通过 `fill_(-1)` / `zero_()` 标记为无效。

**4. `graph_pool` 的作用？**
**答**：多张 CUDAGraph **共享 PyTorch 返回的内存池**，减少碎片、复用分配。具体做法是第一张图 capture 后调用 `graph.pool()` 获取池句柄，后续图的 `torch.cuda.graph(graph, pool)` 传入同一个句柄，使所有图使用同一片 GPU 内存区域。

**5. 为什么要 warmup 再 capture？**
**答**：首次执行会触发 **CUDA 上下文初始化、cuBLAS auto-tuning、torch.compile JIT 编译** 等不确定行为；先 warmup 再 capture，避免把不稳定行为录进图中，保证 replay 行为与预期一致。

**6. `enforce_eager=True` 会带来什么影响？**
**答**：不执行 `capture_cudagraph()`，`run_model` 始终即时前向，便于调试但 **失去 Graph 的 launch 优化**。适用于调试数值错误、Profiling、开发新功能等场景。

**7. 为何 `slot_mapping` 要先 `fill_(-1)`？**
**答**：缓冲区长度为 `max_bs`，实际只用前 `bs`；填充 `-1` 可清除历史残留，且与 **Triton kernel 跳过 `slot == -1`** 的约定一致，避免错误写入 KV Cache。这是 CUDA Graph 使用中"缓冲区复用"带来的工程挑战之一。

**8. batch 超过 512 为何不走 Graph？**
**答**：与 capture 上界一致，控制图数量与显存；超大 batch 时 compute 往往已饱和，launch overhead 占比小，Graph 收益相对小，且保持实现简单可维护。

**9. 为什么从大到小（reversed）顺序 capture？**
**答**：先录制最大 batch 的图，使内存池在首次分配时就获得足够大的空间。后续较小 batch 的图可以在这片已分配的空间内工作，避免内存扩展和碎片问题。

**10. CUDA Graph 能否在训练中使用？**
**答**：理论上可以，但训练涉及反向传播、梯度累积等动态操作，且通常 batch size 固定、shape 较大，launch overhead 占比低，因此训练中使用 Graph 的收益有限且复杂度高。推理（尤其是 decode）才是 CUDA Graph 的主战场。

---

*延伸阅读：PyTorch 文档中 `torch.cuda.graph`、NVIDIA CUDA Graphs 编程指南。*
