# 课程 14：ModelRunner 模型执行器

本课深入剖析 nano-vllm 的 **ModelRunner**——它是**调度器与 GPU 之间的桥梁**，负责将调度器选出的序列转化为 GPU 可执行的张量输入，驱动模型前向传播，并返回采样结果。理解 ModelRunner 是理解推理引擎"执行层"的关键。

---

## 一、ModelRunner 的职责

### 1.1 总览

ModelRunner 承担了推理引擎中**模型执行**的全部职责：

```
Scheduler.schedule()
    ↓ 返回 (seqs: list[Sequence], is_prefill: bool)
ModelRunner.run(seqs, is_prefill)
    ├── prepare_prefill(seqs) / prepare_decode(seqs)   ← 构造输入张量
    ├── run_model(input_ids, positions, is_prefill)     ← 前向传播
    └── sampler(logits, temperatures)                   ← 采样下一个 token
    ↓ 返回 token_ids: list[int]
Scheduler.postprocess(seqs, token_ids)
```

### 1.2 六大职责

| 职责 | 说明 | 对应方法/阶段 |
|------|------|-------------|
| 模型加载 | 加载 HuggingFace 权重到 GPU | `__init__` |
| KV Cache 分配 | 根据 GPU 显存计算并分配 KV Cache 空间 | `allocate_kv_cache()` |
| 输入准备 | 将 Sequence 列表转换为 GPU 张量 | `prepare_prefill()` / `prepare_decode()` |
| 模型执行 | 调用模型前向传播（eager 或 CUDA Graph） | `run_model()` |
| 采样 | 从 logits 采样下一个 token | `sampler()` |
| 多 GPU 通信 | TP > 1 时通过 SharedMemory 同步序列信息 | `write_shm()` / `loop()` |

---

## 二、初始化流程

### 2.1 构造函数全景

```python
class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event):
        self.rank = rank
        self.world_size = config.tensor_parallel_size

        # 1. 初始化分布式通信
        dist.init_process_group("nccl", "tcp://localhost:2333",
                                world_size=self.world_size, rank=rank)

        # 2. 加载模型
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)

        # 3. 初始化采样器
        self.sampler = Sampler()

        # 4. 模型预热
        self.warmup_model()

        # 5. 分配 KV Cache
        self.allocate_kv_cache()

        # 6. 捕获 CUDA Graph（如果不强制 eager 模式）
        if not self.enforce_eager:
            self.capture_cudagraph()

        # 7. 多 GPU 时，非 rank-0 进入事件循环
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
            else:
                self.shm = SharedMemory(name="nanovllm")
                self.loop()  # 非 rank-0 永久阻塞在此
```

### 2.2 初始化顺序的设计考量

初始化各步骤的顺序是精心设计的：

**步骤 1：初始化 NCCL**

```python
dist.init_process_group("nccl", "tcp://localhost:2333",
                        world_size=self.world_size, rank=rank)
```

必须在加载模型之前完成，因为张量并行的线性层（如 `ColumnParallelLinear`）在初始化时需要知道 `tp_rank` 和 `tp_size` 来确定权重分片方式。

**步骤 2：加载模型**

```python
self.model = Qwen3ForCausalLM(hf_config)
load_model(self.model, config.model)
```

先创建模型结构（此时权重为随机值），再从 HuggingFace 权重文件中加载参数。`load_model` 会调用每个层的 `weight_loader` 方法，处理张量并行时的权重分片。

**步骤 3：模型预热（Warmup）**

```python
self.warmup_model()
```

用随机输入做一次前向传播。目的是：
- 触发 PyTorch/CUDA 的 JIT 编译（如 Triton kernel 的编译）
- 让 CUDA 分配器预分配内存池
- 确保后续计算时不会因为首次编译导致延迟抖动

**步骤 4：分配 KV Cache**

必须在 warmup 之后执行，因为 warmup 会占用一些 GPU 显存，分配 KV Cache 时需要知道**剩余**可用显存。

**步骤 5：捕获 CUDA Graph**

必须在 KV Cache 分配之后执行，因为 CUDA Graph 捕获的前向传播中需要使用 KV Cache 张量。

### 2.3 分配 KV Cache：allocate_kv_cache()

这是 ModelRunner 最关键的初始化步骤之一：

```python
def allocate_kv_cache(self):
    # 1. 查询 GPU 显存状态
    free, total = torch.cuda.mem_get_info()

    # 2. 计算每个 KV Cache block 的字节数
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    block_bytes = (2                           # K 和 V
                  * hf_config.num_hidden_layers # 所有 Transformer 层
                  * self.block_size             # block 中的 token 数
                  * num_kv_heads                # KV 注意力头数
                  * head_dim                    # 每个头的维度
                  * hf_config.torch_dtype.itemsize)  # 数据类型字节数

    # 3. 计算可分配的 block 数
    config.num_kvcache_blocks = int(
        total * config.gpu_memory_utilization   # 总显存的可用比例
        - used                                  # 已用显存
        - peak + current                        # 峰值预留
    ) // block_bytes

    # 4. 分配 KV Cache 张量
    self.kv_cache = torch.empty(
        2,                            # K 和 V
        hf_config.num_hidden_layers,  # 层数
        config.num_kvcache_blocks,    # block 数
        self.block_size,              # 每 block 的 token 数
        num_kv_heads,                 # KV 头数
        head_dim                      # 头维度
    )
```

#### KV Cache 张量的形状解读

```
kv_cache 的 6 维张量：
  维度 0: [K, V]                     → 2
  维度 1: [layer_0, ..., layer_N]    → num_hidden_layers
  维度 2: [block_0, ..., block_M]    → num_kvcache_blocks
  维度 3: [token_0, ..., token_B]    → block_size
  维度 4: [head_0, ..., head_H]      → num_kv_heads
  维度 5: [dim_0, ..., dim_D]        → head_dim
```

访问第 `l` 层、第 `b` 个 block、第 `t` 个 token 的 K 向量：

```python
k = kv_cache[0, l, b, t, :, :]  # shape: [num_kv_heads, head_dim]
```

#### 为什么 block 是第 2 维

注意力计算时需要通过 `block_table` 索引物理 block。将 block 维放在靠前的位置，可以利用 `torch.index_select` 或 CUDA kernel 的连续内存访问模式，提高访存效率。

#### 显存使用量的计算

以 Qwen3-7B 为例（FP16, block_size=256）：

```
num_hidden_layers = 32
num_kv_heads = 4（GQA，4 个 KV 头）
head_dim = 128
block_bytes = 2 × 32 × 256 × 4 × 128 × 2 = 16,777,216 bytes = 16 MB/block

GPU 80GB, gpu_memory_utilization=0.9, 模型占用约 14 GB:
可用 = 80 × 0.9 - 14 = 58 GB
num_kvcache_blocks = 58 GB / 16 MB ≈ 3,625 blocks
总 token 容量 = 3,625 × 256 ≈ 928,000 tokens
```

这意味着 KV Cache 可以同时容纳约 93 万个 token——足以支持数百个并发请求。

---

## 三、prepare_prefill 详解

### 3.1 方法签名与作用

```python
def prepare_prefill(self, seqs: list[Sequence]) -> tuple[list[int], list[int]]:
```

将一组待 prefill 的序列转换为模型前向传播所需的输入张量。

### 3.2 核心数据结构

```python
def prepare_prefill(self, seqs):
    input_ids = []       # 所有序列的 token ID 拼接（一维）
    positions = []       # 每个 token 的位置编码索引
    cu_seqlens_q = [0]   # Q 的累积序列长度
    cu_seqlens_k = [0]   # K 的累积序列长度
    slot_mapping = []    # 每个 token 对应的 KV Cache 物理位置
```

### 3.3 逐序列构造

```python
for seq in seqs:
    seqlen = len(seq)
    seqlen_q = seqlen - seq.num_cached_tokens   # Q 的长度（需计算的部分）
    seqlen_k = seqlen                            # K 的长度（包含缓存部分）

    # 1. 构造 input_ids：只包含未缓存的 token
    input_ids.extend(seq[seq.num_cached_tokens:])

    # 2. 构造 positions：从 num_cached_tokens 开始
    positions.extend(list(range(seq.num_cached_tokens, seqlen)))

    # 3. 更新累积序列长度
    cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
    cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

    # 4. 构造 slot_mapping
    for i in range(seq.num_cached_blocks, seq.num_blocks):
        block_id = seq.block_table[i]
        num_tokens = seq.last_block_num_tokens if i == seq.num_blocks - 1 else self.block_size
        for j in range(num_tokens):
            slot_mapping.append(block_id * self.block_size + j)
```

### 3.4 关键概念：cu_seqlens

`cu_seqlens`（cumulative sequence lengths）是 FlashAttention 变长接口的核心输入，用于标记每个序列在拼接后张量中的边界。

```
假设 3 个序列的 Q 长度分别为 [100, 150, 200]：
cu_seqlens_q = [0, 100, 250, 450]

在拼接后的张量中：
- 序列 0 的 Q：位置 [0, 100)
- 序列 1 的 Q：位置 [100, 250)
- 序列 2 的 Q：位置 [250, 450)
```

**为什么 cu_seqlens_q 和 cu_seqlens_k 可能不同？**

当存在前缀缓存命中时：

```
序列长度 = 300, num_cached_tokens = 256
seqlen_q = 300 - 256 = 44   （只计算 44 个新 token 的 Q）
seqlen_k = 300               （注意力需要 attend 到所有 300 个位置的 K）
```

Q 比 K 短是因为：缓存部分的 KV 已经在 KV Cache 中，不需要重新计算 Q，但注意力仍需要与它们交互。此时 FlashAttention 走带 `block_table` 的分页注意力路径。

### 3.5 关键概念：slot_mapping

`slot_mapping` 将每个 token 映射到 KV Cache 中的物理位置（slot）。

```
物理位置 = block_id × block_size + block 内偏移

例如：block_table = [5, 12], block_size = 256
- token 0~255 → block 5, slot 5×256+0 到 5×256+255
- token 256~299 → block 12, slot 12×256+0 到 12×256+43
```

在 Attention 层的前向传播中，新计算的 K 和 V 会被写入 `slot_mapping` 指定的位置：

```python
# attention.py
cache_k = kv_cache[0][layer_idx]  # [num_blocks, block_size, num_kv_heads, head_dim]
cache_k.view(-1, num_kv_heads, head_dim)[slot_mapping] = k
```

### 3.6 set_context 调用

构造完所有输入后，将元数据设置到全局上下文中，供 Attention 层读取：

```python
set_context(True,                   # is_prefill = True
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping=slot_mapping,
            block_tables=block_tables)
```

Attention 层通过 `get_context()` 获取这些信息，决定使用哪种注意力计算路径。

### 3.7 完整示例

假设 2 个序列：
- seq_0: token_ids=[10,20,30,40], block_table=[5], num_cached_tokens=0
- seq_1: token_ids=[50,60,70], block_table=[8], num_cached_tokens=0

构造结果：

```
input_ids  = [10, 20, 30, 40, 50, 60, 70]
positions  = [0, 1, 2, 3, 0, 1, 2]
cu_seqlens_q = [0, 4, 7]
cu_seqlens_k = [0, 4, 7]
slot_mapping = [5×256+0, 5×256+1, 5×256+2, 5×256+3,
                8×256+0, 8×256+1, 8×256+2]
```

---

## 四、prepare_decode 详解

### 4.1 方法实现

```python
def prepare_decode(self, seqs):
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []
    block_tables = []

    for seq in seqs:
        # 1. 输入只有 1 个 token：最后生成的 token
        input_ids.append(seq.last_token)

        # 2. 位置是序列总长度 - 1
        positions.append(len(seq) - 1)

        # 3. slot_mapping：新 token 写入最后一个 block 的下一个位置
        slot_mapping.append(
            seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
        )

        # 4. context_lens：注意力需要 attend 到的历史长度
        context_lens.append(len(seq))

        # 5. block_tables：该序列的所有物理 block ID
        block_tables.append(seq.block_table)
```

### 4.2 与 prepare_prefill 的关键区别

| 特性 | prepare_prefill | prepare_decode |
|------|----------------|----------------|
| 每序列 token 数 | len(seq) - num_cached_tokens | 1（只有 last_token） |
| 位置编码 | range(num_cached_tokens, len(seq)) | len(seq) - 1 |
| slot_mapping 含义 | 所有需写入的 slot | 只有 1 个新 slot |
| cu_seqlens | 需要（变长注意力） | 不需要 |
| context_lens | 不需要（由 cu_seqlens_k 隐含） | 需要（每序列的 KV 长度） |
| 注意力路径 | flash_attn_varlen_func | flash_attn_with_kvcache |

### 4.3 slot_mapping 的计算

```python
slot_mapping.append(
    seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
)
```

这行代码定位新 token 的 KV 应该写入 KV Cache 的哪个位置：

```
假设 seq.block_table = [5, 12], block_size = 256, num_tokens = 300
last_block_num_tokens = 300 - 1×256 = 44
slot = 12 × 256 + 44 - 1 = 12 × 256 + 43 = 3115

这是因为 append_token 在 postprocess 中调用，此时 num_tokens 已经包含了新 token。
所以 last_block_num_tokens 已经包含新 token，slot 指向新 token 的位置。
```

等等，这里有一个微妙之处：`prepare_decode` 在 `append_token` **之前**被调用。此时 `num_tokens` 还没有包含新 token。

实际上在 nano-vllm 中，`may_append` 在 `schedule()` 中已经为新 token 预留了 block（如果需要）。而 `slot_mapping` 的计算使用的是**当前** `last_block_num_tokens`，它指向的是当前最后一个已填充的位置——也就是新 token 应该写入的下一个位置。

具体流程：

```
1. postprocess 上一步：append_token → num_tokens = N
2. schedule 本步：may_append（如果 N % block_size == 1，分配新 block）
3. prepare_decode：slot = block_table[-1] × block_size + last_block_num_tokens - 1
   = block_table[-1] × block_size + (N - (num_blocks-1) × block_size) - 1
```

这里 `last_block_num_tokens - 1` 对应的是 **0-based 索引**中的最后一个已填充位置。新的 KV 将写入这个位置。

### 4.4 block_tables 的处理

```python
block_tables.append(seq.block_table)
```

在 `set_context` 中，所有序列的 `block_table` 会被合并为一个二维张量，传递给 Attention 核：

```python
# 所有序列的 block_table 填充到同一矩阵（Padding 到最长）
block_tables_tensor = torch.zeros(num_seqs, max_num_blocks, dtype=torch.int32)
for i, bt in enumerate(block_tables):
    block_tables_tensor[i, :len(bt)] = torch.tensor(bt)
```

Attention 核通过 `block_tables_tensor[seq_idx]` 查找对应序列的物理 block。

---

## 五、run_model 方法

### 5.1 Eager 模式 vs CUDA Graph 模式

```python
def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager:
        logits = self.model(input_ids, positions)   # 直接调用模型
    else:
        logits = self.graph_runners[len(input_ids)].run(input_ids, positions)
```

**Eager 模式**：直接调用 PyTorch 模型前向传播。每次都会重新发起所有 CUDA kernel，有 CPU→GPU launch 开销。Prefill 阶段**始终使用 Eager 模式**，因为 Prefill 的 batch 形状（总 token 数）变化很大，难以预先捕获 CUDA Graph。

**CUDA Graph 模式**：Decode 阶段使用。预先捕获好的 CUDA Graph 包含了所有 kernel 的调用序列，运行时只需一次 `graph.replay()`，消除了 CPU→GPU 的 launch 开销。对于 Decode 这种单 token 计算量小但 kernel 数量多的场景，CUDA Graph 的加速效果显著。

### 5.2 为什么 Prefill 不用 CUDA Graph

1. **输入长度不固定**：不同请求的 prompt 长度差异巨大（几十到几千 tokens），无法枚举所有可能的输入形状
2. **计算量大**：Prefill 的 kernel 执行时间远大于 launch 开销，CUDA Graph 的收益微小
3. **FlashAttention 的限制**：变长注意力（`flash_attn_varlen_func`）对 CUDA Graph 的支持有限

### 5.3 CUDA Graph 的 batch size 选择

```python
def capture_cudagraph(self):
    for bs in capture_batch_sizes():
        self.graph_runners[bs] = CUDAGraphRunner(self.model, bs)
```

`capture_batch_sizes()` 返回需要预捕获的 batch size 列表。通常包括 1 到 `max_num_seqs` 的所有值（或某些常用值）。

运行时，如果实际 batch size 不在预捕获的列表中，则 fallback 到 Eager 模式。

---

## 六、run 方法完整流程

### 6.1 方法实现

```python
def run(self, seqs, is_prefill):
    # 1. 构造输入
    input_ids, positions = (
        self.prepare_prefill(seqs) if is_prefill
        else self.prepare_decode(seqs)
    )

    # 2. 前向传播
    logits = self.run_model(input_ids, positions, is_prefill)

    # 3. 采样（仅 rank 0）
    token_ids = (
        self.sampler(logits, temperatures).tolist()
        if self.rank == 0
        else None
    )

    return token_ids
```

### 6.2 流程图

```
run(seqs, is_prefill)
    │
    ├── is_prefill == True
    │       └── prepare_prefill(seqs)
    │               ├── 构造 input_ids（拼接所有序列的未缓存 token）
    │               ├── 构造 positions（位置编码索引）
    │               ├── 构造 cu_seqlens_q, cu_seqlens_k（序列边界）
    │               ├── 构造 slot_mapping（KV Cache 写入位置）
    │               └── set_context(is_prefill=True, ...)
    │
    ├── is_prefill == False
    │       └── prepare_decode(seqs)
    │               ├── 构造 input_ids（每序列 1 个 last_token）
    │               ├── 构造 positions（每序列 1 个位置）
    │               ├── 构造 slot_mapping（1 个新 slot）
    │               ├── 构造 context_lens, block_tables
    │               └── set_context(is_prefill=False, ...)
    │
    ├── run_model(input_ids, positions, is_prefill)
    │       ├── Prefill/Eager → self.model(input_ids, positions)
    │       └── Decode/Graph  → self.graph_runners[bs].run(...)
    │       └── 返回 logits: [num_tokens, vocab_size]
    │
    └── sampler(logits, temperatures)
            ├── logits / temperature
            ├── softmax → 概率分布
            ├── multinomial 采样
            └── 返回 token_ids: [num_seqs]
```

### 6.3 为什么只有 rank 0 做采样

```python
token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
```

在张量并行中，所有 rank 运行相同的模型前向传播（只是每个 rank 处理不同的注意力头/FFN 分片）。最终的 logits 在 `lm_head`（`VocabParallelEmbedding`）的 `forward` 中通过 `all_gather` 汇总到所有 rank。

但**采样只需要做一次**——由 rank 0 执行，然后通过 `SharedMemory` 或其他机制将结果分发给其他 rank。这避免了：
1. 重复计算（采样虽然快，但没必要重复）
2. 不一致的随机采样结果（不同 rank 的随机种子可能不同）

### 6.4 Sampler 的实现

```python
class Sampler:
    def __call__(self, logits, temperatures):
        # 1. 温度缩放
        logits = logits / temperatures.unsqueeze(-1)
        # 2. Softmax 转概率
        probs = torch.softmax(logits, dim=-1)
        # 3. 多项式采样
        token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return token_ids
```

温度的作用：
- `temperature = 1.0`：标准采样
- `temperature > 1.0`：分布更平坦，输出更随机
- `temperature → 0`：趋近 argmax（贪心），但 nano-vllm 不允许 temperature=0

---

## 七、多进程通信（SharedMemory）

### 7.1 为什么需要 SharedMemory

在张量并行（TP > 1）中，每个 GPU 运行一个独立进程。调度器运行在 rank 0 的进程中，它需要将序列信息传递给其他 rank 的 `ModelRunner`。

```
Rank 0（主进程）:
  LLMEngine → Scheduler → ModelRunner (rank 0)
                              │
                         SharedMemory
                              │
Rank 1（子进程）:             ModelRunner (rank 1)
Rank 2（子进程）:             ModelRunner (rank 2)
```

### 7.2 通信机制

**Rank 0 写入 SharedMemory**：

```python
def write_shm(self, method_name, *args):
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")   # 前 4 字节存长度
    self.shm.buf[4:n+4] = data                      # 之后存序列化数据
```

**Rank 1+ 读取 SharedMemory**：

```python
def loop(self):
    while True:
        # 等待信号
        event.wait()
        event.clear()

        # 从 SharedMemory 读取
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])

        # 调用对应方法
        getattr(self, method_name)(*args)
```

### 7.3 call 方法：跨进程方法调用

```python
def call(self, method_name, *args):
    if self.world_size > 1:
        self.write_shm(method_name, *args)
        self.event.set()   # 通知其他 rank
    return getattr(self, method_name)(*args)
```

这实现了一种**简单的 RPC**：
1. Rank 0 将方法名和参数序列化写入 SharedMemory
2. 通过 `Event` 通知其他 rank
3. 所有 rank 调用相同的方法（如 `run(seqs, is_prefill)`）
4. 各 rank 独立执行前向传播（张量并行自动处理权重分片和 AllReduce）

### 7.4 Sequence 的序列化优化

在通过 SharedMemory 传递时，`Sequence` 对象需要被 `pickle.dumps()` 序列化。这就是 `__getstate__` 优化的用武之地：

```python
# sequence.py
def __getstate__(self):
    return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token)
```

Decode 阶段只传 5 个值（而非整个 token 列表），大幅减少序列化开销。对于 batch_size=256、平均序列长度 1000 的场景：

```
不优化：256 × 1000 × 4 bytes ≈ 1 MB 序列化数据
优化后：256 × 5 × 8 bytes ≈ 10 KB 序列化数据
减少 100 倍！
```

### 7.5 SharedMemory vs NCCL vs gRPC

| 通信方式 | 适用场景 | 优点 | 缺点 |
|---------|---------|------|------|
| SharedMemory | 同一节点进程间通信 | 零拷贝、低延迟 | 仅限单节点 |
| NCCL | GPU 间张量通信 | 高带宽、支持多节点 | 只能传 tensor |
| gRPC | 跨节点 RPC | 灵活、跨语言 | 延迟较高 |

nano-vllm 的设计：
- **NCCL**：用于张量并行中的 AllReduce（模型前向传播中的张量通信）
- **SharedMemory**：用于传递序列元信息（方法名、Sequence 对象等非张量数据）

---

## 八、Warmup 与显存管理

### 8.1 warmup_model 的作用

```python
def warmup_model(self):
    dummy_input_ids = torch.zeros(self.max_num_batched_tokens, dtype=torch.long, device="cuda")
    dummy_positions = torch.zeros(self.max_num_batched_tokens, dtype=torch.long, device="cuda")
    self.model(dummy_input_ids, dummy_positions)
    torch.cuda.synchronize()
```

Warmup 的目的：

1. **Triton Kernel 编译**：首次执行时，Triton 会 JIT 编译自定义的注意力核、LayerNorm 核等。编译耗时可达数秒，warmup 将这个延迟前置。
2. **CUDA 内存池初始化**：PyTorch 的 CUDA 内存分配器在首次分配时会建立内存池。warmup 后，后续分配会从池中获取，更快。
3. **cuBLAS 句柄初始化**：矩阵乘法库 cuBLAS 在首次调用时需要初始化，warmup 将这个开销前置。
4. **确定显存峰值**：warmup 后调用 `torch.cuda.max_memory_allocated()` 可以得到前向传播的显存峰值，用于后续 KV Cache 的分配计算。

### 8.2 显存预算分配

GPU 显存的分配遵循以下优先级：

```
GPU 总显存（如 80 GB）
├── 模型权重（如 14 GB）          ← 固定
├── 前向传播激活值峰值（如 2 GB）  ← warmup 后确定
├── CUDA 运行时开销（如 1 GB）     ← 固定
├── KV Cache（如 58 GB）          ← allocate_kv_cache 分配
└── 预留余量（如 5 GB）            ← gpu_memory_utilization < 1.0
```

`gpu_memory_utilization` 参数（默认 0.9）控制了 KV Cache 最多使用多少比例的总显存。设为 0.9 意味着保留 10% 的显存余量，防止 OOM。

---

## 九、面试高频考点

### Q1：ModelRunner 的 prepare_prefill 和 prepare_decode 有什么区别？

**参考答案**：

| 维度 | prepare_prefill | prepare_decode |
|------|----------------|----------------|
| 输入 token 数 | 每序列 seqlen - num_cached_tokens | 每序列 1（last_token） |
| 数据格式 | 变长拼接 + cu_seqlens | 等长（每序列 1 token） |
| slot_mapping | 多个 slot（每个需计算的 token 一个） | 1 个 slot |
| 注意力路径 | flash_attn_varlen_func | flash_attn_with_kvcache |
| CUDA Graph | 不使用（形状不固定） | 使用（形状由 batch_size 决定） |

### Q2：为什么 KV Cache 的分配要在 warmup 之后？

**参考答案**：

因为 warmup 会占用 GPU 显存（编译 Triton kernel、建立 CUDA 内存池等），并且 warmup 过程中的显存峰值决定了前向传播所需的最大工作空间。只有在 warmup 之后，才能准确知道"剩余多少显存可以分配给 KV Cache"。如果在 warmup 之前分配，可能分配过多导致 OOM，或分配过少导致浪费。

### Q3：nano-vllm 如何实现多 GPU 推理中的通信？

**参考答案**：

两种通信机制：
1. **NCCL AllReduce**：用于模型前向传播中的张量通信。在 `RowParallelLinear` 和 `VocabParallelEmbedding` 中，各 rank 计算部分结果后通过 AllReduce 汇总。这是高带宽的 GPU 间通信。
2. **SharedMemory + pickle**：用于传递序列元信息。Rank 0 将方法名和 Sequence 对象序列化后写入共享内存，通过 Event 通知其他 rank 读取。`Sequence.__getstate__` 做了优化，decode 阶段只传 5 个基本值。

### Q4：请解释 cu_seqlens 的含义及其在变长注意力中的作用。

**参考答案**：

`cu_seqlens`（cumulative sequence lengths）是 FlashAttention 变长接口的核心输入。它是一个一维整数数组，长度为 `num_seqs + 1`，记录每个序列在拼接后张量中的累积起始位置。

例如 3 个序列长度为 [100, 150, 200]，`cu_seqlens = [0, 100, 250, 450]`。FlashAttention 通过 `cu_seqlens[i]` 和 `cu_seqlens[i+1]` 确定第 i 个序列在拼接张量中的范围，避免了 Padding。

在前缀缓存场景下，`cu_seqlens_q` 和 `cu_seqlens_k` 可能不同：Q 只包含未缓存的 token，K 包含所有 token（含缓存部分）。

### Q5：CUDA Graph 在 Decode 阶段能带来多少加速？为什么 Prefill 不用？

**参考答案**：

加速效果：Decode 阶段每步的计算量小（每序列只有 1 个 token），但 CUDA kernel 数量多（每层有多个 kernel）。CPU→GPU 的 launch 开销可能占总耗时的 30-50%。CUDA Graph 将所有 kernel 打包为一次 replay，消除了 launch 开销，通常可加速 1.5-2 倍。

Prefill 不用的原因：
1. Prefill 的输入长度不固定，无法预先捕获所有可能的形状
2. Prefill 的 kernel 执行时间远大于 launch 开销，CUDA Graph 的边际收益小
3. FlashAttention 的变长接口对 CUDA Graph 支持有限

### Q6：slot_mapping 在 Prefill 和 Decode 阶段分别代表什么？

**参考答案**：

`slot_mapping` 将 token 映射到 KV Cache 的物理位置。计算公式：`slot = block_id × block_size + block内偏移`。

- **Prefill**：为每个需要计算的 token 生成一个 slot。如果有前缀缓存，只为未缓存的 token 生成 slot（从 `num_cached_blocks` 开始）。所有新计算的 K/V 会被批量写入这些 slot。
- **Decode**：每个序列只有 1 个 slot，指向最后一个 block 的当前写入位置。新生成的 K/V 写入这一个 slot。

### Q7：如果让你优化 ModelRunner，你会从哪些方面入手？

**参考答案**：

1. **Chunked Prefill**：将长 Prefill 拆分为 chunk，与 Decode 混合执行，减少 Decode 等待
2. **更高效的序列化**：用 struct.pack 代替 pickle，进一步减少 SharedMemory 传输量
3. **动态 CUDA Graph**：支持可变 batch size 的 CUDA Graph，或使用 CUDA Graph 的条件节点
4. **异步 KV Cache 管理**：在前向传播的同时异步分配/释放 block
5. **投机解码**：集成 draft model，减少大模型的 Decode 步数
6. **量化 KV Cache**：将 KV Cache 从 FP16 压缩为 INT8/FP8，增加可容纳的 token 数

---

## 十、小结

| 要点 | 内容 |
|------|------|
| **核心职责** | 将 Sequence 列表转化为 GPU 可执行的张量输入，驱动模型前向传播 |
| **初始化顺序** | NCCL → 加载模型 → Warmup → 分配 KV Cache → 捕获 CUDA Graph |
| **prepare_prefill** | 变长拼接 + cu_seqlens + slot_mapping，支持前缀缓存 |
| **prepare_decode** | 每序列 1 token + block_tables + context_lens |
| **执行模式** | Prefill = Eager，Decode = CUDA Graph |
| **多 GPU 通信** | NCCL（张量）+ SharedMemory（元信息），Sequence 序列化优化 |
| **KV Cache 分配** | 根据 GPU 剩余显存计算 block 数，6 维张量存储 |

**核心记忆口诀**：

> Prefill **拼一维**，cu_seqlens **划边界**；
> Decode **取末 token**，slot_mapping **定位置**；
> Eager 算 **大矩阵**，Graph 跑 **小向量**；
> SharedMemory **传元信息**，NCCL **搬张量**。

**下一课预告**：理解了 ModelRunner 如何执行模型后，我们将深入 **张量并行（Tensor Parallelism）**——理解 ColumnParallelLinear、RowParallelLinear 和 QKVParallelLinear 如何在多 GPU 间分割权重和计算。
