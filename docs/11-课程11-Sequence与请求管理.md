# 课程 11：Sequence 与请求管理

> **学习目标**：深入理解 nano-vllm 中请求的核心数据结构 `Sequence`；掌握 `SequenceStatus` 状态机的转换逻辑；逐属性、逐方法剖析源码；理解 block_table 与 KV Cache 的映射关系；掌握序列化优化技巧；为后续调度器和连续批处理的学习打下坚实基础。

---

## 一、为什么需要 Sequence 类

### 1.1 推理引擎中的"请求"

在大模型推理服务中，每一条用户输入（prompt）以及后续生成的 token，统一构成一个**序列（Sequence）**。推理引擎需要跟踪每条序列的以下信息：

- **状态**：这条序列当前是等待调度、正在执行、还是已经完成？
- **Token 列表**：包含原始 prompt 的 token 和已经生成的 completion token。
- **KV Cache 映射**：序列的 Key/Value 缓存数据分布在哪些物理 block 中？
- **采样参数**：温度（temperature）、最大生成长度（max_tokens）、是否忽略 EOS 等。
- **缓存信息**：有多少 token 的 KV Cache 已经被计算并缓存？

如果不用统一的数据结构管理这些信息，调度器（Scheduler）和模型执行器（ModelRunner）之间就无法高效协作。因此 nano-vllm 设计了 `Sequence` 类作为贯穿引擎全生命周期的**核心数据结构**。

### 1.2 类比理解

可以把推理引擎想象成一家医院：

| 医院 | 推理引擎 |
|------|---------|
| 患者 | Sequence（用户请求） |
| 挂号信息 | token_ids、sampling_params |
| 病历卡 | block_table（KV Cache 映射） |
| 就诊状态（候诊/就诊中/已完成） | SequenceStatus |
| 分诊台 | Scheduler |
| 诊室 | ModelRunner（GPU） |

### 1.3 Sequence 在系统中的位置

```
用户请求（文本）
  ↓  tokenize
LLMEngine.add_request()
  ↓  创建 Sequence 对象
Scheduler.add(seq)         ← seq 进入 waiting 队列
  ↓
Scheduler.schedule()       ← seq 被调度，分配 KV Cache block
  ↓
ModelRunner.run(seqs, ...)  ← 读取 seq 的 token_ids / block_table
  ↓
Scheduler.postprocess()     ← 追加新 token，判断是否结束
  ↓
seq.status == FINISHED → 返回结果给用户
```

从创建到结束，`Sequence` 实例贯穿了**加入队列 → 调度 → 前向推理 → 后处理 → 完成**的完整流程。它是引擎中所有模块通信的"通用语言"。

---

## 二、SequenceStatus 状态机

### 2.1 三种状态定义

源码路径：`nanovllm/engine/sequence.py`

```python
class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
```

| 状态 | 含义 | 进入条件 | 退出条件 |
|------|------|---------|---------|
| `WAITING` | 在等待队列中，尚未被调度 | 新创建 / 被抢占（preempt） | 被 `schedule()` 选中并分配 block |
| `RUNNING` | 正在参与推理（prefill 或 decode） | `schedule()` 分配 block 后 | 生成完毕（EOS / max_tokens） / 被抢占 |
| `FINISHED` | 生成完毕，等待回收 | 命中 EOS 或达到 max_tokens | 终态，不再变化 |

### 2.2 状态转换图

```
              schedule() 选中
  ┌────────┐  分配 block   ┌────────┐
  │WAITING │──────────────→│RUNNING │
  └────────┘               └────────┘
       ↑                     │    │
       │    preempt()        │    │ 命中 EOS 或
       │    释放 block       │    │ 达到 max_tokens
       └─────────────────────┘    │
                                  ↓
                             ┌──────────┐
                             │ FINISHED │
                             └──────────┘
```

### 2.3 状态转换的触发位置

| 转换 | 触发位置 | 代码片段 |
|------|---------|---------|
| WAITING → RUNNING | `Scheduler.schedule()` | `seq.status = SequenceStatus.RUNNING` |
| RUNNING → WAITING | `Scheduler.preempt()` | `seq.status = SequenceStatus.WAITING` |
| RUNNING → FINISHED | `Scheduler.postprocess()` | `seq.status = SequenceStatus.FINISHED` |

### 2.4 为什么使用 Enum 而非字符串

使用 `Enum` 而非字符串（如 `"waiting"`）的好处：

1. **类型安全**：拼写错误会在访问时立即报错，而非静默失败
2. **性能**：枚举比较是整数比较，比字符串比较快
3. **IDE 支持**：自动补全、重构时可追踪所有引用
4. **可读性**：`SequenceStatus.RUNNING` 比 `"running"` 语义更明确

### 2.5 为什么没有 PREEMPTED 状态

你可能会疑问：被抢占的序列是否需要一个单独的 `PREEMPTED` 状态？

在 nano-vllm 的设计中，抢占后的序列**直接回到 WAITING**。这样做的好处是简化状态机——调度器只需检查 waiting 队列即可，不需要区分"新来的"和"被抢占的"。被抢占的序列通过 `appendleft` 放到 waiting 队列头部，保证它们优先被重新调度。

> **面试提示**：vLLM 的正式版本中有 `SWAPPED` 状态，用于区分被交换到 CPU 内存的序列。nano-vllm 简化了这一设计。

---

## 三、Sequence 类完整源码解读

### 3.1 类级属性

```python
class Sequence:
    block_size = 256
    counter = count()
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `block_size` | int（类变量） | KV Cache block 大小，默认 256 个 token。表示每个物理 block 能容纳多少 token 的 KV 数据 |
| `counter` | itertools.count（类变量） | 全局自增计数器，用于生成唯一的 `seq_id` |

**为什么 block_size 是类变量？**

因为所有序列共享同一套物理 block 管理系统，block 大小必须一致。将其设为类变量确保所有实例使用相同的 block_size，且只需修改一处即可全局生效。

**为什么使用 `itertools.count()` 而非普通计数器？**

`count()` 是一个无限迭代器，线程安全（CPython GIL 下），且生成唯一递增 ID 非常简洁：

```python
from itertools import count
c = count()
next(c)  # 0
next(c)  # 1
next(c)  # 2
```

### 3.2 构造函数 `__init__`

```python
def __init__(self, token_ids, sampling_params=SamplingParams()):
    self.seq_id = next(Sequence.counter)
    self.status = SequenceStatus.WAITING
    self.token_ids = copy(token_ids)
    self.last_token = token_ids[-1]
    self.num_tokens = len(self.token_ids)
    self.num_prompt_tokens = len(token_ids)
    self.num_cached_tokens = 0
    self.block_table = []
    self.temperature = sampling_params.temperature
    self.max_tokens = sampling_params.max_tokens
    self.ignore_eos = sampling_params.ignore_eos
```

逐属性详解：

| 属性 | 类型 | 初始值 | 说明 |
|------|------|--------|------|
| `seq_id` | int | 自增 | 全局唯一标识符，区分不同序列 |
| `status` | SequenceStatus | WAITING | 初始状态：新创建的序列一定是等待状态 |
| `token_ids` | list[int] | prompt tokens 的副本 | 包含 prompt + 已生成的 completion token |
| `last_token` | int | prompt 最后一个 token | 用于 decode 阶段作为下一步输入（避免取列表末尾的开销） |
| `num_tokens` | int | len(token_ids) | 当前总 token 数（prompt + completion） |
| `num_prompt_tokens` | int | len(token_ids) | prompt 的 token 数量，**创建后不变** |
| `num_cached_tokens` | int | 0 | 已被 KV Cache 缓存的 token 数量（用于前缀缓存） |
| `block_table` | list[int] | [] | 物理 block 索引列表，记录 KV Cache 的存储位置 |
| `temperature` | float | 来自 sampling_params | 采样温度：0 为贪心，越大越随机 |
| `max_tokens` | int | 来自 sampling_params | 最大生成 token 数 |
| `ignore_eos` | bool | 来自 sampling_params | 是否忽略 EOS token（强制生成到 max_tokens） |

### 3.3 为什么要 `copy(token_ids)`

```python
self.token_ids = copy(token_ids)
```

使用 `copy()` 创建 token_ids 的浅拷贝，防止外部修改原始列表时影响到 Sequence 内部状态。这是防御性编程的典型实践。

### 3.4 `num_prompt_tokens` vs `num_tokens`

这两个属性的区别是理解 Sequence 的关键：

```
创建时: num_prompt_tokens = 5, num_tokens = 5
生成 1 个 token 后: num_prompt_tokens = 5, num_tokens = 6
生成 2 个 token 后: num_prompt_tokens = 5, num_tokens = 7
...
```

由此可以推导出已生成的 completion token 数量：

```python
@property
def num_completion_tokens(self):
    return self.num_tokens - self.num_prompt_tokens
```

这个值用于判断是否达到 `max_tokens` 限制。

---

## 四、Block 相关属性与方法

### 4.1 block_table 的作用

`block_table` 是 Sequence 与 KV Cache 物理存储之间的桥梁。它是一个整数列表，每个元素是一个**物理 block 的索引号**。

```
假设 block_size = 256, 序列有 600 个 token:

block_table = [3, 7, 15]  # 3个物理block

Block 3: token 0-255 的 KV Cache
Block 7: token 256-511 的 KV Cache
Block 15: token 512-599 的 KV Cache（未满）
```

**关键理解**：block_table 实现了**逻辑 block 到物理 block 的映射**。逻辑上序列的第 0 个 block 可能对应物理内存中的第 3 个 block，第 1 个逻辑 block 对应物理第 7 个 block，以此类推。这就是 PagedAttention 的核心思想——像操作系统的虚拟内存一样管理 KV Cache。

### 4.2 num_blocks 属性

```python
@property
def num_blocks(self):
    return (self.num_tokens + self.block_size - 1) // self.block_size
```

这是经典的**向上取整除法**公式，计算当前序列需要多少个 block：

| num_tokens | block_size | num_blocks | 计算过程 |
|-----------|-----------|-----------|---------|
| 1 | 256 | 1 | (1+255)//256 = 1 |
| 256 | 256 | 1 | (256+255)//256 = 1 |
| 257 | 256 | 2 | (257+255)//256 = 2 |
| 512 | 256 | 2 | (512+255)//256 = 2 |
| 600 | 256 | 3 | (600+255)//256 = 3 |

为什么用 `(n + d - 1) // d` 而不是 `math.ceil(n / d)`？因为整数运算比浮点运算更快更精确，在高频调用的场景下这个微小差异会累积。

### 4.3 last_block_num_tokens 属性

```python
@property
def last_block_num_tokens(self):
    return self.num_tokens - (self.num_blocks - 1) * self.block_size
```

计算最后一个 block 中有多少个 token。这个信息对 decode 阶段的 `slot_mapping` 计算至关重要。

```
示例：num_tokens=600, block_size=256, num_blocks=3
last_block_num_tokens = 600 - 2 * 256 = 88

含义：最后一个 block 中有 88 个 token，下一个新 token 应写入 slot 88
```

**用途**：在 `ModelRunner.prepare_decode()` 中，需要计算新 token 的 KV Cache 应写入哪个物理位置：

```python
slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
```

### 4.4 block(i) 方法

```python
def block(self, i):
    return self.token_ids[i*self.block_size: (i+1)*self.block_size]
```

获取第 `i` 个逻辑 block 对应的 token_ids 切片。主要用于**前缀缓存（Prefix Caching）**：通过比较两个序列相同位置 block 的 token 内容，判断它们是否共享相同的前缀，从而复用 KV Cache。

```
序列 A: [101, 202, 303, ..., 256个token, 401, 402, ...]
序列 B: [101, 202, 303, ..., 256个token, 501, 502, ...]

block(0) 相同 → 可以共享 Block 0 的 KV Cache
block(1) 不同 → Block 1 需要独立计算
```

---

## 五、append_token 流程

### 5.1 源码解读

```python
def append_token(self, token_id):
    self.token_ids.append(token_id)
    self.last_token = token_id
    self.num_tokens += 1
```

这个方法在每一步 decode 后由 `Scheduler.postprocess()` 调用，将新生成的 token 追加到序列中。

### 5.2 执行流程

```
调用前:
  token_ids = [1, 2, 3, 4, 5]  (prompt)
  last_token = 5
  num_tokens = 5

append_token(100):
  token_ids = [1, 2, 3, 4, 5, 100]
  last_token = 100
  num_tokens = 6

append_token(200):
  token_ids = [1, 2, 3, 4, 5, 100, 200]
  last_token = 200
  num_tokens = 7
```

### 5.3 为什么同时维护 last_token 和 token_ids

看似冗余，实则有性能考量：

1. **decode 阶段**只需要最后一个 token 作为输入，直接读 `last_token` 是 O(1) 操作
2. 如果每次都从 `token_ids[-1]` 获取，虽然 Python 列表索引也是 O(1)，但 `last_token` 作为独立属性在序列化传输时可以避免传输整个 token_ids 列表
3. 在 `__getstate__` 中，如果序列已经有 completion token，只传输 `last_token` 而非完整 `token_ids`

### 5.4 append_token 不更新 block_table

注意 `append_token` **不会**修改 `block_table`。block 的分配是 `BlockManager` 的职责，在 `Scheduler.schedule()` 中通过 `block_manager.may_append()` 完成。Sequence 只是数据的持有者，不负责资源管理——这体现了**关注点分离（Separation of Concerns）**的设计原则。

---

## 六、`__len__` 与其他辅助属性

### 6.1 `__len__` 方法

```python
def __len__(self):
    return self.num_tokens
```

让 Sequence 支持 `len(seq)` 语法，在调度器代码中频繁使用，例如：

```python
num_batched_tokens += len(seq) - seq.num_cached_tokens
```

### 6.2 num_completion_tokens 属性

```python
@property
def num_completion_tokens(self):
    return self.num_tokens - self.num_prompt_tokens
```

用于判断生成是否结束：

```python
if seq.num_completion_tokens == seq.max_tokens:
    seq.status = SequenceStatus.FINISHED
```

### 6.3 is_prefill / is_finished 属性

```python
@property
def is_prefill(self):
    return self.num_completion_tokens == 0

@property
def is_finished(self):
    return self.status == SequenceStatus.FINISHED
```

- `is_prefill`：如果还没有生成任何 completion token，说明这个序列还在 prefill 阶段
- `is_finished`：判断序列是否已完成

---

## 七、序列化优化：`__getstate__` 和 `__setstate__`

### 7.1 为什么需要序列化优化

在多进程（张量并行）场景中，rank 0 进程需要将调度结果广播给其他 worker。每一步都要传输所有参与推理的 Sequence 对象。如果不做优化，每次都序列化完整的 `token_ids` 列表（可能有数千个 token），会造成巨大的通信开销。

### 7.2 `__getstate__` 源码

```python
def __getstate__(self):
    return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token)
```

**精妙之处**：

1. **Prefill 阶段**（`num_completion_tokens == 0`）：传输完整 `token_ids`，因为 worker 需要所有 prompt token 来计算 KV Cache
2. **Decode 阶段**（`num_completion_tokens > 0`）：只传输 `last_token`，因为 worker 只需要最后一个 token 作为输入

### 7.3 传输数据量对比

假设一个序列有 1000 个 prompt token，已生成 500 个 token：

| 方式 | 传输数据 | 大小估算 |
|------|---------|---------|
| 不优化 | 完整对象（所有属性 + 1500 token_ids） | ~12KB |
| `__getstate__` 优化 | num_tokens + num_prompt_tokens + num_cached_tokens + block_table + last_token | ~100B |

decode 阶段的传输量减少了约 **100 倍**！

### 7.4 `__setstate__` 源码

```python
def __setstate__(self, state):
    self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table, token_data = state
    if isinstance(token_data, list):
        self.token_ids = token_data
    else:
        self.last_token = token_data
```

接收端根据 `token_data` 的类型判断当前阶段：

- 如果是 `list`：说明是 prefill 阶段，`token_data` 就是完整的 `token_ids`
- 如果是 `int`：说明是 decode 阶段，`token_data` 就是 `last_token`

### 7.5 序列化在系统中的使用场景

```
Rank 0 (主进程)                     Rank 1/2/3 (Worker 进程)
     │                                      │
     │  scheduler.schedule()                │
     │  → 获得 [seq1, seq2, ...]            │
     │                                      │
     │  pickle.dumps(seqs)                  │
     │  → __getstate__ 被调用               │
     │  → 瘦身后的数据                       │
     │                                      │
     │  ════ SharedMemory 传输 ═══════>     │
     │                                      │
     │                           pickle.loads(data)
     │                           → __setstate__ 被调用
     │                           → 恢复 Sequence 对象
     │                                      │
     │                           model.forward(seqs)
```

---

## 八、Sequence 与其他模块的交互

### 8.1 与 Scheduler 的交互

```python
# Scheduler 创建时机
scheduler.add(seq)  # seq 加入 waiting 队列

# Scheduler 调度时
scheduler.schedule()
    → seq.status = SequenceStatus.RUNNING
    → block_manager.allocate(seq)  # 填充 seq.block_table

# Scheduler 后处理时
scheduler.postprocess(seqs, token_ids)
    → seq.append_token(token_id)
    → 检查 seq.num_completion_tokens == seq.max_tokens
```

### 8.2 与 ModelRunner 的交互

```python
# Prefill 准备
model_runner.prepare_prefill(seqs)
    → 读取 seq.token_ids[seq.num_cached_tokens:]  # 跳过已缓存的 token
    → 读取 seq.block_table  # 获取 KV Cache 写入位置

# Decode 准备
model_runner.prepare_decode(seqs)
    → 读取 seq.last_token  # 只需最后一个 token
    → 读取 seq.block_table[-1]  # 最后一个 block
    → 读取 seq.last_block_num_tokens  # 计算 slot_mapping
```

### 8.3 与 BlockManager 的交互

```python
# 分配 block
block_manager.allocate(seq)
    → 计算 seq.num_blocks
    → 分配物理 block
    → 填充 seq.block_table

# 释放 block
block_manager.deallocate(seq)
    → 回收 seq.block_table 中的物理 block
    → seq.block_table = []

# 追加 block
block_manager.may_append(seq)
    → 如果最后一个 block 满了，分配新 block
    → 追加到 seq.block_table
```

---

## 九、num_cached_tokens 的含义与作用

### 9.1 什么是 Prefix Caching

Prefix Caching（前缀缓存）是一种优化技术：如果两个请求有相同的前缀（例如相同的系统提示），可以复用已经计算好的 KV Cache，避免重复计算。

```
请求 A: "你是一个AI助手。请介绍北京。"
请求 B: "你是一个AI助手。请介绍上海。"

共同前缀: "你是一个AI助手。"
→ 这部分的 KV Cache 只需计算一次
```

### 9.2 num_cached_tokens 的作用

`num_cached_tokens` 记录了序列中有多少 token 的 KV Cache **已经可用**（来自前缀缓存或之前的计算）。

```
假设 num_cached_tokens = 100, num_tokens = 500

prefill 时实际需要计算的 token 数: 500 - 100 = 400
→ 节省了 100 个 token 的计算量
```

在 `ModelRunner.prepare_prefill()` 中的使用：

```python
input_ids.extend(seq[seq.num_cached_tokens:])  # 只取未缓存的 token
positions.extend(list(range(seq.num_cached_tokens, len(seq))))  # 位置从缓存结束处开始
```

在 `Scheduler.schedule()` 中计算实际批处理 token 数：

```python
num_batched_tokens += len(seq) - seq.num_cached_tokens  # 只计算需要实际计算的 token
```

### 9.3 被抢占后的 num_cached_tokens

当序列被抢占（preempt）时，它的 KV Cache 被释放，`num_cached_tokens` 会被重置为 0（由 BlockManager 处理）。重新调度时需要重新计算所有 token 的 KV Cache。

---

## 十、SamplingParams 采样参数

### 10.1 参数说明

```python
@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 256
    ignore_eos: bool = False
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `temperature` | 1.0 | 采样温度。0 = 贪心解码，1.0 = 标准采样，> 1.0 = 更随机 |
| `max_tokens` | 256 | 最大生成 token 数量 |
| `ignore_eos` | False | 是否忽略 EOS token。True 时即使遇到 EOS 也继续生成 |

### 10.2 temperature 的数学原理

在采样前，logits 会除以 temperature：

\[
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
\]

- \( T \to 0 \)：概率分布趋于 one-hot，总是选择最大概率的 token（贪心）
- \( T = 1 \)：标准 softmax 分布
- \( T > 1 \)：分布更平坦，更多"创意"但也更不可控

### 10.3 为什么 Sequence 直接存储采样参数

而不是持有 `SamplingParams` 的引用？因为：

1. 序列化时只需传输简单类型（float、int、bool），而非复杂对象
2. 避免 SamplingParams 对象被外部修改导致的副作用
3. Sequence 作为最小数据单元，应自包含所有必要信息

---

## 十一、设计模式与工程实践

### 11.1 数据类（Data Class）模式

`Sequence` 本质上是一个数据类，它的主要职责是**持有数据**而非执行复杂逻辑。复杂的业务逻辑由 Scheduler 和 ModelRunner 完成。这遵循了**贫血模型**的设计风格（虽然有争议，但在高性能系统中很常见）。

### 11.2 不可变 vs 可变

| 属性 | 是否可变 | 修改时机 |
|------|---------|---------|
| `seq_id` | 不可变 | 创建时确定 |
| `num_prompt_tokens` | 不可变 | 创建时确定 |
| `temperature` | 不可变 | 创建时确定 |
| `max_tokens` | 不可变 | 创建时确定 |
| `status` | 可变 | Scheduler 修改 |
| `token_ids` | 可变 | append_token 追加 |
| `num_tokens` | 可变 | append_token 增加 |
| `block_table` | 可变 | BlockManager 修改 |
| `num_cached_tokens` | 可变 | BlockManager 修改 |

### 11.3 全局唯一 ID 的设计

使用 `itertools.count()` 生成全局唯一 `seq_id`，这比 UUID 更轻量，且在单进程环境下足够用。在分布式场景中，由于 Sequence 只在 rank 0 创建，不存在 ID 冲突问题。

---

## 十二、源码对照总结

将完整的 Sequence 源码与引擎各模块的使用场景对照：

| Sequence 属性/方法 | 创建/修改者 | 使用者 | 用途 |
|-------------------|-----------|--------|------|
| `seq_id` | `__init__` | Engine | 唯一标识 |
| `status` | `__init__`, Scheduler | Scheduler, Engine | 生命周期管理 |
| `token_ids` | `__init__`, `append_token` | ModelRunner | prefill 输入 |
| `last_token` | `__init__`, `append_token` | ModelRunner | decode 输入 |
| `num_tokens` | `__init__`, `append_token` | Scheduler, ModelRunner | 批处理计算 |
| `num_prompt_tokens` | `__init__` | Scheduler | 判断 prefill/decode |
| `num_cached_tokens` | `__init__`, BlockManager | Scheduler, ModelRunner | 前缀缓存 |
| `block_table` | `__init__`, BlockManager | ModelRunner | KV Cache 寻址 |
| `num_blocks` | 计算属性 | BlockManager | block 分配 |
| `last_block_num_tokens` | 计算属性 | ModelRunner | slot_mapping 计算 |
| `block(i)` | 方法 | BlockManager | 前缀缓存匹配 |
| `append_token()` | 方法 | Scheduler.postprocess | 追加生成 token |
| `__getstate__` | 方法 | pickle（多进程通信） | 序列化瘦身 |
| `__setstate__` | 方法 | pickle（多进程通信） | 反序列化恢复 |

---

## 十三、面试考点

### 考点 1：请描述 Sequence 的状态机及其转换条件

**标准回答**：Sequence 有三种状态——WAITING、RUNNING、FINISHED。新创建时为 WAITING，被调度器选中并分配 KV Cache block 后变为 RUNNING，生成完毕（遇到 EOS 或达到 max_tokens）后变为 FINISHED。如果 GPU 资源不足，RUNNING 状态的序列可以被抢占回 WAITING 状态（释放其 KV Cache block），等待后续重新调度。

### 考点 2：block_table 是什么？为什么不直接按序存储 KV Cache？

**标准回答**：block_table 是逻辑 block 到物理 block 的映射表，类似操作系统的页表。它实现了 PagedAttention 的核心思想——将 KV Cache 划分为固定大小的 block，非连续地存储在 GPU 显存中。这样做的好处是：(1) 避免显存碎片化，(2) 支持动态增长（每次只需追加一个 block），(3) 支持前缀缓存（不同序列可以共享相同内容的 block）。

### 考点 3：`__getstate__` 的序列化优化策略是什么？为什么这样设计？

**标准回答**：在 decode 阶段，只传输 `last_token` 而非完整的 `token_ids` 列表。因为 decode 阶段 worker 只需要最后一个 token 作为模型输入，不需要完整的历史 token。在 prefill 阶段则传输完整 `token_ids`，因为 worker 需要计算所有 prompt token 的 KV Cache。这个优化可以将 decode 阶段的通信量减少两个数量级。

### 考点 4：num_cached_tokens 的作用是什么？

**标准回答**：`num_cached_tokens` 记录了序列中已有 KV Cache 缓存的 token 数量，用于前缀缓存优化。在 prefill 时，只需要计算 `token_ids[num_cached_tokens:]` 这部分 token 的 KV Cache，已缓存的部分可以跳过。在计算批处理 token 数时，也要减去 `num_cached_tokens`，因为这部分不占用计算资源。

### 考点 5：为什么 Sequence 同时维护 token_ids 和 last_token？

**标准回答**：这是空间换时间的优化。在 decode 阶段，模型输入只需要最后一个 token。维护独立的 `last_token` 属性可以：(1) O(1) 快速访问，(2) 在序列化传输时只传输一个整数而非整个列表，(3) 语义更清晰，代码可读性更好。

### 考点 6：Sequence 的 block_size 为什么是类变量？

**标准回答**：因为整个系统的 KV Cache 使用统一的 block 大小，所有序列必须使用相同的 block_size 才能与 BlockManager 正确交互。设为类变量确保一致性，且只需在一处修改就能全局生效。

### 考点 7：如果让你设计一个生产级别的 Sequence 类，你会做哪些改进？

**参考思路**：
1. 添加 `SWAPPED` 状态，支持将 KV Cache 交换到 CPU 内存（而非直接丢弃）
2. 支持 beam search，添加 `parent_seq_id` 和 `fork()` 方法
3. 添加 `LoRA adapter ID`，支持多 LoRA 推理
4. 添加请求级别的 `priority` 字段，支持优先级调度
5. 添加 `arrival_time`，支持基于等待时间的公平调度
6. 支持 `stop_sequences`（停止词列表），而不仅是 EOS

---

## 十四、小结

| 知识点 | 核心理解 |
|--------|---------|
| Sequence 定位 | 推理引擎的核心数据结构，贯穿请求全生命周期 |
| 状态机 | WAITING → RUNNING → FINISHED，支持抢占回退到 WAITING |
| block_table | 逻辑 block 到物理 block 的映射，实现 PagedAttention |
| num_cached_tokens | 前缀缓存的关键字段，减少重复计算 |
| append_token | 轻量追加操作，不涉及 block 分配 |
| 序列化优化 | decode 阶段只传 last_token，减少 100x 通信量 |
| 设计原则 | 关注点分离、防御性拷贝、空间换时间 |

**下一课预告**：我们将深入 Scheduler 调度器，看它如何利用 Sequence 的这些属性来实现高效的 prefill 优先调度和抢占机制。

---

> **学习建议**：尝试在纸上画出一个 Sequence 从创建到完成的全过程，标注每一步各属性的变化。这个练习对面试中的白板题非常有帮助。
