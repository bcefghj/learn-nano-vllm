# 课程 12：Scheduler 调度器

> **学习目标**：深入理解 nano-vllm 调度器的完整工作机制；掌握 waiting / running 双队列模型；理解 prefill 优先调度策略和 decode 轮转调度的设计思想；掌握抢占（preempt）机制的实现细节；理解 postprocess 后处理流程；能够在面试中对比分析不同调度策略的优劣。

---

## 一、调度器的角色与职责

### 1.1 为什么需要调度器

大模型推理引擎通常同时服务多个用户请求。每个请求在不同时间到达，需要不同长度的 prompt 处理和不同数量的 token 生成。GPU 的显存和算力是有限的，不可能同时处理所有请求。

调度器的核心职责：

1. **决定每一步执行哪些序列**（选取 + 排序）
2. **区分 prefill 和 decode 阶段**（不同阶段的资源特征截然不同）
3. **管理 KV Cache 资源**（通过 BlockManager 分配 / 释放物理 block）
4. **处理资源不足**（抢占低优先级序列，释放 block 给高优先级序列）
5. **后处理**（追加 token、判断终止条件、清理已完成序列）

### 1.2 类比理解

把调度器想象成餐厅的领班：

| 餐厅 | 推理引擎 |
|------|---------|
| 领班 | Scheduler |
| 等位顾客 | waiting 队列 |
| 正在用餐的顾客 | running 队列 |
| 餐桌（有限资源） | KV Cache blocks |
| 安排入座 | schedule() - 分配 block |
| 催促买单让位 | preempt() - 抢占 |
| 上菜 + 确认是否用完 | postprocess() |

### 1.3 调度器在系统中的位置

```
LLMEngine.step()  ← 引擎的"心跳"
    │
    ├── scheduler.schedule()             ← 选出本步参与的序列 + 分配 block
    │       ↓ 返回 (seqs, is_prefill)
    ├── model_runner.run(seqs, is_prefill)  ← 前向推理
    │       ↓ 返回 token_ids
    └── scheduler.postprocess(seqs, token_ids)  ← 追加 token、判断终止
```

每个 `step()` 调用一轮 schedule → run → postprocess，构成引擎的**心跳循环**。调度器是这个循环的**第一个环节**，决定了整个系统的效率。

---

## 二、Scheduler 的数据结构

### 2.1 构造函数

源码路径：`nanovllm/engine/scheduler.py`

```python
class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
```

### 2.2 关键属性详解

| 属性 | 类型 | 说明 |
|------|------|------|
| `max_num_seqs` | int | 单步最多处理的序列数，限制批大小。防止同时处理太多请求导致每个请求延迟过高 |
| `max_num_batched_tokens` | int | 单步最多处理的 token 总数，限制计算量。决定 GPU 单步最大工作负载 |
| `eos` | int | EOS token ID，用于判断生成是否自然终止 |
| `block_manager` | BlockManager | KV Cache 物理 block 管理器，负责分配和回收 block |
| `waiting` | deque[Sequence] | 等待队列：存放尚未开始或被抢占的序列 |
| `running` | deque[Sequence] | 运行队列：存放正在参与推理的序列 |

### 2.3 为什么使用 deque 而非 list

`deque`（双端队列）vs `list` 的性能对比：

| 操作 | deque | list |
|------|-------|------|
| 左端添加 `appendleft` | O(1) | O(n) |
| 左端弹出 `popleft` | O(1) | O(n) |
| 右端添加 `append` | O(1) | 均摊 O(1) |
| 右端弹出 `pop` | O(1) | O(1) |
| 随机访问 `[i]` | O(n) | O(1) |
| 中间删除 `remove` | O(n) | O(n) |

调度器的核心操作是**从队列头部取出序列**和**在头/尾部添加序列**，这些操作在 deque 上都是 O(1)。

### 2.4 两个队列的关系

```
                   schedule()
  ┌──────────┐    选中并分配    ┌──────────┐
  │ waiting  │──────────────→ │ running  │
  │  队列    │                │  队列    │
  └──────────┘                └──────────┘
       ↑                         │  │
       │      preempt()          │  │ postprocess()
       │      资源不足回退        │  │ 完成后移除
       └─────────────────────────┘  │
                                    ↓
                              序列完成，从 running 移除
```

---

## 三、schedule() 方法完整流程

### 3.1 方法签名与返回值

```python
def schedule(self):
    # 返回: (scheduled_seqs: list[Sequence], is_prefill: bool)
```

- `scheduled_seqs`：本步参与推理的序列列表
- `is_prefill`：True 表示本步执行 prefill，False 表示执行 decode

### 3.2 核心设计原则：Prefill 优先

nano-vllm 的调度策略是**prefill 优先**：只要 waiting 队列中有序列可以调度，就优先处理它们（即使 running 队列中有序列在等待 decode）。

**为什么 prefill 优先？**

1. **用户体验**：新请求需要先完成 prefill 才能开始生成，prefill 越快，用户等待首个 token 的时间越短（Time To First Token, TTFT）
2. **计算效率**：prefill 是计算密集型（compute-bound），可以高效利用 GPU 算力
3. **避免饥饿**：如果 decode 优先，新请求可能长时间无法得到处理

### 3.3 Prefill 调度阶段（与源码一致）

以下与 `nano-vllm-main/nanovllm/engine/scheduler.py` 中 `schedule()` 的 **prefill 分支**一致：

```python
def schedule(self):
    # ---------- prefill ----------
    scheduled_seqs = []
    num_seqs = 0
    num_batched_tokens = 0
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]
        if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
            break
        num_seqs += 1
        self.block_manager.allocate(seq)
        num_batched_tokens += len(seq) - seq.num_cached_tokens
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)
    if scheduled_seqs:
        return scheduled_seqs, True
    # ---------- decode 见下一小节 ----------
```

**逐行解析**：

1. **`num_seqs = 0`、`num_batched_tokens = 0`**：本步若走 prefill 分支，计数**仅从 waiting 中即将拉起的序列**算起；**不会**把已在 `running` 里做 decode 的序列算进本步的 batch（见下条）。
2. **`len(seq)`**：对 waiting 中的序列，一般为整段 prompt 长度（首段 prefill）；若曾被抢占后回到 waiting，则长度包含已生成部分，调度逻辑仍用 `len(seq)` 与 `num_cached_tokens` 配合 BlockManager。
3. **`len(seq) - seq.num_cached_tokens`**：前缀缓存命中时，只把**未缓存**的 token 计入本步计算量，从而贴合真实 FLOPs / 显存写入。
4. **两个 break 条件（与运算拆成一行）**：
   - `max_num_batched_tokens`：单步总 token 上限；
   - `can_allocate(seq)`：物理 KV block 是否够整条序列当前所需块数。
5. **Prefill 优先**：只要 `waiting` 里还有**能**被上述条件满足的序列，本函数会**直接** `return ..., True`，**本步不会执行 decode**。因此 nano-vllm 在**单次 `schedule()` 调用内**不做 prefill 与 decode 混合；连续批处理体现在**多次** `step()` 的交替上（先跑完一批 waiting 的 prefill，再在后续步里跑 running 的 decode）。

### 3.4 常见误解：不要把 running 的长度并进 prefill 计数

旧版讲义曾误写为 `num_seqs = len(self.running)`。在当前仓库实现里，**prefill 分支不会**把已在 running 中解码的序列计入 `num_seqs` / `num_batched_tokens`。若 waiting 非空且资源允许，本步**只**调度 waiting → prefill；decode 只在 **waiting 本轮无法调度任何序列** 时，由下一分支处理。

### 3.5 Decode 调度阶段（与源码一致）

```python
    # ---------- decode（仅当上面 prefill 未返回时执行）----------
    while self.running and num_seqs < self.max_num_seqs:
        seq = self.running.popleft()
        while not self.block_manager.can_append(seq):
            if self.running:
                self.preempt(self.running.pop())
            else:
                self.preempt(seq)
                break
        else:
            num_seqs += 1
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
    assert scheduled_seqs
    self.running.extendleft(reversed(scheduled_seqs))
    return scheduled_seqs, False
```

**逐行解析**：

1. **`num_seqs` 沿用 prefill 段结束时的值**：若 prefill 未调度任何序列，此处仍为 `0`；若未来代码改动使两段共用计数，需以仓库为准。
2. **`self.running.popleft()`**：从 running **队头**取序列，保证 **FIFO** 轮转 decode。
3. **内层 `while not can_append`**：KV 无法再 append（例如缺 block）时，通过 **LIFO** `running.pop()` 抢占**队尾**序列，直到能 append 或只能抢占当前 `seq` 并 `break`。
4. **`while ... else`**：仅当内层 while **未**被 `break` 退出时进入 `else`：执行 `may_append`、`append` 到 `scheduled_seqs`，并 `num_seqs += 1`。
5. **`extendleft(reversed(scheduled_seqs))`**：把本步选中的序列按原相对顺序塞回 **running 队头**，下一步仍从队头弹出，实现**轮转**。
6. **`assert scheduled_seqs`**：decode 路径要求至少有一个序列；若 running 非空却因资源死锁，可能触发断言失败——属于容量配置问题，面试可提及。

### 3.6 while...else 语法详解

这是 Python 中一个不太常见但非常优雅的语法：

```python
while condition:
    ...
    if some_check:
        break
else:
    # 只在 while 正常结束（condition 变为 False）时执行
    # 如果 break 退出，则不执行
    ...
```

在 decode 调度中：
- 如果成功释放了足够空间（`can_append` 变为 True，while 正常结束）→ 执行 else，将序列加入调度
- 如果没有可抢占的序列了（break 退出）→ 不执行 else，该序列不参与本步

---

## 四、抢占机制（Preempt）

### 4.1 为什么需要抢占

在 decode 阶段，每个序列每步生成一个 token，需要在 KV Cache 中写入一个新的 KV 对。如果某个序列的最后一个 block 已满，就需要分配新的物理 block。但物理 block 是有限的——如果已经用完，就需要**抢占（preempt）**其他序列来释放 block。

### 4.2 preempt() 源码

```python
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```

三步操作：

1. **状态回退**：将序列状态从 RUNNING 改为 WAITING
2. **释放资源**：通过 BlockManager 释放该序列占用的所有物理 block
3. **重新排队**：使用 `appendleft` 将序列放到 waiting 队列**头部**

### 4.3 为什么用 appendleft

被抢占的序列使用 `appendleft`（放到头部）而非 `append`（放到尾部），是为了**保证公平性**：

- 被抢占的序列已经等待了一段时间，不应该排到新请求后面
- 放到头部确保它们在下一轮调度中**优先被重新调度**
- 这避免了"饥饿"问题——某个序列被反复抢占却永远无法完成

### 4.4 抢占策略：LIFO

```python
self.preempt(self.running.pop())  # pop() 从尾部取出
```

nano-vllm 使用 **LIFO（Last In First Out）** 抢占策略——最后加入 running 队列的序列最先被抢占。

**为什么选择 LIFO？**

1. **最小化浪费**：最后加入的序列可能才刚开始生成，抢占它浪费的计算量最少
2. **资源释放量大**：如果最后加入的序列有较长的 prompt，它的 block 较多，释放后更可能满足空间需求
3. **简单高效**：deque.pop() 是 O(1) 操作

### 4.5 抢占的代价

抢占不是免费的：

```
序列 A 在 running 中，已完成 prefill（1000 token），生成了 200 个 token
  → 占用 ceil(1200/256) = 5 个 block

抢占 A：
  → 释放 5 个 block（KV Cache 全部丢失）
  → A 回到 waiting 队列
  → 重新调度 A 时，需要重新做 1000 token 的 prefill
  → 浪费了之前的全部计算
```

这就是 nano-vllm 的简化设计——**recompute 策略**：被抢占的序列需要完全重新计算。在 vLLM 的完整版本中，还有 **swap 策略**：将 KV Cache 从 GPU 交换到 CPU 内存，避免重复计算。

### 4.6 抢占自己的场景

```python
if self.running:
    self.preempt(self.running.pop())
else:
    self.preempt(seq)  # 抢占自己
    break
```

当 running 队列为空（所有其他序列都被抢占了），但当前序列仍然无法获得足够的 block 时，只能**抢占自己**。这种情况意味着：

- 该序列需要的 block 数量超过了系统总容量
- 或者存在严重的内存碎片

抢占自己后，序列回到 waiting 队列，等待其他序列完成释放更多 block。

---

## 五、Decode 调度的资源检查

### 5.1 can_append vs can_allocate

| 方法 | 调用时机 | 检查内容 |
|------|---------|---------|
| `can_allocate(seq)` | Prefill 调度 | 序列需要 `num_blocks` 个新 block，是否有足够的空闲 block |
| `can_append(seq)` | Decode 调度 | 序列的最后一个 block 是否已满，如果满了是否有 1 个空闲 block |

### 5.2 may_append 的条件性分配

```python
def may_append(self, seq):
    if seq.last_block_num_tokens == self.block_size:
        # 最后一个 block 已满，需要分配新 block
        new_block = self.allocate_block()
        seq.block_table.append(new_block)
```

只有当最后一个 block 恰好满了，才需要新分配。大多数情况下，最后一个 block 还有空位，不需要任何操作。

### 5.3 Decode 调度的完整流程图

```
开始 decode 调度
    │
    ▼
从 running 队列头部取序列 seq
    │
    ▼
can_append(seq)?
    │
    ├── Yes ──→ may_append(seq) → 加入 scheduled_seqs → 取下一个序列
    │
    └── No ──→ running 中有其他序列？
                    │
                    ├── Yes ──→ preempt(running.pop()) → 重新检查 can_append
                    │
                    └── No ──→ preempt(seq) → break（本步无法调度此序列）
```

---

## 六、postprocess 后处理

### 6.1 源码解读

```python
def postprocess(self, seqs, token_ids):
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)
        
        if (not seq.ignore_eos and token_id == self.eos) or \
           seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

### 6.2 处理流程

对于本步参与推理的每个序列：

1. **追加新 token**：调用 `seq.append_token(token_id)` 更新序列状态
2. **检查终止条件**：
   - 条件 1：遇到 EOS token 且未设置 `ignore_eos`
   - 条件 2：已生成的 completion token 数达到 `max_tokens` 上限
3. **如果终止**：
   - 标记状态为 `FINISHED`
   - 释放该序列的所有 KV Cache block
   - 从 running 队列中移除

### 6.3 终止条件的逻辑表达式

```python
(not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens
```

用真值表分析：

| ignore_eos | token == EOS | completion == max | 是否终止 |
|-----------|-------------|------------------|---------|
| False | True | - | **是**（自然结束） |
| False | False | True | **是**（达到上限） |
| False | False | False | 否 |
| True | True | False | 否（忽略了 EOS） |
| True | True | True | **是**（达到上限） |
| True | False | True | **是**（达到上限） |
| True | False | False | 否 |

核心逻辑：`max_tokens` 是硬性上限，无论如何不能超过；`EOS` 是软性终止，可以被 `ignore_eos` 覆盖。

### 6.4 为什么用 `self.running.remove(seq)` 而非 `popleft`

因为完成的序列不一定在队列头部——批次中任何位置的序列都可能先完成。`remove()` 根据值查找并删除，时间复杂度 O(n)，但由于 running 队列通常很短（几十个序列），这不是性能瓶颈。

### 6.5 postprocess 不处理 waiting 队列中的序列

`postprocess` 只处理本步参与推理的序列（`seqs` 参数），不会触及 waiting 队列。这是因为 waiting 中的序列没有参与推理，不会产生新 token。

---

## 七、add 方法

### 7.1 源码

```python
def add(self, seq: Sequence):
    self.waiting.append(seq)
```

极其简单：将新序列加入 waiting 队列尾部。FIFO 顺序保证先到的请求先被处理。

### 7.2 add 的调用时机

```python
# LLMEngine 中
def add_request(self, prompt, sampling_params=SamplingParams()):
    token_ids = self.tokenizer.encode(prompt)
    seq = Sequence(token_ids, sampling_params)
    self.scheduler.add(seq)
    return seq
```

### 7.3 is_running / has_unfinished 属性

```python
@property
def is_running(self):
    return bool(self.running)

@property
def has_unfinished(self):
    return bool(self.waiting) or bool(self.running)
```

- `is_running`：是否有正在执行的序列（用于 `LLMEngine` 判断是否需要继续步进）
- `has_unfinished`：是否有未完成的序列（包括等待和运行中的）

---

## 八、调度器的完整生命周期示例

### 8.1 场景设置

假设系统参数：
- `max_num_seqs = 4`
- `max_num_batched_tokens = 1024`
- `block_size = 256`
- 共有 10 个物理 block

### 8.2 执行时间线

```
T=0: 请求 A 到达 (prompt 300 token)
     waiting: [A(300)]
     running: []

T=1: schedule() — prefill 阶段
     A 需要 2 个 block，分配成功
     waiting: []
     running: [A]
     → 返回 ([A], is_prefill=True)
     → ModelRunner 执行 A 的 prefill

T=1: postprocess()
     A 生成 token_301
     A 未结束
     running: [A(301)]

T=2: 请求 B 到达 (prompt 500 token)
     waiting: [B(500)]
     running: [A(301)]

T=2: schedule() — prefill 优先
     B 需要 2 个 block，分配成功
     waiting: []
     running: [A(301), B(500)]
     → 返回 ([B], is_prefill=True)

T=2: postprocess()
     B 生成 token_501
     running: [A(301), B(501)]

T=3: schedule() — 无 waiting，进入 decode
     A: can_append? Yes (block 未满)
     B: can_append? Yes (block 未满)
     → 返回 ([A, B], is_prefill=False)

T=3: postprocess()
     A 生成 token_302, B 生成 token_502
     running: [A(302), B(502)]

... 正常 decode ...

T=10: A 遇到 EOS
      postprocess: A.status = FINISHED, 释放 2 个 block
      running: [B(510)]

T=11: 请求 C 到达
      waiting: [C]
      schedule(): prefill 优先，调度 C
```

---

## 九、调度策略的深度对比

### 9.1 FCFS（先来先服务）

nano-vllm 的 waiting 队列本质上是 FCFS——先到的请求先被调度。

**优点**：简单、公平
**缺点**：不能区分请求优先级，长 prompt 可能阻塞短 prompt

### 9.2 Prefill 优先 vs Decode 优先

| 策略 | TTFT | TPOT | 吞吐量 | 实现复杂度 |
|------|------|------|--------|-----------|
| Prefill 优先 | 低 | 可能较高 | 中 | 低 |
| Decode 优先 | 高 | 低 | 中 | 低 |
| 混合调度 | 中 | 中 | 高 | 高 |

- **TTFT**（Time To First Token）：用户等待第一个输出 token 的时间
- **TPOT**（Time Per Output Token）：每个输出 token 的生成时间

nano-vllm 选择 prefill 优先是因为 TTFT 对用户体验影响更大——用户更在意"什么时候开始有回复"而非"回复速度有多快"。

### 9.3 chunked prefill

在更先进的系统中，长 prompt 的 prefill 可以被分成多个 chunk，与 decode 序列交错执行。这样既不会因为长 prompt 阻塞 decode 序列，又能保持较低的 TTFT。

```
传统 prefill 优先:
Step 1: [A(prefill 2000 tokens)]  ← decode 序列被阻塞
Step 2: [B(decode), C(decode)]

chunked prefill:
Step 1: [A(prefill chunk1 512 tokens), B(decode), C(decode)]
Step 2: [A(prefill chunk2 512 tokens), B(decode), C(decode)]
Step 3: [A(prefill chunk3 512 tokens), B(decode), C(decode)]
Step 4: [A(prefill chunk4 464 tokens), B(decode), C(decode)]
```

nano-vllm 的简化版本没有实现 chunked prefill。

### 9.4 Priority scheduling

在生产环境中，不同用户/请求可能有不同优先级（如付费用户 > 免费用户）。这需要在 waiting 队列中实现优先队列（如 heap），而非简单的 FIFO。

---

## 十、BlockManager 交互

### 10.1 调度器与 BlockManager 的协作

```python
# 调度器不直接管理物理 block，而是委托给 BlockManager

# Prefill 时：
self.block_manager.can_allocate(seq)   # 询问：有足够 block 给这个序列吗？
self.block_manager.allocate(seq)       # 执行：分配 block 并填充 seq.block_table

# Decode 时：
self.block_manager.can_append(seq)     # 询问：能追加一个 token 吗？
self.block_manager.may_append(seq)     # 执行：如果需要，分配新 block

# 抢占/完成时：
self.block_manager.deallocate(seq)     # 执行：释放该序列的所有 block
```

### 10.2 资源管理的两阶段检查

nano-vllm 采用**先检查后执行**的模式：

1. `can_xxx` 方法：只读查询，不修改状态
2. `allocate/deallocate/may_append` 方法：实际执行资源操作

这种设计允许调度器在决策阶段安全地"试探"资源状况，而不会因为检查操作产生副作用。

---

## 十一、调度器的核心设计原则

### 11.1 Prefill 和 Decode 不混合

在同一步中，要么全做 prefill，要么全做 decode。这简化了 ModelRunner 的实现——不需要在同一个 batch 中混合两种不同的计算模式。

```python
if scheduled_seqs:
    return scheduled_seqs, True   # 有 prefill，本步只做 prefill
# ...
return scheduled_seqs, False      # 否则做 decode
```

但这也意味着 decode 中的序列在有新 prefill 请求到来时会**暂停一步**。

### 11.2 保守调度

调度器倾向于保守——宁可少调度几个序列，也不要因为资源不足导致系统崩溃：

- token 数检查：`num_batched_tokens + len(seq) > self.max_num_batched_tokens` → break
- block 检查：`not self.block_manager.can_allocate(seq)` → break

一旦遇到无法调度的序列，立即停止调度后续序列，即使后续序列可能更小。这是 FCFS 的特性——不会跳过队头的大请求去调度后面的小请求。

### 11.3 单步原子性

每次 `schedule()` 调用产生一个完整的调度结果。调度过程中的所有操作（分配 block、修改状态、移动队列）要么全部成功，要么需要回滚。在 nano-vllm 中，由于是单线程执行，不会出现并发问题。

---

## 十二、调度器的潜在改进

### 12.1 支持 Swap（交换到 CPU）

当前的 preempt 策略是**全部释放**（recompute），被抢占的序列需要重新做 prefill。改进方案是将 KV Cache 从 GPU 交换到 CPU 内存，后续恢复时只需从 CPU 传回 GPU，避免重新计算。

### 12.2 支持优先级调度

将 waiting 队列从 deque 改为优先队列，支持基于优先级或等待时间的调度。

### 12.3 支持 Chunked Prefill

将长 prompt 的 prefill 分块，与 decode 交错执行，平衡 TTFT 和 TPOT。

### 12.4 支持 Speculative Decoding

投机解码需要调度器支持"草稿模型 + 验证"的两阶段执行模式。

---

## 十三、源码对照总结

| Scheduler 方法/属性 | 调用者 | 目的 |
|---------------------|--------|------|
| `__init__` | LLMEngine | 初始化调度器和 BlockManager |
| `add(seq)` | LLMEngine.add_request | 新序列入队 |
| `schedule()` | LLMEngine.step | 选出本步序列，返回 (seqs, is_prefill) |
| `postprocess(seqs, token_ids)` | LLMEngine.step | 追加 token，判断终止 |
| `preempt(seq)` | schedule() 内部 | 抢占序列释放资源 |
| `is_running` | LLMEngine | 判断是否有活跃序列 |
| `has_unfinished` | LLMEngine | 判断是否所有任务完成 |

---

## 十四、面试考点

### 考点 1：请描述 nano-vllm 调度器的 schedule() 方法的完整流程

**标准回答**：schedule() 分两个阶段执行。第一阶段尝试从 waiting 队列调度 prefill 序列：按 FCFS 顺序逐个取出，检查 token 数限制和 block 可用性，通过检查则分配 block、修改状态为 RUNNING、移入 running 队列。如果成功调度了 prefill 序列，直接返回。第二阶段处理 decode：从 running 队列逐个取出，检查是否能追加新 token（可能需要新 block），如果 block 不足则通过 LIFO 策略抢占其他序列释放资源。

### 考点 2：为什么采用 prefill 优先策略？这种策略有什么优缺点？

**标准回答**：Prefill 优先降低了 TTFT（首 token 延迟），用户体验更好。Prefill 是 compute-bound 操作，可以充分利用 GPU 算力。缺点是正在 decode 的序列在有新 prefill 请求时会暂停一步，增加了 TPOT。改进方案是 chunked prefill，将长 prompt 分块与 decode 交错执行。

### 考点 3：描述抢占（preempt）机制的实现，为什么用 LIFO 策略？

**标准回答**：当 decode 阶段需要新 block 但没有空闲 block 时，调度器从 running 队列尾部取出序列进行抢占——释放其所有 KV Cache block，将其状态改回 WAITING，放到 waiting 队列头部。使用 LIFO 策略是因为最后加入的序列生成的 token 最少，被抢占浪费的计算量最小。被抢占序列放到 waiting 头部是为了保证公平性，避免饥饿。

### 考点 4：nano-vllm 的抢占策略是 recompute，与 swap 策略有什么区别？

**标准回答**：Recompute 策略直接丢弃被抢占序列的 KV Cache，重新调度时需要重新做 prefill，计算浪费大但实现简单、不需要额外内存。Swap 策略将 KV Cache 从 GPU 交换到 CPU 内存，恢复时只需传回 GPU，避免重复计算，但需要额外的 CPU 内存和 PCIe 带宽，实现也更复杂。vLLM 同时支持两种策略。

### 考点 5：postprocess 的终止条件有哪些？如何处理 ignore_eos？

**标准回答**：终止条件有两个：(1) 生成了 EOS token 且 `ignore_eos=False`，(2) 已生成的 completion token 数达到 `max_tokens` 限制。`ignore_eos=True` 时会跳过 EOS 检查，强制生成到 `max_tokens`。`max_tokens` 是硬性上限，无论是否设置 `ignore_eos` 都会生效。

### 考点 6：如果让你改进 nano-vllm 的调度器，你会从哪些方面入手？

**参考思路**：
1. 实现 chunked prefill，平衡 TTFT 和 TPOT
2. 添加 swap 策略减少抢占浪费
3. 支持优先级调度（priority queue）
4. 支持 prefix-aware 调度，共享前缀的请求一起调度以最大化缓存命中
5. 支持投机解码的两阶段调度
6. 添加公平性保障（基于等待时间的优先级提升）

### 考点 7：为什么 prefill 和 decode 不在同一步混合执行？

**标准回答**：Prefill 和 decode 的计算模式不同——prefill 处理多个连续 token，使用的是变长序列的注意力计算；decode 每个序列只处理一个 token，使用的是 KV Cache 加速的注意力计算。不混合执行简化了 ModelRunner 的实现和 CUDA kernel 的选择。实际上高性能系统如 vLLM 已支持混合执行以提高 GPU 利用率。

---

## 十五、小结

| 知识点 | 核心理解 |
|--------|---------|
| 调度器角色 | 引擎的"大脑"，决定每一步谁参与计算 |
| 双队列模型 | waiting（等待） + running（运行），用 deque 实现 |
| Prefill 优先 | 新请求优先处理，降低 TTFT |
| Decode 轮转 | FIFO 依次处理 running 中的序列 |
| 抢占机制 | LIFO 策略 + recompute 策略，被抢占者回到 waiting 头部 |
| postprocess | 追加 token + 终止检查 + 资源回收 |
| 资源管理 | 调度器不直接管理 block，委托给 BlockManager |

**下一课预告**：我们将从调度器的视角上升到系统层面，深入理解**连续批处理（Continuous Batching）**——它是调度器和 ModelRunner 配合实现的核心优化策略。

---

> **学习建议**：在纸上模拟一个有 3-4 个请求的调度场景，画出每一步 waiting 和 running 队列的变化，以及 block 的分配和释放过程。这对理解调度器至关重要。
