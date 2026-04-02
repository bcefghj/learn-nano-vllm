# 第18课：LLMEngine 推理循环

面向初学者的说明：本课详解 nano-vllm 的 **核心引擎 `LLMEngine`**，包括初始化、多进程启动、请求添加、推理循环（step）、批量生成（generate），以及权重加载和采样器的实现原理。每节配有源码逐行解析与面试考点。

---

## 一、概念讲解：LLMEngine 的角色

### 1.1 LLMEngine 在架构中的位置

nano-vllm 的整体架构可以抽象为三层：

```
用户接口层：  LLMEngine.generate() / add_request()
     ↓
调度执行层：  Scheduler → ModelRunner → Model
     ↓
底层资源层：  KV Cache (BlockManager) / CUDA Graph / Attention Kernel
```

`LLMEngine` 是 **用户与推理系统之间的桥梁**，它负责：

1. **初始化**：加载模型权重、创建 ModelRunner、创建 Scheduler、加载 Tokenizer。
2. **请求管理**：接收用户的文本/token 输入，包装为 `Sequence` 对象。
3. **推理循环**：反复调用 `step()` 驱动 Scheduler → ModelRunner → 后处理。
4. **结果收集**：检测已完成的序列，返回生成结果。

### 1.2 与 vLLM 的对比

| 特性 | nano-vllm LLMEngine | vLLM LLMEngine |
|------|---------------------|----------------|
| 代码量 | ~200 行 | ~5000 行 |
| 异步支持 | 无（同步循环） | 有（AsyncLLMEngine） |
| API 服务 | 无 | 集成 OpenAI API 服务器 |
| 流式输出 | 简单实现 | 完整 SSE 流式 |
| 多模态 | 不支持 | 支持图像/视频输入 |
| LoRA | 不支持 | 支持动态 LoRA |

nano-vllm 的简洁设计使其成为学习 LLM 推理引擎的最佳入门教材。

---

## 二、源码对照：`LLMEngine.__init__` 初始化流程

下面与仓库 `nanovllm/engine/llm_engine.py` 一致（含 `Config` 字段过滤、`atexit` 清理子进程）。文件头部另含 `from dataclasses import fields`、`import atexit` 等导入。

```python
class LLMEngine:
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)
```

### 2.1 Config 配置对象

`Config` 汇总所有推理参数，核心字段包括：

| 字段 | 默认值 | 含义 |
|------|--------|------|
| `model` | — | 模型路径或 HuggingFace 模型名 |
| `max_num_seqs` | 256 | 最大并发序列数 |
| `max_model_len` | — | 模型支持的最大序列长度 |
| `tensor_parallel_size` | 1 | 张量并行 GPU 数量 |
| `enforce_eager` | False | 是否禁用 CUDA Graph |
| `enable_prefix_caching` | False | 是否启用前缀缓存 |
| `block_size` | 16 | KV Cache 每个 block 的 token 数 |
| `gpu_memory_utilization` | 0.9 | GPU 显存使用比例上限 |

### 2.2 多进程启动（Tensor Parallel）

```python
for i in range(1, config.tensor_parallel_size):
    event = ctx.Event()
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()
```

当 `tensor_parallel_size > 1` 时（例如使用 2 张或更多 GPU），需要启动多个 worker 进程：

**执行流程详解：**

1. `ctx = multiprocessing.get_context("spawn")`：使用 `spawn` 方式创建子进程（比 `fork` 更安全，避免 CUDA 上下文继承问题）。
2. 循环从 `i=1` 开始（`i=0` 是主进程自己的 rank）。
3. 为每个子进程创建一个 `Event` 对象，用于 **主进程与子进程之间的同步**。
4. `ctx.Process(target=ModelRunner, ...)` 创建子进程，入口函数是 `ModelRunner.__init__`。
5. `process.start()` 启动子进程。

**同步机制：**

- 主进程（rank=0）在调用 `model_runner.call("run", ...)` 时，会通过 `Event` 通知各子进程执行相同的前向操作。
- 各子进程执行完后，再通过 `Event` 通知主进程收集结果。
- 这种设计保证了张量并行中 **所有 GPU 同步执行** 相同的前向步骤。

**为什么用多进程而非多线程？**

- Python 的 **GIL（全局解释器锁）** 限制了多线程的并行度。
- 每个 GPU 需要独立的 **CUDA 上下文**，多进程天然隔离。
- `torch.distributed` 的 NCCL 后端在多进程架构下工作最佳。

### 2.3 `atexit` 与子进程清理

```python
atexit.register(self.exit)
```

进程退出时会调用 `exit()`：`model_runner.call("exit")` 通知各 rank 释放资源并 `join` 子进程，避免僵尸进程或 GPU 句柄泄漏。面试时可一句话带过：**spawn 子进程必须配对 join**。

### 2.4 主进程的 ModelRunner

```python
self.model_runner = ModelRunner(config, 0, self.events)
```

主进程（rank=0）自己也创建一个 `ModelRunner`，负责：

- 加载模型权重到第 0 张 GPU。
- 分配 KV Cache。
- capture CUDA Graph（如果启用）。
- 作为前向计算的 **协调者**：发起计算、收集结果。

### 2.5 Tokenizer 与 `eos`

```python
self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
config.eos = self.tokenizer.eos_token_id
```

使用 HuggingFace `transformers` 的 `AutoTokenizer`：

- 自动识别模型类型并加载对应的 tokenizer。
- 支持 `encode`（文本 → token ID 列表）和 `decode`（token ID → 文本）。
- 注意：tokenizer 只在 **主进程** 加载，子进程不需要。

### 2.6 Scheduler 创建

```python
self.scheduler = Scheduler(config)
```

Scheduler（调度器）负责管理请求的生命周期：

- **等待队列**（waiting）：新添加的请求。
- **运行队列**（running）：正在进行推理的请求。
- **调度决策**：每步决定哪些序列参与 prefill、哪些参与 decode。
- **KV Cache 管理**：通过 BlockManager 分配和释放 KV block。

（调度器的详细实现见之前的调度课程）

### 2.7 全局 `Context`（`context.py`）：Attention 的「隐式参数」

`ModelRunner` 在 prefill / decode 准备张量时，会调用 `set_context(...)` 把 **cu_seqlens、slot_mapping、context_lens、block_tables** 等写入进程级全局变量；模型前向里的 Attention 通过 `get_context()` 读取。这样 **无需在每一层函数参数里层层传递** 这些张量，代码更短，但代价是 **单进程内同时只能跑一条调度路径**（教学项目可接受）。

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

与 CUDA Graph 的关系（第16课）：`capture_cudagraph` 里 `set_context(False, slot_mapping=..., ...)` 与 `run_model` replay 前写入 `graph_vars` 的 **是同一份语义**——保证图里录制的 kernel 在 replay 时仍能读到正确的 KV 元数据。

---

## 三、源码对照：`add_request` 方法

```python
def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
    if isinstance(prompt, str):
        prompt = self.tokenizer.encode(prompt)
    seq = Sequence(prompt, sampling_params)
    self.scheduler.add(seq)
```

### 3.1 输入格式灵活性

`add_request` 同时支持两种输入：

| 输入类型 | 示例 | 处理方式 |
|---------|------|---------|
| `str` | `"Hello, world"` | 先 `tokenizer.encode()` 转 token ID 列表 |
| `List[int]` | `[1, 2, 3, 4]` | 直接使用 |

支持 `List[int]` 的好处：在 benchmark 测试中可以直接用随机 token ID，跳过 tokenize 开销。

### 3.2 Sequence 对象

`Sequence` 是 nano-vllm 中表示一个推理请求的核心数据结构：

```python
class Sequence:
    def __init__(self, prompt_token_ids, sampling_params):
        self.seq_id = next_id()              # 全局唯一 ID
        self.prompt_token_ids = prompt_token_ids  # 原始 prompt 的 token ID
        self.completion_token_ids = []        # 已生成的 token ID（逐步追加）
        self.sampling_params = sampling_params # 采样参数
        self.logical_blocks = []             # 分配的 KV block 编号
        self.is_finished = False             # 是否已完成
```

每个 Sequence 在生命周期中经历：**等待 → prefill → decode（循环）→ 完成**。

### 3.3 SamplingParams 采样参数

```python
@dataclass
class SamplingParams:
    temperature: float = 1.0     # 温度，控制随机性
    max_tokens: int = 256        # 最大生成 token 数
    ignore_eos: bool = False     # 是否忽略 EOS token
```

- `temperature > 0`：使用概率采样，值越大越随机。
- `temperature = 0`：退化为 greedy decoding（取概率最大的 token）。
- `ignore_eos = True`：即使生成了 EOS token 也继续生成，用于 benchmark。

---

## 四、源码对照：`step` 方法——推理循环的核心

与源码一致的完整实现如下（注意 **`num_tokens` 的符号约定**，供 `generate` 里统计 prefill/decode 吞吐量）：

```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids)
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
    return outputs, num_tokens
```

### 4.1 第一步：调度 `scheduler.schedule()`

调度器根据当前状态决定本步的执行策略：

**返回值**：
- `seqs`：参与本步计算的序列列表。
- `is_prefill`：布尔值，表示本步是 prefill 还是 decode。

**调度策略**：
1. **优先 prefill**：如果等待队列有新请求且资源允许，先进行 prefill。
2. **然后 decode**：如果所有活跃序列都已 prefill，执行 decode。
3. **资源约束**：受 `max_num_seqs`、可用 KV block 数量限制。

**Prefill vs Decode 的区别**：

| 维度 | Prefill | Decode |
|------|---------|--------|
| 每序列 token 数 | prompt 长度（可达数千） | 1 |
| KV Cache 操作 | 批量写入 | 追加 1 个 slot |
| Attention 类型 | Flash Attention (变长) | Flash Attention with KV Cache |
| CUDA Graph | 不使用 | 使用（如果可用） |

### 4.2 第二步：执行 `model_runner.call("run", seqs, is_prefill)`

`call` 方法封装了多进程通信：

1. 主进程将 `seqs` 和 `is_prefill` 序列化，通过共享内存传递给各子进程。
2. 通过 `Event` 通知所有子进程开始执行。
3. 每个进程（包括主进程）调用 `ModelRunner.run(seqs, is_prefill)`。
4. `run` 方法内部：
   - 从 `seqs` 中提取 `input_ids`、`positions`、`slot_mapping` 等。
   - 调用 `run_model(input_ids, positions, is_prefill)` 执行前向。
   - 对 logits 做采样，得到下一步的 token ID。
5. 主进程等待所有子进程完成，收集结果。

### 4.3 第三步：后处理 `scheduler.postprocess(seqs, token_ids)`

后处理的核心逻辑：

```python
def postprocess(self, seqs, token_ids):
    for seq, token_id in zip(seqs, token_ids):
        # 将新 token 追加到序列
        seq.completion_token_ids.append(token_id)

        # 检查是否应该停止
        if self._should_stop(seq, token_id):
            seq.is_finished = True
            self._free_blocks(seq)  # 释放 KV Cache blocks
```

**停止条件**：
- 生成了 EOS token（且 `ignore_eos=False`）。
- 达到 `max_tokens` 上限。
- 达到模型的 `max_model_len`。

**资源释放**：序列完成后，通过 `BlockManager.free()` 释放其占用的 KV block，供其他请求使用。

### 4.4 第四步：收集输出

```python
outputs = [(seq.seq_id, seq.completion_token_ids)
           for seq in seqs if seq.is_finished]
```

只返回 **本步新完成的** 序列。已经在之前步骤完成的序列不会再出现。

### 4.5 `num_tokens`：吞吐量统计的「编码」

```python
num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
```

| `is_prefill` | 含义 | `num_tokens` |
|--------------|------|----------------|
| `True` | 本步处理 prompt（可能多 token） | **正数** = 本批所有序列当前长度之和（与 prefill 处理的 token 规模相关） |
| `False` | 本步 decode（每序列 1 token） | **负数** = `-batch_size`，用绝对值即本步新生成的 token 数 |

在 `generate` 里配合 `perf_counter()`：**prefill 步**用 `num_tokens / Δt` 得到 **Prefill tok/s**；**decode 步**用 `-num_tokens / Δt`（负负得正）得到 **Decode tok/s**。这是用 **一个标量** 同时区分阶段并携带统计信息的紧凑写法。

---

## 五、源码对照：`generate` 方法——批量推理入口

```python
def generate(self, prompts, sampling_params, use_tqdm=True):
    if use_tqdm:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
    if not isinstance(sampling_params, list):
        sampling_params = [sampling_params] * len(prompts)
    for prompt, sp in zip(prompts, sampling_params):
        self.add_request(prompt, sp)
    outputs = {}
    prefill_throughput = decode_throughput = 0.
    while not self.is_finished():
        t = perf_counter()
        output, num_tokens = self.step()
        if use_tqdm:
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids
            if use_tqdm:
                pbar.update(1)
    outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
    outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
    if use_tqdm:
        pbar.close()
    return outputs
```

### 5.1 完整的生命周期

```
generate() 调用
    ├── add_request(prompt_1, sp_1)  →  Sequence_1 进入 waiting 队列
    ├── add_request(prompt_2, sp_2)  →  Sequence_2 进入 waiting 队列
    └── ...
    │
    ├── step() #1:  schedule → prefill(Seq_1) → postprocess
    ├── step() #2:  schedule → prefill(Seq_2) → postprocess
    ├── step() #3:  schedule → decode([Seq_1, Seq_2]) → postprocess
    ├── step() #4:  schedule → decode([Seq_1, Seq_2]) → postprocess
    │   ...         （Seq_1 完成，释放 KV blocks）
    ├── step() #N:  schedule → decode([Seq_2]) → postprocess
    │   ...         （Seq_2 完成）
    └── is_finished() == True → 返回所有结果
```

### 5.2 `is_finished` 判断

```python
def is_finished(self):
    return self.scheduler.is_finished()
```

当调度器判定 **没有待处理请求** 时返回 `True`，循环结束。（具体条件见 `scheduler.py` 中的 `is_finished` 实现。）

### 5.3 吞吐量统计（与源码一致）

当 `use_tqdm=True` 时，**每一步** `step()` 后根据 `num_tokens` 符号更新进度条上的 **Prefill / Decode** 字符串（见上一节公式）。这不是滑动平均，而是 **当前步的瞬时吞吐**，便于观察调度在 prefill-heavy 与 decode-heavy 之间切换时的波动。

| 指标 | 源码中的实现要点 |
|------|-----------------|
| Prefill | `num_tokens > 0` 时，`prefill_throughput = num_tokens / (perf_counter() - t)` |
| Decode | 否则 `decode_throughput = -num_tokens / (perf_counter() - t)`（`num_tokens` 为负，取负得正） |

**Event 同步**：`tensor_parallel_size > 1` 时，`ModelRunner.call` 内部用 `multiprocessing.Event` 让 rank0 与各 worker **同一步执行 `run`**，保证分布式张量并行时各卡前向对齐。面试回答：**Event 是跨进程栅栏，不是 CUDA 设备同步**。

---

## 六、源码对照：权重加载（loader.py）

### 6.1 为什么需要自定义权重加载

HuggingFace 模型的权重命名与 nano-vllm 内部的参数命名可能不同。特别是 **packed modules**（例如 QKV 被合并为一个线性层）需要特殊处理。

### 6.2 `load_model` 完整解析

```python
def load_model(model, path):
    # 获取模型定义的 packed 映射关系
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # 遍历所有 safetensors 权重文件
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 检查是否属于 packed module
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 普通权重：直接加载
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
```

### 6.3 关键概念解析

#### Safetensors 格式

`safetensors` 是 HuggingFace 推出的安全权重文件格式：

| 特性 | safetensors | pickle (torch.save) |
|------|------------|---------------------|
| 安全性 | 不能执行任意代码 | 可被注入恶意代码 |
| 加载速度 | 支持 mmap，极快 | 需要反序列化 |
| 随机访问 | 支持按 key 读取单个张量 | 需要加载整个文件 |

#### Packed Modules 映射

以 Qwen2 模型为例，HuggingFace 权重中 Q、K、V 是分开的三个矩阵：

```
model.layers.0.self_attn.q_proj.weight  → shape [hidden_size, hidden_size]
model.layers.0.self_attn.k_proj.weight  → shape [kv_hidden_size, hidden_size]
model.layers.0.self_attn.v_proj.weight  → shape [kv_hidden_size, hidden_size]
```

但 nano-vllm 为了效率，将它们合并为一个 `qkv_proj`：

```
model.layers.0.self_attn.qkv_proj.weight  → shape [hidden_size + 2*kv_hidden_size, hidden_size]
```

`packed_modules_mapping` 定义了这种合并关系：

```python
packed_modules_mapping = {
    "q_proj": ("qkv_proj", 0),   # Q 放在 shard 0
    "k_proj": ("qkv_proj", 1),   # K 放在 shard 1
    "v_proj": ("qkv_proj", 2),   # V 放在 shard 2
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
```

#### `weight_loader` 回调

每个参数可以自定义 `weight_loader` 函数，处理权重的切片和放置：

```python
def weight_loader(param, loaded_weight, shard_id):
    # 根据 shard_id 确定在合并参数中的位置
    # 将 loaded_weight 复制到 param 的对应切片
    shard_size = loaded_weight.shape[0]
    param.data[shard_id * shard_size : (shard_id + 1) * shard_size] = loaded_weight
```

#### `default_weight_loader`

对于非 packed 的普通参数，直接整体复制：

```python
def default_weight_loader(param, loaded_weight):
    param.data.copy_(loaded_weight)
```

### 6.4 张量并行下的权重加载

当 `tensor_parallel_size > 1` 时，权重加载还需要考虑 **切分**：

- **列并行（Column Parallel）**：QKV 投影、FFN 的 gate/up 投影按 **输出维度** 切分。
- **行并行（Row Parallel）**：Output 投影、FFN 的 down 投影按 **输入维度** 切分。
- 每个 rank 只加载属于自己的那一份权重。

---

## 七、源码对照：Sampler 采样器

### 7.1 采样在推理中的作用

模型前向输出的是 **logits**（未归一化的对数概率），需要经过 **采样** 得到下一个 token。

### 7.2 Sampler 源码解析

```python
class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits, temperatures):
        # ① 温度缩放
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        # ② 计算概率分布
        probs = torch.softmax(logits, dim=-1)

        # ③ Gumbel-max 采样
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)

        return sample_tokens
```

### 7.3 逐行解析

#### 温度缩放

```python
logits = logits.float().div_(temperatures.unsqueeze(dim=1))
```

- `.float()` 转为 FP32，避免 FP16 下 softmax 溢出。
- `temperatures.unsqueeze(dim=1)` 将 `[batch]` 扩展为 `[batch, 1]`，支持广播。
- `.div_()` 是 in-place 除法，节省内存。
- 温度 \(T\) 的效果：logits 除以 \(T\) 后再 softmax。\(T \to 0\) 趋向 greedy，\(T \to \infty\) 趋向均匀分布。

#### Gumbel-max 技巧

```python
sample_tokens = probs.div_(
    torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
).argmax(dim=-1)
```

这是一种 **高效的分类分布采样方法**，等价于 `torch.multinomial` 但更适合 GPU 并行。

**数学原理**：

Gumbel-max 定理指出：如果 \(G_i \sim \text{Gumbel}(0, 1)\) 是独立同分布的 Gumbel 随机变量，则：

\[
\arg\max_i (\log p_i + G_i) \sim \text{Categorical}(p_1, p_2, \ldots, p_V)
\]

其中 \(p_i\) 是各类别的概率。

**代码中的等价变换**：

1. `exponential_(1)` 生成指数分布随机数 \(E_i \sim \text{Exp}(1)\)。
2. Gumbel 随机变量可以通过 \(G_i = -\log(E_i)\) 生成。
3. \(\arg\max(\log p_i - \log E_i) = \arg\max(p_i / E_i)\)。
4. 因此 `probs / exponential` 然后取 `argmax` 等价于从 `probs` 中按概率采样。
5. `.clamp_min_(1e-10)` 防止除以零。

**为什么不直接用 `torch.multinomial`？**

- `torch.multinomial` 内部需要计算 CDF（累积分布函数），对于大词表（如 150K+）这是 \(O(V)\) 的串行操作。
- Gumbel-max 只需要 **逐元素** 的除法和 argmax，天然适合 GPU 的 SIMD 并行。
- `@torch.compile` 可以进一步将这些操作融合为单个 kernel。

### 7.4 `@torch.compile` 加速

```python
@torch.compile
def forward(self, logits, temperatures):
    ...
```

`torch.compile` 的作用：

1. **算子融合（Operator Fusion）**：将 `.float()` → `.div_()` → `softmax` → `.div_()` → `.argmax()` 融合为更少的 kernel 调用。
2. **内存优化**：减少中间张量的分配和释放。
3. **后端优化**：PyTorch 的 Inductor 后端生成优化的 Triton kernel。

典型加速效果：采样步骤可以加速 2-3 倍。

---

## 八、完整调用链梳理

当用户调用 `llm.generate(["Hello"], [SamplingParams()])` 时：

```
generate()
  ├── add_request("Hello", SamplingParams())
  │     ├── tokenizer.encode("Hello") → [15496]
  │     ├── Sequence([15496], SamplingParams())
  │     └── scheduler.add(seq)
  │
  └── while not is_finished():
        └── step()
              ├── scheduler.schedule()
              │     └── 返回 (seqs, is_prefill=True)  ← 第一步是 prefill
              │
              ├── model_runner.call("run", seqs, True)
              │     ├── 准备 input_ids=[15496], positions=[0]
              │     ├── 分配 slot_mapping
              │     ├── run_model(input_ids, positions, is_prefill=True)
              │     │     └── model(input_ids, positions) → hidden_states
              │     │         └── compute_logits(hidden_states) → logits
              │     └── sampler(logits, temperatures) → token_id
              │
              ├── scheduler.postprocess(seqs, [token_id])
              │     ├── seq.completion_token_ids.append(token_id)
              │     └── 检查停止条件
              │
              └── 后续 step：is_prefill=False，走 decode 路径
                    └── CUDA Graph replay（如果可用）
```

---

## 九、小结

- **LLMEngine** 是 nano-vllm 的核心引擎，串联 Tokenizer、Scheduler、ModelRunner。
- **初始化** 按照"配置 → 多进程 → ModelRunner → Tokenizer → Scheduler"的顺序执行。
- **多进程** 使用 `spawn` + `Event` 实现张量并行的进程同步。
- **`add_request`** 支持字符串和 token ID 列表两种输入，包装为 Sequence 加入调度器。
- **`step`** 是推理循环的最小单元：调度 → 执行 → 后处理 → 收集输出。
- **`generate`** 是批量推理的入口：添加所有请求后反复 step 直到全部完成。
- **权重加载** 处理 packed modules 的映射和切片，支持 safetensors 格式。
- **Sampler** 使用 Gumbel-max 技巧替代 `torch.multinomial`，配合 `@torch.compile` 高效采样。

---

## 十、面试考点（含参考答案）

**1. LLMEngine 的 `step()` 方法做了什么？**
**答**：每次 `step` 完成一轮推理循环：① `scheduler.schedule()` 决定参与计算的序列和是 prefill 还是 decode；② `model_runner.call("run", ...)` 执行前向计算得到 token ID；③ `scheduler.postprocess()` 将新 token 追加到序列并检查停止条件；④ 收集已完成的序列输出。

**2. 为什么多进程用 `spawn` 而非 `fork`？**
**答**：`fork` 会复制父进程的 CUDA 上下文，导致子进程无法正确初始化自己的 GPU。`spawn` 创建全新的进程，每个进程独立初始化 CUDA 上下文，与 NCCL 等多 GPU 通信库的要求一致。

**3. `generate` 和 `step` 的关系是什么？**
**答**：`generate` 是面向用户的高层接口，内部循环调用 `step` 直到所有请求完成。`step` 是推理循环的最小执行单元，每次处理一个调度批次。可以类比为 `generate` 是"整场考试"，`step` 是"做一道题"。

**4. Packed modules mapping 解决什么问题？**
**答**：HuggingFace 模型将 Q、K、V 存为三个独立权重矩阵，但 nano-vllm 为了计算效率将它们合并为一个 `qkv_proj`。`packed_modules_mapping` 定义了映射关系，权重加载时将分开的矩阵正确拼接到合并参数的对应切片中。类似的还有 FFN 的 `gate_proj` + `up_proj` → `gate_up_proj`。

**5. Gumbel-max 采样相比 `torch.multinomial` 有什么优势？**
**答**：`torch.multinomial` 需要计算 CDF 进行串行扫描，复杂度 \(O(V)\)；Gumbel-max 是逐元素除法 + argmax，天然适合 GPU 的 SIMD 并行。配合 `@torch.compile` 可以融合为单个 kernel，在大词表（150K+）场景下显著更快。

**6. Sampler 中 `temperatures.unsqueeze(dim=1)` 的作用是什么？**
**答**：`logits` shape 为 `[batch, vocab_size]`，`temperatures` shape 为 `[batch]`。`unsqueeze(dim=1)` 将其变为 `[batch, 1]`，利用 **广播机制** 实现逐行除以各自的温度值。不同序列可以使用不同的温度参数。

**7. 为什么 tokenizer 只在主进程加载？**
**答**：Tokenizer 的工作是文本与 token ID 之间的转换，属于 CPU 端的预处理和后处理。张量并行的子进程只负责 GPU 上的模型前向计算，不涉及文本处理，因此不需要加载 tokenizer，避免浪费内存。

**8. `@torch.compile` 在 Sampler 中起到什么作用？**
**答**：将多个 PyTorch 算子（类型转换、除法、softmax、指数分布采样、argmax）融合为更少的 GPU kernel，减少中间张量分配和 kernel launch 开销。PyTorch 的 Inductor 后端会生成优化的 Triton 代码，典型加速 2-3 倍。

**9. 序列的停止条件有哪些？**
**答**：三个条件满足任一即停止：① 生成了 EOS token（且 `ignore_eos=False`）；② 生成的 token 数达到 `SamplingParams.max_tokens`；③ 总序列长度（prompt + completion）达到 `Config.max_model_len`。停止后 Scheduler 释放对应的 KV Cache blocks。

**10. safetensors 格式相比 PyTorch 原生 `.pt` 格式有什么优势？**
**答**：① **安全性**：不使用 pickle，无法注入恶意代码；② **加载速度**：支持 mmap（内存映射），可以按需加载单个张量而无需反序列化整个文件；③ **随机访问**：可以按 key 读取特定权重，适合只加载部分权重的场景（如张量并行只加载自己 rank 的切片）。

---

*延伸阅读：HuggingFace safetensors 文档、PyTorch torch.compile 教程。*
