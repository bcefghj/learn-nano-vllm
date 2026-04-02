# 面试问题全集：STAR 法回答（nano-vllm）

> **使用说明**：每题采用 **S–T–A–R** 结构；**Action** 中尽量给出 `nano-vllm-main` 下**可定位的源码线索**（文件名与逻辑）。  
> **注意**：默认 `kvcache_block_size` 以仓库 `config.py` 为准（示例仓库为 **256**）；若你本地改为 16，下列「取模」类结论请同步替换为你们的 `block_size`。

---

## 一、项目整体理解（8 题）

### Q1：请介绍一下 nano-vllm 项目？

**Situation**：大模型推理工程链路长，直接阅读超大规模框架源码成本高；需要一条「模块完整、代码量可控」的学习路径。  

**Task**：在有限时间内建立对「调度—显存—计算—通信」的连贯认知，并能与工业界 vLLM 类架构对照。  

**Action**：nano-vllm 是用约千行量级 Python 组织的教学向推理引擎，入口在 `LLMEngine`：请求进入 `Scheduler` 队列，经 `step()` 驱动 `ModelRunner` 执行前向与采样。核心包括：`BlockManager` 分页 KV、`Scheduler` 连续批处理、`Attention` 中 FlashAttention 双路径 + Triton 写 KV、`ModelRunner` 中 KV 张量分配与可选 CUDA Graph。  

**Result**：能把项目定位为「**架构对齐 vLLM 思想、实现极简**」的学习载体；回答时主动说明**未覆盖**的工业特性（如完整量化、投机解码等）以示边界感。  

**源码锚点**：

```48:54:nano-vllm-main/nanovllm/engine/llm_engine.py
    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens
```

---

### Q2：nano-vllm 的整体架构是怎样的？

**Situation**：面试官希望听到「分层」而非罗列文件名。  

**Task**：用一条主线把模块串起来：**谁持有状态、谁做决策、谁做计算**。  

**Action**：  
- **API / 引擎层**：`LLMEngine` 组合 `Tokenizer`、`Scheduler`、`ModelRunner`（多进程时 rank0 调度，其他 rank 通过 SharedMemory 同步调用）。  
- **调度层**：`Scheduler` 维护 `waiting` 与 `running`，`schedule()` 产出本步参与的 `Sequence` 列表及是否 Prefill。  
- **显存层**：`BlockManager` 管理物理块池与每条序列的 `block_table`，配合前缀哈希。  
- **执行层**：`ModelRunner` 分配 `[2, L, num_blocks, block_size, kv_heads, head_dim]` 形状 KV，构造 `slot_mapping`、`cu_seqlens`、`block_tables` 等上下文，调用模型与 `Sampler`。  

**Result**：图示为「请求 → Sequence → 调度 → 上下文张量 → Attention/FFN → logits → 采样 → postprocess」。  

**源码锚点**：

```33:34:nano-vllm-main/nanovllm/engine/llm_engine.py
        self.scheduler = Scheduler(config)
```

```100:118:nano-vllm-main/nanovllm/engine/model_runner.py
    def allocate_kv_cache(self):
        ...
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
```

---

### Q3：一个请求从输入到输出经历了哪些步骤？

**Situation**：用于检验你是否理解**异步队列 + 分步 decode** 而非一次性 forward。  

**Task**：逐步说明 `add_request` 之后发生的事。  

**Action**：  
1. `add_request` 将 prompt 编码为 `token_ids`，构造 `Sequence`，`scheduler.add` 放入 `waiting`。  
2. 每一轮 `step()`：`schedule()` 选中本批序列；若走 Prefill，`prepare_prefill` 打包多序列；若走 Decode，`prepare_decode` 每序列仅 1 个新 token。  
3. `ModelRunner.run` 内 `run_model` 计算 logits，rank0 用 `Sampler` 采样得 token id。  
4. `postprocess` 把 token 追加到 `Sequence`，判断是否 EOS/达 `max_tokens`，若结束则 `deallocate`。  

**Result**：能强调：**Prefill 可能一次处理多个 token；Decode 每步每序列 1 token**，与 vLLM 一致。  

**源码锚点**：

```42:46:nano-vllm-main/nanovllm/engine/llm_engine.py
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        ...
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
```

```65:72:nano-vllm-main/nanovllm/engine/scheduler.py
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
```

---

### Q4：nano-vllm 和 vLLM 的区别？

**Situation**：避免回答「差不多」或「差很多」两种极端。  

**Task**：从**功能范围、工程化、生态**对比。  

**Action**：nano-vllm **代码量极小、结构清晰**，覆盖 PagedAttention 思想、连续批处理、FlashAttention、TP、CUDA Graph、Triton 写 KV 等主干；vLLM **功能更全**（更多模型、量化、PagedAttention 周边策略、生产级容错与调度变体等），代码量与依赖更重。  

**Result**：表达为「**学习 nano-vllm 建立骨架，再读 vLLM 补工程细节**」—体现成长路径。  

---

### Q5：为什么选择学习 nano-vllm？

**Situation**：面试官考察动机与规划。  

**Task**：给出**可验证的学习计划**，而非空泛兴趣。  

**Action**：说明 vLLM 全量源码阅读周期长；nano-vllm 能在短周期内跑通「调度 → KV → 注意力 → 采样」闭环，且与论文/博客中的概念一一对应，便于做实验与笔记。  

**Result**：可补充你已完成的产出：benchmark、注释 fork、思维导图等。  

---

### Q6：你从这个项目中学到了什么？

**Situation**：行为 + 技术综合题。  

**Task**：总结**可迁移能力**：读代码、做实验、讲清楚。  

**Action**：举例三类收获：  
- **系统**：两阶段调度与显存不足时抢占的取舍；  
- **算子**：FlashAttention 在 prefill/decode API 上的差异；  
- **工程**：CUDA Graph 的前提条件与 `enforce_eager` 开关。  

**Result**：落脚到「能胜任推理引擎**二次开发 / 问题排查**的基础」。  

---

### Q7：如果让你改进 nano-vllm，你会怎么做？

**Situation**：开放题，考察优先级与架构感。  

**Task**：给出**按投入产出排序**的路线图。  

**Action**：  
- 短期：**Chunked Prefill** 降低长 prompt 对 decode 的阻塞；或完善监控与单测。  
- 中期：**量化权重 / KV**、**HTTP API**、流式输出。  
- 长期：**投机解码**、**PD 分离**（需分布式与传输 KV）。  
每条说明会动到的模块（如 `scheduler.py`、`model_runner.py`）。  

**Result**：体现你知道「难点在调度与状态一致性，而非只改一个函数」。  

---

### Q8：nano-vllm 的性能瓶颈可能在哪里？

**Situation**：需区分 Prefill 与 Decode。  

**Task**：结合实现给出**可验证**瓶颈假设。  

**Action**：  
- **Prefill**：序列长时注意力计算量 \(O(n^2)\) 主导；Graph 通常不走（见 `run_model` 条件）。  
- **Decode**：单步算子多、launch 频繁；项目用 **CUDA Graph** 缓解；仍可能受 **显存带宽** 与 **TP 通信** 限制。  
- **调度**：极端负载下抢占与重算 prefill 可能增加尾部延迟。  

**Result**：回答时提到你会用 **Profiler / Nsight** 验证，而不是断言单一瓶颈。  

**源码锚点**：

```190:192:nano-vllm-main/nanovllm/engine/model_runner.py
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
```

---

## 二、KV Cache 和 PagedAttention（8 题）

### Q9：解释 KV Cache 的工作原理？

**Situation**：自回归生成每步依赖历史 token 的 K/V，重复计算代价极高。  

**Task**：说明「缓存什么、何时写入、如何在后续步复用」。  

**Action**：每层先算当前步 K、V；历史步的 K/V 存入缓存；后续步只对**新 query** 与**全历史 K/V** 做注意力。nano-vllm 将缓存放在大块 `kv_cache` 张量中，通过 `slot_mapping` 指定写入位置；Decode 使用 `flash_attn_with_kvcache` 直接读缓存。  

**Result**：能对比「无缓存每步重算整段」与「有缓存只增量」的复杂度差异。  

**源码锚点**：

```59:75:nano-vllm-main/nanovllm/layers/attention.py
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            ...
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
```

---

### Q10：PagedAttention 如何解决显存碎片？

**Situation**：若按最大长度连续分配，短序列会浪费尾部空间，并发多时「总空闲够但无法分配大连续块」问题突出。  

**Task**：说明**分页**如何改为固定大小块分配。  

**Action**：KV 物理上放在统一池中，每条序列用 `block_table` 记录逻辑块到物理块 ID；分配/回收以块为单位，内部碎片被限制在**最后一个不满块**。  

**Result**：对比「连续分配」与「块分配」的碎片来源差异，并能指向 `free_block_ids`。  

**源码锚点**：

```26:33:nano-vllm-main/nanovllm/engine/block_manager.py
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
```

---

### Q11：Block Table 的数据结构和维护逻辑？

**Situation**：面试官考察逻辑块与物理块的映射。  

**Task**：说明**谁持有 table、何时追加、与 Sequence 长度关系**。  

**Action**：`Sequence.block_table` 为物理块 ID 列表；`allocate` 初次为整段 prompt 分配；`may_append` 在块满时追加新物理块 ID；与 `len(seq)`、`num_blocks` 属性联动。  

**Result**：能手算「序列长度从 0 增长时，何时需要新块」——与 `can_append` 联动。  

**源码锚点**：

```93:104:nano-vllm-main/nanovllm/engine/block_manager.py
    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            ...
            block_table.append(block_id)
```

---

### Q12：前缀缓存是如何实现的？

**Situation**：多请求共享相同 prompt 前缀时，重复计算 Prefill 浪费算力。  

**Task**：说明哈希键、命中条件与引用计数。  

**Action**：`compute_hash` 将**前缀哈希 + 当前块 token 字节**结合；满块写入 `hash_to_block_id`；`allocate` 中若哈希命中且 `token_ids` 一致则复用并增加 `ref_count`。  

**Result**：能解释「为何需要 `token_ids` 二次校验」——避免哈希碰撞导致错误复用。  

**源码锚点**：

```35:41:nano-vllm-main/nanovllm/engine/block_manager.py
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()
```

---

### Q13：如何计算需要多少个 KV Cache 块？

**Situation**：调度与 OOM 分析常问。  

**Task**：从**序列长度**与 `block_size` 推导块数，并区分逻辑块与物理池大小。  

**Action**：单条序列约需 \(\lceil \text{len} / \text{block\_size} \rceil\) 个逻辑块；物理池块数在 `allocate_kv_cache` 中由显存预算除以单块字节数得到。  

**Result**：能写出 `block_bytes` 与层数、头维、`block_size` 的关系（见 `allocate_kv_cache`）。  

**源码锚点**：

```107:111:nano-vllm-main/nanovllm/engine/model_runner.py
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
```

---

### Q14：deallocate 时为什么要反向遍历 block_table？

**Situation**：考查引用计数与前缀共享场景。  

**Task**：说明**释放顺序是否影响正确性**；此处主要是按块逐个 `ref_count -= 1`。  

**Action**：反向遍历在链式/共享实现里常与「从序列尾部回退」一致；本实现中每个 `block_id` 独立减引用，**顺序不影响数值**但代码习惯上从后往前释放更贴近序列尾部块先无引用（可与具体缓存策略一起记忆）。  

**Result**：答到「每块 ref 独立递减，为 0 则归还 `free_block_ids`」即可得分；若追问共享，则联系「前缀块可能被多序列引用」。  

**源码锚点**：

```84:91:nano-vllm-main/nanovllm/engine/block_manager.py
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
```

---

### Q15：can_append 的判断逻辑？（len(seq) % block_size == 1 时需要新块）

**Situation**：Decode 每步长度 +1，何时跨块是关键细节。  

**Task**：解释布尔表达式 `len(self.free_block_ids) >= (len(seq) % block_size == 1)`。  

**Action**：在 Python 中 `len(seq) % block_size == 1` 为 True 时，下一步 append 将让新长度满足「新块起点」——需要**提前**再占一块；因此空闲块数至少为 1。若为 False，则仍落在当前块内，不需要新块。  

**Result**：可用手算小例子（block_size=256：长度 256→下一步需要第二块，对应条件在特定余数触发）验证。  

---

### Q16：如果显存不足会怎样？

**Situation**：推理服务必须在压力下可退化而非崩溃。  

**Task**：联系 `can_allocate` / `can_append` 与 `preempt`。  

**Action**：Prefill 时若 `can_allocate` 失败，`schedule` 的 waiting 循环 `break`，请求继续排队；Decode 时若 `can_append` 失败，从 `running` 尾部 `preempt`，`deallocate` 释放 KV，把序列放回 `waiting`（可能重算）。  

**Result**：说明**牺牲部分请求延迟换系统存活**；并诚实说「与 vLLM 的 swap 到 CPU 等机制相比，教学实现更简单」。  

**源码锚点**：

```24:38:nano-vllm-main/nanovllm/engine/scheduler.py
    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        ...
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
```

```44:51:nano-vllm-main/nanovllm/engine/scheduler.py
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
```

---

## 三、调度和批处理（6 题）

### Q17：连续批处理的核心思想？

**Situation**：静态 batch 边界固定，GPU 易在批次尾部空闲。  

**Task**：一句话 + nano-vllm 落点。  

**Action**：每一轮 `step` 重新选择参与计算的序列集合，完成的退出、新请求进入；`waiting`/`running` 实现动态集合。  

**Result**：强调「**以步为单位的动态 batch**」，不是一次 forward 算完所有 token。  

---

### Q18：为什么 Prefill 优先于 Decode？（结合本仓库 schedule）

**Situation**：同时有排队新请求与正在生成的序列时，先服务谁影响 TTFT 与 TPOT。  

**Task**：描述 `schedule` **先扫描 waiting** 的实现。  

**Action**：代码中先执行 prefill 的 `while self.waiting`，若 `scheduled_seqs` 非空则**直接 return**，本轮不执行 decode 段；因此新到达且可分配的 Prefill 会阻塞同一 `step` 内的 Decode。  

**Result**：可讨论利弊：有利于**降低排队请求的 TTFT**；可能增加已在 running 上请求的等待（需结合产品 SLA 说明）。  

**源码锚点**：

```24:41:nano-vllm-main/nanovllm/engine/scheduler.py
        while self.waiting and num_seqs < self.max_num_seqs:
            ...
        if scheduled_seqs:
            return scheduled_seqs, True
```

---

### Q19：抢占机制如何工作？

**Situation**：Decode 需要为新 token 预留块，空闲不足时必须让路。  

**Task**：说明从哪个队列抢、抢完状态如何。  

**Action**：`can_append` 失败时 `preempt(self.running.pop())`，即从 **running 队列尾部**拿走序列；`preempt` 把状态改 `WAITING` 并 `deallocate`。  

**Result**：这是「**牺牲尾部 running 序列**」的策略；可对比 FCFS 叙述（实现上是 pop 尾部）。  

**源码锚点**：

```60:63:nano-vllm-main/nanovllm/engine/scheduler.py
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
```

---

### Q20：schedule 方法的完整流程？

**Situation**：高频考点，需要能口述两段式。  

**Task**：Prefill 段 + Decode 段 + 返回值含义。  

**Action**：  
1. 尝试从 `waiting` 取序列，检查 `max_num_batched_tokens` 与 `can_allocate`，通过则 `allocate` 并入 `running`，收集为 `scheduled_seqs`；若非空，返回 `(scheduled_seqs, True)`。  
2. 否则进入 decode：从 `running` 取序列，确保 `can_append`，否则抢占；`may_append` 后收集；最后 `running.extendleft(reversed(scheduled_seqs))` 维护顺序，返回 `(scheduled_seqs, False)`。  

**Result**：能解释第二个返回值 `is_prefill` 如何传到 `ModelRunner` 影响准备路径。  

---

### Q21：postprocess 如何判断序列结束？

**Situation**：结束条件影响资源释放。  

**Task**：列出条件与副作用。  

**Action**：对每个 `(seq, token_id)`：先 `append_token`；若 `token_id == eos`（且未 `ignore_eos`）或达到 `max_tokens`，置 `FINISHED`，`deallocate`，并从 `running` 移除。  

**Result**：说明 **EOS 与长度上限** 两类结束。  

**源码锚点**：见 Q3 引用 `postprocess`。  

---

### Q22：waiting 和 running 队列的管理策略？

**Situation**：考察队列语义与顺序。  

**Task**：新请求从哪进、Prefill 后去哪、抢占回哪。  

**Action**：`add` 用 `waiting.append`；Prefill 成功后从 `waiting.popleft` 到 `running.append`；抢占用 `waiting.appendleft` 插回队头优先重试。  

**Result**：能描述「running 顺序在 decode 段被 `popleft`/`extendleft` 维护」的直觉。  

---

## 四、模型和计算（8 题）

### Q23：Qwen3 的模型架构？

**Situation**：岗位若考模型侧，需要能概括。  

**Task**：从模块文件回答：RMSNorm、RoPE、SwiGLU、GQA 等。  

**Action**：结合 `nanovllm/models/qwen3.py`：Decoder-only、多层 Transformer Block；注意力里 Q/K/V 与 GQA 头数配置；FFN 用 SwiGLU（gate/up/down）；位置编码 RoPE。  

**Result**：说明「与 LLaMA 系接近的具体变体以 `config.json` 为准」。  

---

### Q24：GQA 的实现和优势？

**Situation**：KV 显存与带宽是推理瓶颈。  

**Task**：解释头数关系与在代码中的体现。  

**Action**：`num_attention_heads` 大于 `num_key_value_heads` 时，多组 Q 共享一组 K/V；`allocate_kv_cache` 用 `num_kv_heads` 计算块大小；FlashAttention 侧支持 GQA。  

**Result**：优势：**KV Cache 更小、访存更少**；代价是表达能力略低于 MHA（由架构设计平衡）。  

---

### Q25：RoPE 如何编码位置信息？

**Situation**：自注意力本身置换不变，需要位置依赖。  

**Task**：说明旋转作用在 Q/K 与相对位置性质。  

**Action**：RoPE 将位置依赖为对 Q/K 的旋转变换；实现上预计算 cos/sin 表并按位置索引应用（见 `rotary_embedding.py`）。  

**Result**：可补充外推话题（NTK/YaRN）为加分。  

---

### Q26：SwiGLU 的计算过程？

**Situation**：FFN 形态高频八股。  

**Task**：写出公式级直觉 + 模块名。  

**Action**：典型形式 \(FFN(x) = (xW_{gate} \odot \sigma(xW_{up})) W_{down}\)（具体以 Qwen3 实现为准）；激活常用 SiLU。  

**Result**：说明「相比单门 FFN，SwiGLU 多一路 gate 控制信息流」。  

---

### Q27：RMSNorm 和 LayerNorm 的区别？

**Situation**：Norm 题。  

**Task**：对比均值方差、是否中心化。  

**Action**：RMSNorm 用均方根缩放、不减均值；计算更省；nano-vllm 中见各层 `RMSNorm` 实现。  

**Result**：一句话：**更轻量、对大模型训练/推理常见**。  

---

### Q28：FlashAttention 的 prefill 和 decode 分支？

**Situation**：同一 `Attention.forward` 内部分支。  

**Task**：对应 API 与上下文构造差异。  

**Action**：Prefill：`flash_attn_varlen_func`，传 `cu_seqlens_q/k`；若带 prefix cache 且 `cu_seqlens_k > cu_seqlens_q`，设置 `block_tables` 走分页读。Decode：`flash_attn_with_kvcache` + `cache_seqlens` + `block_tables`。  

**Result**：强调 **varlen 处理变长打包**；decode **单步单 token 的 KV cache 访问模式**。  

**源码锚点**：见 Q9；以及 `prepare_prefill`/`prepare_decode`。  

---

### Q29：采样策略的实现？

**Situation**：logits → token 的「最后一公里」。  

**Task**：温度、top-k/top-p、greedy 的关系。  

**Action**：读 `layers/sampler.py`：`temperature` 缩放；`top_p`/`top_k` 过滤；`torch.compile` 可选加速。  

**Result**：说明 `temperature=0` 时常对应 greedy（实现细节以代码为准）。  

---

### Q30：LMHead 如何在 prefill 时只取最后一个 token？

**Situation**：Prefill 一次算多个 hidden，但下一步预测只需最后一个位置。  

**Task**：指向 `ParallelLMHead` 分支。  

**Action**：`get_context().is_prefill` 为真时，用 `cu_seqlens_q[1:] - 1` 取每条序列最后一个位置的 hidden，再 `F.linear` 得 logits。  

**Result**：与「packed 序列」表示一致，避免对整个长序列做无意义 logits 计算（仍算 attention，但 logits 降维）。  

**源码锚点**：

```56:61:nano-vllm-main/nanovllm/layers/embed_head.py
    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
```

---

## 五、系统优化（6 题）

### Q31：CUDA Graph 的原理和限制？

**Situation**：Decode launch 开销占比高。  

**Task**：解释 capture/replay 与适用条件。  

**Action**：`capture_cudagraph` 对多个 `bs` 录制；`run_model` 在非 prefill、非 eager、bs≤512 时选图 `replay`。限制：序列长度变化大、图结构变化 → 不适合 prefill；动态 shape 超出录制范围需 fallback。  

**Result**：能复述「**同构图 + 占位缓冲 + copy_**」。  

**源码锚点**：

```216:241:nano-vllm-main/nanovllm/engine/model_runner.py
    def capture_cudagraph(self):
        ...
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            ...
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
```

---

### Q32：Triton Kernel 的设计思路？

**Situation**：把 K/V 写入分页池需要高吞吐 scatter。  

**Task**：说明并行粒度与边界。  

**Action**：`store_kvcache_kernel` 每 program 对应一个 token 位置：读 `slot_mapping`，-1 跳过；把 key/value 向量写入 `k_cache`/`v_cache` 的展平偏移。  

**Result**：强调 **与 `slot_mapping` 同长度**，由 `prepare_prefill/decode` 填好。  

**源码锚点**：

```10:30:nano-vllm-main/nanovllm/layers/attention.py
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    ...
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    ...
    tl.store(k_cache_ptr + cache_offsets, key)
```

---

### Q33：torch.compile 用在了哪些地方？

**Situation**：PyTorch 2.x 编译加速热点。  

**Task**：以仓库为准列举。  

**Action**：在本仓库中可检索 `@torch.compile`，主要出现在：`layers/sampler.py`（采样）、`layers/activation.py`（如 SwiGLU 相关激活）、`layers/layernorm.py`（RMSNorm 等）、`layers/rotary_embedding.py`（RoPE）。原则：**减少 Python 调度开销、便于算子融合**，具体收益依 PyTorch/CUDA 版本与形状而定。  

**Result**：答「覆盖采样、Norm、RoPE、激活等热点子模块」并建议**用 Profiler 实测**再写进简历结论。  

---

### Q34：张量并行的实现细节？

**Situation**：多卡权重切分与通信。  

**Task**：Column/Row 与注意力输出聚合。  

**Action**：`linear.py` 中 ColumnParallel 先局部 matmul 再 gather；RowParallel 先局部再 `all_reduce`；`init_process_group("nccl")` 建立进程组。  

**Result**：能数清「每层大致两次 all_reduce」级别（具体因模块实现而异）。  

---

### Q35：SharedMemory 通信机制？

**Situation**：TP>1 时非主 rank 不直接跑 Python 调度，需要收包执行。  

**Task**：描述 rank0 写共享内存 + Event。  

**Action**：`write_shm` pickle 序列化 `[method_name, args]`；子进程 `read_shm` 等待 `event`；调用 `call` 分发到 `run/exit`。  

**Result**：说明这是**多进程 TP 驱动**的 IPC，而非单进程多线程。  

**源码锚点**：

```76:88:nano-vllm-main/nanovllm/engine/model_runner.py
    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        ...
        for event in self.event:
            event.set()
```

---

### Q36：pin_memory 的作用？

**Situation**：CPU→GPU 拷贝与流水线。  

**Task**：解释 page-locked memory。  

**Action**：`prepare_prefill` 等处 `pin_memory=True` + `non_blocking=True`：**异步 DMA**，与计算重叠，降低每步同步等待。  

**Result**：答「减少 H2D 阻塞、提高吞吐」即可。  

**源码锚点**：

```156:160:nano-vllm-main/nanovllm/engine/model_runner.py
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
```

---

## 六、开放性问题（6 题）

### Q37：如何支持更多模型？

**Situation**：工业需求常是多架构。  

**Task**：指出扩展面：配置、权重、层实现。  

**Action**：新增 `models/xxx.py` 定义 `ForCausalLM`；`load_model`/`weight_loader` 适配命名；在 `ModelRunner` 中实例化对应类；保证 `Attention` 与 KV 形状、RoPE 参数一致。  

**Result**：强调「**引擎层复用，模型层替换**」。  

---

### Q38：如何添加流式输出？

**Situation**：产品体验常见需求。  

**Task**：从「每次 `step` 产出 token」推导接口。  

**Action**：在 `generate` 循环中每步 yield 当前增量 token 或文本；注意线程安全与 HTTP chunked 响应；调度逻辑可不变。  

**Result**：说明 nano-vllm 默认可在 `step` 外加回调实现。  

---

### Q39：如何实现在线服务（API Server）？

**Situation**：部署形态从脚本到服务。  

**Task**：划分进程：请求接入、调度线程、GPU worker。  

**Action**：用 FastAPI/Flask 收请求 → 队列投递 `add_request` → 后台线程循环 `step`；多卡时保持 rank0 统一调度；加健康检查与并发限流。  

**Result**：点出 **batching 与延迟权衡** 与 **OpenAI 兼容层**可作为加分。  

---

### Q40：如何支持量化推理？

**Situation**：nano-vllm 默认可为 FP16/BF16。  

**Task**：说明需要改哪些算子路径。  

**Action**：权重量化为 INT8/FP8；`Linear` 替换为量化内核；KV 量化需改 `store_kvcache` 与注意力 API；校准与缩放策略（GPTQ/AWQ）离线完成。  

**Result**：诚实说「工作量在算子与精度验证」。  

---

### Q41：如何实现投机解码？

**Situation**：加速 decode 的前沿方向。  

**Task**：说明草稿模型 + 验证步。  

**Action**：调度器需插入「草稿生成 k token → 大模型一步验证」；与 `Sampler`、KV 复用策略紧密相关；nano-vllm 需较大改造。  

**Result**：展示你知道**验证失败时如何回滚 token 与 KV**。  

---

### Q42：大规模部署需要考虑什么？

**Situation**：从单机到集群。  

**Task**：SLO、容错、弹性、监控。  

**Action**：  
- **容量**：每卡并发、KV 显存、PD 分离是否需要；  
- **可靠**：进程重启、请求超时、重试；  
- **观测**：TTFT、TPOT、队列长度、GPU SM/显存；  
- **版本**：模型与 tokenizer 版本管理。  

**Result**：把「nano-vllm 学到的是核心算法」，「上线要补运维与 SRE」。  

---

## 附录：备考建议

1. **优先啃透**：`scheduler.py`、`block_manager.py`、`model_runner.py`、`attention.py`。  
2. **对照实验**：`enforce_eager=True` 关闭 CUDA Graph，对比 Decode 吞吐。  
3. **诚实表述**：学习项目与原创工程的边界，用**笔记/实验/commit**证明深度。  

祝面试顺利。
