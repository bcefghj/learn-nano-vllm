# 课程 04：Tokenizer 与 Embedding

> 先把字符串变成 token id（Tokenizer 工作流），再看 **词表并行**：每张卡只存 `V/tp_size` 行嵌入，用 mask + all_reduce 拼出完整向量；最后理解 **ParallelLMHead** 在 Prefill 只取每序列最后一个位置——与自回归「预测下一 token」一致。

## 本课目标

1. 描述 **Tokenizer 工作流**：normalize → 词表映射 → special tokens →（可选）chat 模板。
2. 理解 **VocabParallelEmbedding** 的 **切分方式**、**mask 含义**、**all_reduce 必要性**。
3. 解释 **ParallelLMHead.forward** 中 `cu_seqlens_q` 与 **只取 last token** 的原因。
4. 面试中能对比 **Embedding 前向** 与 **LMHead 前向** 在 TP 下的差异（reduce vs gather）。

## 核心概念

### Tokenizer 工作流（以 HuggingFace 为例）

nano-vllm 的 **`example.py`** 在引擎外使用 **`AutoTokenizer`**，典型步骤：

1. **加载**：`from_pretrained(model_dir)` 读 `tokenizer.json` / `vocab` 等。
2. **对话格式**：`apply_chat_template(messages, ...)` 生成 **带角色标记** 的字符串，便于指令模型理解 **user/assistant** 边界。
3. **编码**：引擎侧或脚本侧 `encode` 得到 **token id 序列**（整数张量），作为 **`embedding` 层输入**。

**面试常问点**：Tokenizer **不属于** `nanovllm` 核心包，但 **与词表大小 V、embed 权重形状** 强相关；**词表并行** 正是按 **V 的维度** 切分。

### 词表并行（Vocab Parallelism）

当 **嵌入矩阵** 过大（大词表 × 隐藏维），可在 **词表维度** 切分：

- 第 `r` 张卡只保存 **行索引区间** `[v_start, v_end)` 对应的 **`V/tp_size` 行**。
- 前向时：若某 token id **落在本卡区间**，用 **本地行** 查表；否则 **本地贡献为 0**。
- 多卡时：每张卡得到 **部分向量**，需 **`all_reduce`（求和）** 合并（因非本卡行权重为 0，求和即等价于拼接后再投影的线性叠加思想）。

**直觉**：每张卡算 **自己那一块词表的 embedding**，其他位置为 0，**加起来** 就是完整 embedding。

### LMHead 与 Embedding 权重共享

许多模型 **输出层与输入嵌入共享权重**（weight tying）。`ParallelLMHead` 继承 `VocabParallelEmbedding`，复用 **分片权重**；但 **前向** 不同：

- **Prefill**：序列中每个位置都有 hidden，但 **训练/推理目标** 常是 **最后一个位置预测下一 token**；nano-vllm 用 **`cu_seqlens_q`** 取 **每条序列最后一个 query 位置**。
- **Decode**：通常每序列 **1 个 token**，形状与上下文 `is_prefill` 由 `model_runner` 设置。

### `cu_seqlens_q` 是什么（与本课相关）

**Cumulative sequence lengths**：长度 `batch+1`，记录 **展平后 token 序列** 每条子序列的起止下标。  
`cu_seqlens_q[1:] - 1` 即 **每条序列最后一个 token 在展平数组中的下标**——用于 **只取 last hidden**。

## 源码解析（带完整源码和逐行注释）

下列代码与仓库 `nanovllm/layers/embed_head.py` 一致（含 `weight_loader` 便于理解权重加载）。

```python
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
```

### VocabParallelEmbedding 逐段注释

| 代码片段 | 解释 |
|----------|------|
| `dist.get_rank()` / `get_world_size()` | 当前 **TP 组** 内 rank 与 **并行度 tp_size** |
| `num_embeddings % self.tp_size == 0` | 词表行数必须 **整除**，否则无法均分 |
| `num_embeddings_per_partition` | 每卡 **本地词表行数** `V/tp_size` |
| `vocab_start_idx` / `vocab_end_idx` | 本卡负责的 **全局 token id 区间** |
| `self.weight` 形状 `(V/tp, D)` | 只存 **本分片** 的嵌入表 |
| `weight_loader` | 从 **完整 HF 权重** 按行切 **`narrow`** 再 `copy_`，与 TP rank 对齐 |
| `mask = (x >= ...) & (x < ...)` | 标记 **哪些位置属于本卡词表** |
| `x = mask * (x - self.vocab_start_idx)` | 将 **全局 id** 转为 **本地行号**；不属于本卡的 id 被置 0（与 mask 配合） |
| `F.embedding(x, self.weight)` | 标准查表；越界 id 行为依赖 mask 与后续乘法 |
| `mask.unsqueeze(1) * y` | 非本卡词表位置 **嵌入置零**，避免脏值进入规约 |
| `dist.all_reduce(y)` | **求和** 合并各卡贡献，得到 **完整 D 维向量** |

### ParallelLMHead 逐段注释

| 代码片段 | 解释 |
|----------|------|
| `assert not bias` | 输出层 **无 bias**，与 Qwen 类实现一致，简化并行 |
| `get_context()` | 取 **全局推理上下文**（prefill/decode、cu_seqlens 等） |
| `if context.is_prefill` | **Prefill**：序列并行展开，`hidden` 形状对应 **所有位置** |
| `last_indices = context.cu_seqlens_q[1:] - 1` | 每条序列 **最后一个 token** 在展平 `hidden` 里的索引 |
| `x = x[last_indices].contiguous()` | 只保留 **last hidden**，形状 `(batch, D)`，准备算 **下一 token logits** |
| `F.linear(x, self.weight)` | 与 embedding 同权重的 **线性层**：`logits = x @ W^T`（形状细节以布局为准） |
| `tp_size > 1` 时 `gather` + `cat` | **词表维切分** 下，每卡只持有 **部分 vocab 列**；需在 **logits 最后一维** 拼接成全词表 logits（**gather 到 rank0** 是常见模式） |

**注意**：多卡时 **rank 非 0** 可能返回 `None`，由引擎保证只在需要处消费 logits；以你阅读的 `sampler`/`engine` 为准。

## 图解（用文字/ASCII 描述）

**词表并行 Embedding（tp=2 示意）**：

```
全局 token id:  0 ... V/2-1  |  V/2 ... V-1
                 ----卡0----    ----卡1----

token 落在卡0区间 -> 卡0算向量，卡1置零 -> all_reduce 相加 -> 完整向量
```

**Prefill 时 LMHead 取 last**：

```
batch 内 3 条序列，展平后 hidden 下标:
  seq0: [0,1,2]
  seq1: [3,4]
  seq2: [5,6,7,8]

cu_seqlens_q 类似 [0,3,5,9]
last_indices = [2,4,8]  -> 只取这三处 hidden 做 logits
```

## 面试考点

- **词表并行 vs 行并行/列并行**：这里并行的是 **嵌入矩阵的行（vocab 维）**。
- **为什么用 mask + all_reduce**：每张卡只负责部分 id，**其余必须为 0** 再规约。
- **LMHead 在 prefill 只取 last**：对齐 **因果 LM 的预测位置**（预测 **下一个** token）。
- **TP>1 时 logits 需 gather/cat**：每张卡 **部分 vocab logits**，拼接成全词表再采样。

## 常见面试题

1. **若没有 `mask * y` 直接 all_reduce 会怎样？**  
   答：非本分片 id 可能产生 **错误非零嵌入**，规约后 **污染结果**。

2. **Decode 阶段 LMHead 还需要 `cu_seqlens_q` 吗？**  
   答：通常 **每序列一步**，`is_prefill` 为 False 时 **不走路径**；以 `context` 为准。

3. **weight tying 时如何加载权重？**  
   答：`weight_loader` 对 **同一份 checkpoint 行切分** 到各卡，**embedding 与 lm_head 共享 Parameter**（若模型实现如此）。

4. **Tokenizer 词表大小与 `num_embeddings` 不一致会怎样？**  
   答：配置/权重不匹配会 **load 失败** 或 **越界**；需与 HF `config.vocab_size` 对齐。

## 小结

- **Tokenizer** 在引擎外把文本变为 **id**；**词表大小** 驱动嵌入形状。
- **VocabParallelEmbedding** 用 **区间 mask + 本地行号 + all_reduce** 实现 **无重复全表存储** 的嵌入。
- **ParallelLMHead** 在 **prefill** 用 **`cu_seqlens_q`** 定位 **每序列最后位置**，与 **自回归目标** 对齐；多卡时对 **logits 维做 gather/拼接**。

## 下一课预告

下一课 **《05-课程05-Attention机制与FlashAttention》** 将拆解 **`store_kvcache` Triton 内核**、`flash_attn_varlen_func` 与 `flash_attn_with_kvcache` 两分支，以及 **prefix cache（`block_tables`）** 下如何从 **KV cache** 读 K、V。
