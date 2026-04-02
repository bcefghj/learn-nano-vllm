# 课程 05：Attention 机制与 FlashAttention

> Attention 前向在 nano-vllm 里分两条：**Prefill** 用变长 FlashAttention 处理整段 prompt（并可带 paged block table）；**Decode** 用带 KV cache 的 kernel 单步续写。中间用 **Triton** 把算出的 K/V 按 slot 写入全局缓存——读完你能讲清 IO、prefix 与两阶段差异。

## 本课目标

1. 讲清 **`Attention.forward`** 中 **先写 KV、再注意力** 的顺序及原因。
2. 对比 **`flash_attn_varlen_func`（prefill）** 与 **`flash_attn_with_kvcache`（decode）** 的输入与适用场景。
3. 理解 **`store_kvcache_kernel`**：每个 program 处理一个 token 位置，按 `slot_mapping` 写入物理 cache。
4. 解释 **prefix cache 分支**：`block_tables is not None` 时 **K/V 来自 cache** 做 varlen attention 的含义。
5. 口述 **FlashAttention 的 IO 优化思想**（tiling、重计算 vs HBM）。

## 核心概念

### 标准注意力与 KV Cache

对单层，给定当前步的 **Q、K、V**（多头形状略），注意力：

\[
\mathrm{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V
\]

**推理时**：历史位置的 K、V 已缓存，只需把 **新 token 的 K、V** 追加进 cache，再用 **当前 Q** 与 **全部历史 K、V** 做注意力（带因果掩码）。

### FlashAttention（直觉）

经典实现：材料化 **完整 \(S \times S\)** 注意力矩阵，**HBM 读写** 成为瓶颈。  
**FlashAttention** 通过 **分块（tiling）**、在 **SRAM** 上完成 softmax 规约，减少 **HBM 访问量**，在 **长序列** 上显著加速。**IO 复杂度** 视角是面试加分点。

### `flash_attn_varlen_func`（Prefill）

- 输入 **多个变长序列拼接** 成的张量，配合 **`cu_seqlens_*`** 描述边界。
- 适合 **prompt 阶段**：一次处理 **多条序列、多个 token**，**因果掩码** 为 True。
- 可选 **`block_table`**：与 **PagedAttention** 一致，把逻辑 token 映射到 **物理块**，支持 **非连续** KV 存储。

### `flash_attn_with_kvcache`（Decode）

- 针对 **已有 KV cache**、当前步 **短 Q**（常每序列 1 token）优化。
- 从 **k_cache / v_cache** 读取历史，**cache_seqlens** 描述当前已缓存长度等。
- **Decode** 路径上 **`q.unsqueeze(1)`** 引入 **长度维**，以匹配 kernel 期望形状。

### Prefix cache（`block_tables is not None` 且 prefill）

当 **部分前缀的 KV 已存在于 cache**（例如 **共享系统提示**、**多轮对话复用**），本轮 **不必重新计算** 前缀 K/V：

- 代码路径：`if context.is_prefill` 且 **`context.block_tables is not None`**，则 **`k, v = k_cache, v_cache`**，直接用 **cache 中已有 KV** 与当前 **Q** 做 **varlen attention**。
- 新算出来的 K/V 仍可通过 **`store_kvcache`** 写入 **新占用的 slot**（与调度分配的块一致）。

## 源码解析（带完整源码和逐行注释）

### `attention.py` 完整源码

```python
import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
```

### Triton `store_kvcache_kernel` 逐行说明

| 行 | 含义 |
|----|------|
| `idx = tl.program_id(0)` | 一维 grid，**第 idx 个 token 位置**（展平后） |
| `slot = tl.load(slot_mapping_ptr + idx)` | 该 token 的 KV 写入 **物理 slot**；**-1 表示不写入**（占位/跳过） |
| `if slot == -1: return` | **无效位置**直接返回，避免污染 cache |
| `key_offsets` / `value_offsets` | 从 **输入 K/V 张量** 按 **idx** 与 **stride** 取向量（展平 head 维为 D） |
| `tl.load(...)` | 读出 **本 token** 的 K、V 向量 |
| `cache_offsets = slot * D + arange(D)` | 物理 cache 以 **slot 为大块**，连续存 **D = num_heads * head_dim** |
| `tl.store(k_cache_ptr + cache_offsets, key)` | 写入 **K cache**；V 同理 |

**`store_kvcache` Python 包装**：

- 校验 **连续内存 stride**，保证 Triton 寻址正确。
- Grid 大小 **`N`** = token 数，每 program **写一对 K/V**。

### `Attention.forward` 两分支说明

| 步骤 | 作用 |
|------|------|
| `get_context()` | 取 **当前步** 的 prefill/decode、cu_seqlens、block_tables、slot_mapping 等 |
| `if k_cache.numel() and v_cache.numel()` | cache 张量已分配则 **先写入** 本步算出的 K/V |
| `store_kvcache(...)` | 按 **`slot_mapping`** 把 **本层** 的 K/V 写入 **全局块缓存** |
| **prefill + `block_tables is not None`** | **prefix cache**：注意力用的 **K/V 直接来自 cache**（已含前缀），避免重复计算前缀 |
| `flash_attn_varlen_func(...)` | **变长 prefill**；`causal=True`；**`block_table`** 支持 paged KV |
| **decode** | `flash_attn_with_kvcache`：**读 cache** + **当前 Q**；`cache_seqlens=context.context_lens` |
| `q.unsqueeze(1)` | 增加 **seqlen=1** 维以符合 API |

## 图解（用文字/ASCII 描述）

**Prefill vs Decode**：

```
Prefill:
  prompt tokens  -->  Q,K,V 线性投影  -->  store_kvcache 写入
                    -->  flash_attn_varlen（整段或带 prefix 的 cache）

Decode:
  1 new token      -->  Q ；K,V 新向量写入 cache
                    -->  flash_attn_with_kvcache（读大块 cache）
```

**Slot 写入**：

```
token index idx ----slot_mapping----> physical slot s
K/V 向量 ----------store-----------> k_cache[s*sD:(s+1)*sD]
```

**FlashAttention IO（口述图）**：

```
朴素: HBM <--> 大注意力矩阵 <--> HBM
Flash: 分块在 SRAM 上算 softmax/规约，减少大矩阵 materialize
```

## 面试考点

- **先 store 再 attention**：prefill 时 **先把本步 K/V 落到 cache**，再参与 **varlen** 或 **prefix** 路径；顺序与 **slot 分配** 一致。
- **prefix 分支**：`k, v = k_cache, v_cache` —— **注意力输入改为缓存** 而非当前线性层刚算出的 **完整 k,v 张量**（当前步 Q 仍新）。
- **`slot == -1`**：padding 或 **不归属本步写入** 的位置。
- **Decode 用 `flash_attn_with_kvcache`**：**低延迟单步**，重用 **HBM 中 paged cache**。
- **FlashAttention**：**减少 HBM 读写**、**数值稳定的 online softmax**（可提一句）。

## 常见面试题

1. **Prefill 为什么可能是 compute-bound？**  
   答：**prompt 长、矩阵大**，GEMM 与注意力 **FLOPs 高**；FlashAttention 降低 IO 后仍可能吃满算力。

2. **Decode 为什么更 memory-bound？**  
   答：每步 **有效计算小**，**读 KV**、**写 KV**、kernel 启动占比高。

3. **没有 Triton store，只用 PyTorch 索引写 cache 可以吗？**  
   答：可以但 **慢**；自定义内核 **融合写**、**stride 可控**，利于 **Paged** 布局。

4. **`block_table` 在两层 API 里分别起什么作用？**  
   答：把 **逻辑序列位置** 映射到 **物理块**，使 **KV 非连续存储** 仍能被 FlashAttention 内核寻址。

5. **GQA（num_kv_heads < num_heads）在哪体现？**  
   答：在 **Qwen3/Linear 投影** 与 **flash-attn** 的 head 布局中；本文件 **Attention** 接收的 **k,v** 已按模型实现。

## 小结

- **Triton `store_kvcache_kernel`**：按 **token 并行** 把 K/V **scatter** 到 **slot 对齐** 的物理 cache。
- **Prefill**：**`flash_attn_varlen_func`** 处理 **变长 batch**，支持 **`block_table`**；若 **已有前缀在 cache**，走 **`k,v = k_cache,v_cache`** 的 **prefix** 优化。
- **Decode**：**`flash_attn_with_kvcache`** 专注 **单步续写** 与 **KV 复用**。
- **FlashAttention**：从 **算法+IO** 两侧优化长序列注意力，是 **推理引擎** 与 **面试** 的双重高频点。

## 下一课预告

下一课通常进入 **RoPE、LayerNorm、SwiGLU** 或 **Qwen3 块级结构**（以你教程目录为准），把 **位置编码与 FFN** 与 **Attention** 拼成完整 Decoder 层，再衔接 **KV 块与 Scheduler**。
