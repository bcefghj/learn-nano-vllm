# 第17课：Triton Kernel 编写

面向初学者的说明：本课从「什么是 Triton」讲起，对比 CUDA 编程，详细解析 nano-vllm 中 **KV Cache 写入 kernel** 的每一行代码，帮你理解 Triton 的编程模型和 GPU 并行思维。每节配有源码对照与面试考点。

下列代码与 `nanovllm/layers/attention.py` 中实现一致（变量名 `N` 与 batch 内 token 数一致）。

```python
@triton.jit
def store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride, k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D: tl.constexpr):
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

def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)
```

---

## 一、概念讲解

### 1.1 什么是 Triton

**Triton** 是由 OpenAI 开发的一门 **GPU 编程语言和编译器**，目标是让开发者用 **接近 Python 的语法** 编写高性能 GPU 内核（kernel），而无需掌握 CUDA C++ 的底层细节。

Triton 的定位：

```
抽象程度：  Python (PyTorch)  >  Triton  >  CUDA C++  >  PTX 汇编
性能上限：  Python (PyTorch)  <  Triton  ≈  CUDA C++  ≈  PTX 汇编
开发效率：  Python (PyTorch)  >  Triton  >  CUDA C++  >  PTX 汇编
```

Triton 能在 **保持接近 CUDA 的性能** 的同时，**大幅降低开发门槛**，这使得它成为深度学习系统中编写自定义 kernel 的首选工具。

### 1.2 Triton vs CUDA 编程对比

| 对比维度 | CUDA C++ | Triton |
|---------|----------|--------|
| **编程语言** | C++ 扩展 | Python（使用装饰器 `@triton.jit`） |
| **并行粒度** | **线程级（Thread-level）** | **Block 级（Block-level）** |
| **内存管理** | 手动管理共享内存、寄存器 | 编译器自动管理 |
| **索引计算** | `threadIdx.x`、`blockIdx.x` 手写 | `tl.program_id()`、`tl.arange()` |
| **内存访问优化** | 手动合并访问（coalescing） | 编译器自动优化 |
| **同步** | `__syncthreads()` 手动同步 | 编译器自动插入 |
| **编译** | nvcc 编译 | JIT（即时编译），首次调用时编译 |
| **调试难度** | 高 | 中等 |

### 1.3 Triton 的 Block-level 并行模型

这是理解 Triton 最核心的概念。

在 CUDA 中，你需要思考「**每个线程做什么**」：

```c++
// CUDA：每个线程处理一个元素
__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

在 Triton 中，你思考的是「**每个 program（block）处理一块数据**」：

```python
# Triton：每个 program 处理 BLOCK_SIZE 个元素
@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)
```

区别总结：
- CUDA：你管理 **每个线程** 的行为，手动处理 **共享内存**、**线程同步**、**内存合并** 等。
- Triton：你只需定义 **每个 block 处理哪块数据**，编译器自动处理线程分配、共享内存、合并访问。

### 1.4 Triton 关键 API 速查

| API | 功能 | 类比 CUDA |
|-----|------|----------|
| `@triton.jit` | 标记函数为 GPU kernel | `__global__` |
| `tl.program_id(axis)` | 获取当前 program 的 ID | `blockIdx.x` |
| `tl.arange(start, end)` | 生成连续整数序列 | 无直接对应，需手写 |
| `tl.load(ptr + offsets)` | 从全局内存加载数据 | 手动 `ptr[idx]` 读取 |
| `tl.store(ptr + offsets, val)` | 向全局内存写入数据 | 手动 `ptr[idx] = val` 赋值 |
| `tl.constexpr` | 编译期常量 | `constexpr` 或模板参数 |
| `tl.zeros(shape, dtype)` | 创建零初始化的 block | CUDA 共享内存手动初始化 |
| `tl.dot(a, b)` | Block 级矩阵乘 | 调用 `wmma` 或手写 |

### 1.5 `@triton.jit` 装饰器

```python
@triton.jit
def my_kernel(arg1, arg2, CONST: tl.constexpr):
    ...
```

这个装饰器的作用：

1. **标记 JIT 编译**：函数体不会立即执行，而是在首次被调用时，Triton 编译器将其编译为 GPU 机器码（PTX → SASS）。
2. **类型推导**：根据调用时传入的参数类型自动推导内部变量类型。
3. **特化（Specialization）**：`tl.constexpr` 参数会在编译期固化，编译器可以据此做常量折叠、循环展开等优化。
4. **缓存**：编译结果会被缓存，相同参数类型的后续调用不需要重新编译。

注意事项：
- kernel 函数 **不能有返回值**，所有输出通过 `tl.store` 写入内存。
- 不能在 kernel 中调用标准 Python 函数（如 `print`、`len`），但可以调用 `tl.` 系列 API。
- 不能使用 Python 的动态特性（如列表推导、字典操作）。

---

## 二、源码对照：`store_kvcache_kernel` 逐行解析

nano-vllm 的 KV Cache 写入 kernel 位于 `nanovllm/model/attention.py`，用 Triton 实现。它负责将 Attention 层计算出的 Key 和 Value 写入 KV Cache 的正确位置。

### 2.1 为什么需要自定义 kernel 来写 KV Cache

在 nano-vllm 的 PagedAttention 架构中（回顾第9课），KV Cache 是按 **slot** 组织的：

- 每个 slot 对应一个 token 的 K/V 向量。
- `slot_mapping[i]` 告诉第 `i` 个 token 应该写入哪个 slot。
- slot 的分配是非连续的（由 BlockManager 管理），不能简单地用连续内存拷贝。

如果用 PyTorch 实现：

```python
for i in range(N):
    slot = slot_mapping[i]
    if slot != -1:
        k_cache[slot] = key[i].reshape(-1)
        v_cache[slot] = value[i].reshape(-1)
```

这种 Python 循环 + 逐元素操作效率极低。用 Triton 可以将其 **并行化**，所有 token 同时写入各自的 slot。

### 2.2 kernel 源码完整解析

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr,           # Key 张量的起始指针
    key_stride,        # Key 张量第 0 维的步长（stride）
    value_ptr,         # Value 张量的起始指针
    value_stride,      # Value 张量第 0 维的步长
    k_cache_ptr,       # K Cache 的起始指针
    v_cache_ptr,       # V Cache 的起始指针
    slot_mapping_ptr,  # slot 映射表的起始指针
    D: tl.constexpr    # 每个 token 的 K/V 向量总维度 (num_heads * head_dim)
):
    # 获取当前 program 的 ID，每个 program 处理一个 token
    idx = tl.program_id(0)

    # 读取这个 token 应写入的 slot 编号
    slot = tl.load(slot_mapping_ptr + idx)

    # 哨兵值：slot == -1 表示无效 token（padding），直接跳过
    if slot == -1:
        return

    # 计算 Key 和 Value 在源张量中的偏移
    # key 形状为 [N, num_heads, head_dim]，展平后 key[idx] 从 idx * key_stride 开始
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    # 从源张量加载 K 和 V 向量
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # 计算在 cache 中的目标偏移
    # cache 形状为 [total_slots, D]，slot 对应的起始位置是 slot * D
    cache_offsets = slot * D + tl.arange(0, D)

    # 写入 K Cache 和 V Cache
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

### 2.3 逐行深度解析

#### `idx = tl.program_id(0)`

- `tl.program_id(axis)` 返回当前 program 在指定轴上的索引。
- `axis=0` 表示第一个维度。Triton 支持最多 3 个轴的 program grid（类似 CUDA 的 3D grid）。
- 在本 kernel 中，grid 大小为 `(N,)`，即每个 program 处理第 `idx` 个 token。

#### `slot = tl.load(slot_mapping_ptr + idx)`

- 从 `slot_mapping` 数组中读取第 `idx` 个 token 对应的目标 slot 编号。
- `slot_mapping` 是一个 1D 张量，`slot_mapping_ptr + idx` 是指针算术。

#### `if slot == -1: return`

- `-1` 是哨兵值（sentinel value），代表这个 token 是 **padding**，不需要写入 KV Cache。
- 在 CUDA Graph 的 `run_model` 中，`graph_vars["slot_mapping"].fill_(-1)` 就是把多余的位置标记为 `-1`，与此处约定一致。
- Triton 支持简单的 `if` 条件分支，但编译器会将其转化为 **predicated execution**（条件执行），而非传统的分支跳转。

#### `key_offsets = idx * key_stride + tl.arange(0, D)`

这是 Triton 中非常典型的 **block-level 索引计算**：

- `key_stride` 是 Key 张量在 token 维度上的步长。对于 shape 为 `[N, num_heads, head_dim]` 的连续张量，`key_stride = num_heads * head_dim = D`。
- `tl.arange(0, D)` 生成 `[0, 1, 2, ..., D-1]` 的向量。
- `idx * key_stride + tl.arange(0, D)` 得到第 `idx` 个 token 的全部 D 个元素的偏移量。

示意图（假设 D=4）：

```
key 内存布局：
[token0_k0, token0_k1, token0_k2, token0_k3, token1_k0, token1_k1, ...]
 ↑ idx=0 的偏移: [0, 1, 2, 3]      ↑ idx=1 的偏移: [4, 5, 6, 7]
```

#### `key = tl.load(key_ptr + key_offsets)`

- **向量化加载**：一次性从全局内存加载 D 个元素到寄存器中。
- Triton 编译器会自动将这些访问 **合并（coalesce）** 为最少次数的内存事务。
- 在 CUDA C++ 中实现等价功能，你需要手动处理 `__shared__` 内存和 `__syncthreads()`。

#### `cache_offsets = slot * D + tl.arange(0, D)`

- KV Cache 的内存布局：每个 slot 占用连续的 D 个元素。
- `slot * D` 是第 `slot` 个槽位的起始位置。
- `+ tl.arange(0, D)` 覆盖该槽位内的所有元素。

#### `tl.store(k_cache_ptr + cache_offsets, key)`

- **向量化写入**：将 D 个元素一次性写入 KV Cache 的目标位置。
- 由于不同 program 写入不同的 slot，不存在写冲突，无需同步。

### 2.4 关键设计总结

| 设计点 | 说明 |
|--------|------|
| 每个 program 处理一个 token | 与 token 数量 N 对应，最大化并行度 |
| `D` 作为 `tl.constexpr` | 编译期固定，允许编译器做循环展开等优化 |
| `slot == -1` 跳过 | 与 padding 和 CUDA Graph 的 `fill_(-1)` 配合 |
| 无共享内存、无同步 | 各 program 完全独立，避免同步开销 |
| 使用 `stride` 而非假设连续 | 通用性更强，支持非连续内存布局 |

---

## 三、源码对照：`store_kvcache` 包装函数

```python
def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    store_kvcache_kernel[(N,)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache,
        slot_mapping,
        D
    )
```

### 3.1 参数解析

| 参数 | 形状 | 含义 |
|------|------|------|
| `key` | `[N, num_heads, head_dim]` | 当前步 Attention 计算的 Key |
| `value` | `[N, num_heads, head_dim]` | 当前步 Attention 计算的 Value |
| `k_cache` | `[total_slots, D]` | 全局 K Cache（所有序列共享） |
| `v_cache` | `[total_slots, D]` | 全局 V Cache |
| `slot_mapping` | `[N]` | token → slot 映射 |

### 3.2 Grid 配置

```python
store_kvcache_kernel[(N,)](...)
```

`[(N,)]` 定义了 Triton kernel 的 **grid 大小**，等价于 CUDA 的 `gridDim`。这里只用了 1 维 grid，`N` 个 program 分别处理 `N` 个 token。

在 CUDA 中，等价写法大致为：

```c++
store_kvcache_kernel<<<N, 1>>>(key, key_stride, value, value_stride, ...);
```

但 Triton 中每个 "program" 内部可以处理多个元素（通过 `tl.arange`），所以不需要像 CUDA 那样显式指定 `blockDim`。

### 3.3 `key.stride(0)` 的含义

`key.stride(0)` 返回 `key` 张量第 0 维的 **步长**（单位是元素个数，不是字节）。

对于 shape `[N, num_heads, head_dim]` 的连续张量：
- `stride(0) = num_heads * head_dim = D`
- `stride(1) = head_dim`
- `stride(2) = 1`

传入 stride 而非硬编码 D，使得 kernel 能处理非连续内存布局（如 `key.permute(...)` 后的张量）。

### 3.4 隐式参数传递

Triton 会自动将 PyTorch tensor 转换为 GPU 指针：

- `key` → `key_ptr`（tensor 的 `.data_ptr()`）
- `key.stride(0)` → `key_stride`（Python int，作为 kernel 参数传入）
- `D` → 编译期常量（`tl.constexpr`）

---

## 四、Triton 编程进阶概念

### 4.1 编译与缓存机制

```
首次调用 → Triton 编译器将 Python kernel → LLVM IR → PTX → SASS
           ↓
      缓存编译结果（~/.triton/cache/）
           ↓
后续调用 → 直接加载缓存的二进制
```

编译触发条件（参数特化）：
- `tl.constexpr` 参数值变化
- 输入张量的 dtype 变化
- 新的 `num_warps`、`num_stages` 配置

### 4.2 内存访问优化

Triton 编译器自动进行：

1. **访问合并（Coalescing）**：当多个线程访问连续地址时，合并为一次内存事务。
2. **向量化（Vectorization）**：将多个标量 load/store 合并为 128-bit 宽的向量操作。
3. **寄存器分配**：自动决定哪些中间值放在寄存器中。

开发者需要注意的：
- 尽量让 `tl.arange` 生成的偏移量是 **连续的**，便于合并访问。
- `D` 对齐到 2 的幂次，有利于编译器优化。

### 4.3 与 CUDA Graph 的配合

Triton kernel 与 CUDA Graph 完全兼容。在 `capture_cudagraph` 期间：

1. kernel 首次执行时完成 JIT 编译（warmup 阶段）。
2. capture 阶段的 `torch.cuda.graph(...)` 会录制 Triton kernel 的调用。
3. replay 时，Triton kernel 像普通 CUDA kernel 一样被重放。

因此 `store_kvcache_kernel` 在 Graph capture 期间的行为与普通 CUDA kernel 一致，不需要特殊处理。

### 4.4 Triton 在 LLM 推理中的其他应用

nano-vllm 中 `store_kvcache_kernel` 只是最简单的 Triton kernel。业界常用 Triton 实现的 LLM 相关 kernel 包括：

| Kernel | 功能 | 复杂度 |
|--------|------|--------|
| Flash Attention | 融合的 QKV 注意力计算 | 高 |
| RMSNorm | Root Mean Square 归一化 | 低 |
| Rotary Embedding | RoPE 位置编码 | 中 |
| SiLU + Mul | 门控激活函数 | 低 |
| FP8 矩阵乘 | 低精度矩阵乘法 | 高 |

---

## 五、动手实验：理解 Triton kernel 的执行

### 5.1 可视化执行过程

假设有 3 个 token 需要写入 KV Cache，`D=4`：

```
输入：
  key = [[k00, k01, k02, k03],    # token 0
         [k10, k11, k12, k13],    # token 1
         [k20, k21, k22, k23]]    # token 2

  slot_mapping = [5, -1, 2]       # token 1 是 padding

Grid = (3,)  →  启动 3 个 program

Program 0 (idx=0):
  slot = slot_mapping[0] = 5
  slot != -1 → 继续
  key_offsets = [0, 1, 2, 3]
  key = [k00, k01, k02, k03]
  cache_offsets = [20, 21, 22, 23]   # slot=5, 5*4=20
  写入 k_cache[20:24] = [k00, k01, k02, k03]

Program 1 (idx=1):
  slot = slot_mapping[1] = -1
  slot == -1 → return（跳过）

Program 2 (idx=2):
  slot = slot_mapping[2] = 2
  slot != -1 → 继续
  key_offsets = [8, 9, 10, 11]
  key = [k20, k21, k22, k23]
  cache_offsets = [8, 9, 10, 11]     # slot=2, 2*4=8
  写入 k_cache[8:12] = [k20, k21, k22, k23]
```

### 5.2 常见调试方法

1. **用 `triton.testing.do_bench` 测性能**：
   ```python
   ms = triton.testing.do_bench(lambda: store_kvcache(key, value, k_cache, v_cache, slot_mapping))
   ```

2. **用 CPU 参考实现验证正确性**：
   ```python
   for i in range(N):
       slot = slot_mapping[i].item()
       if slot != -1:
           assert torch.allclose(k_cache[slot], key[i].reshape(-1))
   ```

3. **查看编译后的 PTX/SASS**：
   ```python
   print(store_kvcache_kernel.cache[0].asm["ptx"])
   ```

---

## 六、小结

- **Triton** 用 Python 语法编写 GPU kernel，比 CUDA C++ 开发效率高，性能接近。
- 核心思维转变：从 CUDA 的 **线程级并行** 到 Triton 的 **Block 级并行**。
- `store_kvcache_kernel` 是一个 **经典的 element-wise scatter kernel**：每个 program 读取一个 token 的 K/V，根据 `slot_mapping` 写入 KV Cache 的目标位置。
- `tl.constexpr` 让 `D` 在编译期固化，编译器可以做循环展开。
- `slot == -1` 的哨兵值设计与 CUDA Graph 的 `fill_(-1)` 配合，确保 padding token 不污染 KV Cache。
- `key.stride(0)` 传入 stride 使 kernel 能处理非连续内存。

---

## 七、面试考点（含参考答案）

**1. Triton 与 CUDA 编程的核心区别是什么？**
**答**：最核心的区别是 **并行粒度**。CUDA 是线程级（thread-level）编程，开发者需要手动管理每个线程的行为、共享内存、线程同步等；Triton 是 Block 级（block-level）编程，开发者定义每个 program 处理一块数据，线程分配、共享内存管理、内存合并等由编译器自动完成。Triton 用 Python 语法，开发效率显著高于 CUDA C++。

**2. `@triton.jit` 的作用是什么？**
**答**：标记函数为 JIT 编译的 GPU kernel。首次调用时 Triton 编译器将 Python 代码编译为 GPU 机器码（经过 LLVM IR → PTX → SASS 流水线），编译结果会被缓存。它类似 CUDA 的 `__global__` 关键字，但支持类型推导和编译期特化。

**3. `tl.program_id(0)` 对应 CUDA 中的什么？**
**答**：对应 CUDA 中的 `blockIdx.x`，返回当前执行单元在 grid 第 0 维上的索引。但与 CUDA 不同的是，Triton 的 "program" 内部通过 `tl.arange` 等操作处理多个元素，所以一个 Triton program 在功能上更像一个 CUDA block。

**4. 为什么 `store_kvcache_kernel` 中 `D` 要声明为 `tl.constexpr`？**
**答**：`tl.constexpr` 使 `D` 在 **编译期固定**，编译器可以据此做 **循环展开、向量化指令选择** 等优化。如果 D 是运行时变量，编译器无法确定循环次数，生成的代码效率会降低。代价是当 D 变化时需要重新编译。

**5. `slot_mapping` 中 `-1` 的作用是什么？**
**答**：`-1` 是哨兵值，表示该位置是 padding token（无效），kernel 检测到 `slot == -1` 时直接 `return`，不写入 KV Cache。这与 CUDA Graph 中 `graph_vars["slot_mapping"].fill_(-1)` 配合：graph_vars 的多余位置被标记为 `-1`，避免向 KV Cache 写入垃圾数据。

**6. `store_kvcache` 包装函数中 `[(N,)]` 是什么意思？**
**答**：定义 Triton kernel 的 **grid 大小**，等价于 CUDA 的 `gridDim`。`(N,)` 表示一维 grid，启动 N 个 program，每个 program 处理一个 token。Triton 中 grid 写在方括号 `[]` 中，跟在 kernel 函数名之后。

**7. 为什么传入 `key.stride(0)` 而不直接用 `D`？**
**答**：使用 `stride` 使 kernel 能处理 **非连续（non-contiguous）** 的张量。虽然对于连续张量 `stride(0) == D`，但如果 key 经过 `transpose`、`permute` 等操作后变成非连续的，stride 与 D 可能不同。传入 stride 是更安全、更通用的做法。

**8. Triton kernel 的 JIT 编译有什么优缺点？**
**答**：
- **优点**：可以针对具体的 `constexpr` 参数、数据类型生成特化代码，性能更优；开发时像写 Python 一样方便。
- **缺点**：首次调用有编译开销（通常几百毫秒到几秒）；不同参数组合需要分别编译；调试比 PyTorch 原生代码更困难。
- 在推理场景中，编译开销是一次性的（且会被缓存），因此 JIT 是可以接受的。

**9. 这个 kernel 的性能瓶颈在哪里？**
**答**：主要是 **全局内存带宽（Global Memory Bandwidth）**。kernel 的计算量极小（只有指针算术和比较），数据量是 `2 * N * D` 个元素的读写（K 和 V 各一份）。优化方向包括：确保内存访问合并、D 对齐到合适的粒度。对于 LLM 推理来说，这个 kernel 通常不是整体瓶颈，因为 Attention 和 FFN 的矩阵乘更耗时。

**10. Triton 生态在 LLM 推理中的地位如何？**
**答**：Triton 已成为 LLM 推理系统编写自定义 kernel 的主流工具。Flash Attention 2/3 有 Triton 实现、vLLM 的多个核心 kernel（如 PagedAttention）使用 Triton、PyTorch 的 `torch.compile` 后端 Inductor 也生成 Triton 代码。相比 CUDA C++，Triton 的开发迭代速度更快，与 PyTorch 生态集成更紧密。

---

*延伸阅读：Triton 官方教程（triton-lang.org）、OpenAI Triton 论文（MAPL 2019）。*
