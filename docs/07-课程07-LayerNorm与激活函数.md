# 课程07：RMSNorm 与激活函数

> 用「均方根缩放」替代 LayerNorm 的均值中心化，配合 SwiGLU 门控，构成现代 LLM 解码层里稳定又高效的标准配方。

## 本课目标

- 对比 **RMSNorm** 与 **LayerNorm** 的公式差异及工程取舍。
- 理解 `rsqrt`、混合精度路径（先 float 再回写 dtype）的原因。
- 掌握 **`add_rms_forward`**：残差与归一化的融合顺序及为何能省一次大张量读写。
- 理解 **SwiGLU**：`SiluAndMul` 与 `gate_up_proj` 如何拼成门控前馈。
- 能说清 **`@torch.compile`** 在本模块中的收益与面试常见追问。

## 核心概念

### 1. LayerNorm 回顾（对照用）

对最后一维向量 \(x \in \mathbb{R}^d\)：

\[
\mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
\]

其中 \(\mu\) 为均值，\(\sigma^2\) 为方差，\(\gamma,\beta\) 为可学习仿射（部分实现可关掉 bias）。

特点：**减均值**使每向量围绕 0 居中，再按标准差缩放。

### 2. RMSNorm 公式

RMSNorm **不减均值**，只用均方根（Root Mean Square）做尺度归一：

\[
\mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}
\]

\[
\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\mathrm{RMS}(x)}
\]

对比：

| 维度 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 是否减均值 | 是 | 否 |
| 缩放依据 | 方差（中心化后） | 二阶矩开方 |
| 参数量（常见） | \(\gamma,\beta\) | 常只有 \(\gamma\)（本实现仅 `weight`） |
| 计算量 | 略高（多一次 mean） | 略低 |

直觉：深层 Transformer 里**尺度稳定**往往比**严格零均值**更关键；RMSNorm 在不少 LLM 上效果与 LN 接近或更好，且更省算。

### 3. 为什么用 `rsqrt` 而不是 `1/sqrt`

\[
\frac{x}{\sqrt{v + \epsilon}} = x \cdot \mathrm{rsqrt}(v + \epsilon)
\]

GPU 上 **`torch.rsqrt`** 常是单指令或融合更好的路径，数值上避免先 `sqrt` 再除带来的额外舍入步骤（实现细节因硬件而异）。面试可答：**融合、习惯写法、与框架算子对齐**。

### 4. `rms_forward` 的 dtype 逻辑

```text
orig_dtype = x.dtype
x = x.float()
... 在 float32 上算 var 与 rsqrt ...
x = x.to(orig_dtype).mul_(self.weight)
```

**原因简述**：fp16/bf16 上直接平方、求均值、再 `rsqrt` 容易溢出或下溢；先在 fp32 算统计量再回写，是混合精度训练/推理的常见模式。

### 5. `add_rms_forward`：融合残差的 Pre-Norm 变体

当存在 `residual` 时：

1. `x = x.float().add_(residual.float())`：在**高精度**上做 **\(x \leftarrow x + \text{residual}\)**（此处命名上 `x` 是子层输入、`residual` 是上一分支未归一化的流）。
2. `residual = x.to(orig_dtype)`：把**相加后的主路径**保存为下一轮子层的「残差分支」（典型 Pre-Norm：**norm 只对一支做，残差另一支绕开**）。
3. 对相加后的 `x` 做 RMSNorm（同 `rms_forward` 的后半段）。

这与「先 norm 再子层再加残差」在数学上需与整体 `forward` 编排一致；本仓库通过 `forward` 分流 `residual is None` 实现第一层与后续层的差异（见下一节源码）。

### 6. SwiGLU 与 `SiluAndMul`

标准 FFN 可写成 \(\mathrm{down}(\sigma(\mathrm{gate}(x)) \odot \mathrm{up}(x))\)。当 \(\sigma\) 为 **SiLU/Swish** \(z \cdot \sigma(z)\) 时，常称 **SwiGLU** 族。

本项目中 `MergedColumnParallelLinear` 一次产出 **gate 与 up 两段拼接**，`SiluAndMul` 做：

```text
x, y = chunk(2, -1)
out = silu(x) * y
```

即 **门控支路** 与 **值支路** 各一半通道，`silu(x)` 作为门。

---

## 源码解析

### `RMSNorm` 结构与 `forward` 分发

源码中 `forward` 根据是否有 `residual` 调用：

- `rms_forward(x)`：仅归一化。
- `add_rms_forward(x, residual)`：先加残差再归一化，并返回更新后的 `residual` 张量。

与 `Qwen3DecoderLayer` 配合时（务必对照 `qwen3.py`）：

```text
if residual is None:
    hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
else:
    hidden_states, residual = self.input_layernorm(hidden_states, residual)
```

- **第一层**：`residual is None` → 先执行 `input_layernorm(hidden_states)`（走 `rms_forward`），再把**未归一化的原** `hidden_states` 保存到 `residual` 变量，供后续层作为捷径分支。
- **后续层**：带 `residual` → 走 `add_rms_forward`，实现「先加残差再 RMSNorm」并更新 `residual`。

读者若只读 `RMSNorm` 会困惑，必须结合 **DecoderLayer** 一起看（面试可主动说明这一点，显得读过全链路）。

### `rms_forward` 逐行

- `var = x.pow(2).mean(dim=-1, keepdim=True)`：最后一维上的 **mean of squares**，即 \(\frac{1}{d}\|x\|_2^2\)（不写 \(1/d\) 的实现在别的库也有，本实现是 mean，与上文公式一致）。
- `x.mul_(torch.rsqrt(var + self.eps))`：原地缩放，省显存。
- `mul_(self.weight)`：可学习缩放 \(\gamma\)。

### `add_rms_forward` 逐行

- 先加再 norm，输出 `(normalized, new_residual)`，`new_residual` 为相加结果（供下一层作为「未归一化捷径」）。

### `SiluAndMul`

- `chunk(2, -1)`：要求最后一维长度为偶数，且前后两半分别对应 gate 与 value。
- `F.silu(x) * y`：逐元素；无额外 bias，简洁。

---

## 图解

### RMSNorm 数据流（无残差）

```text
  x [..., d]
      |
      +-----> pow(2) -> mean(last) -> +eps -> rsqrt ----+
      |                                                    |
      +------------------------ mul <----------------------+
      |
      * weight
      |
  output
```

### SwiGLU 与本仓库线性层

```text
  hidden
     |
  gate_up_proj (MergedColumn: 2 * intermediate 列)
     |
  +------+------+
  gate     up
  |        |
 silu      |
  \       /
   * (逐元素)
      |
   down_proj
```

### 与 DecoderLayer 的残差（概念）

```text
  ---- residual (shortcut, 未 norm) ------------------------+
  |                                                        |
  v                                                        |
  x ----> RMSNorm(add) ----> Attention ----> ...           |
```

（具体张量命名以 `qwen3.py` 为准，上图强调 **norm 与残差的分工**。）

---

## 面试考点

### RMSNorm vs LayerNorm（口述模板）

「LayerNorm 去均值再按方差缩放；RMSNorm 不去均值，只按 RMS 缩放，少一次均值计算，参数常更少。大模型里实践上 RMSNorm 很常用，和 Pre-Norm 搭配稳定。」

### 为什么 Pre-Norm 更好训（简述）

梯度更易穿过残差捷径，深层更不易梯度消失/爆炸（细节可展开 ResNet/Transformer 论文表述）。

### `torch.compile` 在这里优化什么

- 将 `rms_forward` / `add_rms_forward` / `SiluAndMul` 的小算子图**融合**，减少内核启动次数；
- 对固定 rank 形状重复调用时收益更明显。需注意：**首次编译有开销**，动态形状可能导致重编译。

### 数值稳定性

- `eps` 防止除零；
- fp32 上算统计量再 cast 回低精度。

---

## 常见面试题

1. **RMSNorm 少了减均值，会不会破坏表示？**  
   后续线性层与注意力仍可学习任意仿射偏移；实践中 RMSNorm 在 LLM 上表现良好。

2. **`rsqrt` 和 `1/sqrt` 面试怎么答？**  
   等价数学关系；`rsqrt` 是常见融合算子，利于性能与数值习惯。

3. **SwiGLU 相对 ReLU FFN 的优点？**  
   门控机制表达力更强；SiLU 光滑，优化性质较好（可结合论文与工程经验）。

4. **为何 gate 和 up 要合并成一个 `MergedColumnParallelLinear`？**  
   一次矩阵乘从 `hidden` 映射到 `2 * intermediate`，减少 kernel 次数、利于张量并行切分与内存合并访问。

5. **`add_rms_forward` 里为何先 `add_` 再 norm？**  
   与 Pre-Norm 残差编排一致：把子层输入与捷径相加后，再对该流做 RMSNorm 作为下一子层输入。

6. **LayerNorm 的 \(\beta\) 去哪了？**  
   本实现 RMSNorm 只有 `weight`，无 learnable bias，符合常见 LLaMA 系设定。

---

## 小结

RMSNorm 用 RMS 缩放替代 LN 的去中心+方差缩放，计算更轻；`rsqrt` 与 fp32 统计是稳定混合精度的习惯写法；`add_rms_forward` 把「加残差 + 归一化」捆在一起减少中间张量；`SiluAndMul` 完成 SwiGLU 的门控乘。`torch.compile` 适合这类高频小模块，但要在工程上实测编译时间与形状稳定性。

## 下一课预告

下一课 **Qwen3 模型架构**：从 `Qwen3Attention` 的 GQA、张量并行切分，到 `packed_modules_mapping` 与权重加载映射，串起整栈解码器。
