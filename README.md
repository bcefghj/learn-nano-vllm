# nano-vllm 面试导向学习指南

> 面向小白的大模型推理引擎学习指南 | 20节课程 + 60+面试八股文 + STAR面试法 + 项目简历指南

## 项目简介

本项目是一份**面向面试**的 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 全面学习指南。nano-vllm 是由 DeepSeek 工程师俞星凯开发的轻量级 vLLM 实现，仅约 1200 行 Python 代码，却实现了生产级推理框架的核心技术。

通过学习本指南，你将：
- 从零理解大模型推理引擎的核心原理
- 掌握 PagedAttention、连续批处理、张量并行等关键技术
- 准备好面试中关于推理引擎的所有问题
- 获得一份可以直接用于简历的项目经验

## 目录导航

### 课程部分（20节）

| 课程 | 主题 | 核心内容 |
|------|------|---------|
| [导读](docs/00-导读-项目概览.md) | 项目概览 | 学习路线、nano-vllm 全景 |
| [第1课](docs/01-课程01-认识大模型推理.md) | 认识大模型推理 | Prefill/Decode、计算瓶颈 |
| [第2课](docs/02-课程02-nano-vllm项目全景.md) | 项目全景 | 目录结构、架构分层 |
| [第3课](docs/03-课程03-配置与入口.md) | 配置与入口 | Config、SamplingParams、example.py |
| [第4课](docs/04-课程04-Tokenizer与Embedding.md) | Tokenizer与Embedding | 词表并行、VocabParallelEmbedding |
| [第5课](docs/05-课程05-Attention机制与FlashAttention.md) | Attention机制 | FlashAttention、Triton KV存储 |
| [第6课](docs/06-课程06-RoPE旋转位置编码.md) | RoPE旋转位置编码 | 数学推导、apply_rotary_emb |
| [第7课](docs/07-课程07-LayerNorm与激活函数.md) | LayerNorm与激活函数 | RMSNorm、SwiGLU、torch.compile |
| [第8课](docs/08-课程08-Qwen3模型架构.md) | Qwen3模型架构 | GQA、MLP、DecoderLayer |
| [第9课](docs/09-课程09-KV-Cache原理与实现.md) | KV Cache原理 | 显存计算、缓存复用 |
| [第10课](docs/10-课程10-PagedAttention与BlockManager.md) | PagedAttention | Block分配回收、前缀缓存 |
| [第11课](docs/11-课程11-Sequence与请求管理.md) | Sequence请求管理 | 序列状态机、block_table |
| [第12课](docs/12-课程12-Scheduler调度器.md) | Scheduler调度器 | Prefill/Decode调度、抢占 |
| [第13课](docs/13-课程13-连续批处理.md) | 连续批处理 | 动态批处理、GPU利用率 |
| [第14课](docs/14-课程14-ModelRunner模型执行器.md) | ModelRunner执行器 | KV Cache分配、NCCL通信 |
| [第15课](docs/15-课程15-张量并行TP.md) | 张量并行 | 列/行并行、AllReduce |
| [第16课](docs/16-课程16-CUDA-Graph优化.md) | CUDA Graph优化 | 图捕获、replay机制 |
| [第17课](docs/17-课程17-Triton-Kernel编写.md) | Triton Kernel | store_kvcache_kernel |
| [第18课](docs/18-课程18-LLMEngine推理循环.md) | LLMEngine推理循环 | generate→step完整流程 |
| [第19课](docs/19-课程19-性能基准与优化.md) | 性能基准与优化 | bench.py、吞吐量测试 |
| [第20课](docs/20-课程20-完整项目串讲.md) | 完整串讲 | 端到端流程、面试要点 |

### 面试准备

| 文档 | 内容 |
|------|------|
| [面试八股文大全](docs/21-面试八股文大全.md) | 60+道面试题 + 详细答案 |
| [项目简历撰写指南](docs/22-项目简历撰写指南.md) | 如何在简历中写 nano-vllm 项目 |
| [STAR面试稿](docs/23-STAR面试稿.md) | STAR法面试自我介绍模板 |
| [面试问题全集](docs/24-面试问题全集-STAR回答.md) | 全部可能问题 + STAR回答 |
| [岗位需求汇总](docs/25-岗位需求与招聘汇总.md) | 2026年大厂推理岗招聘信息 |
| [学习资源](docs/26-学习资源与参考.md) | 学习路线、参考链接 |

## nano-vllm 架构一览

```
用户请求 → LLM.generate()
              ↓
         LLMEngine.add_request()  → Tokenizer编码
              ↓
         Scheduler.schedule()     → Prefill优先 / Decode轮转
              ↓
         ModelRunner.run()        → 准备输入 → 模型前向 → 采样
              ↓
         Scheduler.postprocess()  → 追加token / 判断结束
              ↓
         返回生成结果
```

## 核心技术栈

| 技术 | nano-vllm中的实现 | 对应文件 |
|------|-------------------|---------|
| PagedAttention | BlockManager + xxhash前缀缓存 | `engine/block_manager.py` |
| 连续批处理 | Scheduler Prefill/Decode调度 | `engine/scheduler.py` |
| FlashAttention | varlen_func + with_kvcache | `layers/attention.py` |
| 张量并行 | Column/Row/QKV ParallelLinear | `layers/linear.py` |
| CUDA Graph | capture + replay | `engine/model_runner.py` |
| Triton Kernel | store_kvcache_kernel | `layers/attention.py` |
| torch.compile | RMSNorm/RoPE/SiLU/Sampler | 各layers文件 |

## 性能对比

在 RTX 4070 Laptop 上（Qwen3-0.6B，256序列）：

| 推理引擎 | 吞吐量 | 显存占用 | 启动时间 |
|---------|--------|---------|---------|
| vLLM | 1361.84 tok/s | ~4.2GB | ~15s |
| nano-vllm | 1434.13 tok/s | ~3.8GB | ~3s |

## 快速开始

```bash
# 克隆本学习指南
git clone https://github.com/bcefghj/learn-nano-vllm-interview.git

# 克隆 nano-vllm 源码
git clone https://github.com/GeeeekExplorer/nano-vllm.git

# 按照课程顺序学习
# 从 docs/00-导读-项目概览.md 开始
```

## 适用人群

- 准备 AI 推理方向面试的求职者
- 想要理解大模型推理引擎原理的初学者
- 希望在简历中展示推理引擎项目经验的同学

## 参考资源

- [nano-vllm GitHub](https://github.com/GeeeekExplorer/nano-vllm)
- [博客园源码解析](https://www.cnblogs.com/cswuyg/p/19471225)
- [d.run 学习教程](https://docs.d.run/blogs/2026/nano-vllm.html)
- [Flaneur2020 Walkthrough](https://flaneur2020.github.io/posts/2025-10-12-nano-vllm/)

## License

MIT License
