# Llama

Meta 的开放权重模型家族。状态：**已实现**。

## 家族特征

- decoder-only
- RMSNorm
- SwiGLU
- RoPE（NeoX）
- GQA（70B/8B 等）
- 无 bias

## 常见型号

| 型号 | 规模 | 上下文 | 注意力 | MoE | 多模态 | StarLight 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| Llama 2 7B/13B/70B | 7B / 13B / 70B | 4K | MHA（7B/13B）、GQA（70B） | 否 | 否 | 结构兼容 |
| Llama 3 8B/70B | 8B / 70B | 8K | GQA | 否 | 否 | 结构兼容 |
| Llama 3.1 8B/70B/405B | 8B / 70B / 405B | 128K | GQA | 否 | 否 | 待适配（rope_scaling） |
| Llama 3.2 1B/3B | 1B / 3B | 128K | GQA | 否 | 否 | 待适配（rope_scaling） |
| Llama 3.3 70B | 70B | 128K | GQA | 否 | 否 | 待适配（rope_scaling） |
| Llama 3.2 Vision 11B/90B | 11B / 90B | 128K | GQA | 否 | 是 | 受限于引擎能力（multimodal） |

状态含义见 [families/README.md](../README.md)。

## 适配笔记

首个已实现家族。`sl-model-adapters` 的 v1 dense lowering 覆盖 RMSNorm + NeoX RoPE + CausalAttention + SwiGLU；Llama 3.1/3.2/3.3 的 rope_scaling 尚未实现。

## 参考

- <https://www.llama.com/models/>
- <https://huggingface.co/meta-llama>
