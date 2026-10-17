# GLM

Zhipu AI 的开放权重模型家族。状态：**待适配**。

## 家族特征

- decoder-only
- RMSNorm
- SwiGLU
- RoPE（部分维度）
- GQA
- MoE（4.5）

## 常见型号

| 型号 | 规模 | 上下文 | 注意力 | MoE | 多模态 | StarLight 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| ChatGLM2/3 6B | 6B | 8K/32K | MHA | 否 | 否 | 待适配（partial RoPE） |
| GLM-4-9B | 9B | 128K | GQA | 否 | 否 | 待适配（partial RoPE） |
| GLM-4.5 355B-A32B | 355B / 32B active | 128K | GQA | 是 | 否 | 受限于引擎能力（MoE + partial RoPE） |
| GLM-4.5-Air 106B-A12B | 106B / 12B active | 128K | GQA | 是 | 否 | 受限于引擎能力（MoE + partial RoPE） |
| GLM-4.1V 9B | 9B | 64K | GQA | 否 | 是 | 受限于引擎能力（multimodal + partial RoPE） |

状态含义见 [families/README.md](../README.md)。

## 适配笔记

ChatGLM/GLM 的 partial RoPE 与 GLM-4.5 的 MoE 都需要新算子；GLM-4-9B 可作为 dense 候选。

## 参考

- <https://github.com/zai-org>
- <https://huggingface.co/zai-org>
