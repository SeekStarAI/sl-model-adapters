# Mistral

Mistral AI 的开放权重模型家族。状态：**待适配**。

## 家族特征

- decoder-only
- RMSNorm
- SwiGLU
- RoPE
- GQA
- 滑动窗口（Mistral 7B/Mixtral）
- MoE（Mixtral）

## 常见型号

| 型号 | 规模 | 上下文 | 注意力 | MoE | 多模态 | StarLight 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| Mistral 7B v0.3 | 7B | 32K | GQA | 否 | 否 | 受限于引擎能力（sliding window） |
| Mixtral 8x7B | 46.7B / 12.9B active | 32K | GQA | 是 | 否 | 受限于引擎能力（MoE + sliding window） |
| Mixtral 8x22B | 141B / 39B active | 64K | GQA | 是 | 否 | 受限于引擎能力（MoE + sliding window） |
| Mistral NeMo 12B | 12B | 128K | GQA | 否 | 否 | 待适配（attention/SWA 待确认） |
| Mistral Small 3 24B | 24B | 32K–128K | GQA | 否 | 否 | 待适配（attention/SWA 待确认） |
| Mistral Large 123B | 123B | 128K | GQA | 否 | 否 | 待适配（attention/SWA 待确认） |
| Pixtral 12B | 12B | 128K | GQA | 否 | 是 | 受限于引擎能力（multimodal） |

状态含义见 [families/README.md](../README.md)。

## 适配笔记

Mistral 7B/Mixtral 的滑动窗口与当前 CausalAttention 语义不兼容；Mistral NeMo/Small/Large 的 attention 与 SWA 细节需要按 config 确认后再标为结构兼容。

## 参考

- <https://mistral.ai/models/>
- <https://huggingface.co/mistralai>
