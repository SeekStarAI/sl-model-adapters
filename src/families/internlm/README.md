# InternLM

Shanghai AI Lab 的开放权重模型家族。状态：**待适配**。

## 家族特征

- decoder-only
- RMSNorm
- SwiGLU
- RoPE
- GQA

## 常见型号

| 型号 | 规模 | 上下文 | 注意力 | MoE | 多模态 | StarLight 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| InternLM2 7B/20B | 7B / 20B | 32K | GQA | 否 | 否 | 待适配 |
| InternLM2.5 7B/20B | 7B / 20B | 32K | GQA | 否 | 否 | 待适配 |
| InternLM3 8B | 8B | 32K | GQA | 否 | 否 | 待适配 |

状态含义见 [families/README.md](../README.md)。

## 适配笔记

整体接近 Llama dense，需要确认 attention bias 与 tokenizer；InternLM3 8B 可作为候选。

## 参考

- <https://github.com/InternLM/InternLM>
- <https://huggingface.co/internlm>
