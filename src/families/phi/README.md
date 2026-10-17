# Phi

Microsoft 的开放权重模型家族。状态：**待适配**。

## 家族特征

- decoder-only
- RMSNorm/LayerNorm（按版本）
- SwiGLU
- RoPE
- GQA（部分型号）

## 常见型号

| 型号 | 规模 | 上下文 | 注意力 | MoE | 多模态 | StarLight 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| Phi-2 2.7B | 2.7B | 2K | MHA | 否 | 否 | 待适配 |
| Phi-3 mini 3.8B | 3.8B | 4K/128K | GQA | 否 | 否 | 待适配 |
| Phi-3 small 7B / medium 14B | 7B / 14B | 8K/128K | GQA/MHA | 否 | 否 | 待适配 |
| Phi-4 14B | 14B | 16K | GQA | 否 | 否 | 待适配 |

状态含义见 [families/README.md](../README.md)。

## 适配笔记

不同 Phi 版本的 norm、位置编码与 tokenizer 差异较大，需要按型号建独立 config。

## 参考

- <https://azure.microsoft.com/en-us/products/phi>
- <https://huggingface.co/microsoft>
