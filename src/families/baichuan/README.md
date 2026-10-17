# Baichuan

Baichuan AI 的开放权重模型家族。状态：**待适配**。

## 家族特征

- decoder-only
- RMSNorm
- SwiGLU
- RoPE/ALiBi（按版本）
- GQA

## 常见型号

| 型号 | 规模 | 上下文 | 注意力 | MoE | 多模态 | StarLight 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| Baichuan 7B/13B | 7B / 13B | 4K | MHA | 否 | 否 | 待适配（attention variant 待确认） |
| Baichuan2 7B/13B | 7B / 13B | 4K | MHA | 否 | 否 | 待适配（attention variant 待确认） |

状态含义见 [families/README.md](../README.md)。

## 适配笔记

Baichuan 1 与 Baichuan2 的 attention/位置编码差异需要分别确认，不能套用同一 config。

## 参考

- <https://github.com/baichuan-inc>
- <https://huggingface.co/baichuan-inc>
