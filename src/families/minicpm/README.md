# MiniCPM

OpenBMB 的开放权重模型家族。状态：**待适配**。

## 家族特征

- decoder-only
- RMSNorm
- SwiGLU
- RoPE
- GQA
- scale_emb
- 多模态（MiniCPM-V）

## 常见型号

| 型号 | 规模 | 上下文 | 注意力 | MoE | 多模态 | StarLight 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| MiniCPM 2.4B | 2.4B | 4K | MHA | 否 | 否 | 待适配（scale_emb） |
| MiniCPM3 4B | 4B | 32K | GQA | 否 | 否 | 待适配（scale_emb） |
| MiniCPM4 0.5B/8B | 0.5B / 8B | 32K/128K | GQA | 否 | 否 | 待适配 |
| MiniCPM-V 2.6/4.0 | 8B | 32K | GQA | 否 | 是 | 受限于引擎能力（multimodal） |

状态含义见 [families/README.md](../README.md)。

## 适配笔记

MiniCPM 使用 embedding scaling 与自定义 tokenizer 细节，需要按版本确认。

## 参考

- <https://github.com/OpenBMB/MiniCPM>
- <https://huggingface.co/openbmb>
