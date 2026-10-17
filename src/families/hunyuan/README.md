# Hunyuan

Tencent 的开放权重模型家族。状态：**待适配**。

## 家族特征

- decoder-only
- RMSNorm
- SwiGLU
- RoPE
- GQA
- MoE（Large）

## 常见型号

| 型号 | 规模 | 上下文 | 注意力 | MoE | 多模态 | StarLight 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| Hunyuan 7B/13B | 7B / 13B | 32K | GQA | 否 | 否 | 待适配 |
| Hunyuan-Large 389B-A52B | 389B / 52B active | 256K | GQA | 是 | 否 | 受限于引擎能力（MoE） |

状态含义见 [families/README.md](../README.md)。

## 适配笔记

Hunyuan-Large 使用 MoE；Hunyuan 7B/13B dense 可作为候选，但需确认 tokenizer 与 config。

## 参考

- <https://github.com/Tencent-Hunyuan>
- <https://huggingface.co/tencent>
