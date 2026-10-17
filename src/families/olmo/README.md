# OLMo

Allen AI 的开放权重模型家族。状态：**待适配**。

## 家族特征

- decoder-only
- RMSNorm
- SwiGLU
- RoPE
- QK-norm（OLMo 2）
- MoE（OLMoE）

## 常见型号

| 型号 | 规模 | 上下文 | 注意力 | MoE | 多模态 | StarLight 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| OLMo 7B | 7B | 2K/4K | MHA | 否 | 否 | 待适配 |
| OLMo 2 7B/13B | 7B / 13B | 4K | GQA | 否 | 否 | 受限于引擎能力（QK-norm） |
| OLMoE 1B-7B | 6.9B / 1.3B active | 4K | GQA | 是 | 否 | 受限于引擎能力（MoE + QK-norm） |
| Molmo 7B/72B | 7B / 72B | 4K | GQA | 否 | 是 | 受限于引擎能力（multimodal） |

状态含义见 [families/README.md](../README.md)。

## 适配笔记

OLMo 2 的 QK-norm 需要新算子；OLMo 1 是较直接的 dense 候选。

## 参考

- <https://allenai.org/olmo>
- <https://huggingface.co/allenai>
