# Qwen

Alibaba 的开放权重模型家族。状态：**待适配**。

## 家族特征

- decoder-only
- RMSNorm
- SwiGLU
- RoPE
- GQA
- Qwen2 系列带 attention bias
- Qwen3 有 MoE 型号

## 常见型号

| 型号 | 规模 | 上下文 | 注意力 | MoE | 多模态 | StarLight 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5 0.5B/1.5B/3B/7B/14B/32B/72B | 0.5B–72B | 32K–128K | GQA | 否 | 否 | 受限于引擎能力（attention bias） |
| Qwen3 0.6B/1.7B/4B/8B/14B/32B | 0.6B–32B | 32K–128K | GQA | 否 | 否 | 待适配（QK-norm/attention 待确认） |
| Qwen3 30B-A3B | 30B / 3B active | 32K | GQA | 是 | 否 | 受限于引擎能力（MoE） |
| Qwen3 235B-A22B | 235B / 22B active | 32K | GQA | 是 | 否 | 受限于引擎能力（MoE） |
| Qwen2.5-VL 3B/7B/72B | 3B–72B | 128K | GQA | 否 | 是 | 受限于引擎能力（multimodal） |

状态含义见 [families/README.md](../README.md)。

## 适配笔记

Qwen2.5 的 q/k/v bias 与当前 `MatMulNT` 无 bias 契约冲突；Qwen3 dense 型号的 QK-Norm/attention 细节需要按 config 确认后再标为结构兼容。MoE 型号需要专家路由能力。

## 参考

- <https://qwen.ai/>
- <https://huggingface.co/Qwen>
