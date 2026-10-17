# ERNIE

Baidu 的开放权重模型家族。状态：**待适配**。

## 家族特征

- decoder-only
- RMSNorm
- SwiGLU
- RoPE
- MoE（多型号）
- 多模态（部分）

## 常见型号

| 型号 | 规模 | 上下文 | 注意力 | MoE | 多模态 | StarLight 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| ERNIE 4.5 0.3B–21B dense | 0.3B–21B | 4K–128K | GQA | 否 | 否 | 待适配 |
| ERNIE 4.5 MoE 系列 | 21B–424B | 128K | GQA | 是 | 否 | 受限于引擎能力（MoE） |
| ERNIE 4.5 VL 系列 | — | — | GQA | 是 | 是 | 受限于引擎能力（MoE + multimodal） |

状态含义见 [families/README.md](../README.md)。

## 适配笔记

ERNIE 4.5 开源系列包含 dense 与 MoE、文本与多模态混合，需要按型号分别建 config。

## 参考

- <https://ernie.baidu.com/>
- <https://github.com/PaddlePaddle/ERNIE>
