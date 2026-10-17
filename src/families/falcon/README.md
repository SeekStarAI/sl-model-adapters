# Falcon

TII 的开放权重模型家族。状态：**待适配**。

## 家族特征

- decoder-only
- multiquery attention
- ALiBi（Falcon 1）
- RoPE（Falcon 3）
- parallel attention

## 常见型号

| 型号 | 规模 | 上下文 | 注意力 | MoE | 多模态 | StarLight 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| Falcon 7B/40B | 7B / 40B | 2K | MQA | 否 | 否 | 受限于引擎能力（ALiBi + MQA） |
| Falcon 2 11B | 11B | 8K | MQA | 否 | 否 | 受限于引擎能力（ALiBi + MQA） |
| Falcon 3 1B/3B/7B/10B | 1B–10B | 32K | GQA | 否 | 否 | 待适配 |

状态含义见 [families/README.md](../README.md)。

## 适配笔记

Falcon 1/2 的 ALiBi 与 multiquery 不在当前算子集内；Falcon 3 需要单独确认位置编码。

## 参考

- <https://github.com/tiiuae>
- <https://huggingface.co/tiiuae>
