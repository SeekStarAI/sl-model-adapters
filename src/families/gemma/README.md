# Gemma

Google 的开放权重模型家族。状态：**待适配**。

## 家族特征

- decoder-only
- RMSNorm
- GeGLU
- RoPE
- GQA（Gemma 2/3）
- 滑动窗口（Gemma 2/3）
- 多模态（Gemma 3）

## 常见型号

| 型号 | 规模 | 上下文 | 注意力 | MoE | 多模态 | StarLight 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma 2B/7B | 2B / 7B | 8K | MHA | 否 | 否 | 受限于引擎能力（GeGLU） |
| Gemma 2 2B/9B/27B | 2B / 9B / 27B | 8K | GQA | 否 | 否 | 受限于引擎能力（GeGLU + sliding window） |
| Gemma 3 1B/4B/12B/27B | 1B–27B | 128K | GQA | 否 | 是 | 受限于引擎能力（GeGLU + sliding window + multimodal） |

状态含义见 [families/README.md](../README.md)。

## 适配笔记

GeGLU 与滑动窗口均未支持；Gemma 3 还需要视觉输入。

## 参考

- <https://ai.google.dev/gemma>
- <https://huggingface.co/google>
