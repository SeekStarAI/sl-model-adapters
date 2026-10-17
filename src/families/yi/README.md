# Yi

01.AI 的开放权重模型家族。状态：**待适配**。

## 家族特征

- decoder-only
- RMSNorm
- SwiGLU
- RoPE
- GQA
- 无 bias

## 常见型号

| 型号 | 规模 | 上下文 | 注意力 | MoE | 多模态 | StarLight 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| Yi 6B/9B/34B | 6B / 9B / 34B | 4K/32K | GQA | 否 | 否 | 结构兼容 |
| Yi-1.5 6B/9B/34B | 6B / 9B / 34B | 4K/32K | GQA | 否 | 否 | 结构兼容 |
| Yi-Coder 1.5B/9B | 1.5B / 9B | 128K | GQA | 否 | 否 | 结构兼容 |
| Yi-VL 6B/34B | 6B / 34B | 4K | GQA | 否 | 是 | 受限于引擎能力（multimodal） |

状态含义见 [families/README.md](../README.md)。

## 适配笔记

结构与 Llama dense 接近，是低成本接入候选；需要确认 RoPE 细节与 tokenizer。

## 参考

- <https://github.com/01-ai/Yi>
- <https://huggingface.co/01-ai>
