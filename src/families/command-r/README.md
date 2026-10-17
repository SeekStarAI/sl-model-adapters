# Command R

Cohere 的开放权重模型家族。状态：**待适配**。

## 家族特征

- decoder-only
- RMSNorm
- SwiGLU
- RoPE
- GQA

## 常见型号

| 型号 | 规模 | 上下文 | 注意力 | MoE | 多模态 | StarLight 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| Command R 35B | 35B | 128K | GQA | 否 | 否 | 待适配 |
| Command R+ 104B | 104B | 128K | GQA | 否 | 否 | 待适配 |

状态含义见 [families/README.md](../README.md)。

## 适配笔记

许可证为非商业；Command R/R+ 面向 RAG 与工具调用，适配优先级取决于商用需求。

## 参考

- <https://cohere.com/command>
- <https://huggingface.co/CohereForAI>
