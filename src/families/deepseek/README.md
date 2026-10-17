# DeepSeek

DeepSeek-AI 的开放权重模型家族。状态：**待适配**。

## 家族特征

- decoder-only
- RMSNorm
- SwiGLU
- RoPE
- MLA（V2+）
- MoE（V2+）

## 常见型号

| 型号 | 规模 | 上下文 | 注意力 | MoE | 多模态 | StarLight 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-LLM 7B/67B | 7B / 67B | 4K | MHA/GQA | 否 | 否 | 待适配 |
| DeepSeek-V2 236B-A21B | 236B / 21B active | 128K | MLA | 是 | 否 | 受限于引擎能力（MLA + MoE） |
| DeepSeek-V3 671B-A37B | 671B / 37B active | 128K | MLA | 是 | 否 | 受限于引擎能力（MLA + MoE） |
| DeepSeek-R1 671B-A37B | 671B / 37B active | 128K | MLA | 是 | 否 | 受限于引擎能力（MLA + MoE） |
| R1-Distill-Llama 8B/70B | 8B / 70B | 8K/128K | GQA | 否 | 否 | 结构兼容（rope_scaling（70B）） |
| R1-Distill-Qwen 1.5B/7B/14B/32B | 1.5B–32B | 32K–128K | GQA | 否 | 否 | 受限于引擎能力（attention bias） |

状态含义见 [families/README.md](../README.md)。

## 适配笔记

V2/V3/R1 使用 MLA 与 MoE，需要新的 attention 与专家路由算子；R1-Distill 系列是 dense，可优先适配。

## 参考

- <https://github.com/deepseek-ai>
- <https://huggingface.co/deepseek-ai>
