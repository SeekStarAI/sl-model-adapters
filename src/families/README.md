# Model family catalog

> 引擎能力边界与缺口见 [ARCH-0008](../../../../docs/ARCH-0008: 推理基础设施能力矩阵与缺口清单.md)。
>
> 型号信息用于适配规划，不作为兼容性承诺。参数规模、上下文长度、许可证与
> 架构字段以官方 `config.json` / model card 为准；本目录最后核对时间：2026-10-18。


每个模型家族一个目录。目录内：

- `README.md`：家族说明、常见型号、适配笔记与参考；
- `models.toml`：机器可读的型号目录；
- 已实现的家族额外包含 `mod.rs` 与版本子目录（如 `v1/`）。

只有已实现的家族才会在 `families/mod.rs` 中声明为 Rust 模块；其余家族保留描述文件，等待适配。

## 状态定义

| 状态 | 含义 |
| --- | --- |
| 已实现 | lowering + `sl-graph::compile` 有测试覆盖 |
| 结构兼容 | 结构与已实现家族一致，但尚无真实 checkpoint 测试 |
| 待适配 | 需要新增 config/张量映射，可能涉及小算子 |
| 受限于引擎能力 | 依赖当前未实现的算子或特性（MoE、MLA、滑动窗口、多模态、QK-norm、bias 等） |

## 家族一览

| 目录 | 家族 | 厂商 | 状态 | 型号数 |
| --- | --- | --- | --- | --- |
| `llama` | Llama | Meta | 已实现 | 6 |
| `qwen` | Qwen | Alibaba | 待适配 | 5 |
| `deepseek` | DeepSeek | DeepSeek-AI | 待适配 | 6 |
| `mistral` | Mistral | Mistral AI | 待适配 | 7 |
| `gemma` | Gemma | Google | 待适配 | 3 |
| `phi` | Phi | Microsoft | 待适配 | 4 |
| `glm` | GLM | Zhipu AI | 待适配 | 5 |
| `internlm` | InternLM | Shanghai AI Lab | 待适配 | 3 |
| `yi` | Yi | 01.AI | 待适配 | 4 |
| `baichuan` | Baichuan | Baichuan AI | 待适配 | 2 |
| `falcon` | Falcon | TII | 待适配 | 3 |
| `olmo` | OLMo | Allen AI | 待适配 | 4 |
| `minicpm` | MiniCPM | OpenBMB | 待适配 | 4 |
| `ernie` | ERNIE | Baidu | 待适配 | 3 |
| `hunyuan` | Hunyuan | Tencent | 待适配 | 2 |
| `exaone` | EXAONE | LG AI Research | 待适配 | 2 |
| `command-r` | Command R | Cohere | 待适配 | 2 |

## 新增家族

1. 新建 `families/<family>/README.md` 与 `models.toml`，先写清架构、型号和 blocker；
2. 结构兼容的家族优先复用 `llama/v1` 的 lowering；
3. 需要新算子时，先在 `docs/ARCH-0004` 与 backend 侧补算子，再写版本目录；
4. 实现完成后在 `families/mod.rs` 注册，并补 `tests/`。
