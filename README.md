# sl-model-adapters

Model-family lowering for StarLight. An adapter turns a
[`sl-model`](https://github.com/SeekStarAI/sl-model) `ModelBundle` into an
`sl-ir` `Program` pair (prefill and decode) plus a shared weight table and KV
cache layout. The contract is defined by
[ARCH-0006](<../../docs/ARCH-0006: 模型家族 Lowering 与 LoweredModel 契约.md>);
the operator and capability ABI is defined by
[ARCH-0004](<../../docs/ARCH-0004: 计算图 IR 与设备命令 ABI 契约.md>).

## Status

| Family | Variant | Scope | State |
| --- | --- | --- | --- |
| `llama` | `dense` | decoder-only, bias-free, dense; RMSNorm + NeoX RoPE + causal attention (GQA allowed) + SwiGLU | lowering and `sl-graph::compile` verified by `tests/lower.rs` |

`ModelMetadata.architecture` may be `llama` or `tiny-llama`; both lower to
`ArchitectureId { family: "llama", variant: Some("dense"), config_version: 1 }`.
`tiny-llama` is the fixture alias used by the tests.

Not implemented: encoder / encoder-decoder models, bias, MoE, sliding-window or
ALiBi attention, non-NeoX RoPE variants, paged KV, multi-device splitting and
variable-length batching.

## Family catalog

`src/families/` currently contains **17 family directories**. Only `llama`
has a Rust implementation; the other families ship `README.md` and
`models.toml` with architecture notes, common variants and explicit blockers.
See [families/README.md](src/families/README.md) for the index and status
definitions (`implemented`, `structural`, `planned`, `blocked`).

Adding a family starts with its catalog, not with code: document the common
models and the blockers first, then implement the version directory once the
required operators exist.

## Repository layout

```text
src/
├── error.rs              # LoweringError (shared across families)
├── model.rs              # LoweredModel / WeightSpec / ProgramIo (shared)
└── families/
    ├── mod.rs            # family registry
    └── llama/
        ├── mod.rs        # family metadata + version dispatch
        └── v1/           # config schema v1 + lowering
            ├── mod.rs
            ├── config.rs
            └── lower.rs
```

Only cross-family contracts live at the crate root. Family-specific tensor
names, config fields and graph construction stay inside
`families/<family>/<version>/`; planned families only provide
`README.md` + `models.toml` until they are implemented.

To add a family:

1. create `src/families/<family>/mod.rs` with `FAMILY`, `VARIANT` and the
   accepted architecture strings;
2. add `<family>/v1/config.rs` and `<family>/v1/lower.rs` for the first schema
   version;
3. register the family in `src/families/mod.rs`;
4. re-export its `lower` from the family module, and keep the shared
   `LoweredModel` types untouched.

To add a new version of an existing family, add a sibling `v2/` directory and
dispatch to it from the family module; older versions stay untouched.

## Entry point

```rust
use sl_model_adapters::lower;

let model = lower(&bundle, /* max_batch = */ 1, /* max_seq = */ 8)?;
```

`max_batch` must be `1`. `max_seq` must be greater than zero and no larger than
the configured `max_position_embeddings`. The prefill program builds a
`Sym("seq")` dimension with `max = max_seq`; the decode program uses
`Const(1)`.

The result is:

```rust
pub struct LoweredModel {
    pub ir_version: u32,
    pub architecture: ArchitectureId,
    pub prefill: LoweredProgram,
    pub decode: LoweredProgram,
    pub weights: Vec<WeightSpec>,
    pub kv_cache: KvCacheSpec,
}
```

Each `LoweredProgram` contains the validated `Program`, its `ProgramIo`
(`token_ids`, `positions`, `seq_len`, `past_len`, `logits`) and one `ParamRef`
per weight role. Value ids are program-local: the prefill and decode programs
have independent value tables.

## Configuration

The adapter reads these fields from `ModelMetadata.config` (a JSON object).
Unknown fields are ignored.

| Field | Rule |
| --- | --- |
| `hidden_size` | `> 0`, divisible by `num_attention_heads` |
| `num_hidden_layers` | `> 0` |
| `num_attention_heads` | `> 0` |
| `num_key_value_heads` | `> 0`, divides `num_attention_heads` |
| `intermediate_size` | `> 0` |
| `vocab_size` | `> 0` |
| `rms_norm_eps` | finite and `> 0` |
| `rope_theta` | finite and `> 0` |
| `max_position_embeddings` | `> 0`, at least `max_seq` |
| `tie_word_embeddings` | optional; missing means `false` |

`head_dim = hidden_size / num_attention_heads`; the division must be exact.

## Weight roles

| Role | Source tensor | Shape |
| --- | --- | --- |
| `embed` | `model.embed_tokens.weight` | `[vocab, hidden]` |
| `final_norm` | `model.norm.weight` | `[hidden]` |
| `lm_head` | `lm_head.weight`, or `model.embed_tokens.weight` when tied | `[vocab, hidden]` |
| `layers.{i}.attn_norm` | `model.layers.{i}.input_layernorm.weight` | `[hidden]` |
| `layers.{i}.attn.q` | `model.layers.{i}.self_attn.q_proj.weight` | `[heads*head_dim, hidden]` |
| `layers.{i}.attn.k` | `model.layers.{i}.self_attn.k_proj.weight` | `[kv_heads*head_dim, hidden]` |
| `layers.{i}.attn.v` | `model.layers.{i}.self_attn.v_proj.weight` | `[kv_heads*head_dim, hidden]` |
| `layers.{i}.attn.o` | `model.layers.{i}.self_attn.o_proj.weight` | `[hidden, heads*head_dim]` |
| `layers.{i}.mlp_norm` | `model.layers.{i}.post_attention_layernorm.weight` | `[hidden]` |
| `layers.{i}.mlp.gate` | `model.layers.{i}.mlp.gate_proj.weight` | `[inter, hidden]` |
| `layers.{i}.mlp.up` | `model.layers.{i}.mlp.up_proj.weight` | `[inter, hidden]` |
| `layers.{i}.mlp.down` | `model.layers.{i}.mlp.down_proj.weight` | `[hidden, inter]` |

All required roles are checked before lowering. Missing tensors are reported
together; tensors not consumed by any role produce `UnexpectedTensor`; shape
and dtype mismatches name the role and source tensor. Dense weights must be
`F32`: the v2 operator ports are F32-only, so `F16`/`BF16` checkpoints must be
converted before lowering (returning `UnsupportedDType` is deliberate, rather
than building a program that `Program::validate` would reject later).

When `tie_word_embeddings` is `true`, `lm_head` and `embed` resolve to the same
source tensor, share one `Param` value in each program, and `lm_head.weight`
must be absent (otherwise it is an unexpected tensor).

## Programs

Both programs are built from the same role table and KV layout:

```text
token_ids [1, seq] U32 Input
positions [1, seq] U32 Input
seq_len, past_len U32 scalars
kv_state [1, layers, 2, max_seq, kv_heads, head_dim] F32 State(KvCache)
x = Gather(embed, token_ids)
per layer:
  RmsNorm -> MatMulNT(q/k/v) -> SplitHeads -> Rope(Neox)
  -> CausalAttention(WriteState) -> MergeHeads -> MatMulNT(o) -> Add
  -> RmsNorm -> SwiGlu -> Add
final: RmsNorm -> LastToken(seq_len) -> MatMulNT(lm_head) -> logits [1, vocab]
```

`sl-graph` compiles the programs for a concrete `ShapeSpec` and a
`DeviceCapabilities`. `CausalAttention` must be declared as a native composite;
`SwiGlu` may be native or decomposed into `MatMulNT`/`Silu`/`Mul`/`MatMulNT`.
Both paths are covered by the tests.

## Quantized weights

`sl-model::TensorEncoding::GroupwiseQuantized` maps to
`sl_ir::TensorEncoding::GroupwiseQuantized { format, group_size }`. The logical
dtype remains the source dtype (normally `F32`), and the program parameter is
created with `ProgramBuilder::param_with_encoding`. `WeightSpec.encoding`
records that encoding.

Limitations:

- `WeightSpec.scales` carries the group-wise scale payload copied from
  `sl_model::TensorEncoding::GroupwiseQuantized`. Binding those scales to the
  runtime is still pending a `PackedLayout` contract; the IR itself only records
  the encoding and the packed payload size.
- `OpaqueQuantized` is rejected with `UnsupportedEncoding`; it is never treated
  as dense.
- `sl-core` currently rejects non-dense `Param`s in `Engine::build`. This crate
  verifies lowering and `sl-graph::compile` only; executing quantized models
  requires backend-side quantized kernels first.

## Errors

`LoweringError` distinguishes unsupported architecture, missing/invalid config
fields, missing and unexpected tensors, shape/dtype mismatches, unsupported
dtype/encoding/feature, and `sl-ir` validation failures. It implements
`Debug + Display + std::error::Error`. `From<sl_ir::IrError>` maps to
`LoweringError::Program`.

## Verification

```sh
cargo test -p sl-model-adapters
cargo clippy -p sl-model-adapters --all-targets -- -D warnings
cargo fmt --all --check
```

`tests/families.rs` checks the family/version layout; `tests/catalog.rs` checks every family directory has a README and a `models.toml`; `tests/lower.rs` builds a tiny fixture (`hidden = 8`, 2 layers, 2 query heads,
1 KV head, `head_dim = 4`, `inter = 16`, `vocab = 16`, `max_seq = 8`) and
covers role coverage, program validation, state shape, native and decomposed
compile paths, missing/shape/tie/quantization/opaque/error cases.
