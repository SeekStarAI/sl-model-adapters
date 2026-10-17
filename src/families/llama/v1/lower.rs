// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Lowering for the `llama` family, variant `dense`.
//!
//! The family covers decoder-only, bias-free, dense transformer blocks with
//! RMSNorm, NeoX-style RoPE, causal attention (GQA allowed) and SwiGLU. Both a
//! symbolic prefill program and a fixed-length decode program are built from
//! the same weight table.

use std::collections::{BTreeMap, BTreeSet};

use sl_ir::{
    DType, Dim, Effect, Op, ProgramBuilder, QuantizationFormat as IrQuantizationFormat,
    RopeVariant, Shape, StateKind, TensorEncoding as IrTensorEncoding, ValueId,
};
use sl_model::{
    ModelBundle, ModelTensor, QuantizationFormat as ModelQuantizationFormat, TensorDType,
    TensorData, TensorEncoding as ModelTensorEncoding,
};

use crate::error::LoweringError;
use crate::families::llama::{ARCHITECTURES, FAMILY, VARIANT};
use crate::model::{
    ArchitectureId, KvCacheSpec, LoweredModel, LoweredProgram, ParamRef, ProgramIo, WeightSpec,
};

use super::config::{LlamaConfig, CONFIG_VERSION};

const EMBED_TENSOR: &str = "model.embed_tokens.weight";
const FINAL_NORM_TENSOR: &str = "model.norm.weight";
const LM_HEAD_TENSOR: &str = "lm_head.weight";

/// Lowers a `llama` dense bundle into prefill/decode programs.
///
/// `max_batch` must currently be `1`. `max_seq` is the symbolic bound of the
/// prefill sequence dimension and must not exceed the model's
/// `max_position_embeddings`. The decode program always uses a fixed sequence
/// length of one.
///
/// # Errors
///
/// Returns [`LoweringError`] for unsupported architectures and variants,
/// malformed configuration, missing or unexpected tensors, shape or dtype
/// mismatches, unsupported encodings, and `sl-ir` validation failures.
pub fn lower(
    bundle: &ModelBundle,
    max_batch: u32,
    max_seq: u32,
) -> Result<LoweredModel, LoweringError> {
    let architecture = bundle.metadata().architecture();
    if !ARCHITECTURES.contains(&architecture) {
        return Err(LoweringError::UnsupportedArchitecture {
            family: FAMILY.to_string(),
            found: architecture.to_string(),
        });
    }

    let config = LlamaConfig::from_json(bundle.metadata().config())?;

    if max_batch != 1 {
        return Err(LoweringError::UnsupportedFeature("max_batch other than 1"));
    }
    if max_seq == 0 {
        return Err(LoweringError::InvalidConfigField {
            field: "max_seq",
            reason: "must be greater than zero",
        });
    }
    if max_seq > config.max_position_embeddings {
        return Err(LoweringError::InvalidConfigField {
            field: "max_seq",
            reason: "must not exceed max_position_embeddings",
        });
    }

    let roles = role_specs(&config);
    let resolved = resolve_weights(bundle, &roles)?;

    let prefill = build_program(&config, &resolved, max_seq, true)?;
    let decode = build_program(&config, &resolved, max_seq, false)?;
    let weights = resolved
        .iter()
        .map(|weight| WeightSpec {
            role: weight.role.clone(),
            tensor: weight.tensor.clone(),
            shape: weight.shape.clone(),
            dtype: weight.dtype,
            encoding: weight.encoding.clone(),
            scales: weight.scales.clone(),
        })
        .collect();

    let kv_cache = KvCacheSpec {
        layers: config.num_hidden_layers,
        kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim(),
        max_seq,
        dtype: DType::F32,
    };

    Ok(LoweredModel {
        ir_version: sl_ir::IR_VERSION,
        architecture: ArchitectureId {
            family: FAMILY,
            variant: Some(VARIANT),
            config_version: CONFIG_VERSION,
        },
        prefill,
        decode,
        weights,
        kv_cache,
    })
}

/// A parameter role and the source tensor plus logical shape it must have.
struct RoleSpec {
    role: String,
    tensor: String,
    shape: Vec<u32>,
}

/// A role whose source tensor has been validated.
struct ResolvedWeight {
    role: String,
    tensor: String,
    shape: Vec<u32>,
    dtype: DType,
    encoding: IrTensorEncoding,
    scales: Option<TensorData>,
}

/// Parameter handles for one decoder layer in one program.
struct LayerParams {
    attn_norm: ValueId,
    q: ValueId,
    k: ValueId,
    v: ValueId,
    o: ValueId,
    mlp_norm: ValueId,
    gate: ValueId,
    up: ValueId,
    down: ValueId,
}

fn role_specs(config: &LlamaConfig) -> Vec<RoleSpec> {
    let hidden = config.hidden_size;
    let vocab = config.vocab_size;
    let q_dim = config.query_dim();
    let kv_dim = config.kv_dim();
    let intermediate = config.intermediate_size;

    let lm_head_tensor = if config.tie_word_embeddings {
        EMBED_TENSOR
    } else {
        LM_HEAD_TENSOR
    };

    let mut roles =
        Vec::with_capacity(3 + 9 * usize::try_from(config.num_hidden_layers).unwrap_or(0));
    roles.push(RoleSpec {
        role: "embed".to_string(),
        tensor: EMBED_TENSOR.to_string(),
        shape: vec![vocab, hidden],
    });
    roles.push(RoleSpec {
        role: "final_norm".to_string(),
        tensor: FINAL_NORM_TENSOR.to_string(),
        shape: vec![hidden],
    });
    roles.push(RoleSpec {
        role: "lm_head".to_string(),
        tensor: lm_head_tensor.to_string(),
        shape: vec![vocab, hidden],
    });

    for layer in 0..config.num_hidden_layers {
        let source = format!("model.layers.{layer}");
        roles.push(RoleSpec {
            role: format!("layers.{layer}.attn_norm"),
            tensor: format!("{source}.input_layernorm.weight"),
            shape: vec![hidden],
        });
        roles.push(RoleSpec {
            role: format!("layers.{layer}.attn.q"),
            tensor: format!("{source}.self_attn.q_proj.weight"),
            shape: vec![q_dim, hidden],
        });
        roles.push(RoleSpec {
            role: format!("layers.{layer}.attn.k"),
            tensor: format!("{source}.self_attn.k_proj.weight"),
            shape: vec![kv_dim, hidden],
        });
        roles.push(RoleSpec {
            role: format!("layers.{layer}.attn.v"),
            tensor: format!("{source}.self_attn.v_proj.weight"),
            shape: vec![kv_dim, hidden],
        });
        roles.push(RoleSpec {
            role: format!("layers.{layer}.attn.o"),
            tensor: format!("{source}.self_attn.o_proj.weight"),
            shape: vec![hidden, q_dim],
        });
        roles.push(RoleSpec {
            role: format!("layers.{layer}.mlp_norm"),
            tensor: format!("{source}.post_attention_layernorm.weight"),
            shape: vec![hidden],
        });
        roles.push(RoleSpec {
            role: format!("layers.{layer}.mlp.gate"),
            tensor: format!("{source}.mlp.gate_proj.weight"),
            shape: vec![intermediate, hidden],
        });
        roles.push(RoleSpec {
            role: format!("layers.{layer}.mlp.up"),
            tensor: format!("{source}.mlp.up_proj.weight"),
            shape: vec![intermediate, hidden],
        });
        roles.push(RoleSpec {
            role: format!("layers.{layer}.mlp.down"),
            tensor: format!("{source}.mlp.down_proj.weight"),
            shape: vec![hidden, intermediate],
        });
    }

    roles
}

fn resolve_weights(
    bundle: &ModelBundle,
    roles: &[RoleSpec],
) -> Result<Vec<ResolvedWeight>, LoweringError> {
    let mut missing: Vec<String> = roles
        .iter()
        .filter(|role| bundle.tensor(&role.tensor).is_none())
        .map(|role| role.tensor.clone())
        .collect();
    if !missing.is_empty() {
        missing.sort();
        missing.dedup();
        return Err(LoweringError::MissingTensor(missing.join(", ")));
    }

    let consumed: BTreeSet<&str> = roles.iter().map(|role| role.tensor.as_str()).collect();
    let unexpected: Vec<String> = bundle
        .tensors()
        .keys()
        .filter(|name| !consumed.contains(name.as_str()))
        .cloned()
        .collect();
    if !unexpected.is_empty() {
        return Err(LoweringError::UnexpectedTensor(unexpected.join(", ")));
    }

    let mut resolved = Vec::with_capacity(roles.len());
    for role in roles {
        let tensor = bundle
            .tensor(&role.tensor)
            .expect("missing tensors were reported before resolution");
        let dtype = logical_dtype(&role.tensor, tensor)?;
        check_shape(role, tensor)?;
        let (encoding, scales) = ir_encoding(&role.tensor, tensor)?;
        resolved.push(ResolvedWeight {
            role: role.role.clone(),
            tensor: role.tensor.clone(),
            shape: role.shape.clone(),
            dtype,
            encoding,
            scales,
        });
    }
    Ok(resolved)
}

fn logical_dtype(tensor_name: &str, tensor: &ModelTensor) -> Result<DType, LoweringError> {
    match tensor.dtype() {
        // IR v2 operator ports are F32-only; accepting F16/BF16 here would
        // build a program that `Program::validate` rejects later.
        TensorDType::F32 => Ok(DType::F32),
        other => Err(LoweringError::UnsupportedDType {
            tensor: tensor_name.to_string(),
            dtype: format!("{other:?}"),
        }),
    }
}

fn check_shape(role: &RoleSpec, tensor: &ModelTensor) -> Result<(), LoweringError> {
    let actual: Vec<u32> = tensor
        .shape()
        .iter()
        .map(|&dimension| u32::try_from(dimension).unwrap_or(u32::MAX))
        .collect();
    if actual != role.shape {
        return Err(LoweringError::ShapeMismatch {
            role: role.role.clone(),
            expected: role.shape.clone(),
            actual,
        });
    }
    Ok(())
}

fn ir_encoding(
    tensor_name: &str,
    tensor: &ModelTensor,
) -> Result<(IrTensorEncoding, Option<TensorData>), LoweringError> {
    match tensor.encoding() {
        ModelTensorEncoding::Dense => Ok((IrTensorEncoding::Dense, None)),
        ModelTensorEncoding::GroupwiseQuantized {
            format,
            group_size,
            scales,
        } => {
            let format = match format {
                ModelQuantizationFormat::Int4 => IrQuantizationFormat::Int4,
                ModelQuantizationFormat::Int8 => IrQuantizationFormat::Int8,
            };
            let group_size =
                u32::try_from(*group_size).map_err(|_| LoweringError::UnsupportedEncoding {
                    tensor: tensor_name.to_string(),
                    encoding: format!("group_size {group_size} does not fit in u32"),
                })?;
            Ok((
                IrTensorEncoding::GroupwiseQuantized { format, group_size },
                Some(scales.clone()),
            ))
        }
        ModelTensorEncoding::OpaqueQuantized { scheme } => {
            Err(LoweringError::UnsupportedEncoding {
                tensor: tensor_name.to_string(),
                encoding: scheme.clone(),
            })
        }
    }
}

fn build_program(
    config: &LlamaConfig,
    weights: &[ResolvedWeight],
    max_seq: u32,
    prefill: bool,
) -> Result<LoweredProgram, LoweringError> {
    let hidden = config.hidden_size;
    let layers = config.num_hidden_layers;
    let heads = config.num_attention_heads;
    let kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim();
    let q_dim = config.query_dim();
    let kv_dim = config.kv_dim();
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut builder = ProgramBuilder::new();
    let seq = if prefill {
        Dim::Sym(builder.sym_dim("seq", max_seq))
    } else {
        Dim::Const(1)
    };

    let token_ids = builder.input(DType::U32, sequence_shape(seq));
    let positions = builder.input(DType::U32, sequence_shape(seq));
    let seq_len = builder.scalar(DType::U32, Shape::default());
    let past_len = builder.scalar(DType::U32, Shape::default());
    let kv_state = builder.state(
        DType::F32,
        Shape::new(vec![
            Dim::Const(1),
            Dim::Const(layers),
            Dim::Const(2),
            Dim::Const(max_seq),
            Dim::Const(kv_heads),
            Dim::Const(head_dim),
        ]),
        StateKind::KvCache {
            batch: 1,
            layers,
            kv_heads,
            head_dim,
            max_seq: Dim::Const(max_seq),
            dtype: DType::F32,
        },
    );

    // `tie_word_embeddings` makes several roles share one source tensor and one
    // `Param` value, so `sl-core` can allocate a single buffer for them.
    let mut tensor_values: BTreeMap<&str, ValueId> = BTreeMap::new();
    for weight in weights {
        if !tensor_values.contains_key(weight.tensor.as_str()) {
            let value = builder.param_with_encoding(
                weight.dtype,
                Shape::consts(&weight.shape),
                weight.encoding.clone(),
            );
            tensor_values.insert(weight.tensor.as_str(), value);
        }
    }

    let mut params = Vec::with_capacity(weights.len());
    let mut values_by_role: BTreeMap<String, ValueId> = BTreeMap::new();
    for weight in weights {
        let value = tensor_values[weight.tensor.as_str()];
        values_by_role.insert(weight.role.clone(), value);
        params.push(ParamRef {
            role: weight.role.clone(),
            value,
        });
    }

    let layer_params: Vec<LayerParams> = (0..layers)
        .map(|layer| LayerParams {
            attn_norm: bound(&values_by_role, &format!("layers.{layer}.attn_norm")),
            q: bound(&values_by_role, &format!("layers.{layer}.attn.q")),
            k: bound(&values_by_role, &format!("layers.{layer}.attn.k")),
            v: bound(&values_by_role, &format!("layers.{layer}.attn.v")),
            o: bound(&values_by_role, &format!("layers.{layer}.attn.o")),
            mlp_norm: bound(&values_by_role, &format!("layers.{layer}.mlp_norm")),
            gate: bound(&values_by_role, &format!("layers.{layer}.mlp.gate")),
            up: bound(&values_by_role, &format!("layers.{layer}.mlp.up")),
            down: bound(&values_by_role, &format!("layers.{layer}.mlp.down")),
        })
        .collect();

    let embed = bound(&values_by_role, "embed");
    let final_norm = bound(&values_by_role, "final_norm");
    let lm_head = bound(&values_by_role, "lm_head");

    let mut hidden_state = builder.temp(DType::F32, activation_shape(seq, hidden));
    builder.node(
        Op::Gather,
        vec![embed, token_ids],
        vec![hidden_state],
        Effect::None,
    );

    for (layer_index, layer) in layer_params.iter().enumerate() {
        let layer_index = u32::try_from(layer_index).expect("layer count exceeds u32::MAX");

        let normed_attention = builder.temp(DType::F32, activation_shape(seq, hidden));
        builder.node(
            Op::RmsNorm {
                eps: config.rms_norm_eps,
            },
            vec![hidden_state, layer.attn_norm],
            vec![normed_attention],
            Effect::None,
        );

        let query = builder.temp(DType::F32, activation_shape(seq, q_dim));
        builder.node(
            Op::MatMulNT,
            vec![normed_attention, layer.q],
            vec![query],
            Effect::None,
        );
        let key = builder.temp(DType::F32, activation_shape(seq, kv_dim));
        builder.node(
            Op::MatMulNT,
            vec![normed_attention, layer.k],
            vec![key],
            Effect::None,
        );
        let value = builder.temp(DType::F32, activation_shape(seq, kv_dim));
        builder.node(
            Op::MatMulNT,
            vec![normed_attention, layer.v],
            vec![value],
            Effect::None,
        );

        let query_heads = builder.temp(DType::F32, heads_shape(seq, heads, head_dim));
        builder.node(
            Op::SplitHeads { heads, head_dim },
            vec![query],
            vec![query_heads],
            Effect::None,
        );
        let key_heads = builder.temp(DType::F32, heads_shape(seq, kv_heads, head_dim));
        builder.node(
            Op::SplitHeads {
                heads: kv_heads,
                head_dim,
            },
            vec![key],
            vec![key_heads],
            Effect::None,
        );
        let value_heads = builder.temp(DType::F32, heads_shape(seq, kv_heads, head_dim));
        builder.node(
            Op::SplitHeads {
                heads: kv_heads,
                head_dim,
            },
            vec![value],
            vec![value_heads],
            Effect::None,
        );

        let rotary_query = builder.temp(DType::F32, heads_shape(seq, heads, head_dim));
        builder.node(
            Op::Rope {
                variant: RopeVariant::Neox,
                theta: config.rope_theta,
            },
            vec![query_heads, positions],
            vec![rotary_query],
            Effect::None,
        );
        let rotary_key = builder.temp(DType::F32, heads_shape(seq, kv_heads, head_dim));
        builder.node(
            Op::Rope {
                variant: RopeVariant::Neox,
                theta: config.rope_theta,
            },
            vec![key_heads, positions],
            vec![rotary_key],
            Effect::None,
        );

        let attention = builder.temp(DType::F32, heads_shape(seq, heads, head_dim));
        builder.node(
            Op::CausalAttention {
                layer: layer_index,
                heads,
                kv_heads,
                head_dim,
                scale,
            },
            vec![rotary_query, rotary_key, value_heads, past_len],
            vec![attention],
            Effect::WriteState(kv_state),
        );

        let merged = builder.temp(DType::F32, activation_shape(seq, q_dim));
        builder.node(
            Op::MergeHeads { heads, head_dim },
            vec![attention],
            vec![merged],
            Effect::None,
        );
        let projected = builder.temp(DType::F32, activation_shape(seq, hidden));
        builder.node(
            Op::MatMulNT,
            vec![merged, layer.o],
            vec![projected],
            Effect::None,
        );
        let attention_residual = builder.temp(DType::F32, activation_shape(seq, hidden));
        builder.node(
            Op::Add,
            vec![hidden_state, projected],
            vec![attention_residual],
            Effect::None,
        );

        let normed_mlp = builder.temp(DType::F32, activation_shape(seq, hidden));
        builder.node(
            Op::RmsNorm {
                eps: config.rms_norm_eps,
            },
            vec![attention_residual, layer.mlp_norm],
            vec![normed_mlp],
            Effect::None,
        );
        let feed_forward = builder.temp(DType::F32, activation_shape(seq, hidden));
        builder.node(
            Op::SwiGlu,
            vec![normed_mlp, layer.gate, layer.up, layer.down],
            vec![feed_forward],
            Effect::None,
        );
        let mlp_residual = builder.temp(DType::F32, activation_shape(seq, hidden));
        builder.node(
            Op::Add,
            vec![attention_residual, feed_forward],
            vec![mlp_residual],
            Effect::None,
        );
        hidden_state = mlp_residual;
    }

    let normed_output = builder.temp(DType::F32, activation_shape(seq, hidden));
    builder.node(
        Op::RmsNorm {
            eps: config.rms_norm_eps,
        },
        vec![hidden_state, final_norm],
        vec![normed_output],
        Effect::None,
    );
    let last_token = builder.temp(DType::F32, Shape::consts(&[1, hidden]));
    builder.node(
        Op::LastToken,
        vec![normed_output, seq_len],
        vec![last_token],
        Effect::None,
    );
    let logits = builder.output(DType::F32, Shape::consts(&[1, config.vocab_size]));
    builder.node(
        Op::MatMulNT,
        vec![last_token, lm_head],
        vec![logits],
        Effect::None,
    );

    builder
        .declare_input(token_ids)
        .declare_input(positions)
        .declare_output(logits);

    let program = builder.build()?;
    let io = ProgramIo {
        token_ids,
        positions,
        seq_len,
        past_len,
        logits,
    };
    Ok(LoweredProgram {
        program,
        io,
        params,
    })
}

fn sequence_shape(seq: Dim) -> Shape {
    Shape::new(vec![Dim::Const(1), seq])
}

fn activation_shape(seq: Dim, hidden: u32) -> Shape {
    Shape::new(vec![Dim::Const(1), seq, Dim::Const(hidden)])
}

fn heads_shape(seq: Dim, heads: u32, head_dim: u32) -> Shape {
    Shape::new(vec![
        Dim::Const(1),
        Dim::Const(heads),
        seq,
        Dim::Const(head_dim),
    ])
}

fn bound(values: &BTreeMap<String, ValueId>, role: &str) -> ValueId {
    *values
        .get(role)
        .expect("every required role has a bound parameter")
}
