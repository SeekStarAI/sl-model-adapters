// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Lowering and compile tests for the tiny `llama` dense fixture.

use std::collections::BTreeSet;

use serde_json::{json, Value};
use sl_graph::compile;
use sl_ir::{
    BackendTag, DType, DTypeSet, DeviceCapabilities, Dim, Op, OpCode, OpSet,
    QuantizationFormat as IrQuantizationFormat, ShapeConstraints, ShapeSpec, ShapeSpecEntry,
    StateKind, TensorEncoding as IrTensorEncoding, IR_VERSION,
};
use sl_model::{
    ModelBundle, ModelMetadata, ModelTensor, QuantizationFormat, TensorDType, TensorData,
    TensorEncoding,
};
use sl_model_adapters::{lower, LoweredModel, LoweringError};

const HIDDEN: usize = 8;
const LAYERS: usize = 2;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 4;
const INTERMEDIATE: usize = 16;
const VOCAB: usize = 16;
const MAX_SEQ: u32 = 8;

const EMBED: &str = "model.embed_tokens.weight";
const FINAL_NORM: &str = "model.norm.weight";
const LM_HEAD: &str = "lm_head.weight";
const LAYER_ROLE_SUFFIXES: [&str; 9] = [
    "attn_norm",
    "attn.q",
    "attn.k",
    "attn.v",
    "attn.o",
    "mlp_norm",
    "mlp.gate",
    "mlp.up",
    "mlp.down",
];

fn dense_f32(shape: &[usize]) -> ModelTensor {
    let elements: usize = shape.iter().product();
    ModelTensor::new(
        shape.to_vec(),
        TensorDType::F32,
        TensorEncoding::Dense,
        TensorData::from_vec(vec![0; elements * 4]),
    )
    .expect("dense F32 payload matches its shape")
}

fn quantized_int4(shape: &[usize], group_size: usize) -> ModelTensor {
    let elements: usize = shape.iter().product();
    let payload_bytes = elements.div_ceil(2);
    let scale_groups = elements.div_ceil(group_size);
    ModelTensor::new(
        shape.to_vec(),
        TensorDType::F32,
        TensorEncoding::GroupwiseQuantized {
            format: QuantizationFormat::Int4,
            group_size,
            scales: TensorData::from_vec(vec![0; scale_groups * 4]),
        },
        TensorData::from_vec(vec![0; payload_bytes]),
    )
    .expect("quantized Int4 payload and scales match the shape")
}

fn config_json(tie_word_embeddings: bool) -> Value {
    json!({
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "intermediate_size": INTERMEDIATE,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": MAX_SEQ,
        "tie_word_embeddings": tie_word_embeddings,
    })
}

fn tiny_tensor_list(tie_word_embeddings: bool) -> Vec<(String, ModelTensor)> {
    let mut tensors = vec![
        (EMBED.to_string(), dense_f32(&[VOCAB, HIDDEN])),
        (FINAL_NORM.to_string(), dense_f32(&[HIDDEN])),
    ];
    if !tie_word_embeddings {
        tensors.push((LM_HEAD.to_string(), dense_f32(&[VOCAB, HIDDEN])));
    }

    for layer in 0..LAYERS {
        let source = format!("model.layers.{layer}");
        tensors.push((
            format!("{source}.input_layernorm.weight"),
            dense_f32(&[HIDDEN]),
        ));
        tensors.push((
            format!("{source}.self_attn.q_proj.weight"),
            dense_f32(&[HEADS * HEAD_DIM, HIDDEN]),
        ));
        tensors.push((
            format!("{source}.self_attn.k_proj.weight"),
            dense_f32(&[KV_HEADS * HEAD_DIM, HIDDEN]),
        ));
        tensors.push((
            format!("{source}.self_attn.v_proj.weight"),
            dense_f32(&[KV_HEADS * HEAD_DIM, HIDDEN]),
        ));
        tensors.push((
            format!("{source}.self_attn.o_proj.weight"),
            dense_f32(&[HIDDEN, HEADS * HEAD_DIM]),
        ));
        tensors.push((
            format!("{source}.post_attention_layernorm.weight"),
            dense_f32(&[HIDDEN]),
        ));
        tensors.push((
            format!("{source}.mlp.gate_proj.weight"),
            dense_f32(&[INTERMEDIATE, HIDDEN]),
        ));
        tensors.push((
            format!("{source}.mlp.up_proj.weight"),
            dense_f32(&[INTERMEDIATE, HIDDEN]),
        ));
        tensors.push((
            format!("{source}.mlp.down_proj.weight"),
            dense_f32(&[HIDDEN, INTERMEDIATE]),
        ));
    }

    tensors
}

fn bundle_with_architecture(
    architecture: &str,
    config: Value,
    tensors: Vec<(String, ModelTensor)>,
) -> ModelBundle {
    let metadata =
        ModelMetadata::new(architecture, config).expect("architecture and config are valid");
    let mut bundle = ModelBundle::new(metadata);
    for (name, tensor) in tensors {
        bundle
            .insert_tensor(name, tensor)
            .expect("fixture tensor names are unique");
    }
    bundle
}

fn bundle(config: Value, tensors: Vec<(String, ModelTensor)>) -> ModelBundle {
    bundle_with_architecture("llama", config, tensors)
}

fn tiny_bundle(tie_word_embeddings: bool) -> ModelBundle {
    bundle(
        config_json(tie_word_embeddings),
        tiny_tensor_list(tie_word_embeddings),
    )
}

fn expected_roles() -> BTreeSet<String> {
    let mut roles: BTreeSet<String> = ["embed", "final_norm", "lm_head"]
        .iter()
        .map(|role| (*role).to_string())
        .collect();
    for layer in 0..LAYERS {
        for suffix in LAYER_ROLE_SUFFIXES {
            roles.insert(format!("layers.{layer}.{suffix}"));
        }
    }
    roles
}

fn capabilities(native_swiglu: bool) -> DeviceCapabilities {
    let mut composite_ops = vec![OpCode::CausalAttention];
    if native_swiglu {
        composite_ops.push(OpCode::SwiGlu);
    }

    DeviceCapabilities {
        backend: BackendTag("test-cpu"),
        ir_version: IR_VERSION,
        core_ops: OpSet::new(vec![
            OpCode::Add,
            OpCode::Gather,
            OpCode::LastToken,
            OpCode::MatMulNT,
            OpCode::MergeHeads,
            OpCode::Mul,
            OpCode::RmsNorm,
            OpCode::Rope,
            OpCode::Silu,
            OpCode::SplitHeads,
        ]),
        composite_ops: OpSet::new(composite_ops),
        dtypes: DTypeSet::ONLY_F32.union(DTypeSet::ONLY_U32),
        min_buffer_align: 64,
        max_scratchpad_bytes: 1 << 30,
        max_value_bytes: 0,
        constraints: ShapeConstraints {
            head_dim_multiple_of: 2,
            matmul_k_multiple_of: 2,
            max_heads: 8,
        },
    }
}

fn prefill_shape_spec(model: &LoweredModel) -> ShapeSpec {
    let id = model.prefill.program.sym_dims[0].id;
    ShapeSpec::new(vec![ShapeSpecEntry {
        id,
        max: MAX_SEQ,
        value: 2,
    }])
}

fn replace_tensor(
    tensors: &mut [(String, ModelTensor)],
    tensor_name: &str,
    replacement: ModelTensor,
) {
    for (name, tensor) in tensors {
        if name.as_str() == tensor_name {
            *tensor = replacement.clone();
        }
    }
}

#[test]
fn lowers_tiny_llama_into_valid_prefill_and_decode() {
    let model = lower(&tiny_bundle(false), 1, MAX_SEQ).expect("lowering succeeds");

    assert_eq!(model.ir_version, IR_VERSION);
    assert_eq!(model.architecture.family, "llama");
    assert_eq!(model.architecture.variant, Some("dense"));
    assert_eq!(model.architecture.config_version, 1);

    let roles: BTreeSet<String> = model.weights.iter().map(|spec| spec.role.clone()).collect();
    assert_eq!(roles, expected_roles());
    assert_eq!(model.weights.len(), expected_roles().len());

    for program in [&model.prefill, &model.decode] {
        program
            .program
            .validate()
            .expect("lowered program is valid");
        assert_eq!(program.program.states.len(), 1);
        assert_eq!(program.params.len(), model.weights.len());
        assert_eq!(program.program.outputs, vec![program.io.logits]);

        let state = &program.program.states[0];
        assert_eq!(
            program
                .program
                .value(state.value)
                .expect("state value exists")
                .shape
                .to_consts(),
            Some(vec![
                1,
                LAYERS as u32,
                2,
                MAX_SEQ,
                KV_HEADS as u32,
                HEAD_DIM as u32
            ])
        );

        match state.kind {
            StateKind::KvCache {
                batch,
                layers,
                kv_heads,
                head_dim,
                max_seq,
                dtype,
            } => {
                assert_eq!(batch, 1);
                assert_eq!(layers, LAYERS as u32);
                assert_eq!(kv_heads, KV_HEADS as u32);
                assert_eq!(head_dim, HEAD_DIM as u32);
                assert_eq!(max_seq, Dim::Const(MAX_SEQ));
                assert_eq!(dtype, DType::F32);
            }
        }
    }

    assert_eq!(model.kv_cache.layers, LAYERS as u32);
    assert_eq!(model.kv_cache.kv_heads, KV_HEADS as u32);
    assert_eq!(model.kv_cache.head_dim, HEAD_DIM as u32);
    assert_eq!(model.kv_cache.max_seq, MAX_SEQ);
    assert_eq!(model.kv_cache.dtype, DType::F32);

    let prefill = &model.prefill;
    assert_eq!(prefill.program.sym_dims.len(), 1);
    assert_eq!(prefill.program.sym_dims[0].name, "seq");
    assert_eq!(prefill.program.sym_dims[0].max, MAX_SEQ);
    assert_eq!(
        prefill
            .program
            .value(prefill.io.token_ids)
            .expect("token ids value exists")
            .shape
            .to_consts(),
        None
    );
    assert_eq!(
        prefill
            .program
            .value(prefill.io.logits)
            .expect("logits value exists")
            .shape
            .to_consts(),
        Some(vec![1, VOCAB as u32])
    );

    let decode = &model.decode;
    assert!(decode.program.sym_dims.is_empty());
    assert_eq!(
        decode
            .program
            .value(decode.io.token_ids)
            .expect("token ids value exists")
            .shape
            .to_consts(),
        Some(vec![1, 1])
    );
}

#[test]
fn compile_accepts_native_composites() {
    let model = lower(&tiny_bundle(false), 1, MAX_SEQ).expect("lowering succeeds");
    let caps = capabilities(true);

    let prefill = compile(&model.prefill.program, &prefill_shape_spec(&model), &caps)
        .expect("prefill compiles");
    prefill
        .validate(64)
        .expect("prefill command buffer is valid");
    assert!(prefill
        .commands
        .iter()
        .any(|command| matches!(command.op, Op::CausalAttention { .. })));
    assert!(prefill
        .commands
        .iter()
        .any(|command| matches!(command.op, Op::SwiGlu)));

    let decode =
        compile(&model.decode.program, &ShapeSpec::default(), &caps).expect("decode compiles");
    decode.validate(64).expect("decode command buffer is valid");
}

#[test]
fn compile_decomposes_swiglu_when_backend_lacks_it() {
    let model = lower(&tiny_bundle(false), 1, MAX_SEQ).expect("lowering succeeds");
    let buffer = compile(
        &model.prefill.program,
        &prefill_shape_spec(&model),
        &capabilities(false),
    )
    .expect("prefill compiles with SwiGlu decomposition");
    buffer
        .validate(64)
        .expect("decomposed command buffer is valid");

    assert!(!buffer
        .commands
        .iter()
        .any(|command| matches!(command.op, Op::SwiGlu)));
    assert!(buffer
        .commands
        .iter()
        .any(|command| matches!(command.op, Op::Silu)));
    assert!(buffer
        .commands
        .iter()
        .any(|command| matches!(command.op, Op::Mul)));
    assert!(buffer
        .commands
        .iter()
        .any(|command| matches!(command.op, Op::CausalAttention { .. })));
}

#[test]
fn missing_tensor_is_reported_with_its_name() {
    let q_proj = "model.layers.0.self_attn.q_proj.weight";
    let tensors: Vec<(String, ModelTensor)> = tiny_tensor_list(false)
        .into_iter()
        .filter(|(name, _)| name.as_str() != q_proj)
        .collect();

    let error = lower(&bundle(config_json(false), tensors), 1, MAX_SEQ)
        .expect_err("removing q_proj must fail lowering");
    match error {
        LoweringError::MissingTensor(names) => assert_eq!(names, q_proj),
        other => panic!("expected MissingTensor, got {other:?}"),
    }
}

#[test]
fn reports_all_missing_tensors_at_once() {
    let missing = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
    ];
    let tensors: Vec<(String, ModelTensor)> = tiny_tensor_list(false)
        .into_iter()
        .filter(|(name, _)| !missing.contains(&name.as_str()))
        .collect();

    let error = lower(&bundle(config_json(false), tensors), 1, MAX_SEQ)
        .expect_err("two missing tensors must fail lowering");
    match error {
        LoweringError::MissingTensor(names) => {
            for tensor in missing {
                assert!(names.contains(tensor), "missing report omits {tensor}");
            }
        }
        other => panic!("expected MissingTensor, got {other:?}"),
    }
}

#[test]
fn shape_mismatch_is_reported() {
    let q_proj = "model.layers.0.self_attn.q_proj.weight";
    let mut tensors = tiny_tensor_list(false);
    replace_tensor(
        &mut tensors,
        q_proj,
        dense_f32(&[HEADS * HEAD_DIM, HIDDEN + 1]),
    );

    let error = lower(&bundle(config_json(false), tensors), 1, MAX_SEQ)
        .expect_err("a wrong q_proj shape must fail lowering");
    match error {
        LoweringError::ShapeMismatch {
            role,
            expected,
            actual,
        } => {
            assert_eq!(role, "layers.0.attn.q");
            assert_eq!(expected, vec![(HEADS * HEAD_DIM) as u32, HIDDEN as u32]);
            assert_eq!(actual, vec![(HEADS * HEAD_DIM) as u32, (HIDDEN + 1) as u32]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn tied_word_embeddings_reuse_the_embedding_weight() {
    let model = lower(&tiny_bundle(true), 1, MAX_SEQ).expect("tied weights lower successfully");

    let embed = model
        .weights
        .iter()
        .find(|spec| spec.role == "embed")
        .expect("embed role exists");
    let lm_head = model
        .weights
        .iter()
        .find(|spec| spec.role == "lm_head")
        .expect("lm_head role exists");
    assert_eq!(embed.tensor, EMBED);
    assert_eq!(lm_head.tensor, EMBED);
    assert_eq!(embed.shape, lm_head.shape);

    for program in [&model.prefill, &model.decode] {
        let embed_value = program
            .params
            .iter()
            .find(|param| param.role == "embed")
            .expect("embed parameter exists")
            .value;
        let lm_head_value = program
            .params
            .iter()
            .find(|param| param.role == "lm_head")
            .expect("lm_head parameter exists")
            .value;
        assert_eq!(
            embed_value, lm_head_value,
            "tied lm_head must reuse the embedding Param"
        );
    }
}

#[test]
fn groupwise_quantized_weight_is_lowered() {
    let q_proj = "model.layers.0.self_attn.q_proj.weight";
    let mut tensors = tiny_tensor_list(false);
    replace_tensor(
        &mut tensors,
        q_proj,
        quantized_int4(&[HEADS * HEAD_DIM, HIDDEN], 4),
    );

    let model = lower(&bundle(config_json(false), tensors), 1, MAX_SEQ)
        .expect("groupwise quantized weights lower successfully");

    let spec = model
        .weights
        .iter()
        .find(|spec| spec.role == "layers.0.attn.q")
        .expect("q role exists");
    assert_eq!(spec.dtype, DType::F32);
    assert_eq!(
        spec.encoding,
        IrTensorEncoding::GroupwiseQuantized {
            format: IrQuantizationFormat::Int4,
            group_size: 4,
        }
    );
    let scales = spec.scales.as_ref().expect("scales are recorded");
    assert_eq!(scales.len(), (HEADS * HEAD_DIM * HIDDEN / 4) * 4);

    model
        .prefill
        .program
        .validate()
        .expect("quantized prefill program is valid");
    model
        .decode
        .program
        .validate()
        .expect("quantized decode program is valid");

    let param = model
        .prefill
        .params
        .iter()
        .find(|param| param.role == "layers.0.attn.q")
        .expect("q parameter exists");
    assert!(matches!(
        model
            .prefill
            .program
            .value(param.value)
            .expect("q parameter value exists")
            .encoding,
        IrTensorEncoding::GroupwiseQuantized { .. }
    ));
}

#[test]
fn opaque_quantized_weight_is_rejected() {
    let mut tensors = tiny_tensor_list(false);
    replace_tensor(
        &mut tensors,
        EMBED,
        ModelTensor::new(
            vec![VOCAB, HIDDEN],
            TensorDType::F32,
            TensorEncoding::OpaqueQuantized {
                scheme: "awq".to_string(),
            },
            TensorData::from_vec(vec![0; 8]),
        )
        .expect("opaque tensor is structurally valid"),
    );

    let error = lower(&bundle(config_json(false), tensors), 1, MAX_SEQ)
        .expect_err("opaque encodings must be rejected");
    match error {
        LoweringError::UnsupportedEncoding { tensor, encoding } => {
            assert_eq!(tensor, EMBED);
            assert_eq!(encoding, "awq");
        }
        other => panic!("expected UnsupportedEncoding, got {other:?}"),
    }
}

#[test]
fn invalid_config_is_rejected() {
    let mut config = config_json(false);
    config["hidden_size"] = json!(10);
    config["num_attention_heads"] = json!(4);
    let error = lower(&bundle(config, tiny_tensor_list(false)), 1, MAX_SEQ)
        .expect_err("a non-integral head_dim must be rejected");
    assert!(
        matches!(
            error,
            LoweringError::InvalidConfigField {
                field: "hidden_size",
                ..
            }
        ),
        "expected InvalidConfigField(hidden_size), got {error:?}"
    );

    let mut config = config_json(false);
    config["num_key_value_heads"] = json!(0);
    let error = lower(&bundle(config, tiny_tensor_list(false)), 1, MAX_SEQ)
        .expect_err("zero kv heads must be rejected");
    assert!(
        matches!(
            error,
            LoweringError::InvalidConfigField {
                field: "num_key_value_heads",
                ..
            }
        ),
        "expected InvalidConfigField(num_key_value_heads), got {error:?}"
    );
}

#[test]
fn missing_config_field_is_reported() {
    let mut config = config_json(false);
    config
        .as_object_mut()
        .expect("config is an object")
        .remove("vocab_size");

    let error = lower(&bundle(config, tiny_tensor_list(false)), 1, MAX_SEQ)
        .expect_err("a missing vocab_size must be rejected");
    assert_eq!(error, LoweringError::MissingConfigField("vocab_size"));
}

#[test]
fn unexpected_tensor_is_reported() {
    let mut tensors = tiny_tensor_list(false);
    tensors.push(("checkpoint.extra.weight".to_string(), dense_f32(&[HIDDEN])));

    let error = lower(&bundle(config_json(false), tensors), 1, MAX_SEQ)
        .expect_err("unconsumed tensors must be rejected");
    match error {
        LoweringError::UnexpectedTensor(names) => {
            assert_eq!(names, "checkpoint.extra.weight");
        }
        other => panic!("expected UnexpectedTensor, got {other:?}"),
    }
}

#[test]
fn unsupported_dtype_is_rejected() {
    let mut tensors = tiny_tensor_list(false);
    replace_tensor(
        &mut tensors,
        EMBED,
        ModelTensor::new(
            vec![VOCAB, HIDDEN],
            TensorDType::U8,
            TensorEncoding::Dense,
            TensorData::from_vec(vec![0; VOCAB * HIDDEN]),
        )
        .expect("U8 tensor is structurally valid"),
    );

    let error = lower(&bundle(config_json(false), tensors), 1, MAX_SEQ)
        .expect_err("U8 weights must be rejected");
    match error {
        LoweringError::UnsupportedDType { tensor, dtype } => {
            assert_eq!(tensor, EMBED);
            assert_eq!(dtype, "U8");
        }
        other => panic!("expected UnsupportedDType, got {other:?}"),
    }
}

#[test]
fn unsupported_architecture_is_rejected() {
    let bundle = bundle_with_architecture("gpt2", config_json(false), tiny_tensor_list(false));
    let error = lower(&bundle, 1, MAX_SEQ).expect_err("unknown architectures must be rejected");
    assert_eq!(
        error,
        LoweringError::UnsupportedArchitecture {
            family: "llama".to_string(),
            found: "gpt2".to_string(),
        }
    );
}

#[test]
fn tiny_llama_architecture_alias_is_accepted() {
    let bundle =
        bundle_with_architecture("tiny-llama", config_json(false), tiny_tensor_list(false));
    let model = lower(&bundle, 1, MAX_SEQ).expect("tiny-llama is an accepted alias");
    assert_eq!(model.architecture.family, "llama");
}

#[test]
fn limits_are_enforced() {
    let error = lower(&tiny_bundle(false), 1, MAX_SEQ + 1)
        .expect_err("max_seq beyond max_position_embeddings must be rejected");
    assert!(
        matches!(
            error,
            LoweringError::InvalidConfigField {
                field: "max_seq",
                ..
            }
        ),
        "expected InvalidConfigField(max_seq), got {error:?}"
    );

    let error = lower(&tiny_bundle(false), 2, MAX_SEQ)
        .expect_err("max_batch other than one must be rejected");
    assert!(
        matches!(error, LoweringError::UnsupportedFeature(_)),
        "expected UnsupportedFeature, got {error:?}"
    );
}
