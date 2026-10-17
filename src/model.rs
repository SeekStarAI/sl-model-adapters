// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Lowered model-family artifacts shared by all adapters.

use sl_ir::{DType, Program, TensorEncoding, ValueId};
use sl_model::TensorData;

/// Stable identity of the family and variant that produced a [`LoweredModel`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchitectureId {
    /// Family name, for example `"llama"`.
    pub family: &'static str,
    /// Variant name, for example `"dense"`; `None` when the family has no variant.
    pub variant: Option<&'static str>,
    /// Version of the architecture configuration schema.
    pub config_version: u32,
}

/// Per-program entry points produced by lowering.
///
/// Ids index the value table of the owning [`LoweredProgram::program`]. The
/// prefill and decode programs have independent value tables, so the same role
/// has different ids in each program.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProgramIo {
    /// Token ids, `U32 [batch, seq]`.
    pub token_ids: ValueId,
    /// Absolute positions, `U32 [batch, seq]`.
    pub positions: ValueId,
    /// Number of tokens in this step, `U32 []`.
    pub seq_len: ValueId,
    /// Number of already-cached tokens, `U32 []`.
    pub past_len: ValueId,
    /// Output logits, `F32 [batch, vocab]`.
    pub logits: ValueId,
}

/// Binding between a stable parameter role and a program's `Param` value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParamRef {
    /// Role path, for example `"layers.0.attn.q"`.
    pub role: String,
    /// `Param` value in the owning program.
    pub value: ValueId,
}

/// One program produced by lowering, its IO handles and parameter bindings.
#[derive(Clone, Debug)]
pub struct LoweredProgram {
    /// Validated `sl-ir` program.
    pub program: Program,
    /// Per-step inputs and output handles.
    pub io: ProgramIo,
    /// Parameter roles bound in this program.
    pub params: Vec<ParamRef>,
}

/// One weight role and the source tensor that supplies it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightSpec {
    /// Stable role path, for example `"layers.0.attn.q"`.
    pub role: String,
    /// Source tensor name, for example `"model.layers.0.self_attn.q_proj.weight"`.
    pub tensor: String,
    /// Logical shape, output dimension first for matrix weights.
    pub shape: Vec<u32>,
    /// Logical element type. Quantized weights keep their source logical dtype.
    pub dtype: DType,
    /// Physical storage encoding declared on the program `Param`.
    pub encoding: TensorEncoding,
    /// Group-wise scale payload copied from the source tensor, if any.
    ///
    /// Scales are not yet bound to the IR; this keeps the asset available for
    /// the runtime once a packed-layout contract exists.
    pub scales: Option<TensorData>,
}

/// Shape and dtype of the KV cache shared by prefill and decode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvCacheSpec {
    /// Number of decoder layers.
    pub layers: u32,
    /// Number of key/value heads.
    pub kv_heads: u32,
    /// Per-head width.
    pub head_dim: u32,
    /// Maximum sequence length.
    pub max_seq: u32,
    /// Element type of the cache.
    pub dtype: DType,
}

/// Family lowering result: two programs, one shared weight table and KV layout.
#[derive(Clone, Debug)]
pub struct LoweredModel {
    /// `sl-ir` ABI version the programs were built against.
    pub ir_version: u32,
    /// Family and variant identity.
    pub architecture: ArchitectureId,
    /// Program that processes a prompt of one or more tokens.
    pub prefill: LoweredProgram,
    /// Program that processes exactly one token.
    pub decode: LoweredProgram,
    /// Weight roles shared by both programs.
    pub weights: Vec<WeightSpec>,
    /// KV-cache layout shared by both programs.
    pub kv_cache: KvCacheSpec,
}
