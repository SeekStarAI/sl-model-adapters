// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Model-family lowering from [`sl_model::ModelBundle`] to [`sl_ir::Program`].
//!
//! The first implemented family is `llama`, variant `dense`: decoder-only,
//! bias-free dense transformer blocks with RMSNorm, NeoX RoPE, causal attention
//! (GQA allowed) and SwiGLU. Lowering produces a symbolic prefill program and a
//! fixed-length decode program that share one weight table and one KV-cache
//! layout. See `docs/ARCH-0006` for the contract.

#![deny(unsafe_code)]

pub mod config;
pub mod error;
pub mod llama;
pub mod model;

pub use config::{LlamaConfig, CONFIG_VERSION};
pub use error::LoweringError;
pub use llama::{lower, FAMILY, VARIANT};
pub use model::{
    ArchitectureId, KvCacheSpec, LoweredModel, LoweredProgram, ParamRef, ProgramIo, WeightSpec,
};
