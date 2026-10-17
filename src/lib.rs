// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Model-family lowering from [`sl_model::ModelBundle`] to [`sl_ir::Program`].
//!
//! Layout:
//!
//! ```text
//! src/
//! ├── error.rs          # LoweringError (shared)
//! ├── model.rs          # LoweredModel / WeightSpec / ... (shared)
//! └── families/
//!     └── llama/
//!         ├── mod.rs    # family metadata, version dispatch
//!         └── v1/       # config schema v1 + lowering
//!             ├── config.rs
//!             └── lower.rs
//! ```
//!
//! Only cross-family contracts live at the crate root. Family rules, tensor
//! names, configuration fields and lowering stay inside the family directory so
//! that adding a family is additive. See `docs/ARCH-0006` for the contract.

#![deny(unsafe_code)]

pub mod error;
pub mod families;
pub mod model;

pub use error::LoweringError;
pub use families::llama::v1::{LlamaConfig, CONFIG_VERSION};
pub use families::llama::{lower, FAMILY, VARIANT};
pub use model::{
    ArchitectureId, KvCacheSpec, LoweredModel, LoweredProgram, ParamRef, ProgramIo, WeightSpec,
};
