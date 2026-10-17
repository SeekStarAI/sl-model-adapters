// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Version 1 of the `llama` dense family.
//!
//! This module owns the v1 configuration schema and its lowering to
//! `sl-ir::Program`. Adding a new schema version means adding a sibling
//! `v2/` directory and dispatching to it from the family module, without
//! touching older versions.

pub mod config;
mod lower;

pub use config::{LlamaConfig, CONFIG_VERSION};
pub use lower::lower;
