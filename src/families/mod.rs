// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Model families.
//!
//! Each family owns a directory. Shared types (`LoweredModel`, `WeightSpec`,
//! `LoweringError`) live at the crate root; family rules and configuration
//! schemas live under `families/<family>/<version>/`.
//!
//! Every family has `README.md` and `models.toml`. Only families with an
//! implementation are declared as Rust modules here; planned families keep
//! their catalog until an adapter lands. See `README.md` in this directory for
//! the full catalog.

pub mod llama;
