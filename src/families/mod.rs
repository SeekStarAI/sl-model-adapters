// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Model families.
//!
//! Each family owns a directory. Shared types (`LoweredModel`, `WeightSpec`,
//! `LoweringError`) live at the crate root; family rules and configuration
//! schemas live under `families/<family>/<version>/`.

pub mod llama;
