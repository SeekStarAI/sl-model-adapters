// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Layout tests: every family owns a directory and every supported schema
//! version owns a subdirectory with its own entry point.

use sl_model::ModelBundle;
use sl_model_adapters::families::llama::{self, v1};
use sl_model_adapters::{LoweredModel, LoweringError};

#[test]
fn family_directory_exposes_versioned_entry_point() {
    assert_eq!(llama::FAMILY, "llama");
    assert_eq!(llama::VARIANT, "dense");
    assert_eq!(v1::CONFIG_VERSION, 1);

    // The versioned entry point must keep the shared lowering signature.
    let entry: fn(&ModelBundle, u32, u32) -> Result<LoweredModel, LoweringError> = v1::lower;
    let _ = entry;
}
