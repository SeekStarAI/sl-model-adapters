// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! The `llama` model family.
//!
//! Family-wide metadata lives here; each supported configuration schema version
//! has its own submodule (`v1`, `v2`, ...) containing the typed config and the
//! lowering implementation for that version.

pub mod v1;

pub use v1::lower;

/// Family name recorded on [`crate::ArchitectureId`].
pub const FAMILY: &str = "llama";

/// Variant name recorded on [`crate::ArchitectureId`].
pub const VARIANT: &str = "dense";

/// Architecture strings accepted by this family.
pub(crate) const ARCHITECTURES: [&str; 2] = ["llama", "tiny-llama"];
