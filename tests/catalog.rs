// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Catalog layout test: every family directory must ship both a human-readable
//! README and a machine-readable `models.toml`, and only implemented families
//! are Rust modules.

use std::fs;
use std::path::{Path, PathBuf};

fn family_dirs() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families");
    let mut dirs: Vec<PathBuf> = fs::read_dir(&root)
        .expect("families directory exists")
        .map(|entry| entry.expect("readable entry").path())
        .filter(|path| path.is_dir())
        .collect();
    dirs.sort();
    dirs
}

#[test]
fn every_family_has_a_catalog() {
    let dirs = family_dirs();
    assert!(dirs.len() >= 10, "expected the initial family catalog");

    for dir in &dirs {
        let name = dir
            .file_name()
            .and_then(|name| name.to_str())
            .expect("family directory name is UTF-8");
        let readme = dir.join("README.md");
        let models = dir.join("models.toml");
        assert!(readme.is_file(), "{} is missing README.md", name);
        assert!(models.is_file(), "{} is missing models.toml", name);

        let catalog = fs::read_to_string(&models).expect("models.toml is readable");
        assert!(
            catalog.contains(&format!("family = \"{name}\"")),
            "{} does not declare its family name",
            name
        );
        assert!(
            catalog.contains("[[variants]]"),
            "{} does not list any variants",
            name
        );
    }

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families");
    assert!(root.join("README.md").is_file(), "catalog index is missing");
    assert!(
        root.join("llama/mod.rs").is_file(),
        "implemented family must be a Rust module"
    );
    assert!(
        dirs.iter().any(
            |dir| dir.file_name().and_then(|name| name.to_str()) != Some("llama")
                && !dir.join("mod.rs").exists()
        ),
        "expected at least one not-yet-implemented family"
    );
}
