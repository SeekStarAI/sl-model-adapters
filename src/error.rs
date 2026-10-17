// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Structured errors returned by model-family lowering.

use std::error::Error;
use std::fmt;

use sl_ir::{DType, IrError};

/// Failure while lowering a model bundle into `sl-ir` programs.
///
/// Every variant carries the role, source tensor name, or config field needed
/// to locate the offending input. Missing tensors are reported together in a
/// single [`Self::MissingTensor`] value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LoweringError {
    /// The bundle declares an architecture this adapter does not implement.
    UnsupportedArchitecture { family: String, found: String },
    /// A required architecture config field is absent.
    MissingConfigField(&'static str),
    /// A config field is present but invalid.
    InvalidConfigField {
        field: &'static str,
        reason: &'static str,
    },
    /// One or more tensors required by a role are absent.
    MissingTensor(String),
    /// One or more tensors are not consumed by any role.
    UnexpectedTensor(String),
    /// A tensor's logical shape differs from the role's expected shape.
    ShapeMismatch {
        role: String,
        expected: Vec<u32>,
        actual: Vec<u32>,
    },
    /// A tensor's dtype differs from the role's expected dtype.
    DTypeMismatch {
        role: String,
        expected: DType,
        actual: DType,
    },
    /// A tensor uses a dtype this adapter cannot lower.
    UnsupportedDType { tensor: String, dtype: String },
    /// A tensor uses a storage encoding this adapter cannot lower.
    UnsupportedEncoding { tensor: String, encoding: String },
    /// A configuration or limit combination is not supported yet.
    UnsupportedFeature(&'static str),
    /// The constructed program failed `sl-ir` validation.
    Program(IrError),
}

impl fmt::Display for LoweringError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedArchitecture { family, found } => write!(
                formatter,
                "unsupported architecture `{found}`; this adapter implements family `{family}`"
            ),
            Self::MissingConfigField(field) => {
                write!(formatter, "missing required config field `{field}`")
            }
            Self::InvalidConfigField { field, reason } => {
                write!(formatter, "invalid config field `{field}`: {reason}")
            }
            Self::MissingTensor(tensors) => {
                write!(formatter, "missing tensor(s) required by roles: {tensors}")
            }
            Self::UnexpectedTensor(tensors) => {
                write!(formatter, "tensor(s) not consumed by any role: {tensors}")
            }
            Self::ShapeMismatch {
                role,
                expected,
                actual,
            } => write!(
                formatter,
                "shape mismatch for role `{role}`: expected {expected:?}, actual {actual:?}"
            ),
            Self::DTypeMismatch {
                role,
                expected,
                actual,
            } => write!(
                formatter,
                "dtype mismatch for role `{role}`: expected {expected}, actual {actual}"
            ),
            Self::UnsupportedDType { tensor, dtype } => {
                write!(
                    formatter,
                    "unsupported dtype `{dtype}` for tensor `{tensor}`"
                )
            }
            Self::UnsupportedEncoding { tensor, encoding } => write!(
                formatter,
                "unsupported encoding `{encoding}` for tensor `{tensor}`"
            ),
            Self::UnsupportedFeature(feature) => {
                write!(formatter, "unsupported feature: {feature}")
            }
            Self::Program(error) => write!(formatter, "invalid lowered program: {error}"),
        }
    }
}

impl Error for LoweringError {}

impl From<IrError> for LoweringError {
    fn from(error: IrError) -> Self {
        Self::Program(error)
    }
}
