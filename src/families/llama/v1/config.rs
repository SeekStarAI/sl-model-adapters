// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Typed architecture configuration for the `llama` dense family.

use serde_json::{Map, Value};

use crate::error::LoweringError;

/// Version of the configuration schema understood by this adapter.
pub const CONFIG_VERSION: u32 = 1;

/// Typed configuration for a Llama-style decoder-only dense model.
///
/// All dimensions are validated to be non-zero and mutually consistent by
/// [`LlamaConfig::from_json`]. [`Self::head_dim`] therefore never divides by
/// zero for configs produced by that constructor.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LlamaConfig {
    /// Hidden width of the residual stream.
    pub hidden_size: u32,
    /// Number of decoder layers.
    pub num_hidden_layers: u32,
    /// Number of query attention heads.
    pub num_attention_heads: u32,
    /// Number of key/value attention heads; equal to the query heads unless GQA.
    pub num_key_value_heads: u32,
    /// Width of the SwiGLU intermediate projection.
    pub intermediate_size: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE theta (base frequency).
    pub rope_theta: f32,
    /// Largest supported sequence length, including the prompt.
    pub max_position_embeddings: u32,
    /// Whether the output projection shares the embedding table.
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl LlamaConfig {
    /// Parses and validates a Hugging Face-style `config.json` document.
    ///
    /// Unknown fields are ignored. Missing fields produce
    /// [`LoweringError::MissingConfigField`]; present but malformed fields
    /// produce [`LoweringError::InvalidConfigField`].
    pub fn from_json(config: &Value) -> Result<Self, LoweringError> {
        let object = config
            .as_object()
            .ok_or(LoweringError::InvalidConfigField {
                field: "config",
                reason: "expected a JSON object",
            })?;

        let config = Self {
            hidden_size: required_positive_u32(object, "hidden_size")?,
            num_hidden_layers: required_positive_u32(object, "num_hidden_layers")?,
            num_attention_heads: required_positive_u32(object, "num_attention_heads")?,
            num_key_value_heads: required_positive_u32(object, "num_key_value_heads")?,
            intermediate_size: required_positive_u32(object, "intermediate_size")?,
            vocab_size: required_positive_u32(object, "vocab_size")?,
            rms_norm_eps: required_positive_f32(object, "rms_norm_eps")?,
            rope_theta: required_positive_f32(object, "rope_theta")?,
            max_position_embeddings: required_positive_u32(object, "max_position_embeddings")?,
            tie_word_embeddings: optional_bool(object, "tie_word_embeddings")?.unwrap_or(false),
        };

        config.validate()?;
        Ok(config)
    }

    /// Returns the per-head width, `hidden_size / num_attention_heads`.
    pub const fn head_dim(&self) -> u32 {
        self.hidden_size / self.num_attention_heads
    }

    /// Returns the query projection width, `num_attention_heads * head_dim`.
    pub const fn query_dim(&self) -> u32 {
        self.num_attention_heads * self.head_dim()
    }

    /// Returns the key/value projection width, `num_key_value_heads * head_dim`.
    pub const fn kv_dim(&self) -> u32 {
        self.num_key_value_heads * self.head_dim()
    }

    fn validate(&self) -> Result<(), LoweringError> {
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(LoweringError::InvalidConfigField {
                field: "hidden_size",
                reason: "must be divisible by num_attention_heads so head_dim is an integer",
            });
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(LoweringError::InvalidConfigField {
                field: "num_key_value_heads",
                reason: "must divide num_attention_heads",
            });
        }
        if !self.rms_norm_eps.is_finite() {
            return Err(LoweringError::InvalidConfigField {
                field: "rms_norm_eps",
                reason: "must be finite",
            });
        }
        if !self.rope_theta.is_finite() {
            return Err(LoweringError::InvalidConfigField {
                field: "rope_theta",
                reason: "must be finite",
            });
        }
        Ok(())
    }
}

fn required_field<'a>(
    object: &'a Map<String, Value>,
    field: &'static str,
) -> Result<&'a Value, LoweringError> {
    object
        .get(field)
        .ok_or(LoweringError::MissingConfigField(field))
}

fn required_positive_u32(
    object: &Map<String, Value>,
    field: &'static str,
) -> Result<u32, LoweringError> {
    let value = required_field(object, field)?;
    let number = value.as_u64().ok_or(LoweringError::InvalidConfigField {
        field,
        reason: "expected a non-negative integer",
    })?;
    if number == 0 {
        return Err(LoweringError::InvalidConfigField {
            field,
            reason: "must be greater than zero",
        });
    }
    u32::try_from(number).map_err(|_| LoweringError::InvalidConfigField {
        field,
        reason: "does not fit in u32",
    })
}

fn required_positive_f32(
    object: &Map<String, Value>,
    field: &'static str,
) -> Result<f32, LoweringError> {
    let value = required_field(object, field)?;
    let number = value.as_f64().ok_or(LoweringError::InvalidConfigField {
        field,
        reason: "expected a number",
    })?;
    if !number.is_finite() || number <= 0.0 {
        return Err(LoweringError::InvalidConfigField {
            field,
            reason: "must be a finite number greater than zero",
        });
    }
    let number = number as f32;
    if !number.is_finite() || number <= 0.0 {
        return Err(LoweringError::InvalidConfigField {
            field,
            reason: "is outside the representable f32 range",
        });
    }
    Ok(number)
}

fn optional_bool(
    object: &Map<String, Value>,
    field: &'static str,
) -> Result<Option<bool>, LoweringError> {
    match object.get(field) {
        None => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(LoweringError::InvalidConfigField {
            field,
            reason: "expected a boolean",
        }),
    }
}
