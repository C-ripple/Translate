//! Versioned diagnostics shared by import, verification and SPMD lowering.
use serde::Serialize;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum Code {
    #[serde(rename = "RIPPLE-D000")]
    Pipeline,
    #[serde(rename = "RIPPLE-D001")]
    InvalidInput,
    #[serde(rename = "RIPPLE-I001")]
    InvalidIr,
    #[serde(rename = "RIPPLE-I002")]
    ControlFlow,
    #[serde(rename = "RIPPLE-I003")]
    Type,
    #[serde(rename = "RIPPLE-I004")]
    Initialization,
    #[serde(rename = "RIPPLE-I005")]
    Call,
    #[serde(rename = "RIPPLE-I006")]
    IrLimit,
    #[serde(rename = "RIPPLE-M001")]
    UnsupportedType,
    #[serde(rename = "RIPPLE-M002")]
    UnsupportedOperation,
    #[serde(rename = "RIPPLE-M003")]
    Assertion,
    #[serde(rename = "RIPPLE-M004")]
    Drop,
    #[serde(rename = "RIPPLE-M005")]
    Unwind,
    #[serde(rename = "RIPPLE-M006")]
    UnsupportedCall,
    #[serde(rename = "RIPPLE-S001")]
    Bounds,
    #[serde(rename = "RIPPLE-S002")]
    Address,
    #[serde(rename = "RIPPLE-S003")]
    Expansion,
    #[serde(rename = "RIPPLE-S004")]
    Entry,
}

#[derive(Clone, Debug, Serialize)]
pub struct Diagnostic {
    pub schema_version: u32,
    pub code: Code,
    pub message: String,
    /// Exact span display from rustc (or source label for caller-supplied IR).
    /// This is opaque text, not a machine-readable file/line range.
    pub source: Option<String>,
    pub function: Option<String>,
}

impl Diagnostic {
    pub fn new(code: Code, message: impl Into<String>) -> Self {
        Self {
            schema_version: 1,
            code,
            message: message.into(),
            source: None,
            function: None,
        }
    }
    pub fn at(mut self, source: impl Into<String>) -> Self {
        if self.source.is_none() {
            self.source = Some(source.into());
        }
        self
    }
    pub fn in_function(mut self, function: impl Into<String>) -> Self {
        if self.function.is_none() {
            self.function = Some(function.into());
        }
        self
    }
}

impl fmt::Display for Code {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Pipeline => "RIPPLE-D000",
            Self::InvalidInput => "RIPPLE-D001",
            Self::InvalidIr => "RIPPLE-I001",
            Self::ControlFlow => "RIPPLE-I002",
            Self::Type => "RIPPLE-I003",
            Self::Initialization => "RIPPLE-I004",
            Self::Call => "RIPPLE-I005",
            Self::IrLimit => "RIPPLE-I006",
            Self::UnsupportedType => "RIPPLE-M001",
            Self::UnsupportedOperation => "RIPPLE-M002",
            Self::Assertion => "RIPPLE-M003",
            Self::Drop => "RIPPLE-M004",
            Self::Unwind => "RIPPLE-M005",
            Self::UnsupportedCall => "RIPPLE-M006",
            Self::Bounds => "RIPPLE-S001",
            Self::Address => "RIPPLE-S002",
            Self::Expansion => "RIPPLE-S003",
            Self::Entry => "RIPPLE-S004",
        })
    }
}
impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)?;
        if let Some(source) = &self.source {
            write!(f, " at {source}")?;
        }
        if let Some(function) = &self.function {
            write!(f, " in {function}")?;
        }
        Ok(())
    }
}
impl std::error::Error for Diagnostic {}
impl From<&str> for Diagnostic {
    fn from(message: &str) -> Self {
        Self::new(Code::Pipeline, message)
    }
}
impl From<String> for Diagnostic {
    fn from(message: String) -> Self {
        Self::new(Code::Pipeline, message)
    }
}
