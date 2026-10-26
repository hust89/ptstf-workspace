use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum PtstfError {
    EmptyDataset,
    DimensionMismatch { expected: usize, found: usize },
    InvalidParameter(String),
}

impl fmt::Display for PtstfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PtstfError::EmptyDataset => write!(f, "empty dataset"),
            PtstfError::DimensionMismatch { expected, found } => {
                write!(f, "dimension mismatch: expected {expected}, found {found}")
            }
            PtstfError::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
        }
    }
}

impl std::error::Error for PtstfError {}
