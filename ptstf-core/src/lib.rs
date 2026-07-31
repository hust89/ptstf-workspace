pub mod graph;
pub mod tensor;
pub mod embedding;
pub mod flow;
pub mod optimization;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum PtstfError {
    #[error("Dimension mismatch: expected {expected}, found {found}")]
    DimensionMismatch { expected: usize, found: usize },
    #[error("Dataset is empty")]
    EmptyDataset,
    #[error("Mathematical or optimization parameter is invalid: {0}")]
    InvalidParameter(String),
}

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyclass(name = "ConductanceField")]
pub struct PyConductanceField {
    inner: embedding::ConductanceField,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyConductanceField {
    #[new]
    pub fn new(embeddings: Vec<(f64, f64)>, discrete_conductances: Vec<f64>, eta: f64) -> PyResult<Self> {
        use nalgebra::Vector2;
        let converted: Vec<Vector2<f64>> = embeddings.into_iter()
            .map(|(x, y)| Vector2::new(x, y))
            .collect();
        let field = embedding::ConductanceField::new(converted, discrete_conductances, eta)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyConductanceField { inner: field })
    }

    pub fn evaluate(&self, x: f64, y: f64) -> f64 {
        use nalgebra::Vector2;
        self.inner.evaluate(Vector2::new(x, y))
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "TriadicFlowCurve")]
pub struct PyTriadicFlowCurve {
    inner: flow::TriadicFlowCurve,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTriadicFlowCurve {
    #[new]
    pub fn new(p0: (f64, f64), p1: (f64, f64), p2: (f64, f64)) -> Self {
        use nalgebra::Vector2;
        PyTriadicFlowCurve {
            inner: flow::TriadicFlowCurve::new(
                Vector2::new(p0.0, p0.1),
                Vector2::new(p1.0, p1.1),
                Vector2::new(p2.0, p2.1),
            )
        }
    }

    pub fn bending_energy(&self, a: f64, b: f64) -> f64 {
        self.inner.bending_energy(a, b)
    }

    pub fn detect_crossing(&self) -> bool {
        self.inner.detect_crossing()
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn ptstf_core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyConductanceField>()?;
    m.add_class::<PyTriadicFlowCurve>()?;
    Ok(())
}
