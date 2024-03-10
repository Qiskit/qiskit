// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::Python;

/// Affect the dynamic scaling of the weight of node-set-based heuristics (basic and lookahead).
#[pyclass]
#[pyo3(module = "qiskit._accelerate.sabre", frozen)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SetScaling {
    /// No dynamic scaling of the weight.
    Constant,
    /// Scale the weight by the current number of nodes in the set (e.g., if it contains 5 nodes,
    /// the weight will be multiplied by ``0.2``).
    Size,
}
#[pymethods]
impl SetScaling {
    pub fn __reduce__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let name = match self {
            SetScaling::Constant => "Constant",
            SetScaling::Size => "Size",
        };
        Ok((
            py.import("builtins")?.getattr("getattr")?,
            (py.get_type::<Self>(), name),
        )
            .into_py(py))
    }
}

#[pyclass]
#[pyo3(module = "qiskit._accelerate.sabre", frozen)]
#[derive(Clone, Copy, PartialEq)]
pub struct BasicHeuristic {
    pub weight: f64,
    pub scale: SetScaling,
}
#[pymethods]
impl BasicHeuristic {
    #[new]
    pub fn new(weight: f64, scale: SetScaling) -> Self {
        Self { weight, scale }
    }

    pub fn __getnewargs__(&self, py: Python) -> Py<PyAny> {
        (self.weight, self.scale).into_py(py)
    }

    pub fn __eq__(&self, py: Python, other: Py<PyAny>) -> bool {
        if let Ok(other) = other.extract::<Self>(py) {
            self == &other
        } else {
            false
        }
    }

    pub fn __repr__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let fmt = "BasicHeuristic(weight={!r}, scale={!r})";
        Ok(PyString::new(py, fmt)
            .call_method1("format", (self.weight, self.scale))?
            .into_py(py))
    }
}

#[pyclass]
#[pyo3(module = "qiskit._accelerate.sabre", frozen)]
#[derive(Clone, Copy, PartialEq)]
pub struct LookaheadHeuristic {
    pub weight: f64,
    /// Number of gates to consider in the heuristic.
    pub size: usize,
    pub scale: SetScaling,
}
#[pymethods]
impl LookaheadHeuristic {
    #[new]
    pub fn new(weight: f64, size: usize, scale: SetScaling) -> Self {
        Self {
            weight,
            size,
            scale,
        }
    }

    pub fn __getnewargs__(&self, py: Python) -> Py<PyAny> {
        (self.weight, self.size, self.scale).into_py(py)
    }

    pub fn __eq__(&self, py: Python, other: Py<PyAny>) -> bool {
        if let Ok(other) = other.extract::<Self>(py) {
            self == &other
        } else {
            false
        }
    }

    pub fn __repr__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let fmt = "LookaheadHeuristic(weight={!r}, size={!r}, scale={!r})";
        Ok(PyString::new(py, fmt)
            .call_method1("format", (self.weight, self.size, self.scale))?
            .into_py(py))
    }
}

#[pyclass]
#[pyo3(module = "qiskit._accelerate.sabre", frozen)]
#[derive(Clone, Copy, PartialEq)]
pub struct DecayHeuristic {
    pub increment: f64,
    pub reset: usize,
}
#[pymethods]
impl DecayHeuristic {
    #[new]
    pub fn new(increment: f64, reset: usize) -> Self {
        Self { increment, reset }
    }

    pub fn __getnewargs__(&self, py: Python) -> Py<PyAny> {
        (self.increment, self.reset).into_py(py)
    }

    pub fn __eq__(&self, py: Python, other: Py<PyAny>) -> bool {
        if let Ok(other) = other.extract::<Self>(py) {
            self == &other
        } else {
            false
        }
    }

    pub fn __repr__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let fmt = "DecayHeuristic(increment={!r}, reset={!r})";
        Ok(PyString::new(py, fmt)
            .call_method1("format", (self.increment, self.reset))?
            .into_py(py))
    }
}

#[pyclass]
#[pyo3(module = "qiskit._accelerate.sabre", frozen)]
#[derive(Clone, PartialEq)]
pub struct Heuristic {
    pub basic: Option<BasicHeuristic>,
    pub lookahead: Option<LookaheadHeuristic>,
    pub decay: Option<DecayHeuristic>,
    pub best_epsilon: f64,
    pub attempt_limit: usize,
}

#[pymethods]
impl Heuristic {
    /// Construct a new Sabre heuristic.  This can either be made directly of the desired
    /// components, or you can make an empty heuristic and use the ``with_*`` methods to add
    /// components to it.
    ///
    /// Args:
    ///     attempt_limit (int): the maximum number of swaps to attempt before using a fallback
    ///         "escape" mechanism to forcibly route a gate.  Set this to ``None`` to entirely
    ///         disable the mechanism, but beware that it's possible (on large coupling maps with a
    ///         lookahead heuristic component) for Sabre to get stuck in an inescapable arbitrarily
    ///         deep local minimum of the heuristic.  If this happens, and the escape mechanism is
    ///         disabled entirely, Sabre will enter an infinite loop.
    ///     best_epsilon (float): the floating-point epsilon to use when comparing scores to find
    ///         the best value.
    #[new]
    #[pyo3(signature = (basic=None, lookahead=None, decay=None, attempt_limit=1000, best_epsilon=1e-10))]
    pub fn new(
        basic: Option<BasicHeuristic>,
        lookahead: Option<LookaheadHeuristic>,
        decay: Option<DecayHeuristic>,
        attempt_limit: Option<usize>,
        best_epsilon: f64,
    ) -> Self {
        Self {
            basic,
            lookahead,
            decay,
            best_epsilon,
            attempt_limit: attempt_limit.unwrap_or(usize::MAX),
        }
    }

    pub fn __getnewargs__(&self, py: Python) -> Py<PyAny> {
        (
            self.basic,
            self.lookahead,
            self.decay,
            self.attempt_limit,
            self.best_epsilon,
        )
            .into_py(py)
    }

    /// Set the weight of the ``basic`` heuristic (the sum of distances of gates in the front
    /// layer).  This is often set to ``1.0``.  You almost certainly should enable this part of the
    /// heuristic, or it's highly unlikely that Sabre will be able to make any progress.
    pub fn with_basic(&self, weight: f64, scale: SetScaling) -> Self {
        Self {
            basic: Some(BasicHeuristic { weight, scale }),
            ..self.clone()
        }
    }

    /// Set the weight and extended-set size of the ``lookahead`` heuristic.  The weight here
    /// should typically be less than that of ``basic``.
    pub fn with_lookahead(&self, weight: f64, size: usize, scale: SetScaling) -> Self {
        Self {
            lookahead: Some(LookaheadHeuristic {
                weight,
                size,
                scale,
            }),
            ..self.clone()
        }
    }

    /// Set the multiplier increment and reset interval of the decay heuristic.  The reset interval
    /// must be non-zero.
    pub fn with_decay(&self, increment: f64, reset: usize) -> PyResult<Self> {
        if reset == 0 {
            Err(PyValueError::new_err("decay reset interval cannot be zero"))
        } else {
            Ok(Self {
                decay: Some(DecayHeuristic { increment, reset }),
                ..self.clone()
            })
        }
    }

    pub fn __eq__(&self, py: Python, other: Py<PyAny>) -> bool {
        if let Ok(other) = other.extract::<Self>(py) {
            self == &other
        } else {
            false
        }
    }

    pub fn __repr__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let fmt = "Heuristic(basic={!r}, lookahead={!r}, decay={!r}, attempt_limit={!r}, best_epsilon={!r})";
        Ok(PyString::new(py, fmt)
            .call_method1(
                "format",
                (
                    self.basic,
                    self.lookahead,
                    self.decay,
                    self.attempt_limit,
                    self.best_epsilon,
                ),
            )?
            .into_py(py))
    }
}
