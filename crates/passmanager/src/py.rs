// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::sync::Arc;

use hashbrown::HashMap;
use pyo3::{
    exceptions::{PyIndexError, PyRuntimeError, PyTypeError, PyValueError},
    intern,
    prelude::*,
    types::PyString,
};

use crate::{Pass, PassContext, PassManager, PassManagerContext, PassManagerError, Value};

impl From<PassManagerError> for PyErr {
    fn from(value: PassManagerError) -> Self {
        match value {
            // TODO introduce a Python error for the errors here (or re-use PassManagerError)
            PassManagerError::EmptyTask => PyValueError::new_err("Empty tasks are not allowed."),
            PassManagerError::FailedOutputConversion | PassManagerError::IncompatibleTypes => {
                PyTypeError::new_err("Incompatible types.")
            }
            PassManagerError::IndexError { index, len } => {
                PyIndexError::new_err(format!("Index {index} out of bounds ({len})"))
            }
            PassManagerError::PassError(p) => PyRuntimeError::new_err(p.to_string()),
        }
    }
}

// #[cfg(feature = "python")]
#[pyclass]
#[pyo3(name = "PassContext")]
#[derive(Debug)]
pub struct PyPassContext {
    global: Arc<PassManagerContext>,
    data: HashMap<String, Py<PyAny>>,
    /// If ``True``, the pass has changed the IR.  If ``False``, no updates have been made.
    pub has_changed: bool,
    /// If ``True``, all values have been drained into the global pass manager context and it is
    /// no longer to read from or write to this pass context.
    pub is_drained: bool,
}

impl PyPassContext {
    fn new_bound<'py>(py: Python<'py>, pass_context: &PassContext) -> PyResult<Bound<'py, Self>> {
        Bound::new(
            py,
            PyPassContext {
                global: Arc::clone(&pass_context.global_context),
                data: HashMap::new(),
                has_changed: pass_context.has_changed,
                is_drained: false,
            },
        )
    }

    fn drain_into_context<'py>(slf: Bound<'py, Self>, pass_context: &mut PassContext) {
        let mut borrowed = slf.borrow_mut();
        for (key, value) in borrowed.data.drain() {
            pass_context.set(key, Value::PyCompatible(Box::new(value)));
        }
        borrowed.is_drained = true;
    }
}

#[pymethods]
impl PyPassContext {
    #[pyo3(signature = (key,))]
    /// Get a value from the pass context.
    ///
    /// Args:
    ///     key: The lookup key.
    ///
    /// Returns:
    ///     The value, if it exists and can be represented in Python. Else ``None`` is returned.
    ///
    /// Raises:
    ///     RuntimeError: If the pass context instance has already been drained into the global
    ///         pass manager execution context. See also :attr:`is_drained`.
    pub fn get(&self, py: Python<'_>, key: Bound<'_, PyString>) -> PyResult<Option<Py<PyAny>>> {
        if self.is_drained {
            return Err(PyRuntimeError::new_err(
                "PassContext is already drained and cannot be read",
            ));
        }
        let key = key.to_str()?;

        if let Some(value) = self.data.get(key) {
            Ok(Some(value.clone_ref(py)))
        } else if let Some(Value::PyCompatible(py_value)) = self.global.data.get(key) {
            Ok(Some(py_value.to_py_any(py)?.clone_ref(py)))
        } else {
            Ok(None)
        }
    }

    /// Set a value in the pass context.
    pub fn set(&mut self, key: Bound<'_, PyString>, value: Bound<'_, PyAny>) {
        let key = key.to_string();
        self.data.insert(key, value.unbind());
    }
}

pub struct PassFromPy {
    /// A handle to the Python Pass class.
    py_obj: Py<PyAny>,
}

impl PassFromPy {
    fn from_bound(py_pass: Bound<'_, PyAny>) -> Self {
        Self {
            py_obj: py_pass.unbind(),
        }
    }
}

impl Pass for PassFromPy {
    type InputIR = Py<PyAny>;
    type OutputIR = Py<PyAny>;

    fn run(
        &self,
        ir: Self::InputIR,
        context: &mut crate::PassContext,
    ) -> anyhow::Result<Self::OutputIR> {
        // We create a PyPassContext from a PassContext, which clones the Arc<PassManagerContext>
        // and keeps a local update HashMap with PyAny values. After the pass is run, we drain
        // the PyAny values into the &mut PassContext, which is then handled in the main loop.
        // This means that the values are no longer valid to read from Python.
        let ir_out = Python::attach(|py| -> PyResult<_> {
            let py_context = PyPassContext::new_bound(py, context)?;
            let ir_out = self
                .py_obj
                .bind(py)
                .call_method1(intern!(py, "run"), (ir, py_context.clone()))?
                .unbind();
            PyPassContext::drain_into_context(py_context, context);
            Ok(ir_out)
        })?;

        Ok(ir_out)
    }
}

#[pymethods]
impl PassManager {
    #[new]
    fn py_new() -> Self {
        PassManager::default()
    }

    #[pyo3(name = "push")]
    fn py_push_pass(&mut self, py_pass: Bound<'_, PyAny>) -> PyResult<()> {
        // TODO instance-check
        self.try_push_pass(Box::new(PassFromPy::from_bound(py_pass)))
            .map_err(|e| e.into())
    }

    #[pyo3(name = "run")]
    fn py_run(&self, ir: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let ir_in = ir.unbind();
        let (ir_out, _) = self.run::<_, Py<PyAny>>(ir_in, None)?;

        Ok(ir_out)
    }
}

pub fn passmanager(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyPassContext>()?;
    m.add_class::<PassManager>()?;
    Ok(())
}
