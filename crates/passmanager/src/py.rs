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
use pyo3::{intern, prelude::*, types::PyString};

use crate::{Pass, PassContext, PassManagerContext, Value};

// #[cfg(feature = "python")]
#[pyclass]
#[derive(Debug)]
pub struct PyPassContext {
    pub global: Arc<PassManagerContext>,
    pub data: HashMap<String, Py<PyAny>>,
    pub has_changed: bool,
    is_drained: bool, // if `false` we can no longer read from/write to this object
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

    pub fn get(&self, key: Bound<'_, PyString>) -> PyResult<Option<Py<PyAny>>> {
        // add getter from global context, try casting to PyAny
        let key = key.to_str()?;

        if let Some(value) = self.data.get(key) {
            Ok(Some(value.clone())) // return ref?
        } else if let Some(Value::PyCompatible(py_value)) = self.global.data.get(key) {
            Ok(Some(Python::attach(|py| py_value.to_py_any(py))?))
        } else {
            Ok(None)
        }
    }

    fn drain_into_context<'py>(slf: Bound<'py, Self>, pass_context: &mut PassContext) {
        let mut borrowed = slf.borrow_mut();
        for (key, value) in borrowed.data.drain() {
            pass_context.set(key, Value::PyCompatible(Box::new(value)));
        }
        borrowed.is_drained = true;
    }
}

#[pyclass]
pub struct PassFromPy {
    /// A handle to the Python Pass class.
    py_obj: Py<PyAny>,
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
