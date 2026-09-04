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

use std::{any::Any, fmt::Debug, sync::Arc};

use hashbrown::{HashMap, HashSet};
use pyo3::{
    BoundObject,
    exceptions::{PyIndexError, PyRuntimeError, PyTypeError, PyValueError},
    intern,
    prelude::*,
    types::{PyString, PyTuple},
};

use crate::{
    Callback, CallbackError, CallbackType, Pass, PassContext, PassManager, PassManagerContext,
    PassManagerError, Value, imports,
};

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
            PassManagerError::CallbackError(e) => PyRuntimeError::new_err(e.to_string()),
        }
    }
}

pub trait PyConvertible: Any + Send + Sync + Debug {
    fn as_any(&self) -> &(dyn Any + Send + Sync);
    fn to_py_any(&self, py: Python<'_>) -> PyResult<Py<PyAny>>;
}

impl<T> PyConvertible for T
where
    T: Any + Send + Sync + Clone + Debug + for<'py> IntoPyObject<'py>,
{
    fn as_any(&self) -> &(dyn Any + Send + Sync) {
        self
    }

    fn to_py_any(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.clone().into_pyobject(py) {
            Ok(value) => Ok(value.into_any().unbind()),
            Err(e) => Err(e.into()),
        }
    }
}

/// Context information provided to passes during their execution.
///
/// This local context can hold free-form data like a dictionary using :meth:`get` and :meth:`set`.
/// The :meth:`get` method also has access to data from the global execution context of the pass
/// manager, which is queried if the data is not available locally. The global execution
/// context is backed by Rust and might contain data that is not compatible with Python, in
/// which case ``None`` is returned.
///
/// This local context is drained into the global context after the pass is executed. All data
/// in this object will be removed and writing new data into this object will have no effect.
/// Drained contexts are marked with :attr:`is_drained` set to ``True``.
// #[cfg(feature = "python")]
#[pyclass]
#[pyo3(name = "PassContext")]
#[derive(Debug)]
pub struct PyPassContext {
    global: Option<Arc<PassManagerContext>>,
    data: HashMap<String, Py<PyAny>>,
    deletions: HashSet<String>,
    /// If ``True``, the pass has changed the IR.  If ``False``, no updates have been made.
    pub has_changed: bool,
    /// If ``True``, all values have been drained into the global pass manager context and it is
    /// no longer to read from or write to this pass context.
    pub is_drained: bool,
}

impl PyPassContext {
    fn new_bound<'py>(py: Python<'py>, pass_context: &PassContext) -> PyResult<Bound<'py, Self>> {
        // build the Python data dict from the PassContext
        let py_data = pass_context
            .updates
            .insertions
            .iter()
            .filter_map(|(key, value)| {
                if let Value::PyCompatible(py_value) = value {
                    match py_value.to_py_any(py) {
                        Err(e) => Some(Err(e)),
                        Ok(py_any) => Some(Ok((key.clone(), py_any))),
                    }
                } else {
                    None
                }
            })
            .collect::<PyResult<_>>()?;

        Bound::new(
            py,
            PyPassContext {
                global: Some(Arc::clone(&pass_context.global_context)),
                data: py_data,
                deletions: HashSet::new(),
                has_changed: pass_context.has_changed,
                is_drained: false,
            },
        )
    }

    fn drain_into_context<'py>(slf: Bound<'py, Self>, pass_context: &mut PassContext) {
        let mut borrowed = slf.borrow_mut();
        borrowed.global = None;
        for (key, value) in borrowed.data.drain() {
            pass_context.set(key, Value::PyCompatible(Box::new(value)));
        }
        for key in borrowed.deletions.drain() {
            pass_context.delete(key);
        }
        borrowed.is_drained = true;
    }
}

#[pymethods]
impl PyPassContext {
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
    ///     ValueError: If the value exists but it not Python compatible.
    #[pyo3(signature = (key, default=None))]
    pub fn get(
        &self,
        py: Python<'_>,
        key: Bound<'_, PyString>,
        default: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Option<Py<PyAny>>> {
        if self.is_drained {
            return Err(PyRuntimeError::new_err(
                "PassContext is already drained and cannot be read",
            ));
        }

        if let Some(value) = self.data.get(key.to_str()?) {
            Ok(Some(value.clone_ref(py)))
        } else if let Some(global) = self.global.as_ref() {
            global.get(py, key, default)
        } else {
            Ok(None)
        }
    }

    /// Set a value in the pass context.
    pub fn set(&mut self, key: Bound<'_, PyString>, value: Bound<'_, PyAny>) {
        let key = key.to_string();
        self.deletions.remove(&key);
        self.data.insert(key, value.unbind());
    }

    pub fn __getitem__(
        &self,
        py: Python<'_>,
        key: Bound<'_, PyString>,
    ) -> PyResult<Option<Py<PyAny>>> {
        self.get(py, key, None)
    }

    pub fn __setitem__(&mut self, key: Bound<'_, PyString>, value: Bound<'_, PyAny>) {
        self.set(key, value)
    }

    pub fn __delitem__(&mut self, key: Bound<'_, PyString>) {
        let key = key.to_string();
        self.data.remove(&key);
        self.deletions.insert(key);
    }

    pub fn __contains__(&self, key: Bound<'_, PyString>) -> bool {
        let key = key.to_string();
        if self.data.contains_key(&key) {
            return true;
        }
        if !self.deletions.contains(&key)
            && let Some(global) = self.global.as_ref()
        {
            global.data.contains_key(&key)
        } else {
            false
        }
    }
}

#[pyclass]
pub struct PyCallback {
    /// A handle to the Python class implementing the callback interface.
    py_obj: Py<PyAny>,
}

impl PyCallback {
    fn from_bound(py_obj: Bound<'_, PyAny>) -> Self {
        // TODO do instance check
        Self {
            py_obj: py_obj.unbind(),
        }
    }
}

impl Callback for PyCallback {
    fn trigger(&self, hookpoint: &CallbackType) -> Result<bool, CallbackError> {
        Python::attach(|py| -> PyResult<_> {
            self.py_obj
                .bind(py)
                .call_method1(intern!(py, "trigger"), (*hookpoint,))?
                .extract::<bool>()
        })
        .map_err(|e| e.into())
    }

    fn ir_and_context(&self, ir: &dyn Any, context: &PassContext) -> Result<(), CallbackError> {
        let Some(py_ir) = ir.downcast_ref::<Py<PyAny>>() else {
            return Err(CallbackError::IRCastingError);
        };

        Python::attach(|py| -> PyResult<_> {
            let py_context = PyPassContext::new_bound(py, context)?;
            self.py_obj
                .bind(py)
                .call_method1(intern!(py, "ir_and_context"), (py_ir, py_context))?;
            Ok(())
        })
        .map_err(|e| e.into())
    }

    fn with_pass(
        &self,
        pass: &dyn crate::AnyPass,
        ir: &dyn Any,
        context: &PassContext,
    ) -> Result<(), CallbackError> {
        let py_pass = if let Some(from_py) = pass.as_any().downcast_ref::<PassFromPy>() {
            &from_py.py_obj
        } else if let Some(from_task) = pass.as_any().downcast_ref::<PassFromLegacy>() {
            &from_task.py_obj
        } else {
            return Err(CallbackError::PassCastingError);
        };
        let Some(py_ir) = ir.downcast_ref::<Py<PyAny>>() else {
            return Err(CallbackError::IRCastingError);
        };

        Python::attach(|py| -> PyResult<_> {
            let py_context = PyPassContext::new_bound(py, context)?;
            self.py_obj
                .bind(py)
                .call_method1(intern!(py, "with_pass"), (py_pass, py_ir, py_context))?;
            Ok(())
        })
        .map_err(|e| e.into())
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

pub struct PassFromLegacy {
    /// A handle to the Python `GenericPass` instance
    py_obj: Py<PyAny>,
}

impl PassFromLegacy {
    fn from_bound(py_pass: Bound<'_, PyAny>) -> Self {
        Self {
            py_obj: py_pass.unbind(),
        }
    }
}

impl Pass for PassFromLegacy {
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
            let py_state = imports::STATE_FROM_CONTEXT
                .get_bound(py)
                .call1((py_context,))?;

            let result = self
                .py_obj
                .bind(py)
                .call_method1(intern!(py, "execute"), (ir, py_state))?;
            let result_tuple = result.cast::<PyTuple>()?;

            let ir_out = result_tuple.get_item(0)?.unbind();
            // TODO replace this by just getattr("property_set")?
            let py_state_out = result_tuple.get_item(1)?;
            let py_context_out = imports::CONTEXT_FROM_STATE
                .get_bound(py)
                .call1((py_state_out,))?
                .cast_into::<PyPassContext>()?;

            PyPassContext::drain_into_context(py_context_out, context);
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
    fn py_push_pass(&mut self, py: Python<'_>, py_pass: Bound<'_, PyAny>) -> PyResult<()> {
        if py_pass.is_instance(imports::PASS.get_bound(py))? {
            self.try_push_pass(Box::new(PassFromPy::from_bound(py_pass)))
                .map_err(|e| e.into())
        } else if py_pass.is_instance(imports::TASK.get_bound(py))? {
            self.try_push_pass(Box::new(PassFromLegacy::from_bound(py_pass)))
                .map_err(|e| e.into())
        } else {
            Err(PyTypeError::new_err("Unsupported pass type."))
        }
    }

    #[pyo3(name = "run", signature = (ir, callback=None))]
    fn py_run(
        &self,
        ir: Bound<'_, PyAny>,
        callback: Option<Bound<'_, PyAny>>,
    ) -> PyResult<(Py<PyAny>, PassManagerContext)> {
        let ir_in = ir.unbind();
        let py_callback = callback.map(PyCallback::from_bound);
        let (ir_out, context) =
            self.run::<_, Py<PyAny>>(ir_in, py_callback.as_ref().map(|cb| cb as &dyn Callback))?;

        Ok((ir_out, context))
    }
}

#[pymethods]
impl PassManagerContext {
    /// Get a value from the pass manager context.
    ///
    /// Args:
    ///     key: The lookup key.
    ///
    /// Returns:
    ///     The value, if it exists and can be represented in Python. Else ``None`` is returned.
    ///
    /// Raises:
    ///     ValueError: If the value exists but it not Python compatible.
    #[pyo3(signature = (key, default=None))]
    pub fn get(
        &self,
        py: Python<'_>,
        key: Bound<'_, PyString>,
        default: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let key = key.to_str()?;

        match self.data.get(key) {
            Some(Value::PyCompatible(value)) => Ok(Some(value.to_py_any(py)?.clone_ref(py))),
            Some(Value::Any(_)) => Err(PyValueError::new_err(format!(
                "The value of {key} is Python incompatible."
            ))),
            None => {
                if let Some(default) = default {
                    Ok(Some(default.unbind().clone_ref(py)))
                } else {
                    Ok(None)
                }
            }
        }
    }
}

pub fn passmanager(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyPassContext>()?;
    m.add_class::<PassManager>()?;
    m.add_class::<CallbackType>()?;
    Ok(())
}
