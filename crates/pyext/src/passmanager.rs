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

use std::any::Any;
use std::mem;
use std::sync::{Arc, RwLock};

use anyhow::anyhow;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyType;

use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::dag_circuit::{DAGCircuit, PyDAGCircuit};
use qiskit_circuit::operations::Param;
use qiskit_passmanager::{IR, PassContext};
use qiskit_util::dyn_types::*;
use qiskit_util::py::ImportOnceCell;

// TODO: there'll be an abstraction we can make about `PyIrExposer` and the idea of
// "Python-exposable types", but we're not solving everything in a PR series that's running late.
trait PyIrExposer: Send + Sync + 'static {
    fn object_dyn_type_id(&self) -> DynTypeId<'_>;
    /// Move the data in `ob` into a suitable Python object.
    fn to_py<'py>(&self, py: Python<'py>, ob: Box<dyn IR>) -> PyResult<Bound<'py, PyAny>>;
    /// Extract the object from Python, without cloning.
    ///
    /// This is permitted to assume that it is given the only copy of `ob` from which reading is
    /// permitted.  In particular, implementers may replace the data in `ob` with (valid) nonsense,
    /// to leave the object in a valid state while having extracted all its data.
    ///
    /// Implementers can assume that they should be able to take out a mutable reference to any
    /// inner data, and should return a Python exception if they cannot.
    fn steal_from_py<'py>(&self, ob: Bound<'py, PyAny>) -> PyResult<Box<dyn IR>>;
}

struct PyDagExposer;
impl PyIrExposer for PyDagExposer {
    fn object_dyn_type_id(&self) -> DynTypeId<'_> {
        DynTypeId::of::<DAGCircuit>()
    }
    fn to_py<'py>(&self, py: Python<'py>, ob: Box<dyn IR>) -> PyResult<Bound<'py, PyAny>> {
        let owned = (ob as Box<dyn Any>)
            .downcast::<DAGCircuit>()
            .expect("caller should ensure typing");
        PyDAGCircuit::from(*owned)
            .into_pyobject(py)
            .map(|ob| ob.into_any())
    }
    fn steal_from_py<'py>(&self, ob: Bound<'py, PyAny>) -> PyResult<Box<dyn IR>> {
        let py_dag = ob.cast_into::<PyDAGCircuit>()?;
        let mut py_dag = py_dag.try_borrow_mut()?;
        // Per documentation, we're not required to leave `ob`'s data intact.
        let dag =
            mem::replace(&mut *py_dag, PyDAGCircuit::from(DAGCircuit::default())).into_inner();
        Ok(Box::new(dag))
    }
}

struct PyCircuitExposer;
impl PyIrExposer for PyCircuitExposer {
    fn object_dyn_type_id(&self) -> DynTypeId<'_> {
        DynTypeId::of::<CircuitData>()
    }
    fn to_py<'py>(&self, py: Python<'py>, ob: Box<dyn IR>) -> PyResult<Bound<'py, PyAny>> {
        let owned = (ob as Box<dyn Any>)
            .downcast::<CircuitData>()
            .expect("caller should ensure typing");
        PyCircuitData::from(*owned)
            .into_pyobject(py)
            .map(|ob| ob.into_any())
    }
    fn steal_from_py<'py>(&self, ob: Bound<'py, PyAny>) -> PyResult<Box<dyn IR>> {
        let py_circuit = ob.cast_into::<PyCircuitData>()?;
        let mut py_circuit = py_circuit.try_borrow_mut()?;
        // TODO: this should be the infallible implementation of `Default for CircuitData`.
        let empty = CircuitData::with_capacity(0, 0, 0, Param::Float(0.0))?;
        // Per documentation, we're not required to leave `ob`'s data intact.
        let circuit = mem::replace(&mut *py_circuit, PyCircuitData::from(empty)).inner;
        Ok(Box::new(circuit))
    }
}

struct PyObjectExposer(Arc<PyIrBase>);
impl PyIrExposer for PyObjectExposer {
    fn object_dyn_type_id(&self) -> DynTypeId<'_> {
        PyIr::dyn_type_for_base(&self.0)
    }
    fn to_py<'py>(&self, py: Python<'py>, ob: Box<dyn IR>) -> PyResult<Bound<'py, PyAny>> {
        let py_ir = (ob as Box<dyn Any>)
            .downcast::<PyIr>()
            .expect("caller should ensure typing");
        Ok(py_ir.ob.into_bound(py))
    }
    fn steal_from_py<'py>(&self, ob: Bound<'py, PyAny>) -> PyResult<Box<dyn IR>> {
        Ok(Box::new(PyIr {
            ob: ob.unbind(),
            base: Arc::clone(&self.0),
        }))
    }
}

pub struct PyIr {
    ob: Py<PyAny>,
    base: Arc<PyIrBase>,
}
impl PyIr {
    fn dyn_type_for_base(base: &PyIrBase) -> DynTypeId<'_> {
        DynTypeId::of::<Self>().with_dynamic(base.ty.as_ptr().cast(), &base.name)
    }
}
impl DynTyped for PyIr {
    fn dyn_type_id(&self) -> DynTypeId<'_> {
        Self::dyn_type_for_base(&self.base)
    }
}
impl IR for PyIr {}

struct PyIrBase {
    name: String,
    ty: Py<PyType>,
}

// TODO: we might want to cache these, rather than recreating them every time.  If we want to store
// them on the Python `type` object, we a) need to do proper integration with the GC so we don't
// make the type immortal via a reference cycle and b) handle the case of static extension types
// which are already immortal but don't support attaching arbitrary Python attributes.  In the
// static-type case, since they're immortal, though, it doesn't matter if we store things in a
// static; the memory use can't be unbounded because static types (naturally) have a bounded number
// in the process.
fn ir_exposer(mut ty: Bound<PyType>) -> PyResult<Box<dyn PyIrExposer>> {
    static PY_IR: ImportOnceCell = ImportOnceCell::new("qiskit.passmanager", "IR");
    let py = ty.py();
    if ty.hasattr(intern!(py, "_qiskit_ir_builtin_"))? {
        if ty.is_subclass_of::<PyCircuitData>()? {
            return Ok(Box::new(PyCircuitExposer));
        }
        if ty.is_subclass_of::<PyDAGCircuit>()? {
            return Ok(Box::new(PyDagExposer));
        }
        return Err(PyTypeError::new_err(format!(
            "{} declares it is a builtin IR type, but it is not handled",
            ty.repr()?
        )));
    }
    if !ty.is_subclass(PY_IR.get_bound(py))? {
        return Err(PyTypeError::new_err(format!(
            "{} does not implement `IR`",
            ty.repr()?
        )));
    }
    // TODO: if the logic gets more complex, we might want to consider the creation of the
    // `PyIrBase` object into Python via some `pyclass`.
    if let Some(base_ty) = ty
        .getattr(intern!(py, "_qiskit_ir_base_"))?
        .extract::<Option<Bound<PyType>>>()?
    {
        ty = base_ty;
    }
    let base = PyIrBase {
        name: ty.getattr(intern!(py, "_qiskit_ir_name_"))?.extract()?,
        ty: ty.unbind(),
    };
    Ok(Box::new(PyObjectExposer(Arc::new(base))))
}

/// Private Rust-Python boundary object that represents a handle to a pass coming from Python.
#[pyclass]
pub struct PyPass(Option<PyPassInner>);
#[pymethods]
impl PyPass {
    #[new]
    fn py_new(
        ob: Bound<PyAny>,
        name: String,
        ir_in: Bound<PyType>,
        ir_out: Bound<PyType>,
    ) -> PyResult<Self> {
        let inner = PyPassInner {
            ob: ob.unbind(),
            name,
            ir_in: ir_exposer(ir_in)?,
            ir_out: ir_exposer(ir_out)?,
        };
        Ok(Self(Some(inner)))
    }
}

/// The actual logic of Rust wrapping passes that come from Python.  This is what implements `Pass`.
struct PyPassInner {
    name: String,
    ob: Py<PyAny>,
    ir_in: Box<dyn PyIrExposer>,
    ir_out: Box<dyn PyIrExposer>,
}
impl PyPassInner {
    /// The internal logic of the [`Pass::run`](qiskit_passmanager::Pass::run) method, but with an
    /// explicit attachment to an interpreter, and returning `PyErr` so we can centralise the error
    /// handling.
    fn run_py(
        &self,
        py: Python,
        ir: Box<dyn IR>,
        context: PassContextHandle,
    ) -> PyResult<Box<dyn IR>> {
        static CONTEXT_WRAPPER: ImportOnceCell =
            ImportOnceCell::new("qiskit.passmanager", "PassContextHandle");
        let ir = self.ir_in.to_py(py, ir)?;
        let ctx = CONTEXT_WRAPPER.get_bound(py).call1((context,))?;
        let res = self
            .ob
            .bind(py)
            .getattr(intern!(py, "_qiskit_pass_run_"))?
            .call1((ir, ctx))?;
        self.ir_out.steal_from_py(res)
    }
}
impl qiskit_passmanager::Pass for PyPassInner {
    #[inline]
    fn ir_id_in(&self) -> DynTypeId<'_> {
        self.ir_in.object_dyn_type_id()
    }

    #[inline]
    fn ir_id_out(&self) -> DynTypeId<'_> {
        self.ir_out.object_dyn_type_id()
    }

    #[inline]
    fn name(&self) -> &str {
        &self.name
    }

    #[inline]
    fn run(
        &self,
        ir: Box<dyn IR>,
        context: &mut PassContext,
    ) -> Result<Box<dyn IR>, qiskit_passmanager::PassError> {
        // SAFETY: we guarantee that `stolen` will drop (and therefore replace `context`) before
        // control flow leaves this function by leaving ownership of it here.
        let stolen = unsafe { StolenContext::new(context) };

        Python::attach(|py| {
            // SAFETY: per documentation of `PassContextHandle`, no methods allow moving the
            // internal `PassContext` from it, and we ensured that `stolen` drops as this function
            // ends, so the lifetime of the internal object actually ends with `'a`.
            let handle = unsafe { stolen.handle() };
            self.run_py(py, ir, handle).map_err(|e| {
                // `<PyErr as Display>` internally requires the interpreter, so we'll pull out the
                // message while we know we're attached.
                let native = e.to_string();
                // Stash the exception back into the global state; when we next return to the Python
                // interpreter we should be able to retrieve it and attach it as the cause.
                e.restore(py);
                anyhow!("Python execution raised an exception.")
                    .context(native)
                    .into()
            })
        })
    }
}

struct StolenContext<'a, 'b> {
    whence: &'b mut PassContext<'a>,
    shared: Arc<RwLock<Option<PassContext<'static>>>>,
}
impl<'a, 'b> StolenContext<'a, 'b> {
    unsafe fn new(whence: &'b mut PassContext<'a>) -> Self {
        let stolen = mem::replace(whence, PassContext::dummy());
        // SAFETY: per lifetime rules, 'a is longer lived than us.  If nothing else
        // moves the `PassContext` out from us, we can put in back in `whence` in our
        // `Drop`, guaranteeing the lifetime.
        let stolen = unsafe { mem::transmute::<PassContext<'a>, PassContext<'static>>(stolen) };
        Self {
            whence,
            shared: Arc::new(RwLock::new(Some(stolen))),
        }
    }

    unsafe fn handle(&self) -> PassContextHandle {
        PassContextHandle(Arc::clone(&self.shared))
    }
}
impl<'a, 'b> Drop for StolenContext<'a, 'b> {
    fn drop(&mut self) {
        let mut lock = self.shared.write().expect("lock should not poison");
        _ = mem::replace(
            self.whence,
            lock.take()
                .expect("no other references should allow taking the inner object"),
        );
    }
}

/// Rust-native wrapper that mediates access to a lifetime-bound `&mut PassContext`.
///
/// # Safety
///
/// This is in general *unsafe*: the [`PassContext`] object is actually backed by a lifetime-bound
/// object which is type-erased to `'static`.  The creator of this object must ensure that the inner
/// `PassContext` is removed before its lifetime actually expires.
///
/// No method on this class may ever move a `PassContext` out of the `Option` (or delete it, etc).
///
/// Methods on this handle object should attempt to hold locks from the [`RwLock`] for as short
/// periods as possible.
#[pyclass]
pub struct PassContextHandle(Arc<RwLock<Option<PassContext<'static>>>>);
impl PassContextHandle {
    #[inline]
    fn with_borrow<F, T>(&self, use_fn: F) -> PyResult<T>
    where
        F: FnOnce(&PassContext<'static>) -> T,
    {
        let Ok(inner) = self.0.read() else {
            return Err(PyRuntimeError::new_err("internal lock was poisoned"));
        };
        let ctx = inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("attempted to use handle after pass returned")
        })?;
        Ok(use_fn(ctx))
    }

    #[inline]
    fn with_borrow_mut<F, T>(&self, use_fn: F) -> PyResult<T>
    where
        F: FnOnce(&mut PassContext<'static>) -> T,
    {
        let Ok(mut inner) = self.0.write() else {
            return Err(PyRuntimeError::new_err("internal lock was poisoned"));
        };
        let ctx = inner.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("attempted to use handle after pass returned")
        })?;
        Ok(use_fn(ctx))
    }
}
#[pymethods]
impl PassContextHandle {
    pub fn get_context(&self, py: Python, key: String, default: Py<PyAny>) -> PyResult<Py<PyAny>> {
        self.with_borrow(|ctx| {
            let Some(val) = ctx.get(&key) else {
                return Ok(default);
            };
            let Some(ob) = val.downcast_ref::<Py<PyAny>>() else {
                return Err(PyTypeError::new_err(format!(
                    "'{}' did not correspond to a Python object",
                    &key
                )));
            };
            Ok(ob.clone_ref(py))
        })?
    }
    pub fn set_context(&self, key: String, val: Py<PyAny>) -> PyResult<()> {
        let val = Box::new(val);
        self.with_borrow_mut(|ctx| ctx.set(key, val))
    }
    pub fn del_context(&self, key: String) -> PyResult<()> {
        self.with_borrow_mut(|ctx| ctx.delete(key))
    }
    pub fn get_ir_modified(&self) -> PyResult<bool> {
        self.with_borrow(|ctx| ctx.ir_modified)
    }
    pub fn set_ir_modified(&self, val: bool) -> PyResult<()> {
        self.with_borrow_mut(|ctx| ctx.ir_modified = val)
    }
}

/// Wrapper around the Rust-native pass manager; we can take this as owned when we create it from
/// Python or give a Rust-created one to Python.
#[pyclass(name = "PassManager")]
pub struct PyPassManager;

#[pymodule(name = "passmanager")]
pub fn passmanager_mod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPass>()?;
    m.add_class::<PyPassManager>()?;
    m.add_class::<PassContextHandle>()?;
    Ok(())
}
