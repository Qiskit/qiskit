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
use std::sync::Arc;

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyType;

use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::dag_circuit::{DAGCircuit, PyDAGCircuit};
use qiskit_circuit::imports;
use qiskit_circuit::operations::Param;
use qiskit_passmanager::IR;
use qiskit_util::dyn_types::*;
use qiskit_util::py::ImportOnceCell;

// TODO: there'll be an abstraction we can make about `PyIrExposer` and the idea of
// "Python-exposable types", but we're not solving everything in a PR series that's running late.
trait PyIrExposer {
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
        imports::QUANTUM_CIRCUIT.get_bound(py).call_method1(
            intern!(py, "_from_circuit_data"),
            (PyCircuitData::from(*owned),),
        )
    }
    fn steal_from_py<'py>(&self, ob: Bound<'py, PyAny>) -> PyResult<Box<dyn IR>> {
        let ob = ob.getattr(intern!(ob.py(), "_data"))?;
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
        if ty.is_subclass(imports::QUANTUM_CIRCUIT.get_bound(py))? {
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

/// The pass type that we define the `dyn Pass` stuff on.
#[pyclass]
pub struct PyPass;
/// Rust-native wrapper that mediates access to the lifetime-bound `&mut PassContext` (somehow -
/// possibly unsafe).
#[pyclass]
pub struct PassContextHandle;
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
