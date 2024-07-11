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

use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple, PyType};

use crate::error::QASM3ImporterError;

pub trait PyRegister {
    // This really should be
    //      fn iter<'a>(&'a self, py: Python<'a>) -> impl Iterator<Item = &'a PyAny>;
    // or at a minimum
    //      fn iter<'a>(&'a self, py: Python<'a>) -> ::pyo3::types::iter::PyListIterator<'a>;
    // but we can't use the former before Rust 1.75 and the latter before PyO3 0.21.
    fn bit_list<'a>(&'a self, py: Python<'a>) -> &Bound<'a, PyList>;
}

macro_rules! register_type {
    ($name: ident) => {
        /// Rust-space wrapper around Qiskit `Register` objects.
        pub struct $name {
            /// The actual register instance.
            object: Py<PyAny>,
            /// A pointer to the inner list of bits.  We keep a handle to this for lookup
            /// efficiency; we can use direct list methods to retrieve the bit instances, rather
            /// than needing to indirect through the general `__getitem__` of the register, or
            /// looking up the qubit instances on the circuit.
            items: Py<PyList>,
        }

        impl PyRegister for $name {
            fn bit_list<'a>(&'a self, py: Python<'a>) -> &Bound<'a, PyList> {
                self.items.bind(py)
            }
        }

        impl ::pyo3::IntoPy<Py<PyAny>> for $name {
            fn into_py(self, _py: Python) -> Py<PyAny> {
                self.object
            }
        }

        impl ::pyo3::ToPyObject for $name {
            fn to_object(&self, py: Python) -> Py<PyAny> {
                // _Technically_, allowing access this internal object can let the Rust-space
                // wrapper get out-of-sync since we keep a direct handle to the list, but in
                // practice, the field it's viewing is private and "inaccessible" from Python.
                self.object.clone_ref(py)
            }
        }
    };
}

register_type!(PyQuantumRegister);
register_type!(PyClassicalRegister);

/// Information received from Python space about how to construct a Python-space object to
/// represent a given gate that might be declared.
#[pyclass(module = "qiskit._accelerate.qasm3", frozen, name = "CustomGate")]
#[derive(Clone, Debug)]
pub struct PyGate {
    constructor: Py<PyAny>,
    name: String,
    num_params: usize,
    num_qubits: usize,
}

impl PyGate {
    pub fn new<T: IntoPy<Py<PyAny>>, S: AsRef<str>>(
        py: Python,
        constructor: T,
        name: S,
        num_params: usize,
        num_qubits: usize,
    ) -> Self {
        Self {
            constructor: constructor.into_py(py),
            name: name.as_ref().to_owned(),
            num_params,
            num_qubits,
        }
    }

    /// Construct a Python-space instance of the custom gate.
    pub fn construct<A>(&self, py: Python, args: A) -> PyResult<Py<PyAny>>
    where
        A: IntoPy<Py<PyTuple>>,
    {
        let args = args.into_py(py);
        let received_num_params = args.bind(py).len();
        if received_num_params == self.num_params {
            self.constructor.call1(py, args.bind(py))
        } else {
            Err(QASM3ImporterError::new_err(format!(
                "internal error: wrong number of params for {} (got {}, expected {})",
                &self.name, received_num_params, self.num_params
            )))
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn num_params(&self) -> usize {
        self.num_params
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

#[pymethods]
impl PyGate {
    #[new]
    #[pyo3(signature=(/, constructor, name, num_params, num_qubits))]
    fn __new__(constructor: Py<PyAny>, name: String, num_params: usize, num_qubits: usize) -> Self {
        Self {
            constructor,
            name,
            num_params,
            num_qubits,
        }
    }

    fn __repr__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        PyString::new_bound(py, "CustomGate(name={!r}, num_params={}, num_qubits={})").call_method1(
            "format",
            (
                PyString::new_bound(py, &self.name),
                self.num_params,
                self.num_qubits,
            ),
        )
    }

    fn __reduce__(&self, py: Python) -> Py<PyTuple> {
        (
            PyType::new_bound::<PyGate>(py),
            (
                self.constructor.clone_ref(py),
                &self.name,
                self.num_params,
                self.num_qubits,
            ),
        )
            .into_py(py)
    }
}

/// Wrapper around various Python-space imports. This is just a convenience wrapper to save us
/// needing to `getattr` things off a Python-space module quite so frequently.  This is
/// give-or-take just a manual lookup for a few `import` items at the top of a Python module, and
/// the attached constructor functions produce (when appropriate), Rust-space wrappers around the
/// Python objects.
pub struct PyCircuitModule {
    circuit: Py<PyType>,
    qreg: Py<PyType>,
    qubit: Py<PyType>,
    creg: Py<PyType>,
    clbit: Py<PyType>,
    circuit_instruction: Py<PyType>,
    barrier: Py<PyType>,
    // The singleton object.
    measure: Py<PyAny>,
}

impl PyCircuitModule {
    /// Import the necessary components from `qiskit.circuit`.
    pub fn import(py: Python) -> PyResult<Self> {
        let module = PyModule::import_bound(py, "qiskit.circuit")?;
        Ok(Self {
            circuit: module
                .getattr("QuantumCircuit")?
                .downcast_into::<PyType>()?
                .unbind(),
            qreg: module
                .getattr("QuantumRegister")?
                .downcast_into::<PyType>()?
                .unbind(),
            qubit: module.getattr("Qubit")?.downcast_into::<PyType>()?.unbind(),
            creg: module
                .getattr("ClassicalRegister")?
                .downcast_into::<PyType>()?
                .unbind(),
            clbit: module.getattr("Clbit")?.downcast_into::<PyType>()?.unbind(),
            circuit_instruction: module
                .getattr("CircuitInstruction")?
                .downcast_into::<PyType>()?
                .unbind(),
            barrier: module
                .getattr("Barrier")?
                .downcast_into::<PyType>()?
                .unbind(),
            // Measure is a singleton, so just store the object.
            measure: module.getattr("Measure")?.call0()?.into_py(py),
        })
    }

    pub fn new_circuit(&self, py: Python) -> PyResult<PyCircuit> {
        self.circuit.call0(py).map(PyCircuit)
    }

    pub fn new_qreg<T: IntoPy<Py<PyString>>>(
        &self,
        py: Python,
        name: T,
        size: usize,
    ) -> PyResult<PyQuantumRegister> {
        let qreg = self.qreg.call1(py, (size, name.into_py(py)))?;
        Ok(PyQuantumRegister {
            items: qreg
                .bind(py)
                .getattr("_bits")?
                .downcast_into::<PyList>()?
                .unbind(),
            object: qreg,
        })
    }

    pub fn new_qubit(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.qubit.call0(py)
    }

    pub fn new_creg<T: IntoPy<Py<PyString>>>(
        &self,
        py: Python,
        name: T,
        size: usize,
    ) -> PyResult<PyClassicalRegister> {
        let creg = self.creg.call1(py, (size, name.into_py(py)))?;
        Ok(PyClassicalRegister {
            items: creg
                .bind(py)
                .getattr("_bits")?
                .downcast_into::<PyList>()?
                .unbind(),
            object: creg,
        })
    }

    pub fn new_clbit(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.clbit.call0(py)
    }

    pub fn new_instruction<O, Q, C>(
        &self,
        py: Python,
        operation: O,
        qubits: Q,
        clbits: C,
    ) -> PyResult<Py<PyAny>>
    where
        O: IntoPy<Py<PyAny>>,
        Q: IntoPy<Py<PyTuple>>,
        C: IntoPy<Py<PyTuple>>,
    {
        self.circuit_instruction
            .call1(py, (operation, qubits.into_py(py), clbits.into_py(py)))
    }

    pub fn new_barrier(&self, py: Python, num_qubits: usize) -> PyResult<Py<PyAny>> {
        self.barrier.call1(py, (num_qubits,)).map(|x| x.into_py(py))
    }

    pub fn measure(&self, py: Python) -> Py<PyAny> {
        self.measure.clone_ref(py)
    }
}

/// Circuit construction context object to provide an easier Rust-space interface for us to
/// construct the Python :class:`.QuantumCircuit`.  The idea of doing this from Rust space like
/// this is that we might steadily be able to move more and more of it into being native Rust as
/// the Rust-space APIs around the internal circuit data stabilize.
pub struct PyCircuit(Py<PyAny>);

impl PyCircuit {
    /// Untyped access to the inner Python object.
    pub fn inner<'a>(&'a self, py: Python<'a>) -> &Bound<'a, PyAny> {
        self.0.bind(py)
    }

    pub fn add_qreg(&self, py: Python, qreg: &PyQuantumRegister) -> PyResult<()> {
        self.inner(py)
            .call_method1("add_register", (qreg.to_object(py),))
            .map(|_| ())
    }

    pub fn add_qubit(&self, py: Python, qubit: Py<PyAny>) -> PyResult<()> {
        self.inner(py)
            .call_method1("add_bits", ((qubit,),))
            .map(|_| ())
    }

    pub fn add_creg(&self, py: Python, creg: &PyClassicalRegister) -> PyResult<()> {
        self.inner(py)
            .call_method1("add_register", (creg.to_object(py),))
            .map(|_| ())
    }

    pub fn add_clbit<T: IntoPy<Py<PyAny>>>(&self, py: Python, clbit: T) -> PyResult<()> {
        self.inner(py)
            .call_method1("add_bits", ((clbit,),))
            .map(|_| ())
    }

    pub fn append<T: IntoPy<Py<PyAny>>>(&self, py: Python, instruction: T) -> PyResult<()> {
        self.inner(py)
            .call_method1("_append", (instruction.into_py(py),))
            .map(|_| ())
    }
}

impl ::pyo3::IntoPy<Py<PyAny>> for PyCircuit {
    fn into_py(self, py: Python) -> Py<PyAny> {
        self.0.clone_ref(py)
    }
}
