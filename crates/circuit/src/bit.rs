use std::hash::{DefaultHasher, Hash, Hasher};

use pyo3::{exceptions::PyTypeError, prelude::*, types::PyDict};

use crate::{
    circuit_data::CircuitError,
    interner::Interned,
    register::{Register, RegisterAsKey},
};

/// Object representing a Python bit, that allows us to keep backwards compatibility
/// with the previous structure.
#[pyclass(name = "Bit", module = "qiskit._accelerate.bit", subclass)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PyBit {
    register: Option<RegisterAsKey>, // Register identifier
    index: Option<u32>,              // Index within Register
}

#[pymethods]
impl PyBit {
    #[new]
    #[pyo3(signature=(register=None, index=None))]
    pub fn new(register: Option<RegisterAsKey>, index: Option<u32>) -> PyResult<Self> {
        match (&register, index) {
            (None, None) => Ok(Self { register, index }),
            (Some(_), Some(_)) => Ok(Self { register, index }),
            _ => Err(CircuitError::new_err(
                "You should provide both an index and a register, not just one of them.",
            )),
        }
    }

    fn __eq__<'py>(slf: Bound<'py, Self>, other: Bound<'py, Self>) -> bool {
        let borrowed = slf.borrow();
        let other_borrowed = other.borrow();
        if borrowed.register.is_some() && borrowed.index.is_some() {
            return borrowed.register == other_borrowed.register
                && borrowed.index == other_borrowed.index;
        }

        slf.is(&other)
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        if let (Some(reg), Some(idx)) = (self.register.as_ref(), self.index) {
            return (reg.reduce(), idx).to_object(py).bind(py).hash();
        }

        // If registers are unavailable, hash by pointer value.
        let mut hasher = DefaultHasher::new();
        let pointer_val = self as *const Self;
        pointer_val.hash(&mut hasher);
        Ok(hasher.finish() as isize)
    }

    fn __copy__(slf: Bound<Self>) -> Bound<Self> {
        slf
    }

    #[pyo3(signature = (_memo=None))]
    fn __deepcopy__<'py>(
        slf: Bound<'py, Self>,
        _memo: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyBit>> {
        let borrowed: PyRef<Self> = slf.borrow();
        if borrowed.index.is_none() && borrowed.register.is_none() {
            return Ok(slf);
        }
        let copy = slf
            .get_type()
            .call_method1("__new__", (slf.get_type(),))?
            .downcast_into::<PyBit>()?;
        let mut copy_mut = copy.borrow_mut();
        copy_mut.register = borrowed.register.clone();
        copy_mut.index = borrowed.index;
        Ok(copy)
    }

    fn __getstate__(slf: PyRef<'_, Self>) -> (Option<(String, u32)>, Option<u32>) {
        (
            slf.register.as_ref().map(|reg| {
                let (name, num_qubits) = reg.reduce();
                (name.to_string(), num_qubits)
            }),
            slf.index.as_ref().copied(),
        )
    }

    fn __setstate__(mut slf: PyRefMut<'_, Self>, state: (Option<(String, u32)>, Option<u32>)) {
        slf.register = state
            .0
            .map(|(name, num_qubits)| RegisterAsKey::Register((name, num_qubits)));
        slf.index = state.1;
    }

    fn __repr__(slf: Bound<Self>) -> PyResult<String> {
        let borrowed = slf.borrow();
        if borrowed.register.is_none() && borrowed.index.is_none() {
            return Ok(slf.py_super()?.repr()?.to_string());
        }
        let reg = borrowed.register.as_ref().unwrap();
        Ok(format!(
            "{}({}({}, '{}'), {})",
            slf.get_type().name()?,
            reg.type_identifier(),
            reg.index(),
            reg.name(),
            borrowed.index.unwrap()
        ))
    }

    pub fn is_new(&self) -> bool {
        self.index.is_none() && self.register.is_none()
    }
}

macro_rules! create_py_bit {
    ($name:ident, $pyname:literal, $reg_type:pat, $module:literal) => {
        #[pyclass(name=$pyname, extends=PyBit, subclass, module=$module)]
        #[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
        pub struct $name();

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (register=None, index=None))]
            pub fn py_new(
                register: Option<RegisterAsKey>,
                index: Option<u32>,
            ) -> PyResult<(Self, PyBit)> {
                if register.is_none() || matches!(register, Some($reg_type)) {
                    Ok((Self(), PyBit::new(register, index)?))
                } else {
                    Err(PyTypeError::new_err(format!(
                        "The incorrect register was assigned. Bit type {}, Register type {}",
                        $pyname,
                        register.unwrap().type_identifier()
                    )))
                }
            }
        }
    };
}

// Create python instances
create_py_bit! {PyQubit, "Qubit", RegisterAsKey::Quantum(_), "qiskit._accelerate.bit"}
create_py_bit! {PyClbit, "Clbit", RegisterAsKey::Classical(_), "qiskit._accelerate.bit"}

/// Keeps information about where a qubit is located within the circuit.
#[derive(Debug, Clone)]
pub struct BitInfo<T: Register + Hash + Eq> {
    register_idx: Interned<T>,
    index: u32,
}

impl<T: Register + Hash + Eq> BitInfo<T> {
    pub fn new(register_idx: Interned<T>, index: u32) -> Self {
        Self {
            register_idx,
            index,
        }
    }

    pub fn register_index(&self) -> Interned<T> {
        self.register_idx
    }

    pub fn index(&self) -> u32 {
        self.index
    }
}
