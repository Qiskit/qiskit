// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::classical::types::types::Type;
use crate::imports;

use pyo3::types::PyDict;
use pyo3::{intern, prelude::*};
use pyo3::exceptions::PyTypeError;


/// The Rust-side enum indicating a [Ordering] expression's kind.
///
/// // TODO is this 100% true? Copied doc from binary.rs BinaryOp enum and modified slightly.
/// The values are part of the public Qiskit Python interface, since
/// they are public in the sister Python enum `Ordering` in `ordering.py`
/// and used in our QPY serialization format. 
///
/// WARNING: If you add more, **be sure to update ordering.py** as well
/// as the implementation of [::bytemuck::CheckedBitPattern]
/// below.
/// 
/// Enumeration listing the possible relations between two types.  Types only have a partial
/// ordering, so it's possible for two types to have no sub-typing relationship.
///
/// Note that the sub-/supertyping relationship is not the same as whether a type can be explicitly
/// cast from one to another.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ordering {
    /// The left type is a strict subtype of the right type.
    Less = 1,
    /// The two types are equal.
    Equal = 2,
    /// The left type is a strict supertype of the right type.
    Greater = 3,
    /// There is no typing relationship between the two types.
    None = 4,
}

unsafe impl ::bytemuck::CheckedBitPattern for Ordering {
    type Bits = u8;

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits > 0 && *bits < 5
    }
}

impl Ordering {
    /// Get the ordering relationship between the two types as an enumeration value.
    pub fn order(left: Type, right: Type) -> Self {
        match(left, right) {
            // Bool <-> Bool
            (Type::Bool, Type::Bool) |
            // Float <-> Float 
            (Type::Float, Type::Float) |
            // Duration <-> Duration
            (Type::Duration, Type::Duration) => Ordering::Equal,

            // Uint(w1) vs Uint(w2)
            // replaces _order_uint_uint in Python implementation
            (Type::Uint(w1), Type::Uint(w2)) if w1 < w2 => Ordering::Less,
            (Type::Uint(w1), Type::Uint(w2)) if w1 > w2 => Ordering::Greater,
            (Type::Uint(_), Type::Uint(_)) => Ordering::Equal,

            // Remainder unordered
            _ => Ordering::None,
        }
    }

    /// Does the relation :math:`\text{left} \le \text{right}` hold?  If there is no ordering
    /// relation between the two types, then this returns ``False``.  If ``strict``, then the equality
    /// is also forbidden.
    pub fn is_subtype(left: Type, right: Type, strict: bool) -> bool {
        let ord = Self::order(left, right);
        ord == Ordering::Less || (ord == Ordering::Equal && !strict)
    }

    /// Does the relation :math:`\text{left} \ge \text{right}` hold?  If there is no ordering
    /// relation between the two types, then this returns ``False``.  If ``strict``, then the equality
    /// is also forbidden.
    pub fn is_supertype(left: Type, right: Type, strict: bool) -> bool {
        let ord = Self::order(left, right);
        ord == Ordering::Greater || (ord == Ordering::Equal && !strict)
    }

    /// Get the greater of the two types, assuming that there is an ordering relation between them.
    /// Technically, this is a slightly restricted version of the concept of the 'meet' of the two
    /// types in that the return value must be one of the inputs. In practice in the type system there
    /// is no concept of a 'sum' type, so the 'meet' exists if and only if there is an ordering between
    // the two types, and is equal to the greater of the two types.
    pub fn greater(left: Type, right: Type) -> PyResult<Type> {
        match Self::order(left, right) {
            Ordering::Less => Ok(right),
            Ordering::Equal => Ok(right),
            Ordering::Greater => Ok(left),
            Ordering::None    => Err(PyTypeError::new_err(
                format!("no ordering exists between '{:?}' and '{:?}'", left, right)
            )),
        }

    }
}

impl<'py> IntoPyObject<'py> for Ordering {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        imports::ORDERING.get_bound(py).call1((self as usize,))
    }
}

impl<'py> FromPyObject<'py> for Ordering {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let val = ob.getattr(intern!(ob.py(), "int"))?;
        Ok(bytemuck::checked::cast(val.extract::<u8>()?))
    }
}

/// A Python descriptor to prevent PyO3 from attempting to import the Python-side
/// enum before we're initialized.
#[pyclass(
    module = "qiskit._accelerate.circuit.classical.types.ordering",
    name = "Ordering"
)]
struct PyOrdering;

#[pymethods]
impl PyOrdering {
    fn __get__(&self, _obj: &Bound<PyAny>, _obj_type: &Bound<PyAny>) -> Py<PyAny> {
        imports::ORDERING.get_bound(_obj.py()).clone().unbind()
    } 
}

#[pyfunction(name = "order")]
#[pyo3(signature=(left, right))]
fn py_order(left: Type, right: Type) -> Ordering {
    Ordering::order(left, right)
}

#[pyfunction(name = "is_subtype")]
#[pyo3(signature=(left, right, strict=false))]
fn py_is_subtype(left: Type, right: Type, strict: bool) -> bool {
    Ordering::is_subtype(left, right, strict)
}

#[pyfunction(name = "is_supertype")]
#[pyo3(signature=(left, right, strict=false))]
fn py_is_supertype(left: Type, right: Type, strict: bool) -> bool {
    Ordering::is_supertype(left, right, strict)
}

#[pyfunction(name = "greater")]
#[pyo3(signature=(left, right))]
fn py_greater(left: Type, right: Type) -> PyResult<Type> {
    Ordering::greater(left, right)
        .map_err(|msg| PyTypeError::new_err(msg))
}


/// The Rust-side enum indicating a [CastKind] expression's kind.
///
/// // TODO is this 100% true? Copied doc from binary.rs BinaryOp enum and modified slightly.
/// The values are part of the public Qiskit Python interface, since
/// they are public in the sister Python enum `CastKind` in `ordering.py`
/// and used in our QPY serialization format. 
///
/// WARNING: If you add more, **be sure to update ordering.py** as well
/// as the implementation of [::bytemuck::CheckedBitPattern]
/// below.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CastKind {
    Equal = 1,
    Implicit = 2,
    Lossless = 3,
    Dangerous = 4,
    None = 5,
}

unsafe impl bytemuck::CheckedBitPattern for CastKind {
    type Bits = u8;

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits > 0 && *bits < 6
    }
}

impl CastKind {
    /// Determine the sort of cast that is required to move from the left type to the right type.
    pub fn cast_kind(from: Type, to: Type) -> Self {
        match(from, to) {
            // Bool <-> Bool
            (Type::Bool, Type::Bool) => CastKind::Equal,
            // Bool <-> Uint
            (Type::Bool, Type::Uint(_)) => CastKind::Lossless,
            // Bool <-> Float
            (Type::Bool, Type::Float) => CastKind::Lossless,
            // Uint <-> Bool
            (Type::Uint(_), Type::Bool) => CastKind::Implicit,
            // Uint <-> Uint
            (Type::Uint(w1), Type::Uint(w2)) => {
                if w1 == w2 {
                    CastKind::Equal
                } else if w1 < w2 {
                    CastKind::Lossless
                } else {
                    CastKind::Dangerous
                }
            }
            // Uint <-> Float
            (Type::Uint(_), Type::Float) => CastKind::Dangerous,
            // Float <-> Float
            (Type::Float, Type::Float) => CastKind::Equal,
            // Float <-> Uint
            (Type::Float, Type::Uint(_)) => CastKind::Dangerous,
            // Float <-> Bool
            (Type::Float, Type::Bool) => CastKind::Dangerous,
            // Duration <-> Duration
            (Type::Duration, Type::Duration) => CastKind::Equal,
            // Remainder unordered
            _ => CastKind::None,
        }
    }
}

impl<'py> IntoPyObject<'py> for CastKind {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        imports::CAST_KIND.get_bound(py).call1((self as usize,))
    }
}

impl<'py> FromPyObject<'py> for CastKind {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let val = ob.getattr(intern!(ob.py(), "int"))?;
        Ok(bytemuck::checked::cast(val.extract::<u8>()?))
    }
}

/// A Python descriptor to prevent PyO3 from attempting to import the Python-side
/// enum before we're initialized.
#[pyclass(
    module = "qiskit._accelerate.circuit.classical.types.ordering",
    name = "CastKind"
)]
struct PyCastKind;

#[pymethods]
impl PyCastKind {
    fn __get__(&self, _obj: &Bound<PyAny>, _obj_type: &Bound<PyAny>) -> Py<PyAny> {
        imports::CAST_KIND.get_bound(_obj.py()).clone().unbind()
    }
}

#[pyfunction(name = "cast_kind")]
#[pyo3(signature=(from, to))]
fn py_cast_kind(from: Type, to: Type) -> CastKind {
    CastKind::cast_kind(from, to)
}

// #[pymodule]
// fn qiskit_accelerate_circuit_classical_types(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
//     // Build Python’s enum.Enum metaclass
//     let enum_mod = py.import("enum")?;
//     let enum_meta = enum_mod.getattr("Enum")?;

//     // Create the Ordering class
//     let members = PyDict::new(py);
//     members.set_item("LESS",    Ordering::Less    as u8)?;
//     members.set_item("EQUAL",   Ordering::Equal   as u8)?;
//     members.set_item("GREATER", Ordering::Greater as u8)?;
//     members.set_item("NONE",    Ordering::None    as u8)?;
//     let ordering_cls = enum_meta.call1(("Ordering", members))?;
//     // patch repr so repr(x) == "Ordering.LESS"
//     use std::ffi::CString;
//     ordering_cls.setattr(
//         "__repr__",
//         py.eval(
//             CString::new("lambda self: f'Ordering.{self.name}'").unwrap().as_c_str(),
//             None,
//             None
//         )?
//     )?;
//     m.add("Ordering", ordering_cls)?;

//     // 3) Create the CastKind class
//     let members = PyDict::new(py);
//     members.set_item("EQUAL",     CastKind::Equal     as u8)?;
//     members.set_item("IMPLICIT",  CastKind::Implicit  as u8)?;
//     members.set_item("LOSSLESS",  CastKind::Lossless  as u8)?;
//     members.set_item("DANGEROUS", CastKind::Dangerous as u8)?;
//     members.set_item("NONE",      CastKind::None      as u8)?;
//     let castkind_cls = enum_meta.call1(("CastKind", members))?;
//     // default repr "<CastKind.EQUAL: 1>" is fine, or override similarly:
//     m.add("CastKind", castkind_cls)?;

//     // 4) Export your functions
//     m.add_function(wrap_pyfunction!(py_order, m)?)?;
//     m.add_function(wrap_pyfunction!(py_is_subtype, m)?)?;
//     m.add_function(wrap_pyfunction!(py_is_supertype, m)?)?;
//     m.add_function(wrap_pyfunction!(py_greater, m)?)?;
//     m.add_function(wrap_pyfunction!(py_cast_kind, m)?)?;

//     Ok(())
// }

pub(crate) fn register_python(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyOrdering>()?;
    m.add_function(wrap_pyfunction!(py_order, m)?)?;
    m.add_function(wrap_pyfunction!(py_is_subtype, m)?)?;
    m.add_function(wrap_pyfunction!(py_is_supertype, m)?)?;
    m.add_function(wrap_pyfunction!(py_greater, m)?)?;

    m.add_class::<PyCastKind>()?;
    m.add_function(wrap_pyfunction!(py_cast_kind, m)?)?;

    Ok(())
}