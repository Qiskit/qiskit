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

use pyo3::sync::GILOnceCell;
use pyo3::{intern, prelude::*, IntoPyObjectExt};
use pyo3::exceptions::{PyTypeError, PyValueError};

static CK_EQUAL: GILOnceCell<Py<PyCastKind>> = GILOnceCell::new();
static CK_IMPLICIT: GILOnceCell<Py<PyCastKind>> = GILOnceCell::new();
static CK_LOSSLESS: GILOnceCell<Py<PyCastKind>> = GILOnceCell::new();
static CK_DANGEROUS: GILOnceCell<Py<PyCastKind>> = GILOnceCell::new();
static CK_NONE: GILOnceCell<Py<PyCastKind>> = GILOnceCell::new();

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
/// WARNING: If you add more, **be sure to update**
/// the implementation of [::bytemuck::CheckedBitPattern]
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
        match self {
            CastKind::Equal => CastKind::Equal.into_bound_py_any(py),
            CastKind::Implicit => CastKind::Implicit.into_bound_py_any(py),
            CastKind::Lossless => CastKind::Lossless.into_bound_py_any(py),
            CastKind::Dangerous => CastKind::Dangerous.into_bound_py_any(py),
            CastKind::None => CastKind::None.into_bound_py_any(py),
        }
    }
}

impl<'py> FromPyObject<'py> for CastKind {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let cc: PyRef<'_, PyCastKind> = ob.downcast()?.borrow();
        match cc.inner {
            CastKind::Equal => Ok(CastKind::Equal),
            CastKind::Implicit => Ok(CastKind::Implicit),
            CastKind::Lossless => Ok(CastKind::Lossless),
            CastKind::Dangerous => Ok(CastKind::Dangerous),
            CastKind::None => Ok(CastKind::None),
        }
        // is this faster / better?
        // let val = ob.getattr(intern!(ob.py(), "int"))?;
        // Ok(bytemuck::checked::cast(val.extract::<u8>()?))
    }
}

/// A Python descriptor to prevent PyO3 from attempting to import the Python-side
/// enum before we're initialized.
#[pyclass(
    module = "qiskit._accelerate.circuit.classical.types.ordering",
    name = "CastKind",
    eq,
    hash,
    frozen,
    // subclass,
)]
#[derive(PartialEq, Debug, Hash)]
// struct PyCastKind(pub CastKind);
struct PyCastKind{
    inner: CastKind,
}

#[pymethods]
impl PyCastKind {
    #[new]
    fn new(py: Python) -> PyResult<Py<Self>> {
        // Problem: Which variant should this return?
        // You can't determine the variant from the constructor arguments alone
        Err(pyo3::exceptions::PyTypeError::new_err(
            "CastKind cannot be instantiated directly. Use CastKind.EQUAL, etc."
        ))
    }

    #[getter]
    fn name(&self) -> &'static str {
        match self.inner {
            CastKind::Equal => "EQUAL",
            CastKind::Implicit => "IMPLICIT",
            CastKind::Lossless => "LOSSLESS",
            CastKind::Dangerous => "DANGEROUS",
            CastKind::None => "NONE",
        }
    }

    #[getter]
    fn value(&self) -> usize {
        self.inner as usize
    }    

    #[staticmethod]
    fn from_value(py: Python<'_>, val: u8) -> PyResult<Py<PyCastKind>> {
        if val == 0 || val > 5 {
            return Err(PyValueError::new_err(format!(
                "Invalid CastKind value: {} (expected 1-5)", 
                val
            )));
        }

        let kind = bytemuck::checked::try_cast::<u8, CastKind>(val)
            .map_err(|_| PyValueError::new_err(format!("Invalid CastKind value: {}", val)))?;

        Ok(PyCastKind::get_singleton(py, kind))
    }

    fn __repr__(&self) -> String {
        format!("CastKind.{}", self.name())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl PyCastKind {
    fn get_singleton(py: Python<'_>, kind: CastKind) -> Py<PyCastKind> {
        match kind {
            CastKind::Equal => {
                CK_EQUAL.get_or_init(py, || {
                    Py::new(py, PyCastKind { inner: CastKind::Equal }).unwrap()
                }).clone_ref(py)
            }
            CastKind::Implicit => {
                CK_IMPLICIT.get_or_init(py, || {
                    Py::new(py, PyCastKind { inner: CastKind::Implicit }).unwrap()
                }).clone_ref(py)
            }
            CastKind::Lossless => {
                CK_LOSSLESS.get_or_init(py, || {
                    Py::new(py, PyCastKind { inner: CastKind::Lossless }).unwrap()
                }).clone_ref(py)
            }
            CastKind::Dangerous => {
                CK_DANGEROUS.get_or_init(py, || {
                    Py::new(py, PyCastKind { inner: CastKind::Dangerous }).unwrap()
                }).clone_ref(py)
            }
            CastKind::None => {
                CK_NONE.get_or_init(py, || {
                    Py::new(py, PyCastKind { inner: CastKind::None }).unwrap()
                }).clone_ref(py)
            }
        }
    }
}

#[pyfunction(name = "cast_kind")]
#[pyo3(signature=(from, to))]
fn py_cast_kind(py: Python<'_>, from: Type, to: Type) -> Py<PyCastKind> {
    // PyCastKind{ inner: CastKind::cast_kind(from, to) }
    // PyCastKind(CastKind::cast_kind(from, to))
    let kind = CastKind::cast_kind(from, to);
    PyCastKind::get_singleton(py, kind)
}

pub(crate) fn register_python(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyOrdering>()?;
    m.add_function(wrap_pyfunction!(py_order, m)?)?;
    m.add_function(wrap_pyfunction!(py_is_subtype, m)?)?;
    m.add_function(wrap_pyfunction!(py_is_supertype, m)?)?;
    m.add_function(wrap_pyfunction!(py_greater, m)?)?;

    m.add_class::<PyCastKind>()?;
    m.add_function(wrap_pyfunction!(py_cast_kind, m)?)?;

    let cast_kind_class = m.getattr("CastKind")?;
    
    // Add singleton instances as class attributes
    cast_kind_class.setattr("EQUAL", PyCastKind::get_singleton(m.py(), CastKind::Equal))?;
    cast_kind_class.setattr("IMPLICIT", PyCastKind::get_singleton(m.py(), CastKind::Implicit))?;
    cast_kind_class.setattr("LOSSLESS", PyCastKind::get_singleton(m.py(), CastKind::Lossless))?;
    cast_kind_class.setattr("DANGEROUS", PyCastKind::get_singleton(m.py(), CastKind::Dangerous))?;
    cast_kind_class.setattr("NONE", PyCastKind::get_singleton(m.py(), CastKind::None))?;

    Ok(())
}