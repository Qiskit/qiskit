// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::classical::types::types::Type;

use pyo3::sync::GILOnceCell;
use pyo3::{prelude::*, IntoPyObjectExt};
use pyo3::exceptions::{PyTypeError};
use std::ffi::CString;

static ORD_LESS: GILOnceCell<Py<PyOrdering>> = GILOnceCell::new();
static ORD_EQUAL: GILOnceCell<Py<PyOrdering>> = GILOnceCell::new();
static ORD_GREATER: GILOnceCell<Py<PyOrdering>> = GILOnceCell::new();
static ORD_NONE: GILOnceCell<Py<PyOrdering>> = GILOnceCell::new();

static CK_EQUAL: GILOnceCell<Py<PyCastKind>> = GILOnceCell::new();
static CK_IMPLICIT: GILOnceCell<Py<PyCastKind>> = GILOnceCell::new();
static CK_LOSSLESS: GILOnceCell<Py<PyCastKind>> = GILOnceCell::new();
static CK_DANGEROUS: GILOnceCell<Py<PyCastKind>> = GILOnceCell::new();
static CK_NONE: GILOnceCell<Py<PyCastKind>> = GILOnceCell::new();

/// The Rust-side enum indicating a [Ordering] expression's kind.
///
/// WARNING: If you add more, **be sure to update**
/// the implementations of [ALL_ORDERINGS], [register_python], 
/// and [::bytemuck::CheckedBitPattern] below.
/// 
/// Enumeration listing the possible relations between two types.  Types only have a partial
/// ordering, so it's possible for two types to have no sub-typing relationship.
///
/// Note that the sub-/supertyping relationship is not the same as whether a type can be explicitly
/// cast from one to another.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Hash, PartialEq)]
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

// Used to construct iterators for Python
const ALL_ORDERINGS: &[Ordering] = &[
    Ordering::Less,
    Ordering::Equal,
    Ordering::Greater,
    Ordering::None,
];

unsafe impl ::bytemuck::CheckedBitPattern for Ordering {
    type Bits = u8;

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits > 0 && *bits < 5
    }
}

/// Get the ordering relationship between the two types as an enumeration value.
pub fn order(left: Type, right: Type) -> Ordering {
    match(left, right) {
        // Bool <-> Bool
        (Type::Bool, Type::Bool) |
        // Float <-> Float 
        (Type::Float, Type::Float) |
        // Duration <-> Duration
        (Type::Duration, Type::Duration) => Ordering::Equal,

        // Uint(w1) <-> Uint(w2)
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
    let ord = order(left, right);
    ord == Ordering::Less || (ord == Ordering::Equal && !strict)
}

/// Does the relation :math:`\text{left} \ge \text{right}` hold?  If there is no ordering
/// relation between the two types, then this returns ``False``.  If ``strict``, then the equality
/// is also forbidden.
pub fn is_supertype(left: Type, right: Type, strict: bool) -> bool {
    let ord = order(left, right);
    ord == Ordering::Greater || (ord == Ordering::Equal && !strict)
}

/// Get the greater of the two types, assuming that there is an ordering relation between them.
/// Technically, this is a slightly restricted version of the concept of the 'meet' of the two
/// types in that the return value must be one of the inputs. In practice in the type system there
/// is no concept of a 'sum' type, so the 'meet' exists if and only if there is an ordering between
// the two types, and is equal to the greater of the two types.
pub fn greater(left: Type, right: Type) -> PyResult<Type> {
    match order(left, right) {
        Ordering::Less => Ok(right),
        Ordering::Equal => Ok(right),
        Ordering::Greater => Ok(left),
        Ordering::None    => Err(PyTypeError::new_err(
            format!("no ordering exists between '{:?}' and '{:?}'", left, right)
        )),
    }
}

impl<'py> IntoPyObject<'py> for Ordering {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Ordering::Less => Ordering::Less.into_bound_py_any(py),
            Ordering::Equal => Ordering::Equal.into_bound_py_any(py),
            Ordering::Greater => Ordering::Greater.into_bound_py_any(py),
            Ordering::None => Ordering::None.into_bound_py_any(py),
        }
    }
}

impl<'py> FromPyObject<'py> for Ordering {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let ord: PyRef<'_, PyOrdering> = ob.downcast()?.borrow();
        match ord.inner {
            Ordering::Less => Ok(Ordering::Less),
            Ordering::Equal => Ok(Ordering::Equal),
            Ordering::Greater => Ok(Ordering::Greater),
            Ordering::None => Ok(Ordering::None),
        }
        // is this faster / better?
        // let val = ob.getattr(intern!(ob.py(), "int"))?;
        // Ok(bytemuck::checked::cast(val.extract::<u8>()?))
    }
}

#[pyclass(
    module = "qiskit._accelerate.circuit.classical.types.ordering",
    name = "Ordering",
    eq,
    hash,
    frozen,
    subclass,
)]
#[derive(PartialEq, Debug, Hash)]
struct PyOrdering {
    inner: Ordering,
}

#[pymethods]
impl PyOrdering {
    #[getter]
    fn name(&self) -> &'static str {
        match self.inner {
            Ordering::Less => "LESS",
            Ordering::Equal => "EQUAL",
            Ordering::Greater => "GREATER",
            Ordering::None => "NONE",
        }
    }

    #[getter]
    fn value(&self) -> usize {
        self.inner as usize
    }

    #[getter]
    fn _name_(&self) -> &'static str {
        self.name()
    }

    #[getter]
    fn _value_(&self) -> usize {
        self.value()
    }

    fn __repr__(&self) -> String {
        format!("Ordering.{}", self.name())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __format__(&self, _spec: &str) -> String {
        self.__repr__()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyOrderingIter>> {
        let ords: Vec<Py<PyOrdering>> = ALL_ORDERINGS.iter()
            .map(|&ord| PyOrdering::get_singleton(slf.py(), ord))
            .collect();

        Py::new(slf.py(), PyOrderingIter { items: ords.into_iter() })
    }
}

impl PyOrdering {
    fn get_singleton(py: Python<'_>, ord: Ordering) -> Py<PyOrdering> {
        match ord {
            Ordering::Less => {
                ORD_LESS.get_or_init(py, || {
                    Py::new(py, PyOrdering { inner: Ordering::Less }).unwrap()
                }).clone_ref(py)
            }
            Ordering::Equal => {
                ORD_EQUAL.get_or_init(py, || {
                    Py::new(py, PyOrdering { inner: Ordering::Equal }).unwrap()
                }).clone_ref(py)
            }
            Ordering::Greater => {
                ORD_GREATER.get_or_init(py, || {
                    Py::new(py, PyOrdering { inner: Ordering::Greater }).unwrap()
                }).clone_ref(py)
            }
            Ordering::None => {
                ORD_NONE.get_or_init(py, || {
                    Py::new(py, PyOrdering { inner: Ordering::None }).unwrap()
                }).clone_ref(py)
            }
        }
    }
}

#[pyclass(
    module = "qiskit._accelerate.circuit.classical.types.ordering",
    name = "OrderingIter",
)]
struct PyOrderingIter {
    items: std::vec::IntoIter<Py<PyOrdering>>,
}

#[pymethods]
impl PyOrderingIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Py<PyOrdering>> {
        slf.items.next()
    }
}

#[pyfunction]
fn _ord_iter(py: Python<'_>) -> PyResult<Py<PyOrderingIter>> {
    let ords: Vec<Py<PyOrdering>> = ALL_ORDERINGS.iter()
        .map(|&ord| PyOrdering::get_singleton(py, ord))
        .collect();

    Py::new(py, PyOrderingIter { items: ords.into_iter() })
}

#[pyfunction]
fn _ord_len() -> usize {
    ALL_ORDERINGS.len()
}

#[pyfunction]
fn _ord_contains(py: Python<'_>, obj: Bound<'_, PyAny>) -> PyResult<bool> {
    // Accept PyOrdering member
    if obj.is_instance(&<PyOrdering as pyo3::PyTypeInfo>::type_object(py))? {
        return Ok(true);
    }

    // Accept raw integer
    if let Ok(v) = obj.extract::<u8>() {
        return Ok(bytemuck::checked::try_cast::<u8, Ordering>(v).is_ok());
    }

    // Anything with a `.value` attribute
    if let Ok(val) = obj.getattr("value") {
        if let Ok(v) = val.extract::<u8>() {
            return Ok(bytemuck::checked::try_cast::<u8, Ordering>(v).is_ok())
        }
    }

    Ok(false)
}

#[pyfunction(name = "order")]
#[pyo3(signature=(left, right))]
fn py_order(py: Python<'_>, left: Type, right: Type) -> Py<PyOrdering> {
    let order = order(left, right);
    PyOrdering::get_singleton(py, order)
}

#[pyfunction(name = "is_subtype")]
#[pyo3(signature=(left, right, strict=false))]
fn py_is_subtype(left: Type, right: Type, strict: bool) -> bool {
    is_subtype(left, right, strict)
}

#[pyfunction(name = "is_supertype")]
#[pyo3(signature=(left, right, strict=false))]
fn py_is_supertype(left: Type, right: Type, strict: bool) -> bool {
    is_supertype(left, right, strict)
}

#[pyfunction(name = "greater")]
#[pyo3(signature=(left, right))]
fn py_greater(left: Type, right: Type) -> PyResult<Type> {
    greater(left, right)
        .map_err(|msg| PyTypeError::new_err(msg))
}

/// The Rust-side enum indicating a [CastKind] expression's kind.
///
/// WARNING: If you add more, **be sure to update**
/// the implementations of [ALL_CAST_KINDS], [register_python], 
/// and [::bytemuck::CheckedBitPattern] below.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Hash)]
pub enum CastKind {
    /// The two types are equal; no cast node is required at all.
    Equal = 1,
    /// The 'from' type can be cast to the 'to' type implicitly.  A :class:`~.expr.Cast` node with
    /// ``implicit==True`` is the minimum required to specify this.
    Implicit = 2,
    /// The 'from' type can be cast to the 'to' type explicitly, and the cast will be lossless.  This
    /// requires a :class:`~.expr.Cast`` node with ``implicit=False``, but there's no danger from
    /// inserting one.
    Lossless = 3,
    /// The 'from' type has a defined cast to the 'to' type, but depending on the value, it may lose
    /// data.  A user would need to manually specify casts.
    Dangerous = 4,
    /// There is no casting permitted from the 'from' type to the 'to' type.
    None = 5,
}

// Used to construct iterators for Python
const ALL_CAST_KINDS: &[CastKind] = &[
    CastKind::Equal,
    CastKind::Implicit,
    CastKind::Lossless,
    CastKind::Dangerous,
    CastKind::None,
];

unsafe impl ::bytemuck::CheckedBitPattern for CastKind {
    type Bits = u8;

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits > 0 && *bits < 6
    }
}

impl CastKind {
    /// Determine the sort of cast that is required to move from the left type to the right type.
    pub fn cast_kind(from: Type, to: Type) -> Self {
        match(from, to) {
            // Bool -> Bool
            (Type::Bool, Type::Bool) => CastKind::Equal,
            // Bool -> Uint
            (Type::Bool, Type::Uint(_)) => CastKind::Lossless,
            // Bool -> Float
            (Type::Bool, Type::Float) => CastKind::Lossless,
            // Uint -> Bool
            (Type::Uint(_), Type::Bool) => CastKind::Implicit,
            // Uint -> Uint
            (Type::Uint(w1), Type::Uint(w2)) => {
                if w1 == w2 {
                    CastKind::Equal
                } else if w1 < w2 {
                    CastKind::Lossless
                } else {
                    CastKind::Dangerous
                }
            }
            // Uint -> Float
            (Type::Uint(_), Type::Float) => CastKind::Dangerous,
            // Float -> Float
            (Type::Float, Type::Float) => CastKind::Equal,
            // Float -> Uint
            (Type::Float, Type::Uint(_)) => CastKind::Dangerous,
            // Float -> Bool
            (Type::Float, Type::Bool) => CastKind::Dangerous,
            // Duration -> Duration
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

#[pyclass(
    module = "qiskit._accelerate.circuit.classical.types.ordering",
    name = "CastKind",
    eq,
    hash,
    frozen,
    subclass,
)]
#[derive(Debug, Hash, PartialEq)]
struct PyCastKind{
    inner: CastKind,
}

#[pymethods]
impl PyCastKind {
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

    #[getter]
    fn _name_(&self) -> &'static str {
        self.name()
    }

    #[getter]
    fn _value_(&self) -> usize {
        self.value()
    }

    fn __repr__(&self) -> String {
        format!("CastKind.{}", self.name())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __format__(&self, _spec: &str) -> String {
        self.__repr__()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyCastKindIter>> {
        let cks: Vec<Py<PyCastKind>> = ALL_CAST_KINDS.iter()
            .map(|&kind| PyCastKind::get_singleton(slf.py(), kind))
            .collect();

        Py::new(slf.py(), PyCastKindIter { items: cks.into_iter() })
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

#[pyclass(
    module = "qiskit._accelerate.circuit.classical.types.ordering",
    name = "CastKindIter",
)]
struct PyCastKindIter {
    items: std::vec::IntoIter<Py<PyCastKind>>,
}

#[pymethods]
impl PyCastKindIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Py<PyCastKind>> {
        slf.items.next()
    }
}

#[pyfunction]
fn _ck_iter(py: Python<'_>) -> PyResult<Py<PyCastKindIter>> {
    let cks: Vec<Py<PyCastKind>> = ALL_CAST_KINDS
        .iter()
        .map(|&k| PyCastKind::get_singleton(py, k))
        .collect();
    Py::new(py, PyCastKindIter { items: cks.into_iter() })
}

#[pyfunction]
fn _ck_len() -> usize {
    ALL_CAST_KINDS.len()
}

#[pyfunction]
fn _ck_contains(py: Python<'_>, obj: Bound<'_, PyAny>) -> PyResult<bool> {
    // Accept PyCastKind member
    if obj.is_instance(&<PyCastKind as pyo3::PyTypeInfo>::type_object(py))? {
        return Ok(true)
    }
    
    // Accept raw integer
    if let Ok(v) = obj.extract::<u8>() {
        return Ok(bytemuck::checked::try_cast::<u8, CastKind>(v).is_ok())
    }

    // Anything with a `.value` attribute
    if let Ok(val) = obj.getattr("value") {
        if let Ok(v) = val.extract::<u8>() {
            return Ok(bytemuck::checked::try_cast::<u8, CastKind>(v).is_ok())
        }
    }

    Ok(false)
}

#[pyfunction(name = "cast_kind")]
#[pyo3(signature=(from, to))]
fn py_cast_kind(py: Python<'_>, from: Type, to: Type) -> Py<PyCastKind> {
    let kind = CastKind::cast_kind(from, to);
    PyCastKind::get_singleton(py, kind)
}

pub(crate) fn register_python(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyOrdering>()?;
    m.add_function(wrap_pyfunction!(py_order, m)?)?;
    m.add_function(wrap_pyfunction!(py_is_subtype, m)?)?;
    m.add_function(wrap_pyfunction!(py_is_supertype, m)?)?;
    m.add_function(wrap_pyfunction!(py_greater, m)?)?;

    let ordering_class = m.getattr("Ordering")?;
    // Add singleton instances as class attributes
    ordering_class.setattr("LESS", PyOrdering::get_singleton(m.py(), Ordering::Less))?;
    ordering_class.setattr("EQUAL", PyOrdering::get_singleton(m.py(), Ordering::Equal))?;
    ordering_class.setattr("GREATER", PyOrdering::get_singleton(m.py(), Ordering::Greater))?;
    ordering_class.setattr("NONE", PyOrdering::get_singleton(m.py(), Ordering::None))?;

    m.add_class::<PyOrderingIter>()?;
    m.add_function(wrap_pyfunction!(_ord_iter, m)?)?;
    m.add_function(wrap_pyfunction!(_ord_len, m)?)?;
    m.add_function(wrap_pyfunction!(_ord_contains, m)?)?;

    // create a metaclass: subclass of `type`
    // !! the whitespace in meta_src is critical and cannot be      !!
    // !! modified or Python gets angry about incorrect indentation !!
    let ord_meta_src = r#"
def _ord_make_meta(iter_impl, len_impl, contains_impl):
  class OrderingMeta(type):
    def __iter__(cls):
      return iter_impl()
    def __len__(cls):
      return len_impl()
    def __contains__(cls, x):
      return contains_impl(x)
  return OrderingMeta
"#;
    let ord_filename = CString::new("ordering_meta.py").unwrap();
    let ord_module_name = CString::new("ordering_meta").unwrap();
    let ord_meta_mod = PyModule::from_code(
        m.py(),
        CString::new(ord_meta_src).unwrap().as_c_str(),
        ord_filename.as_c_str(),
        ord_module_name.as_c_str()
    )?;
    let ord_meta = ord_meta_mod
        .getattr("_ord_make_meta")?
        .call1((m.getattr("_ord_iter")?, m.getattr("_ord_len")?, m.getattr("_ord_contains")?))?;

    // Build `Ordering` as a subclass of PyOrdering, with that metaclass
    let types = PyModule::import(m.py(), "types")?;
    let bases = pyo3::types::PyTuple::new(m.py(), [<PyOrdering as pyo3::PyTypeInfo>::type_object(m.py())])?;
    let kwargs = pyo3::types::PyDict::new(m.py());
    kwargs.set_item("metaclass", ord_meta)?;
    let ord_cls = types.getattr("new_class")?.call(("Ordering", bases, kwargs), None)?;

    // nice repr/pickling
    ord_cls.setattr("__module__", m.name()?)?;

    // re-export
    m.add("Ordering", ord_cls)?;

    m.add_class::<PyCastKind>()?;
    m.add_function(wrap_pyfunction!(py_cast_kind, m)?)?;

    let cast_kind_class = m.getattr("CastKind")?;    
    // Add singleton instances as class attributes
    cast_kind_class.setattr("EQUAL", PyCastKind::get_singleton(m.py(), CastKind::Equal))?;
    cast_kind_class.setattr("IMPLICIT", PyCastKind::get_singleton(m.py(), CastKind::Implicit))?;
    cast_kind_class.setattr("LOSSLESS", PyCastKind::get_singleton(m.py(), CastKind::Lossless))?;
    cast_kind_class.setattr("DANGEROUS", PyCastKind::get_singleton(m.py(), CastKind::Dangerous))?;
    cast_kind_class.setattr("NONE", PyCastKind::get_singleton(m.py(), CastKind::None))?;

    m.add_class::<PyCastKindIter>()?;
    m.add_function(pyo3::wrap_pyfunction!(_ck_iter, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(_ck_len, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(_ck_contains, m)?)?;

    // create a metaclass: subclass of `type`
    // !! the whitespace in meta_src is critical and cannot be      !!
    // !! modified or Python gets angry about incorrect indentation !!
    let ck_meta_src = r#"
def _ck_make_meta(iter_impl, len_impl, contains_impl):
  class CastKindMeta(type):
    def __iter__(cls):
      return iter_impl()
    def __len__(cls):
      return len_impl()
    def __contains__(cls, x):
      return contains_impl(x)
  return CastKindMeta
"#;    
    let ck_filename = CString::new("castkind_meta.py").unwrap();
    let ck_module_name = CString::new("castkind_meta").unwrap();
    let ck_meta_mod = PyModule::from_code(
        m.py(),
        CString::new(ck_meta_src).unwrap().as_c_str(),
        ck_filename.as_c_str(),
        ck_module_name.as_c_str()
    )?;
    let ck_meta = ck_meta_mod
        .getattr("_ck_make_meta")?
        .call1((m.getattr("_ck_iter")?, m.getattr("_ck_len")?, m.getattr("_ck_contains")?))?;

    // Build `CastKind` as a subclass of PyCastKind, with that metaclass
    // let types = PyModule::import(m.py(), "types")?;
    let bases = pyo3::types::PyTuple::new(m.py(), [<PyCastKind as pyo3::PyTypeInfo>::type_object(m.py())])?;
    let kwargs = pyo3::types::PyDict::new(m.py());
    kwargs.set_item("metaclass", ck_meta)?;
    let ck_cls = types.getattr("new_class")?.call(("CastKind", bases, kwargs), None)?;

    // nice repr/pickling
    ck_cls.setattr("__module__", m.name()?)?;

    // re-export
    m.add("CastKind", ck_cls)?;

    Ok(())
}