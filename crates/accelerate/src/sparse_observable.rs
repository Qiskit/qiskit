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

use num_complex::Complex64;
use thiserror::Error;

use numpy::{PyArray1, PyArrayDescr, PyArrayDescrMethods, PyReadonlyArray1, PyUntypedArrayMethods};

use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{IntoPyDict, PyList, PyType};

use qiskit_circuit::imports::NUMPY_COPY_ONLY_IF_REQUIRED;
use qiskit_circuit::slice::{PySequenceIndex, SequenceIndex};

/// Named handle to the alphabet of single-qubit terms.
///
/// This is just the Rust-space representation.  We make a separate Python-space `enum.IntEnum` to
/// represent the same information, since we enforce strongly typed interactions in Rust, including
/// not allowing the stored values to be outside the valid `BitTerm`s, but doing so in Python would
/// make it very difficult to use the class efficiently with Numpy array views.  We attach this
/// sister class of `BitTerm` to `SparseObservable` as a scoped class.
///
/// # Representation
///
/// The `u8` representation and the exact numerical values of these are part of the public API.  The
/// low two bits are the symplectic Pauli representation of the required measurement basis with Z in
/// the Lsb0 and X in the Lsb1 (e.g. X and its eigenstate projectors all have their two low bits as
/// `0b10`).  The high two bits are `00` for the operator, `10` for the projector to the positive
/// eigenstate, and `01` for the projector to the negative eigenstate.
///
/// The `0b00_00` representation thus ends up being the natural representation of the `I` operator,
/// but this is never stored, and is not named in the enumeration.
///
/// # Dev notes
///
/// This type is required to be `u8`, but it's a subtype of `u8` because not all `u8` are valid
/// `BitTerm`s.  For interop with Python space, we accept Numpy arrays of `u8` to represent this,
/// which we transmute into slices of `BitTerm`, after checking that all the values are correct (or
/// skipping the check if Python space promises that it upheld the checks).
///
/// We deliberately _don't_ impl `numpy::Element` for `BitTerm` (which would let us accept and
/// return `PyArray1<BitTerm>` at Python-space boundaries) so that it's clear when we're doing
/// the transmute, and we have to be explicit about the safety of that.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum BitTerm {
    /// Pauli X operator.
    X = 0b00_10,
    /// Projector to the positive eigenstate of Pauli X.
    Plus = 0b10_10,
    /// Projector to the negative eigenstate of Pauli X.
    Minus = 0b01_10,
    /// Pauli Y operator.
    Y = 0b00_11,
    /// Projector to the positive eigenstate of Pauli Y.
    Right = 0b10_11,
    /// Projector to the negative eigenstate of Pauli Y.
    Left = 0b01_11,
    /// Pauli Z operator.
    Z = 0b00_01,
    /// Projector to the positive eigenstate of Pauli Z.
    Zero = 0b10_01,
    /// Projector to the negative eigenstate of Pauli Z.
    One = 0b01_01,
}
impl<'py> FromPyObject<'py> for BitTerm {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        ob.extract::<u8>()?.try_into().map_err(PyErr::from)
    }
}

impl BitTerm {
    /// Get the name of this `BitTerm` used in Python-space applications.  This is a single-letter
    /// string.
    fn py_name(&self) -> &'static str {
        match self {
            Self::X => "X",
            Self::Plus => "+",
            Self::Minus => "-",
            Self::Y => "Y",
            Self::Right => "r",
            Self::Left => "l",
            Self::Z => "Z",
            Self::Zero => "0",
            Self::One => "1",
        }
    }
}

/// Lookup table for the conversion from `u8`.  We only store a small part of the table because
/// there's not really any need to do more than this; lookups can be done by masking first.
static BIT_TERM_FROM_U8: [Option<BitTerm>; 16] = [
    None,                 // 0b00_00
    Some(BitTerm::Z),     // 0b00_01
    Some(BitTerm::X),     // 0b00_10
    Some(BitTerm::Y),     // 0b00_11
    None,                 // 0b01_00
    Some(BitTerm::One),   // 0b01_01
    Some(BitTerm::Minus), // 0b01_10
    Some(BitTerm::Left),  // 0b01_11
    None,                 // 0b10_00
    Some(BitTerm::Zero),  // 0b10_01
    Some(BitTerm::Plus),  // 0b10_10
    Some(BitTerm::Right), // 0b10_11
    None,                 // 0b11_00
    None,                 // 0b11_01
    None,                 // 0b11_10
    None,                 // 0b11_11
];

static BIT_TERM_PY_ENUM: GILOnceCell<Py<PyType>> = GILOnceCell::new();
static BIT_TERM_INTO_PY: GILOnceCell<[Option<Py<PyAny>>; 16]> = GILOnceCell::new();

/// Construct the Python-space `IntEnum` that represents the same values as the Rust-spce `BitTerm`.
/// We don't make `BitTerm` a direct `pyclass` because we want the behaviour of `IntEnum`, which
/// specifically also makes its variants subclasses of the Python `int` type; we use a type-safe
/// enum in Rust, but from Python space we expect people to (carefully) deal with the raw ints in
/// Numpy arrays for efficiency.
///
/// The resulting class is attached to `SparseObservable` as a class attribute, and its
/// `__qualname__` is set to reflect this.
fn make_py_bit_term(py: Python) -> PyResult<Py<PyType>> {
    let terms = [
        BitTerm::X,
        BitTerm::Plus,
        BitTerm::Minus,
        BitTerm::Y,
        BitTerm::Right,
        BitTerm::Left,
        BitTerm::Z,
        BitTerm::Zero,
        BitTerm::One,
    ]
    .map(|term| (term.py_name(), term as u8));

    py.import_bound("enum")?
        .getattr("IntEnum")?
        .call(
            ("BitTerm", terms),
            Some(
                &[
                    ("module", "qiskit._accelerate.sparse_observable"),
                    ("qualname", "SparseObservable.BitTerm"),
                ]
                .into_py_dict_bound(py),
            ),
        )
        .and_then(|obj| {
            obj.downcast_into::<PyType>()
                .map(Bound::unbind)
                .map_err(PyErr::from)
        })
}

// Return the relevant value from the Python-space sister enumeration.  These are Python-space
// singletons and subclasses of Python `int`.  We only use this for interaction with "high level"
// Python space; the efficient Numpy-like array paths use `u8` directly so Numpy can act on it
// efficiently.
impl IntoPy<Py<PyAny>> for BitTerm {
    fn into_py(self, py: Python) -> Py<PyAny> {
        let terms = BIT_TERM_INTO_PY.get_or_init(py, || {
            let py_enum = BIT_TERM_PY_ENUM
                .get_or_try_init(py, || make_py_bit_term(py))
                .unwrap()
                .bind(py);
            BIT_TERM_FROM_U8
                .map(|val| val.map(|term| py_enum.getattr(term.py_name()).unwrap().unbind()))
        });
        terms[self as usize].as_ref().unwrap().clone_ref(py)
    }
}
impl ToPyObject for BitTerm {
    fn to_object(&self, py: Python) -> Py<PyAny> {
        self.into_py(py)
    }
}

#[derive(Error, Debug)]
#[error("{0} is not a valid letter of the single-qubit alphabet")]
pub struct BitTermFromU8Error(u8);
impl From<BitTermFromU8Error> for PyErr {
    fn from(value: BitTermFromU8Error) -> PyErr {
        PyValueError::new_err(value.to_string())
    }
}

// `BitTerm` allows safe `as` casting into `u8`.  This is the reverse, which is fallible, because
// `BitTerm` is a value-wise subtype of `u8`.
impl ::std::convert::TryFrom<u8> for BitTerm {
    type Error = BitTermFromU8Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        BIT_TERM_FROM_U8
            .get(value as usize)
            .copied()
            .flatten()
            .ok_or(BitTermFromU8Error(value))
    }
}

#[derive(Error, Debug)]
pub enum CoherenceError {
    #[error("`boundaries` ({boundaries}) must be one element longer than `coeffs` ({coeffs})")]
    MismatchedTermCount { coeffs: usize, boundaries: usize },
    #[error("`bit_terms` ({bit_terms}) and `indices` ({indices}) must be the same length")]
    MismatchedItemCount { bit_terms: usize, indices: usize },
    #[error("the first item of `boundaries` ({0}) must be 0")]
    BadInitialBoundary(usize),
    #[error("the last item of `boundaries` ({last}) must match the length of `bit_terms` and `indices` ({items})")]
    BadFinalBoundary { last: usize, items: usize },
    #[error("all qubit indices must be less than the number of qubits")]
    BitIndexTooHigh,
    #[error("the values in `boundaries` include backwards slices")]
    DecreasingBoundaries,
    #[error("the values in `indices` are not term-wise increasing")]
    UnsortedIndices,
}
impl From<CoherenceError> for PyErr {
    fn from(value: CoherenceError) -> PyErr {
        PyValueError::new_err(value.to_string())
    }
}

/// An observable over Pauli bases that stores its data in a qubit-sparse format.
///
/// TODO: write complete documentation.
#[pyclass(module = "qiskit.quantum_info")]
#[derive(Clone, Debug, PartialEq)]
pub struct SparseObservable {
    /// The number of qubits the operator acts on.  This is not inferable from any other shape or
    /// values, since identities are not stored explicitly.
    #[pyo3(get)]
    pub num_qubits: u32,
    /// The coefficients of each abstract term in in the sum.  This has as many elements as terms in
    /// the sum.
    pub coeffs: Vec<Complex64>,
    /// A flat list of single-qubit terms.  This is more naturally a list of lists, but is stored flat
    /// for memory usage and locality reasons, with the sublists denoted by `boundaries.`
    pub bit_terms: Vec<BitTerm>,
    /// A flat list of the qubit indices that the corresponding entries in `bit_terms` act on.  This
    /// list must always be term-wise sorted, where a term is a sublist as denoted by `boundaries`.
    pub indices: Vec<u32>,
    /// Indices that partition `bit_terms` and `indices` into sublists for each individual term in
    /// the sum.  `boundaries[0]..boundaries[1]` is the range of indices into `bit_terms` and
    /// `indices` that correspond to the first term of the sum.  All unspecified qubit indices are
    /// implicitly the identity.  This is one item longer than `coeffs`, since `boundaries[0]` is
    /// always an explicit zero (for algorithmic ease).
    pub boundaries: Vec<usize>,
}

impl SparseObservable {
    pub fn new_checked(
        num_qubits: u32,
        coeffs: Vec<Complex64>,
        bit_terms: Vec<BitTerm>,
        indices: Vec<u32>,
        boundaries: Vec<usize>,
    ) -> Result<Self, CoherenceError> {
        if coeffs.len() + 1 != boundaries.len() {
            return Err(CoherenceError::MismatchedTermCount {
                coeffs: coeffs.len(),
                boundaries: boundaries.len(),
            });
        }
        if bit_terms.len() != indices.len() {
            return Err(CoherenceError::MismatchedItemCount {
                bit_terms: bit_terms.len(),
                indices: indices.len(),
            });
        }
        // We already checked that `boundaries` is at least length 1.
        if *boundaries.first().unwrap() != 0 {
            return Err(CoherenceError::BadInitialBoundary(boundaries[0]));
        }
        if *boundaries.last().unwrap() != indices.len() {
            return Err(CoherenceError::BadFinalBoundary {
                last: *boundaries.last().unwrap(),
                items: indices.len(),
            });
        }
        for (&left, &right) in boundaries[..].iter().zip(&boundaries[1..]) {
            if right < left {
                return Err(CoherenceError::DecreasingBoundaries);
            }
            let indices = &indices[left..right];
            for (index_left, index_right) in indices[..].iter().zip(&indices[1..]) {
                if index_left >= index_right {
                    return Err(CoherenceError::UnsortedIndices);
                }
            }
            if indices.last().map(|&ix| ix >= num_qubits).unwrap_or(false) {
                return Err(CoherenceError::BitIndexTooHigh);
            }
        }
        Ok(Self {
            num_qubits,
            coeffs,
            bit_terms,
            indices,
            boundaries,
        })
    }
}

#[pymethods]
impl SparseObservable {
    #[getter]
    fn get_coeffs(slf_: Py<Self>) -> ArrayView {
        ArrayView {
            base: slf_,
            slot: ArraySlot::Coeffs,
        }
    }

    #[getter]
    fn get_bit_terms(slf_: Py<Self>) -> ArrayView {
        ArrayView {
            base: slf_,
            slot: ArraySlot::BitTerms,
        }
    }

    #[getter]
    fn get_indices(slf_: Py<Self>) -> ArrayView {
        ArrayView {
            base: slf_,
            slot: ArraySlot::Indices,
        }
    }

    #[getter]
    fn get_boundaries(slf_: Py<Self>) -> ArrayView {
        ArrayView {
            base: slf_,
            slot: ArraySlot::Boundaries,
        }
    }

    #[allow(non_snake_case)]
    #[classattr]
    fn BitTerm(py: Python) -> PyResult<Py<PyType>> {
        BIT_TERM_PY_ENUM
            .get_or_try_init(py, || make_py_bit_term(py))
            .map(|obj| obj.clone_ref(py))
    }

    fn __reduce__(&self, py: Python) -> PyResult<Py<PyAny>> {
        // SAFETY: `BitTerm` is compatible with `u8`.
        let bit_terms: &[u8] = unsafe { ::std::mem::transmute(self.bit_terms.as_slice()) };
        Ok((
            py.get_type_bound::<Self>().getattr("from_raw_parts")?,
            (
                self.num_qubits,
                PyArray1::from_slice_bound(py, &self.coeffs),
                PyArray1::from_slice_bound(py, bit_terms),
                PyArray1::from_slice_bound(py, &self.indices),
                PyArray1::from_slice_bound(py, &self.boundaries),
            ),
            [("check", false)].into_py_dict_bound(py),
        )
            .into_py(py))
    }

    fn __eq__(slf: Bound<Self>, other: Bound<PyAny>) -> bool {
        if slf.is(&other) {
            return true;
        }
        let Ok(other) = other.downcast_into::<Self>() else { return false };
        slf.borrow().eq(&other.borrow())
    }

    /// Get a copy of this observable.
    fn copy(&self) -> Self {
        self.clone()
    }

    // SAFETY: this cannot invoke undefined behaviour if `check = true`, but if `check = false` then
    // the `bit_terms` must all be valid `BitTerm` representations.
    /// Construct a :class:`.SparseObservable` from raw Numpy arrays that match the required data
    /// representation described in the class-level documentation.
    ///
    /// The data from each array is copied into fresh, growable Rust-space allocations.
    ///
    /// Args:
    ///     num_qubits: number of qubits in the observable.
    ///     coeffs: complex coefficients of each term of the observable.
    ///     bit_terms: flattened list of the single-qubit terms comprising all complete terms.
    ///     indices: flattened term-wise sorted list of the qubits each single-qubit term corresponds
    ///         to.
    ///     boundaries: the indices that partition ``bit_terms`` and ``indices`` into terms.
    ///     check: if ``True`` (the default), validate that the data satisfies all coherence
    ///         guarantees.  If ``False``, no checks are done.
    ///
    ///         .. warning::
    ///
    ///             If ``check=False``, the ``bit_terms`` absolutely *must* be all be valid values
    ///             of :class:`.SparseObservable.BitTerm`.  If they are not, Rust-space undefined
    ///             behavior may occur, entirely invalidating the program execution.
    #[deny(unsafe_op_in_unsafe_fn)]
    #[staticmethod]
    #[pyo3(signature = (/, num_qubits, coeffs, bit_terms, indices, boundaries, *, check=true))]
    pub unsafe fn from_raw_parts(
        num_qubits: u32,
        coeffs: PyReadonlyArray1<Complex64>,
        bit_terms: PyReadonlyArray1<u8>,
        indices: PyReadonlyArray1<u32>,
        boundaries: PyReadonlyArray1<usize>,
        check: bool,
    ) -> PyResult<Self> {
        let coeffs = coeffs.as_array().to_vec();
        let bit_terms = if check {
            bit_terms
                .as_array()
                .into_iter()
                .copied()
                .map(BitTerm::try_from)
                .collect::<Result<_, _>>()?
        } else {
            let bit_terms_as_u8 = bit_terms.as_array().to_vec();
            // SAFETY: the caller enforced that each `u8` is a valid `BitTerm`, and `BitTerm` is be
            // represented by a `u8`.
            unsafe { ::std::mem::transmute(bit_terms_as_u8) }
        };
        let indices = indices.as_array().to_vec();
        let boundaries = boundaries.as_array().to_vec();

        if check {
            Self::new_checked(num_qubits, coeffs, bit_terms, indices, boundaries)
                .map_err(PyErr::from)
        } else {
            Ok(Self {
                num_qubits,
                coeffs,
                bit_terms,
                indices,
                boundaries,
            })
        }
    }
}

/// Helper class of `ArrayView` that denotes the slot of the `SparseObservable` we're looking at.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ArraySlot {
    Coeffs,
    BitTerms,
    Indices,
    Boundaries,
}

/// Custom wrapper sequence class to get safe views onto the Rust-space data.  We can't directly
/// expose Python-managed wrapped pointers without introducing some form of runtime exclusion on the
/// ability of `SparseObservable` to re-allocate in place; we can't leave dangling pointers for
/// Python space.
#[pyclass(frozen, sequence)]
struct ArrayView {
    base: Py<SparseObservable>,
    slot: ArraySlot,
}
#[pymethods]
impl ArrayView {
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let obs = self.base.borrow(py);
        let data = match self.slot {
            // Simple integers look the same in Rust-space debug as Python.
            ArraySlot::Indices => format!("{:?}", obs.indices),
            ArraySlot::Boundaries => format!("{:?}", obs.boundaries),
            // Complexes don't have a nice repr in Rust, so just delegate the whole load to Python
            // and convert back.
            ArraySlot::Coeffs => PyList::new_bound(py, &obs.coeffs).repr()?.to_string(),
            ArraySlot::BitTerms => format!(
                "[{}]",
                obs.bit_terms
                    .iter()
                    .map(BitTerm::py_name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        Ok(format!(
            "<observable {} view: {}>",
            match self.slot {
                ArraySlot::Coeffs => "coeffs",
                ArraySlot::BitTerms => "bit_terms",
                ArraySlot::Indices => "indices",
                ArraySlot::Boundaries => "boundaries",
            },
            data,
        ))
    }

    fn __getitem__(&self, py: Python, index: PySequenceIndex) -> PyResult<Py<PyAny>> {
        fn get_from_slice<T: ToPyObject>(
            py: Python,
            slice: &[T],
            index: PySequenceIndex,
        ) -> PyResult<Py<PyAny>> {
            match index.with_len(slice.len())? {
                SequenceIndex::Int(index) => Ok(slice[index].to_object(py)),
                indices => Ok(PyList::new_bound(
                    py,
                    indices.iter().map(|index| slice[index].to_object(py)),
                )
                .into_py(py)),
            }
        }

        let obs = self.base.borrow(py);
        match self.slot {
            ArraySlot::Coeffs => get_from_slice(py, &obs.coeffs, index),
            ArraySlot::BitTerms => get_from_slice(py, &obs.bit_terms, index),
            ArraySlot::Indices => get_from_slice(py, &obs.indices, index),
            ArraySlot::Boundaries => get_from_slice(py, &obs.boundaries, index),
        }
    }

    fn __setitem__(&self, index: PySequenceIndex, values: &Bound<PyAny>) -> PyResult<()> {
        /// Set values of a slice according to the indexer, using `extract` to retrieve the
        /// Rust-space object from the collection of Python-space values.
        ///
        /// This allows broadcasting a single item into many locations in a slice (like Numpy), but
        /// otherwise requires that the index and values are the same length (unlike Python's
        /// `list`) because that would change the length.
        fn set_in_slice<'py, T: Copy + FromPyObject<'py>>(
            slice: &mut [T],
            index: PySequenceIndex<'py>,
            values: &Bound<'py, PyAny>,
        ) -> PyResult<()> {
            match index.with_len(slice.len())? {
                SequenceIndex::Int(index) => {
                    slice[index] = values.extract()?;
                    Ok(())
                }
                indices => {
                    if let Ok(value) = values.extract() {
                        for index in indices {
                            slice[index] = value;
                        }
                    } else {
                        let values = values
                            .iter()?
                            .map(|value| value?.extract())
                            .collect::<PyResult<Vec<_>>>()?;
                        if indices.len() != values.len() {
                            return Err(PyValueError::new_err(format!(
                                "tried to set a slice of length {} with a sequence of length {}",
                                indices.len(),
                                values.len(),
                            )));
                        }
                        for (index, value) in indices.into_iter().zip(values) {
                            slice[index] = value;
                        }
                    }
                    Ok(())
                }
            }
        }

        let mut obs = self.base.borrow_mut(values.py());
        match self.slot {
            ArraySlot::Coeffs => set_in_slice(&mut obs.coeffs, index, values),
            ArraySlot::BitTerms => set_in_slice(&mut obs.bit_terms, index, values),
            ArraySlot::Indices => set_in_slice(&mut obs.indices, index, values),
            ArraySlot::Boundaries => set_in_slice(&mut obs.boundaries, index, values),
        }
    }

    fn __len__(&self, py: Python) -> usize {
        let obs = self.base.borrow(py);
        match self.slot {
            ArraySlot::Coeffs => obs.coeffs.len(),
            ArraySlot::BitTerms => obs.bit_terms.len(),
            ArraySlot::Indices => obs.indices.len(),
            ArraySlot::Boundaries => obs.boundaries.len(),
        }
    }

    fn __eq__(slf: Bound<Self>, other: Bound<PyAny>) -> bool {
        let py = slf.py();
        if slf.is(&other) {
            return true;
        }
        let Ok(other) = other.downcast_into::<Self>() else { return false };
        let slf = slf.borrow();
        let other = other.borrow();
        if slf.slot != other.slot {
            return false;
        } else if slf.base.is(&other.base) {
            return true;
        }
        let obs_left = slf.base.borrow(py);
        let obs_right = slf.base.borrow(py);
        match slf.slot {
            ArraySlot::Coeffs => obs_left.coeffs == obs_right.coeffs,
            ArraySlot::BitTerms => obs_left.bit_terms == obs_right.bit_terms,
            ArraySlot::Indices => obs_left.indices == obs_right.indices,
            ArraySlot::Boundaries => obs_left.boundaries == obs_right.boundaries,
        }
    }

    #[pyo3(signature = (/, dtype=None, copy=None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // This method always copies, so we don't leave dangling pointers lying around in Numpy
        // arrays; it's not enough just to set the `base` of the Numpy array to the
        // `SparseObservable`, since the `Vec` we're referring to might re-allocate and invalidate
        // the pointer the Numpy array is wrapping.
        if !copy.unwrap_or(true) {
            return Err(PyValueError::new_err(
                "cannot produce a safe view onto movable memory",
            ));
        }
        let obs = self.base.borrow(py);
        match self.slot {
            ArraySlot::Coeffs => {
                cast_array_type(py, PyArray1::from_slice_bound(py, &obs.coeffs), dtype)
            }
            ArraySlot::Indices => {
                cast_array_type(py, PyArray1::from_slice_bound(py, &obs.indices), dtype)
            }
            ArraySlot::Boundaries => {
                cast_array_type(py, PyArray1::from_slice_bound(py, &obs.boundaries), dtype)
            }
            ArraySlot::BitTerms => {
                // SAFETY: `BitTerm` is a subtype of `u8`, and this copy object doesn't allow
                // writeback to our buffers.
                let bit_terms: &[u8] = unsafe { ::std::mem::transmute(obs.bit_terms.as_slice()) };
                cast_array_type(py, PyArray1::from_slice_bound(py, bit_terms), dtype)
            }
        }
    }
}

/// Use the Numpy Python API to convert a `PyArray` into a dynamically chosen `dtype`, copying only
/// if required.
fn cast_array_type<'py, T>(
    py: Python<'py>,
    array: Bound<'py, PyArray1<T>>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let base_dtype = array.dtype();
    let dtype = dtype
        .map(|dtype| PyArrayDescr::new_bound(py, dtype))
        .unwrap_or_else(|| Ok(base_dtype.clone()))?;
    if dtype.is_equiv_to(&base_dtype) {
        return Ok(array.into_any());
    }
    PyModule::import_bound(py, intern!(py, "numpy"))?
        .getattr(intern!(py, "array"))?
        .call(
            (array,),
            Some(
                &[
                    (
                        intern!(py, "copy"),
                        NUMPY_COPY_ONLY_IF_REQUIRED.get_bound(py),
                    ),
                    (intern!(py, "dtype"), dtype.as_any()),
                ]
                .into_py_dict_bound(py),
            ),
        )
        .map(|obj| obj.into_any())
}

#[pymodule]
pub fn sparse_observable(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<SparseObservable>()?;
    Ok(())
}
