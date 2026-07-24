// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::{
    IntoPyObjectExt, PyErr,
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    intern,
    prelude::*,
    sync::PyOnceLock,
    types::{IntoPyDict, PyList, PyString, PyTuple, PyType},
};
use std::collections::btree_map;

use std::{
    iter::zip,
    sync::{Arc, RwLock},
};

use qiskit_util::py::{PySequenceIndex, SequenceIndex};

use crate::pauli_lindblad_map::qubit_sparse_pauli::{
    ArithmeticError, CoherenceError, InnerReadError, InnerWriteError, LabelError, Pauli,
    PauliFromU8Error, QubitSparsePauli, QubitSparsePauliList, QubitSparsePauliView,
    raw_parts_from_sparse_list,
};
use crate::python::imports;

pub(super) static PAULI_PY_ENUM: PyOnceLock<Py<PyType>> = PyOnceLock::new();
pub(super) static PAULI_INTO_PY: PyOnceLock<[Option<Py<PyAny>>; 16]> = PyOnceLock::new();

impl From<InnerReadError> for PyErr {
    fn from(value: InnerReadError) -> PyErr {
        PyRuntimeError::new_err(value.to_string())
    }
}

impl From<InnerWriteError> for PyErr {
    fn from(value: InnerWriteError) -> PyErr {
        PyRuntimeError::new_err(value.to_string())
    }
}

impl From<PauliFromU8Error> for PyErr {
    fn from(value: PauliFromU8Error) -> PyErr {
        PyValueError::new_err(value.to_string())
    }
}

impl From<CoherenceError> for PyErr {
    fn from(value: CoherenceError) -> PyErr {
        PyValueError::new_err(value.to_string())
    }
}

impl From<LabelError> for PyErr {
    fn from(value: LabelError) -> PyErr {
        PyValueError::new_err(value.to_string())
    }
}

impl From<ArithmeticError> for PyErr {
    fn from(value: ArithmeticError) -> PyErr {
        PyValueError::new_err(value.to_string())
    }
}

/// The single-character string label used to represent this term in the
/// :class:`QubitSparsePauliList` alphabet.
#[pyfunction]
#[pyo3(name = "label")]
fn pauli_label(py: Python<'_>, slf: Pauli) -> &Bound<'_, PyString> {
    // This doesn't use `py_label` so we can use `intern!`.
    match slf {
        Pauli::X => intern!(py, "X"),
        Pauli::Y => intern!(py, "Y"),
        Pauli::Z => intern!(py, "Z"),
    }
}
/// Construct the Python-space `IntEnum` that represents the same values as the Rust-spce `Pauli`.
///
/// We don't make `Pauli` a direct `pyclass` because we want the behaviour of `IntEnum`, which
/// specifically also makes its variants subclasses of the Python `int` type; we use a type-safe
/// enum in Rust, but from Python space we expect people to (carefully) deal with the raw ints in
/// Numpy arrays for efficiency.
///
/// The resulting class is attached to `QubitSparsePauliList` as a class attribute, and its
/// `__qualname__` is set to reflect this.
fn make_py_pauli(py: Python) -> PyResult<Py<PyType>> {
    let terms = [Pauli::X, Pauli::Y, Pauli::Z]
        .into_iter()
        .flat_map(|term| {
            let mut out = vec![(term.py_label(), term as u8)];
            if term.py_label() != term.py_label() {
                // Also ensure that the labels are created as aliases.  These can't be (easily) accessed
                // by attribute-getter (dot) syntax, but will work with the item-getter (square-bracket)
                // syntax, or programmatically with `getattr`.
                out.push((term.py_label(), term as u8));
            }
            out
        })
        .collect::<Vec<_>>();
    let obj = py.import("enum")?.getattr("IntEnum")?.call(
        ("Pauli", terms),
        Some(
            &[
                ("module", "qiskit.quantum_info"),
                ("qualname", "QubitSparsePauliList.Pauli"),
            ]
            .into_py_dict(py)?,
        ),
    )?;
    let label_property = py
        .import("builtins")?
        .getattr("property")?
        .call1((wrap_pyfunction!(pauli_label, py)?,))?;
    obj.setattr("label", label_property)?;
    Ok(obj.cast_into::<PyType>()?.unbind())
}

// Return the relevant value from the Python-space sister enumeration.  These are Python-space
// singletons and subclasses of Python `int`.  We only use this for interaction with "high level"
// Python space; the efficient Numpy-like array paths use `u8` directly so Numpy can act on it
// efficiently.
impl<'py> IntoPyObject<'py> for Pauli {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let terms = PAULI_INTO_PY.get_or_init(py, || {
            let py_enum = PAULI_PY_ENUM
                .get_or_try_init(py, || make_py_pauli(py))
                .expect("creating a simple Python enum class should be infallible")
                .bind(py);
            ::std::array::from_fn(|val| {
                ::bytemuck::checked::try_cast(val as u8)
                    .ok()
                    .map(|term: Pauli| {
                        py_enum
                            .getattr(term.py_label())
                            .expect("the created `Pauli` enum should have matching attribute names to the terms")
                            .unbind()
                    })
            })
        });
        Ok(terms[self as usize]
            .as_ref()
            .expect("the lookup table initializer populated a 'Some' in all valid locations")
            .bind(py)
            .clone())
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Pauli {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let value = ob
            .extract::<isize>()
            .map_err(|_| match ob.get_type().repr() {
                Ok(repr) => PyTypeError::new_err(format!("bad type for 'Pauli': {repr}")),
                Err(err) => err,
            })?;
        let value_error = || {
            PyValueError::new_err(format!(
                "value {value} is not a valid letter of the single-qubit alphabet for 'Pauli'"
            ))
        };
        let value: u8 = value.try_into().map_err(|_| value_error())?;
        value.try_into().map_err(|_| value_error())
    }
}

/// A phase-less Pauli operator stored in a qubit-sparse format.
///
/// Representation
/// ==============
///
/// A Pauli operator is a tensor product of single-qubit Pauli operators of the form :math:`P =
/// \bigotimes_n A^{(n)}_i`, for :math:`A^{(n)}_i \in \{I, X, Y, Z\}`. The internal representation
/// of a :class:`QubitSparsePauli` stores only the non-identity single-qubit Pauli operators.
///
/// Internally, each single-qubit Pauli operator is stored with a numeric value, explicitly:
///
/// .. _qubit-sparse-pauli-alphabet:
/// .. table:: Alphabet of single-qubit Pauli operators used in :class:`QubitSparsePauliList`
///
///   =======  =======================================  ===============  ===========================
///   Label    Operator                                 Numeric value    :class:`.Pauli` attribute
///   =======  =======================================  ===============  ===========================
///   ``"I"``  :math:`I` (identity)                     Not stored.      Not stored.
///
///   ``"X"``  :math:`X` (Pauli X)                      ``0b10`` (2)     :attr:`~.Pauli.X`
///
///   ``"Y"``  :math:`Y` (Pauli Y)                      ``0b11`` (3)     :attr:`~.Pauli.Y`
///
///   ``"Z"``  :math:`Z` (Pauli Z)                      ``0b01`` (1)     :attr:`~.Pauli.Z`
///
///   =======  =======================================  ===============  ===========================
///
/// .. _qubit-sparse-pauli-arrays:
/// .. table:: Data arrays used to represent :class:`.QubitSparsePauli`
///
///   ==================  ===========  =============================================================
///   Attribute           Length       Description
///   ==================  ===========  =============================================================
///   :attr:`paulis`      :math:`s`    Each of the non-identity single-qubit Pauli operators.  These
///                                    correspond to the non-identity :math:`A^{(n)}_i` in the list,
///                                    where the entries are stored in order of increasing :math:`i`
///                                    first, and in order of increasing :math:`n` within each term.
///
///   :attr:`indices`     :math:`s`    The corresponding qubit (:math:`n`) for each of the operators
///                                    in :attr:`paulis`.  :class:`QubitSparsePauli` requires
///                                    that this list is term-wise sorted, and algorithms can rely
///                                    on this invariant being upheld.
///   ==================  ===========  =============================================================
///
/// The parameter :math:`s` is the total number of non-identity single-qubit terms.
///
/// The scalar item of the :attr:`paulis` array is stored as a numeric byte.  The numeric values
/// are related to the symplectic Pauli representation that :class:`.SparsePauliOp` uses, and are
/// accessible with named access by an enumeration:
///
/// ..
///     This is documented manually here because the Python-space `Enum` is generated
///     programmatically from Rust - it'd be _more_ confusing to try and write a docstring somewhere
///     else in this source file. The use of `autoattribute` is because it pulls in the numeric
///     value.
///
/// .. py:class:: QubitSparsePauli.Pauli
///
///
///     An :class:`~enum.IntEnum` that provides named access to the numerical values used to
///     represent each of the single-qubit alphabet terms enumerated in
///     :ref:`qubit-sparse-pauli-alphabet`.
///
///     This class is attached to :class:`.QubitSparsePauli`.  Access it as
///     :class:`.QubitSparsePauli.Pauli`.  If this is too much typing, and you are solely
///     dealing with :class:`QubitSparsePauliList` objects and the :class:`Pauli` name is not
///     ambiguous, you might want to shorten it as::
///
///         >>> ops = QubitSparsePauli.Pauli
///         >>> assert ops.X is QubitSparsePauli.Pauli.X
///
///     You can access all the values of the enumeration either with attribute access, or with
///     dictionary-like indexing by string::
///
///         >>> assert QubitSparsePauli.Pauli.X is QubitSparsePauli.Pauli["X"]
///
///     The bits representing each single-qubit Pauli are the (phase-less) symplectic representation
///     of the Pauli operator.
///
///     Values
///     ------
///
///     .. autoattribute:: qiskit.quantum_info::QubitSparsePauli.Pauli.X
///
///         The Pauli :math:`X` operator.  Uses the single-letter label ``"X"``.
///
///     .. autoattribute:: qiskit.quantum_info::QubitSparsePauli.Pauli.Y
///
///         The Pauli :math:`Y` operator.  Uses the single-letter label ``"Y"``.
///
///     .. autoattribute:: qiskit.quantum_info::QubitSparsePauli.Pauli.Z
///
///         The Pauli :math:`Z` operator.  Uses the single-letter label ``"Z"``.
///
///
/// Each of the array-like attributes behaves like a Python sequence.  You can index and slice these
/// with standard :class:`list`-like semantics.  Slicing an attribute returns a Numpy
/// :class:`~numpy.ndarray` containing a copy of the relevant data with the natural ``dtype`` of the
/// field; this lets you easily do mathematics on the results, like bitwise operations on
/// :attr:`paulis`.
///
/// Construction
/// ============
///
/// :class:`QubitSparsePauli` defines several constructors.  The default constructor will
/// attempt to delegate to one of the more specific constructors, based on the type of the input.
/// You can always use the specific constructors to have more control over the construction.
///
/// .. _qubit-sparse-pauli-convert-constructors:
/// .. table:: Construction from other objects
///
///   ============================  ================================================================
///   Method                        Summary
///   ============================  ================================================================
///   :meth:`from_label`            Convert a dense string label into a :class:`~.QubitSparsePauli`.
///
///   :meth:`from_sparse_label`     Build a :class:`.QubitSparsePauli` from a tuple of a sparse
///                                 string label and the qubits they apply to.
///
///   :meth:`from_pauli`            Raise a single :class:`~.quantum_info.Pauli` into a
///                                 :class:`.QubitSparsePauli`.
///
///   :meth:`from_raw_parts`        Build the operator from :ref:`the raw data arrays
///                                 <qubit-sparse-pauli-arrays>`.
///   ============================  ================================================================
///
/// .. py:function:: QubitSparsePauli.__new__(data, /, num_qubits=None)
///
///     The default constructor of :class:`QubitSparsePauli`.
///
///     This delegates to one of :ref:`the explicit conversion-constructor methods
///     <qubit-sparse-pauli-convert-constructors>`, based on the type of the ``data`` argument.
///     If ``num_qubits`` is supplied and constructor implied by the type of ``data`` does not
///     accept a number, the given integer must match the input.
///
///     :param data: The data type of the input.  This can be another :class:`QubitSparsePauli`,
///         in which case the input is copied, or it can be a valid format for either
///         :meth:`from_label` or :meth:`from_sparse_label`.
///     :param int|None num_qubits: Optional number of qubits for the operator.  For most data
///         inputs, this can be inferred and need not be passed.  It is only necessary for the
///         sparse-label format.  If given unnecessarily, it must match the data input.
#[pyclass(
    name = "QubitSparsePauli",
    frozen,
    module = "qiskit.quantum_info",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyQubitSparsePauli {
    inner: QubitSparsePauli,
}

impl PyQubitSparsePauli {
    pub fn inner(&self) -> &QubitSparsePauli {
        &self.inner
    }
}

#[pymethods]
impl PyQubitSparsePauli {
    #[new]
    #[pyo3(signature = (data, /, num_qubits=None))]
    fn py_new(data: &Bound<PyAny>, num_qubits: Option<u32>) -> PyResult<Self> {
        let py = data.py();
        let check_num_qubits = |data: &Bound<PyAny>| -> PyResult<()> {
            let Some(num_qubits) = num_qubits else {
                return Ok(());
            };
            let other_qubits = data.getattr(intern!(py, "num_qubits"))?.extract::<u32>()?;
            if num_qubits == other_qubits {
                return Ok(());
            }
            Err(PyValueError::new_err(format!(
                "explicitly given 'num_qubits' ({num_qubits}) does not match operator ({other_qubits})"
            )))
        };
        if data.is_instance(imports::PAULI_TYPE.get_bound(py))? {
            check_num_qubits(data)?;
            return Self::from_pauli(data);
        }
        if let Ok(label) = data.extract::<String>() {
            let num_qubits = num_qubits.unwrap_or(label.len() as u32);
            if num_qubits as usize != label.len() {
                return Err(PyValueError::new_err(format!(
                    "explicitly given 'num_qubits' ({}) does not match label ({})",
                    num_qubits,
                    label.len(),
                )));
            }
            return Self::from_label(&label);
        }
        if let Ok(sparse_label) = data.extract() {
            let Some(num_qubits) = num_qubits else {
                return Err(PyValueError::new_err(
                    "if using the sparse-label form, 'num_qubits' must be provided",
                ));
            };
            return Self::from_sparse_label(sparse_label, num_qubits);
        }
        Err(PyTypeError::new_err(format!(
            "unknown input format for 'QubitSparsePauli': {}",
            data.get_type().repr()?,
        )))
    }

    /// Construct a :class:`.QubitSparsePauli` from raw Numpy arrays that match :ref:`the required
    /// data representation described in the class-level documentation
    /// <qubit-sparse-pauli-arrays>`.
    ///
    /// The data from each array is copied into fresh, growable Rust-space allocations.
    ///
    /// Args:
    ///     num_qubits: number of qubits the operator acts on.
    ///     paulis: list of the single-qubit terms.  This should be a Numpy array with dtype
    ///         :attr:`~numpy.uint8` (which is compatible with :class:`.Pauli`).
    ///     indices: sorted list of the qubits each single-qubit term corresponds to.  This should
    ///         be a Numpy array with dtype :attr:`~numpy.uint32`.
    ///
    /// Examples:
    ///
    ///     Construct a :math:`Z` operator acting on qubit 50 of 100 qubits.
    ///
    ///         >>> num_qubits = 100
    ///         >>> terms = np.array([QubitSparsePauli.Pauli.Z], dtype=np.uint8)
    ///         >>> indices = np.array([50], dtype=np.uint32)
    ///         >>> QubitSparsePauli.from_raw_parts(num_qubits, terms, indices)
    ///         <QubitSparsePauli on 100 qubits: Z_50>
    #[staticmethod]
    #[pyo3(signature = (/, num_qubits, paulis, indices))]
    fn from_raw_parts(num_qubits: u32, paulis: Vec<Pauli>, indices: Vec<u32>) -> PyResult<Self> {
        if paulis.len() != indices.len() {
            return Err(CoherenceError::MismatchedItemCount {
                paulis: paulis.len(),
                indices: indices.len(),
            }
            .into());
        }
        let mut order = (0..paulis.len()).collect::<Vec<_>>();
        order.sort_unstable_by_key(|a| indices[*a]);
        let paulis = order.iter().map(|i| paulis[*i]).collect();
        let mut sorted_indices = Vec::<u32>::with_capacity(order.len());
        for i in order {
            let index = indices[i];
            if sorted_indices
                .last()
                .map(|prev| *prev >= index)
                .unwrap_or(false)
            {
                return Err(CoherenceError::UnsortedIndices.into());
            }
            sorted_indices.push(index)
        }
        let inner = QubitSparsePauli::new(num_qubits, paulis, sorted_indices.into_boxed_slice())?;
        Ok(PyQubitSparsePauli { inner })
    }

    /// Construct from a dense string label.
    ///
    /// The label must be a sequence of the alphabet ``'IXYZ'``.  The label is interpreted
    /// analogously to a bitstring.  In other words, the right-most letter is associated with qubit
    /// 0, and so on.  This is the same as the labels for :class:`~.quantum_info.Pauli` and
    /// :class:`.SparsePauliOp`.
    ///
    /// Args:
    ///     label (str): the dense label.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> QubitSparsePauli.from_label("IIIIXZI")
    ///         <QubitSparsePauli on 7 qubits: X_2 Z_1>
    ///         >>> label = "IYXZI"
    ///         >>> pauli = Pauli(label)
    ///         >>> assert QubitSparsePauli.from_label(label) == QubitSparsePauli.from_pauli(pauli)
    #[staticmethod]
    #[pyo3(signature = (label, /))]
    fn from_label(label: &str) -> PyResult<Self> {
        let inner = QubitSparsePauli::from_dense_label(label)?;
        Ok(inner.into())
    }

    /// Construct a :class:`.QubitSparsePauli` from a single :class:`~.quantum_info.Pauli` instance.
    ///
    /// Note that the phase of the Pauli is dropped.
    ///
    /// Args:
    ///     pauli (:class:`~.quantum_info.Pauli`): the single Pauli to convert.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> label = "IYXZI"
    ///         >>> pauli = Pauli(label)
    ///         >>> QubitSparsePauli.from_pauli(pauli)
    ///         <QubitSparsePauli on 5 qubits: Y_3 X_2 Z_1>
    ///         >>> assert QubitSparsePauli.from_label(label) == QubitSparsePauli.from_pauli(pauli)
    #[staticmethod]
    #[pyo3(signature = (pauli, /))]
    pub fn from_pauli(pauli: &Bound<PyAny>) -> PyResult<Self> {
        let py = pauli.py();
        let num_qubits = pauli.getattr(intern!(py, "num_qubits"))?.extract::<u32>()?;
        let z = pauli
            .getattr(intern!(py, "z"))?
            .extract::<PyReadonlyArray1<bool>>()?;
        let x = pauli
            .getattr(intern!(py, "x"))?
            .extract::<PyReadonlyArray1<bool>>()?;
        let mut paulis = Vec::new();
        let mut indices = Vec::new();
        for (i, (x, z)) in x.as_array().iter().zip(z.as_array().iter()).enumerate() {
            // The only failure case possible here is the identity, because of how we're
            // constructing the value to convert.
            let Ok(term) = ::bytemuck::checked::try_cast(((*x as u8) << 1) | (*z as u8)) else {
                continue;
            };
            indices.push(i as u32);
            paulis.push(term);
        }
        let inner = QubitSparsePauli::new(
            num_qubits,
            paulis.into_boxed_slice(),
            indices.into_boxed_slice(),
        )?;
        Ok(inner.into())
    }

    /// Construct a qubit sparse Pauli from a sparse label, given as a tuple of a string of Paulis,
    /// and the indices of the corresponding qubits.
    ///
    /// This is analogous to :meth:`.SparsePauliOp.from_sparse_list`.
    ///
    /// Args:
    ///     sparse_label (tuple[str, Sequence[int]]): labels and the qubits each single-qubit term
    ///         applies to.
    ///
    ///     num_qubits (int): the number of qubits the operator acts on.
    ///
    /// Examples:
    ///
    ///     Construct a simple Pauli::
    ///
    ///         >>> QubitSparsePauli.from_sparse_label(
    ///         ...     ("ZX", (1, 4)),
    ///         ...     num_qubits=5,
    ///         ... )
    ///         <QubitSparsePauli on 5 qubits: X_4 Z_1>
    ///
    ///     This method can replicate the behavior of :meth:`from_label`, if the qubit-arguments
    ///     field of the tuple is set to decreasing integers::
    ///
    ///         >>> label = "XYXZ"
    ///         >>> from_label = QubitSparsePauli.from_label(label)
    ///         >>> from_sparse_label = QubitSparsePauli.from_sparse_label(
    ///         ...     (label, (3, 2, 1, 0)),
    ///         ...     num_qubits=4
    ///         ... )
    ///         >>> assert from_label == from_sparse_label
    #[staticmethod]
    #[pyo3(signature = (/, sparse_label, num_qubits))]
    fn from_sparse_label(sparse_label: (String, Vec<u32>), num_qubits: u32) -> PyResult<Self> {
        let label = sparse_label.0;
        let indices = sparse_label.1;
        let mut paulis = Vec::new();
        let mut sorted_indices = Vec::new();

        let label: &[u8] = label.as_ref();
        let mut sorted = btree_map::BTreeMap::new();
        if label.len() != indices.len() {
            return Err(LabelError::WrongLengthIndices {
                label: label.len(),
                indices: indices.len(),
            }
            .into());
        }
        for (letter, index) in label.iter().zip(indices) {
            if index >= num_qubits {
                return Err(LabelError::BadIndex { index, num_qubits }.into());
            }
            let btree_map::Entry::Vacant(entry) = sorted.entry(index) else {
                return Err(LabelError::DuplicateIndex { index }.into());
            };
            entry.insert(Pauli::try_from_u8(*letter).map_err(|_| LabelError::OutsideAlphabet)?);
        }
        for (index, term) in sorted.iter() {
            let Some(term) = term else {
                continue;
            };
            sorted_indices.push(*index);
            paulis.push(*term);
        }

        let inner = QubitSparsePauli::new(
            num_qubits,
            paulis.into_boxed_slice(),
            sorted_indices.into_boxed_slice(),
        )?;
        Ok(inner.into())
    }

    /// Get the identity operator for a given number of qubits.
    ///
    /// Examples:
    ///
    ///     Get the identity on 100 qubits::
    ///
    ///         >>> QubitSparsePauli.identity(100)
    ///         <QubitSparsePauli on 100 qubits: >
    #[pyo3(signature = (/, num_qubits))]
    #[staticmethod]
    pub fn identity(num_qubits: u32) -> Self {
        QubitSparsePauli::identity(num_qubits).into()
    }

    /// Convert this Pauli into a single element :class:`QubitSparsePauliList`.
    fn to_qubit_sparse_pauli_list(&self) -> PyResult<PyQubitSparsePauliList> {
        let qubit_sparse_pauli_list = QubitSparsePauliList::new(
            self.inner.num_qubits(),
            self.inner.paulis().to_vec(),
            self.inner.indices().to_vec(),
            vec![0, self.inner.paulis().len()],
        )?;
        Ok(qubit_sparse_pauli_list.into())
    }

    /// Phaseless composition with another :class:`QubitSparsePauli`.
    ///
    /// Args:
    ///     other (QubitSparsePauli): the qubit sparse Pauli to compose with.
    fn compose(&self, other: &PyQubitSparsePauli) -> PyResult<Self> {
        Ok(PyQubitSparsePauli {
            inner: self.inner.compose(&other.inner)?,
        })
    }

    fn __matmul__(&self, other: &PyQubitSparsePauli) -> PyResult<Self> {
        self.compose(other)
    }

    /// Check if `self`` commutes with another qubit sparse Pauli.
    ///
    /// Args:
    ///     other (QubitSparsePauli): the qubit sparse Pauli to check for commutation with.
    fn commutes(&self, other: &PyQubitSparsePauli) -> PyResult<bool> {
        Ok(self.inner.commutes(&other.inner)?)
    }

    fn __eq__(slf: Bound<Self>, other: Bound<PyAny>) -> PyResult<bool> {
        if slf.is(&other) {
            return Ok(true);
        }
        let Ok(other) = other.cast_into::<Self>() else {
            return Ok(false);
        };
        let slf = slf.borrow();
        let other = other.borrow();
        Ok(slf.inner.eq(&other.inner))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "<{} on {} qubit{}: {}>",
            "QubitSparsePauli",
            self.inner.num_qubits(),
            if self.inner.num_qubits() == 1 {
                ""
            } else {
                "s"
            },
            self.inner.view().to_sparse_str(),
        ))
    }

    fn __getnewargs__(slf_: Bound<Self>) -> PyResult<Bound<PyTuple>> {
        let py = slf_.py();
        let borrowed = slf_.borrow();
        (
            borrowed.inner.num_qubits(),
            Self::get_paulis(slf_.clone()),
            Self::get_indices(slf_),
        )
            .into_pyobject(py)
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let paulis: &[u8] = ::bytemuck::cast_slice(self.inner.paulis());
        (
            py.get_type::<Self>().getattr("from_raw_parts")?,
            (
                self.inner.num_qubits(),
                PyArray1::from_slice(py, paulis),
                PyArray1::from_slice(py, self.inner.indices()),
            ),
        )
            .into_pyobject(py)
    }

    /// Return a :class:`~.quantum_info.Pauli` representing the same phaseless Pauli.
    fn to_pauli<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        imports::PAULI_TYPE
            .get_bound(py)
            .call1((self.inner.to_dense_label(),))
    }

    /// Get a copy of this term.
    fn copy(&self) -> Self {
        self.clone()
    }

    /// Read-only view onto the individual single-qubit terms.
    ///
    /// The only valid values in the array are those with a corresponding
    /// :class:`~QubitSparsePauli.Pauli`.
    #[getter]
    fn get_paulis(slf_: Bound<Self>) -> Bound<PyArray1<u8>> {
        let borrowed = slf_.borrow();
        let paulis = borrowed.inner.paulis();
        let arr = ::ndarray::aview1(::bytemuck::cast_slice::<_, u8>(paulis));
        // SAFETY: in order to call this function, the lifetime of `self` must be managed by Python.
        // We tie the lifetime of the array to `slf_`, and there are no public ways to modify the
        // `Box<[Pauli]>` allocation (including dropping or reallocating it) other than the entire
        // object getting dropped, which Python will keep safe.
        let out = unsafe { PyArray1::borrow_from_array(&arr, slf_.into_any()) };
        out.readwrite().make_nonwriteable();
        out
    }

    /// The number of qubits the term is defined on.
    #[getter]
    fn get_num_qubits(&self) -> u32 {
        self.inner.num_qubits()
    }

    /// Read-only view onto the indices of each non-identity single-qubit term.
    ///
    /// The indices will always be in sorted order.
    #[getter]
    fn get_indices(slf_: Bound<Self>) -> Bound<PyArray1<u32>> {
        let borrowed = slf_.borrow();
        let indices = borrowed.inner.indices();
        let arr = ::ndarray::aview1(indices);
        // SAFETY: in order to call this function, the lifetime of `self` must be managed by Python.
        // We tie the lifetime of the array to `slf_`, and there are no public ways to modify the
        // `Box<[u32]>` allocation (including dropping or reallocating it) other than the entire
        // object getting dropped, which Python will keep safe.
        let out = unsafe { PyArray1::borrow_from_array(&arr, slf_.into_any()) };
        out.readwrite().make_nonwriteable();
        out
    }

    // The documentation for this is inlined into the class-level documentation of
    // :class:`QubitSparsePauliList`.
    #[allow(non_snake_case)]
    #[classattr]
    pub fn Pauli(py: Python) -> PyResult<Py<PyType>> {
        PAULI_PY_ENUM
            .get_or_try_init(py, || make_py_pauli(py))
            .map(|obj| obj.clone_ref(py))
    }
}

/// A list of phase-less Pauli operators stored in a qubit-sparse format.
///
/// Representation
/// ==============
///
/// Each individual Pauli operator in the list is a tensor product of single-qubit Pauli operators
/// of the form :math:`P = \bigotimes_n A^{(n)}_i`, for :math:`A^{(n)}_i \in \{I, X, Y, Z\}`. The
/// internal representation of a :class:`QubitSparsePauliList` stores only the non-identity
/// single-qubit Pauli operators.  This makes it significantly more efficient to represent lists of
/// Pauli operators with low weights on a large number of qubits. For example, the list of
/// :math`n`-qubit operators :math:`[Z^{(0)}, \dots Z^{(n-1)}]`, where :math:`Z^{(j)}` represents
/// The :math:`Z` operator on qubit :math:`j` and identity on all others, can be stored in
/// :class:`QubitSparsePauliList` with a linear amount of memory in the number of qubits.
///
/// Indexing
/// --------
///
/// :class:`QubitSparsePauliList` behaves as `a Python sequence
/// <https://docs.python.org/3/glossary.html#term-sequence>`__ (the standard form, not the expanded
/// :class:`collections.abc.Sequence`).  The elements of the list can be indexed by integers, as
/// well as iterated through. Whether through indexing or iterating, elements of the list are
/// returned as :class:`QubitSparsePauli` instances.
///
/// Construction
/// ============
///
/// :class:`QubitSparsePauliList` defines several constructors.  The default constructor will
/// attempt to delegate to one of the more specific constructors, based on the type of the input.
/// You can always use the specific constructors to have more control over the construction.
///
/// .. _qubit-sparse-pauli-list-convert-constructors:
/// .. table:: Construction from other objects
///
///   ================================  ============================================================
///   Method                            Summary
///   ================================  ============================================================
///   :meth:`from_label`                Convert a dense string label into a single-element
///                                     :class:`.QubitSparsePauliList`.
///
///   :meth:`from_list`                 Construct from a list of dense string labels.
///
///   :meth:`from_sparse_list`          Elements given as a list of tuples of sparse string labels
///                                     and the qubits they apply to.
///
///   :meth:`from_pauli`                Raise a single :class:`~.quantum_info.Pauli` into a
///                                     single-element :class:`.QubitSparsePauliList`.
///
///   :meth:`from_qubit_sparse_paulis`  Construct from a list of :class:`.QubitSparsePauli`\s.
///   ================================  ============================================================
///
/// .. py:function:: QubitSparsePauliList.__new__(data, /, num_qubits=None)
///
///     The default constructor of :class:`QubitSparsePauliList`.
///
///     This delegates to one of :ref:`the explicit conversion-constructor methods
///     <qubit-sparse-pauli-list-convert-constructors>`, based on the type of the ``data`` argument.
///     If ``num_qubits`` is supplied and constructor implied by the type of ``data`` does not
///     accept a number, the given integer must match the input.
///
///     :param data: The data type of the input.  This can be another :class:`QubitSparsePauliList`,
///         in which case the input is copied, or it can be a list in a valid format for either
///         :meth:`from_list` or :meth:`from_sparse_list`.
///     :param int|None num_qubits: Optional number of qubits for the list.  For most data
///         inputs, this can be inferred and need not be passed.  It is only necessary for empty
///         lists or the sparse-list format.  If given unnecessarily, it must match the data input.
///
/// In addition to the conversion-based constructors, the method :meth:`empty` can be used to
/// construct an empty list of qubit-sparse Paulis acting on a given number of qubits.
///
/// Conversions
/// ===========
///
/// An existing :class:`QubitSparsePauliList` can be converted into other formats.
///
/// .. table:: Conversion methods to other observable forms.
///
///   ===========================  =================================================================
///   Method                       Summary
///   ===========================  =================================================================
///   :meth:`to_sparse_list`       Express the observable in a sparse list format with elements
///                                ``(paulis, indices)``.
///   ===========================  =================================================================
#[pyclass(
    name = "QubitSparsePauliList",
    module = "qiskit.quantum_info",
    sequence
)]
#[derive(Debug)]
pub struct PyQubitSparsePauliList {
    // This class keeps a pointer to a pure Rust-SparseTerm and serves as interface from Python.
    pub inner: Arc<RwLock<QubitSparsePauliList>>,
}

#[pymethods]
impl PyQubitSparsePauliList {
    #[pyo3(signature = (data, /, num_qubits=None))]
    #[new]
    fn py_new(data: &Bound<PyAny>, num_qubits: Option<u32>) -> PyResult<Self> {
        let py = data.py();
        let check_num_qubits = |data: &Bound<PyAny>| -> PyResult<()> {
            let Some(num_qubits) = num_qubits else {
                return Ok(());
            };
            let other_qubits = data.getattr(intern!(py, "num_qubits"))?.extract::<u32>()?;
            if num_qubits == other_qubits {
                return Ok(());
            }
            Err(PyValueError::new_err(format!(
                "explicitly given 'num_qubits' ({num_qubits}) does not match operator ({other_qubits})"
            )))
        };
        if data.is_instance(imports::PAULI_TYPE.get_bound(py))? {
            check_num_qubits(data)?;
            return Self::from_pauli(data);
        }
        if let Ok(label) = data.extract::<String>() {
            let num_qubits = num_qubits.unwrap_or(label.len() as u32);
            if num_qubits as usize != label.len() {
                return Err(PyValueError::new_err(format!(
                    "explicitly given 'num_qubits' ({}) does not match label ({})",
                    num_qubits,
                    label.len(),
                )));
            }
            return Self::from_label(&label).map_err(PyErr::from);
        }
        if let Ok(pauli_list) = data.cast_exact::<Self>() {
            check_num_qubits(data)?;
            let borrowed = pauli_list.borrow();
            let inner = borrowed.inner.read().map_err(|_| InnerReadError)?;
            return Ok(inner.clone().into());
        }
        // The type of `vec` is inferred from the subsequent calls to `Self::from_list` or
        // `Self::from_sparse_list` to be either the two-tuple or the three-tuple form during the
        // `extract`.  The empty list will pass either, but it means the same to both functions.
        if let Ok(vec) = data.extract() {
            return Self::from_list(vec, num_qubits);
        }
        if let Ok(vec) = data.extract() {
            let Some(num_qubits) = num_qubits else {
                return Err(PyValueError::new_err(
                    "if using the sparse-list form, 'num_qubits' must be provided",
                ));
            };
            return Self::from_sparse_list(vec, num_qubits);
        }
        if let Ok(term) = data.cast_exact::<PyQubitSparsePauli>() {
            return term.borrow().to_qubit_sparse_pauli_list();
        };
        if let Ok(pauli_list) = Self::from_qubit_sparse_paulis(data, num_qubits) {
            return Ok(pauli_list);
        }
        Err(PyTypeError::new_err(format!(
            "unknown input format for 'QubitSparsePauliList': {}",
            data.get_type().repr()?,
        )))
    }

    /// Get a copy of this qubit sparse Pauli list.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> qubit_sparse_pauli_list = QubitSparsePauliList.from_list(["IXZXYYZZ", "ZXIXYYZZ"])
    ///         >>> assert qubit_sparse_pauli_list == qubit_sparse_pauli_list.copy()
    ///         >>> assert qubit_sparse_pauli_list is not qubit_sparse_pauli_list.copy()
    fn copy(&self) -> PyResult<Self> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.clone().into())
    }

    /// The number of qubits the operators in the list act on.
    ///
    /// This is not inferable from any other shape or values, since identities are not stored
    /// explicitly.
    #[getter]
    #[inline]
    pub fn num_qubits(&self) -> PyResult<u32> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.num_qubits())
    }

    /// The number of elements in the list.
    #[getter]
    #[inline]
    pub fn num_terms(&self) -> PyResult<usize> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        Ok(inner.num_terms())
    }

    /// Get the empty list for a given number of qubits.
    ///
    /// The empty list contains no elements, and is the identity element for joining two
    /// :class:`QubitSparsePauliList` instances.
    ///
    /// Examples:
    ///
    ///     Get the empty list on 100 qubits::
    ///
    ///         >>> QubitSparsePauliList.empty(100)
    ///         <QubitSparsePauliList with 0 elements on 100 qubits: []>
    #[pyo3(signature = (/, num_qubits))]
    #[staticmethod]
    pub fn empty(num_qubits: u32) -> Self {
        QubitSparsePauliList::empty(num_qubits).into()
    }

    /// Construct a :class:`.QubitSparsePauliList` from a single :class:`~.quantum_info.Pauli`
    /// instance.
    ///
    /// The output list will have a single term. Note that the phase is dropped.
    ///
    /// Args:
    ///     pauli (:class:`~.quantum_info.Pauli`): the single Pauli to convert.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> label = "IYXZI"
    ///         >>> pauli = Pauli(label)
    ///         >>> QubitSparsePauliList.from_pauli(pauli)
    ///         <QubitSparsePauliList with 1 element on 5 qubits: [Y_3 X_2 Z_1]>
    ///         >>> assert QubitSparsePauliList.from_label(label) == QubitSparsePauliList.from_pauli(pauli)
    #[staticmethod]
    #[pyo3(signature = (pauli, /))]
    fn from_pauli(pauli: &Bound<PyAny>) -> PyResult<Self> {
        let py = pauli.py();
        let num_qubits = pauli.getattr(intern!(py, "num_qubits"))?.extract::<u32>()?;
        let z = pauli
            .getattr(intern!(py, "z"))?
            .extract::<PyReadonlyArray1<bool>>()?;
        let x = pauli
            .getattr(intern!(py, "x"))?
            .extract::<PyReadonlyArray1<bool>>()?;
        let mut paulis = Vec::new();
        let mut indices = Vec::new();
        for (i, (x, z)) in x.as_array().iter().zip(z.as_array().iter()).enumerate() {
            // The only failure case possible here is the identity, because of how we're
            // constructing the value to convert.
            let Ok(term) = ::bytemuck::checked::try_cast(((*x as u8) << 1) | (*z as u8)) else {
                continue;
            };
            indices.push(i as u32);
            paulis.push(term);
        }
        let boundaries = vec![0, indices.len()];
        let inner = QubitSparsePauliList::new(num_qubits, paulis, indices, boundaries)?;
        Ok(inner.into())
    }

    /// Construct a list with a single-term from a dense string label.
    ///
    /// The label must be a sequence of the alphabet ``'IXYZ'``.  The label is interpreted
    /// analogously to a bitstring.  In other words, the right-most letter is associated with qubit
    /// 0, and so on.  This is the same as the labels for :class:`~.quantum_info.Pauli` and
    /// :class:`.SparsePauliOp`.
    ///
    /// Args:
    ///     label (str): the dense label.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> QubitSparsePauliList.from_label("IIIIXZI")
    ///         <QubitSparsePauliList with 1 element on 7 qubits: [X_2 Z_1]>
    ///         >>> label = "IYXZI"
    ///         >>> pauli = Pauli(label)
    ///         >>> assert QubitSparsePauliList.from_label(label) == QubitSparsePauliList.from_pauli(pauli)
    ///
    /// See also:
    ///     :meth:`from_list`
    ///         A generalization of this method that constructs a list from multiple labels.
    #[staticmethod]
    #[pyo3(signature = (label, /))]
    fn from_label(label: &str) -> Result<Self, LabelError> {
        let mut inner = QubitSparsePauliList::empty(label.len() as u32);
        inner.add_dense_label(label)?;
        Ok(inner.into())
    }

    /// Construct a qubit-sparse Pauli list from a list of dense labels.
    ///
    /// This is analogous to :meth:`.SparsePauliOp.from_list`. In this dense form, you must supply
    /// all identities explicitly in each label.
    ///
    /// The label must be a sequence of the alphabet ``'IXYZ'``.  The label is interpreted
    /// analogously to a bitstring.  In other words, the right-most letter is associated with qubit
    /// 0, and so on.  This is the same as the labels for :class:`~.quantum_info.Pauli` and
    /// :class:`.SparsePauliOp`.
    ///
    /// Args:
    ///     iter (list[str]): List of dense string labels.
    ///     num_qubits (int | None): It is not necessary to specify this if you are sure that
    ///         ``iter`` is not an empty sequence, since it can be inferred from the label lengths.
    ///         If ``iter`` may be empty, you must specify this argument to disambiguate how many
    ///         qubits the operators act on.  If this is given and ``iter`` is not empty, the value
    ///         must match the label lengths.
    ///
    /// Examples:
    ///
    ///     Construct a qubit sparse Pauli list from a list of labels::
    ///
    ///         >>> QubitSparsePauliList.from_list([
    ///         ...     "IIIXX",
    ///         ...     "IIYYI",
    ///         ...     "IXXII",
    ///         ...     "ZZIII",
    ///         ... ])
    ///         <QubitSparsePauliList with 4 elements on 5 qubits:
    ///             [X_1 X_0, Y_2 Y_1, X_3 X_2, Z_4 Z_3]>
    ///
    ///     Use ``num_qubits`` to disambiguate potentially empty inputs::
    ///
    ///         >>> QubitSparsePauliList.from_list([], num_qubits=10)
    ///         <QubitSparsePauliList with 0 elements on 10 qubits: []>
    ///
    ///     This method is equivalent to calls to :meth:`from_sparse_list` with the explicit
    ///     qubit-arguments field set to decreasing integers::
    ///
    ///         >>> labels = ["XYXZ", "YYZZ", "XYXZ"]
    ///         >>> from_list = QubitSparsePauliList.from_list(labels)
    ///         >>> from_sparse_list = QubitSparsePauliList.from_sparse_list([
    ///         ...     (label, (3, 2, 1, 0))
    ///         ...     for label in labels
    ///         ... ])
    ///         >>> assert from_list == from_sparse_list
    ///
    /// See also:
    ///     :meth:`from_sparse_list`
    ///         Construct the list from labels without explicit identities, but with the qubits each
    ///         single-qubit operator term applies to listed explicitly.
    #[staticmethod]
    #[pyo3(signature = (iter, /, *, num_qubits=None))]
    fn from_list(iter: Vec<String>, num_qubits: Option<u32>) -> PyResult<Self> {
        if iter.is_empty() && num_qubits.is_none() {
            return Err(PyValueError::new_err(
                "cannot construct a QubitSparsePauliList from an empty list without knowing `num_qubits`",
            ));
        }
        let num_qubits = match num_qubits {
            Some(num_qubits) => num_qubits,
            None => iter[0].len() as u32,
        };
        let mut inner = QubitSparsePauliList::with_capacity(num_qubits, iter.len(), 0);
        for label in iter {
            inner.add_dense_label(&label)?;
        }
        Ok(inner.into())
    }

    /// Construct a :class:`QubitSparsePauliList` out of individual :class:`QubitSparsePauli`
    /// instances.
    ///
    /// All the terms must have the same number of qubits.  If supplied, the ``num_qubits`` argument
    /// must match the terms.
    ///
    /// Args:
    ///     obj (Iterable[QubitSparsePauli]): Iterable of individual terms to build the list from.
    ///     num_qubits (int | None): The number of qubits the elements of the list should act on.
    ///         This is usually inferred from the input, but can be explicitly given to handle the
    ///         case of an empty iterable.
    ///
    /// Returns:
    ///     The corresponding list.
    #[staticmethod]
    #[pyo3(signature = (obj, /, num_qubits=None))]
    fn from_qubit_sparse_paulis(obj: &Bound<PyAny>, num_qubits: Option<u32>) -> PyResult<Self> {
        let mut iter = obj.try_iter()?;
        let mut inner = match num_qubits {
            Some(num_qubits) => QubitSparsePauliList::empty(num_qubits),
            None => {
                let Some(first) = iter.next() else {
                    return Err(PyValueError::new_err(
                        "cannot construct an empty QubitSparsePauliList without knowing `num_qubits`",
                    ));
                };
                let py_term = first?.cast::<PyQubitSparsePauli>()?.borrow();
                py_term.inner.to_qubit_sparse_pauli_list()
            }
        };
        for bound_py_term in iter {
            let py_term = bound_py_term?.cast::<PyQubitSparsePauli>()?.borrow();
            inner.add_qubit_sparse_pauli(py_term.inner.view())?;
        }
        Ok(inner.into())
    }

    /// Clear all the elements from the list, making it equal to the empty list again.
    ///
    /// This does not change the capacity of the internal allocations, so subsequent addition or
    /// subtraction operations resulting from composition may not need to reallocate.
    ///
    /// Examples:
    ///
    ///     .. code-block:: python
    ///
    ///         >>> pauli_list = QubitSparsePauliList.from_list(["IXXXYY", "ZZYZII"])
    ///         >>> pauli_list.clear()
    ///         >>> assert pauli_list == QubitSparsePauliList.empty(pauli_list.num_qubits)
    pub fn clear(&mut self) -> PyResult<()> {
        let mut inner = self.inner.write().map_err(|_| InnerWriteError)?;
        inner.clear();
        Ok(())
    }

    /// Construct a qubit sparse Pauli list from a list of labels and the qubits each item applies
    /// to.
    ///
    /// This is analogous to :meth:`.SparsePauliOp.from_sparse_list`.
    ///
    /// The "labels" and "indices" fields of the tuples are associated by zipping them together.
    /// For example, this means that a call to :meth:`from_list` can be converted to the form used
    /// by this method by setting the "indices" field of each triple to ``(num_qubits-1, ..., 1,
    /// 0)``.
    ///
    /// Args:
    ///     iter (list[tuple[str, Sequence[int]]]): tuples of labels and the qubits each
    ///         single-qubit term applies to.
    ///
    ///     num_qubits (int): the number of qubits the operators in the list act on.
    ///
    /// Examples:
    ///
    ///     Construct a simple list::
    ///
    ///         >>> QubitSparsePauliList.from_sparse_list(
    ///         ...     [("ZX", (1, 4)), ("YY", (0, 3))],
    ///         ...     num_qubits=5,
    ///         ... )
    ///         <QubitSparsePauliList with 2 elements on 5 qubits: [X_4 Z_1, Y_3 Y_0]>
    ///
    ///     This method can replicate the behavior of :meth:`from_list`, if the qubit-arguments
    ///     field of the tuple is set to decreasing integers::
    ///
    ///         >>> labels = ["XYXZ", "YYZZ", "XYXZ"]
    ///         >>> from_list = QubitSparsePauliList.from_list(labels)
    ///         >>> from_sparse_list = QubitSparsePauliList.from_sparse_list([
    ///         ...     (label, (3, 2, 1, 0))
    ///         ...     for label in labels
    ///         ... ])
    ///         >>> assert from_list == from_sparse_list
    ///
    /// See also:
    ///     :meth:`to_sparse_list`
    ///         The reverse of this method.
    #[staticmethod]
    #[pyo3(signature = (iter, /, num_qubits))]
    fn from_sparse_list(iter: Vec<(String, Vec<u32>)>, num_qubits: u32) -> PyResult<Self> {
        let (paulis, indices, boundaries) = raw_parts_from_sparse_list(iter, num_qubits)?;
        let inner = QubitSparsePauliList::new(num_qubits, paulis, indices, boundaries)?;
        Ok(inner.into())
    }

    /// Express the list in terms of a sparse list format.
    ///
    /// This can be seen as counter-operation of :meth:`.QubitSparsePauliList.from_sparse_list`,
    /// however the order of terms is not guaranteed to be the same at after a roundtrip to a sparse
    /// list and back.
    ///
    /// Examples:
    ///
    ///     >>> qubit_sparse_list = QubitSparsePauliList.from_list(["IIXIZ", "IIZIX"])
    ///     >>> reconstructed = QubitSparsePauliList.from_sparse_list(qubit_sparse_list.to_sparse_list(), qubit_sparse_list.num_qubits)
    ///
    /// See also:
    ///     :meth:`from_sparse_list`
    ///         The constructor that can interpret these lists.
    #[pyo3(signature = ())]
    fn to_sparse_list(&self, py: Python) -> PyResult<Py<PyList>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;

        // turn a SparseView into a Python tuple of (paulis, indices, coeff)
        let to_py_tuple = |view: QubitSparsePauliView| {
            let mut pauli_string = String::with_capacity(view.paulis.len());

            for bit in view.paulis.iter() {
                pauli_string.push_str(bit.py_label());
            }
            let py_string = PyString::new(py, &pauli_string).unbind();
            let py_indices = PyList::new(py, view.indices.iter())?.unbind();

            PyTuple::new(py, vec![py_string.as_any(), py_indices.as_any()])
        };

        let out = PyList::empty(py);
        for view in inner.iter() {
            out.append(to_py_tuple(view)?)?;
        }
        Ok(out.unbind())
    }

    /// Express the list in a dense array format.
    ///
    /// Each entry is a u8 following the :class:`Pauli` representation, while the rows index
    /// distinct Paulis and the columns distinct qubits.
    ///
    /// Examples:
    ///
    ///         >>> paulis = QubitSparsePauliList.from_sparse_list(
    ///         ...     [("ZX", (1, 4)), ("YY", (0, 3)), ("XX", (0, 1))],
    ///         ...     num_qubits=5,
    ///         ... )
    ///         >>> paulis.to_dense_array()
    #[pyo3(signature = ())]
    fn to_dense_array(&self, py: Python) -> PyResult<Py<PyArray2<u8>>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let mut out = Array2::zeros((inner.num_terms(), inner.num_qubits().try_into().unwrap()));
        for (idx, paulis) in inner.iter().enumerate() {
            for (p, p_idx) in zip(paulis.paulis, paulis.indices) {
                out[[idx, *p_idx as usize]] = *p as u8;
            }
        }
        Ok(out.into_pyarray(py).unbind())
    }

    /// Check if the elements of `self`` commute with another qubit sparse Pauli list.
    ///
    /// Args:
    ///     other (QubitSparsePauliList): the qubit sparse Pauli list to check for commutation with.
    #[pyo3(signature = (other))]
    fn commutes(&self, py: Python, other: &PyQubitSparsePauliList) -> PyResult<Py<PyArray2<bool>>> {
        let slf_inner = self.inner.read().map_err(|_| InnerReadError)?;
        let other_inner = other.inner.read().map_err(|_| InnerReadError)?;
        Ok(slf_inner.commutes(&other_inner)?.into_pyarray(py).unbind())
    }

    /// Return a :class:`~.quantum_info.PauliList` representing the same phaseless list of Paulis.
    fn to_pauli_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        imports::PAULI_LIST_TYPE
            .get_bound(py)
            .call1((inner.to_dense_label_list(),))
    }

    /// Apply a transpiler layout to this qubit sparse Pauli list.
    ///
    /// This enables remapping of qubit indices, e.g. if the list is defined in terms of virtual
    /// qubit labels.
    ///
    /// Args:
    ///     layout (TranspileLayout | list[int] | None): The layout to apply.  Most uses of this
    ///         function should pass the :attr:`.QuantumCircuit.layout` field from a circuit that
    ///         was transpiled for hardware.  In addition, you can pass a list of new qubit indices.
    ///         If given as explicitly ``None``, no remapping is applied (but you can still use
    ///         ``num_qubits`` to expand the qubits in the list).
    ///     num_qubits (int | None): The number of qubits to expand the list elements to.  If not
    ///         supplied, the output will be as wide as the given :class:`.TranspileLayout`, or the
    ///         same width as the input if the ``layout`` is given in another form.
    ///
    /// Returns:
    ///     A new :class:`QubitSparsePauli` with the provided layout applied.
    #[pyo3(signature = (/, layout, num_qubits=None))]
    fn apply_layout(&self, layout: Bound<PyAny>, num_qubits: Option<u32>) -> PyResult<Self> {
        let py = layout.py();
        let inner = self.inner.read().map_err(|_| InnerReadError)?;

        // A utility to check the number of qubits is compatible with the map.
        let check_inferred_qubits = |inferred: u32| -> PyResult<u32> {
            if inferred < inner.num_qubits() {
                return Err(CoherenceError::NotEnoughQubits {
                    current: inner.num_qubits() as usize,
                    target: inferred as usize,
                }
                .into());
            }
            Ok(inferred)
        };

        // Normalize the number of qubits in the layout and the layout itself, depending on the
        // input types, before calling QubitSparsePauliList.apply_layout to do the actual work.
        let (num_qubits, layout): (u32, Option<Vec<u32>>) = if layout.is_none() {
            (num_qubits.unwrap_or(inner.num_qubits()), None)
        } else if layout.is_instance(
            &py.import(intern!(py, "qiskit.transpiler"))?
                .getattr(intern!(py, "TranspileLayout"))?,
        )? {
            (
                check_inferred_qubits(
                    layout.getattr(intern!(py, "_output_qubit_list"))?.len()? as u32
                )?,
                Some(
                    layout
                        .call_method0(intern!(py, "final_index_layout"))?
                        .extract::<Vec<u32>>()?,
                ),
            )
        } else {
            (
                check_inferred_qubits(num_qubits.unwrap_or(inner.num_qubits()))?,
                Some(layout.extract()?),
            )
        };

        let out = inner.apply_layout(layout.as_deref(), num_qubits)?;
        Ok(out.into())
    }

    fn __len__(&self) -> PyResult<usize> {
        self.num_terms()
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        (
            py.get_type::<Self>().getattr("from_sparse_list")?,
            (self.to_sparse_list(py)?, inner.num_qubits()),
        )
            .into_pyobject(py)
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: PySequenceIndex<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let indices = match index.with_len(inner.num_terms())? {
            SequenceIndex::Int(index) => {
                return PyQubitSparsePauli {
                    inner: inner.term(index).to_term(),
                }
                .into_bound_py_any(py);
            }
            indices => indices,
        };
        let mut out = QubitSparsePauliList::empty(inner.num_qubits());
        for index in indices.iter() {
            out.add_qubit_sparse_pauli(inner.term(index))?;
        }
        out.into_bound_py_any(py)
    }

    fn __eq__(slf: Bound<Self>, other: Bound<PyAny>) -> PyResult<bool> {
        // this is also important to check before trying to read both slf and other
        if slf.is(&other) {
            return Ok(true);
        }
        let Ok(other) = other.cast_into::<Self>() else {
            return Ok(false);
        };
        let slf_borrowed = slf.borrow();
        let other_borrowed = other.borrow();
        let slf_inner = slf_borrowed.inner.read().map_err(|_| InnerReadError)?;
        let other_inner = other_borrowed.inner.read().map_err(|_| InnerReadError)?;
        Ok(slf_inner.eq(&other_inner))
    }

    fn __repr__(&self) -> PyResult<String> {
        let num_terms = self.num_terms()?;
        let num_qubits = self.num_qubits()?;

        let str_num_terms = format!(
            "{} element{}",
            num_terms,
            if num_terms == 1 { "" } else { "s" }
        );
        let str_num_qubits = format!(
            "{} qubit{}",
            num_qubits,
            if num_qubits == 1 { "" } else { "s" }
        );

        let inner = self.inner.read().map_err(|_| InnerReadError)?;
        let str_terms = if num_terms == 0 {
            "".to_owned()
        } else {
            inner
                .iter()
                .map(QubitSparsePauliView::to_sparse_str)
                .collect::<Vec<_>>()
                .join(", ")
        };
        Ok(format!(
            "<QubitSparsePauliList with {str_num_terms} on {str_num_qubits}: [{str_terms}]>"
        ))
    }
}

impl From<QubitSparsePauli> for PyQubitSparsePauli {
    fn from(val: QubitSparsePauli) -> PyQubitSparsePauli {
        PyQubitSparsePauli { inner: val }
    }
}

impl<'py> IntoPyObject<'py> for QubitSparsePauli {
    type Target = PyQubitSparsePauli;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        PyQubitSparsePauli::from(self).into_pyobject(py)
    }
}

impl From<QubitSparsePauliList> for PyQubitSparsePauliList {
    fn from(val: QubitSparsePauliList) -> PyQubitSparsePauliList {
        PyQubitSparsePauliList {
            inner: Arc::new(RwLock::new(val)),
        }
    }
}

impl<'py> IntoPyObject<'py> for QubitSparsePauliList {
    type Target = PyQubitSparsePauliList;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        PyQubitSparsePauliList::from(self).into_pyobject(py)
    }
}
