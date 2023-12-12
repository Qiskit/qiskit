// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::quantum_circuit::circuit_instruction::CircuitInstruction;
use crate::quantum_circuit::intern_context::{BitType, IndexType, InternContext};
use crate::quantum_circuit::py_ext;
use hashbrown::HashMap;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyList, PySlice, PyTuple, PyType};
use pyo3::{PyObject, PyResult, PyTraverseError, PyVisit};
use std::hash::{Hash, Hasher};

/// Private type used to store instructions with interned arg lists.
#[derive(Clone, Debug)]
struct PackedInstruction {
    /// The Python-side operation instance.
    op: PyObject,
    /// The index under which the interner has stored `qubits`.
    qubits_id: IndexType,
    /// The index under which the interner has stored `clbits`.
    clbits_id: IndexType,
}

/// Private wrapper for Python-side Bit instances that implements
/// [Hash] and [Eq], allowing them to be used in Rust hash-based
/// sets and maps.
///
/// Python's `hash()` is called on the wrapped Bit instance during
/// construction and returned from Rust's [Hash] trait impl.
/// The impl of [PartialEq] first compares the native Py pointers
/// to determine equality. If these are not equal, only then does
/// it call `repr()` on both sides, which has a significant
/// performance advantage.
#[derive(Clone, Debug)]
struct BitAsKey {
    /// Python's `hash()` of the wrapped instance.
    hash: isize,
    /// The wrapped instance.
    bit: PyObject,
}

impl BitAsKey {
    fn new(bit: &PyAny) -> PyResult<Self> {
        Ok(BitAsKey {
            hash: bit.hash()?,
            bit: bit.into_py(bit.py()),
        })
    }
}

impl Hash for BitAsKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_isize(self.hash);
    }
}

impl PartialEq for BitAsKey {
    fn eq(&self, other: &Self) -> bool {
        self.bit.is(&other.bit)
            || Python::with_gil(|py| {
                self.bit
                    .as_ref(py)
                    .repr()
                    .unwrap()
                    .eq(other.bit.as_ref(py).repr().unwrap())
                    .unwrap()
            })
    }
}

impl Eq for BitAsKey {}

/// A container for :class:`.QuantumCircuit` instruction listings that stores
/// :class:`.CircuitInstruction` instances in a packed form by interning
/// their :attr:`~.CircuitInstruction.qubits` and
/// :attr:`~.CircuitInstruction.clbits` to native vectors of indices.
///
/// Before adding a :class:`.CircuitInstruction` to this container, its
/// :class:`.Qubit` and :class:`.Clbit` instances MUST be registered via the
/// constructor or via :meth:`.CircuitData.add_qubit` and
/// :meth:`.CircuitData.add_clbit`. This is because the order in which
/// bits of the same type are added to the container determines their
/// associated indices used for storage and retrieval.
///
/// Once constructed, this container behaves like a Python list of
/// :class:`.CircuitInstruction` instances. However, these instances are
/// created and destroyed on the fly, and thus should be treated as ephemeral.
///
/// For example,
///
/// .. code-block::
///
///     qubits = [Qubit()]
///     data = CircuitData(qubits)
///     data.append(CircuitInstruction(XGate(), (qubits[0],), ()))
///     assert(data[0] == data[0]) # => Ok.
///     assert(data[0] is data[0]) # => PANICS!
///
/// .. warning::
///
///     This is an internal interface and no part of it should be relied upon
///     outside of Qiskit.
///
/// Args:
///     qubits (Iterable[:class:`.Qubit`] | None): The initial sequence of
///         qubits, used to map :class:`.Qubit` instances to and from its
///         indices.
///     clbits (Iterable[:class:`.Clbit`] | None): The initial sequence of
///         clbits, used to map :class:`.Clbit` instances to and from its
///         indices.
///     data (Iterable[:class:`.CircuitInstruction`]): An initial instruction
///         listing to add to this container. All bits appearing in the
///         instructions in this iterable must also exist in ``qubits`` and
///         ``clbits``.
///     reserve (int): The container's initial capacity. This is reserved
///         before copying instructions into the container when ``data``
///         is provided, so the initialized container's unused capacity will
///         be ``max(0, reserve - len(data))``.
///
/// Raises:
///     KeyError: if ``data`` contains a reference to a bit that is not present
///         in ``qubits`` or ``clbits``.
#[pyclass(sequence, module = "qiskit._accelerate.quantum_circuit")]
#[derive(Clone, Debug)]
pub struct CircuitData {
    /// The packed instruction listing.
    data: Vec<PackedInstruction>,
    /// The intern context used to intern instruction bits.
    intern_context: InternContext,
    /// The qubits registered (e.g. through :meth:`~.CircuitData.add_qubit`).
    qubits_native: Vec<PyObject>,
    /// The clbits registered (e.g. through :meth:`~.CircuitData.add_clbit`).
    clbits_native: Vec<PyObject>,
    /// Map of :class:`.Qubit` instances to their index in
    /// :attr:`.CircuitData.qubits`.
    qubit_indices_native: HashMap<BitAsKey, BitType>,
    /// Map of :class:`.Clbit` instances to their index in
    /// :attr:`.CircuitData.clbits`.
    clbit_indices_native: HashMap<BitAsKey, BitType>,
    /// The qubits registered, cached as a ``list[Qubit]``.
    qubits: Py<PyList>,
    /// The clbits registered, cached as a ``list[Clbit]``.
    clbits: Py<PyList>,
}

/// A private enumeration type used to extract arguments to pymethods
/// that may be either an index or a slice.
#[derive(FromPyObject)]
pub enum SliceOrInt<'a> {
    Slice(&'a PySlice),
    Int(isize),
}

#[pymethods]
impl CircuitData {
    #[new]
    #[pyo3(signature = (qubits=None, clbits=None, data=None, reserve=0))]
    pub fn new(
        py: Python<'_>,
        qubits: Option<&PyAny>,
        clbits: Option<&PyAny>,
        data: Option<&PyAny>,
        reserve: usize,
    ) -> PyResult<Self> {
        let mut self_ = CircuitData {
            data: Vec::new(),
            intern_context: InternContext::new(),
            qubits_native: Vec::new(),
            clbits_native: Vec::new(),
            qubit_indices_native: HashMap::new(),
            clbit_indices_native: HashMap::new(),
            qubits: PyList::empty(py).into_py(py),
            clbits: PyList::empty(py).into_py(py),
        };
        if let Some(qubits) = qubits {
            for bit in qubits.iter()? {
                self_.add_qubit(py, bit?)?;
            }
        }
        if let Some(clbits) = clbits {
            for bit in clbits.iter()? {
                self_.add_clbit(py, bit?)?;
            }
        }
        if let Some(data) = data {
            self_.reserve(py, reserve);
            self_.extend(py, data)?;
        }
        Ok(self_)
    }

    pub fn __reduce__(self_: &PyCell<CircuitData>, py: Python<'_>) -> PyResult<PyObject> {
        let ty: &PyType = self_.get_type();
        let args = {
            let self_ = self_.borrow();
            (
                self_.qubits.clone_ref(py),
                self_.clbits.clone_ref(py),
                None::<()>,
                self_.data.len(),
            )
        };
        Ok((ty, args, None::<()>, self_.iter()?).into_py(py))
    }

    /// Returns the current sequence of registered :class:`.Qubit`
    /// instances as a list.
    ///
    /// .. note::
    ///
    ///     This list is not kept in sync with the container.
    ///
    /// Returns:
    ///     list(:class:`.Qubit`): The current sequence of registered qubits.
    #[getter]
    pub fn qubits(&self, py: Python<'_>) -> PyObject {
        PyList::new(py, self.qubits.as_ref(py)).into_py(py)
    }

    /// Returns the current sequence of registered :class:`.Clbit`
    /// instances as a list.
    ///
    /// .. note::
    ///
    ///     This list is not kept in sync with the container.
    ///
    /// Returns:
    ///     list(:class:`.Clbit`): The current sequence of registered clbits.
    #[getter]
    pub fn clbits(&self, py: Python<'_>) -> PyObject {
        PyList::new(py, self.clbits.as_ref(py)).into_py(py)
    }

    /// Registers a :class:`.Qubit` instance.
    ///
    /// Args:
    ///     bit (:class:`.Qubit`): The qubit to register.
    pub fn add_qubit(&mut self, py: Python<'_>, bit: &PyAny) -> PyResult<()> {
        let idx: BitType = self.qubits_native.len().try_into().map_err(|_| {
            PyRuntimeError::new_err(
                "The number of qubits in the circuit has exceeded the maximum capacity",
            )
        })?;
        self.qubit_indices_native.insert(BitAsKey::new(bit)?, idx);
        self.qubits_native.push(bit.into_py(py));
        self.qubits = PyList::new(py, &self.qubits_native).into_py(py);
        Ok(())
    }

    /// Registers a :class:`.Clbit` instance.
    ///
    /// Args:
    ///     bit (:class:`.Clbit`): The clbit to register.
    pub fn add_clbit(&mut self, py: Python<'_>, bit: &PyAny) -> PyResult<()> {
        let idx: BitType = self.clbits_native.len().try_into().map_err(|_| {
            PyRuntimeError::new_err(
                "The number of clbits in the circuit has exceeded the maximum capacity",
            )
        })?;
        self.clbit_indices_native.insert(BitAsKey::new(bit)?, idx);
        self.clbits_native.push(bit.into_py(py));
        self.clbits = PyList::new(py, &self.clbits_native).into_py(py);
        Ok(())
    }

    /// Performs a shallow copy.
    ///
    /// Returns:
    ///     CircuitData: The shallow copy.
    pub fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(CircuitData {
            data: self.data.clone(),
            intern_context: self.intern_context.clone(),
            qubits_native: self.qubits_native.clone(),
            clbits_native: self.clbits_native.clone(),
            qubit_indices_native: self.qubit_indices_native.clone(),
            clbit_indices_native: self.clbit_indices_native.clone(),
            qubits: PyList::new(py, &self.qubits_native).into_py(py),
            clbits: PyList::new(py, &self.clbits_native).into_py(py),
        })
    }

    /// Reserves capacity for at least ``additional`` more
    /// :class:`.CircuitInstruction` instances to be added to this container.
    ///
    /// Args:
    ///     additional (int): The additional capacity to reserve. If the
    ///         capacity is already sufficient, does nothing.
    pub fn reserve(&mut self, _py: Python<'_>, additional: usize) {
        self.data.reserve(additional);
    }

    pub fn __len__(&self) -> usize {
        self.data.len()
    }

    // Note: we also rely on this to make us iterable!
    pub fn __getitem__(&self, py: Python, index: &PyAny) -> PyResult<PyObject> {
        // Internal helper function to get a specific
        // instruction by index.
        fn get_at(
            self_: &CircuitData,
            py: Python<'_>,
            index: isize,
        ) -> PyResult<Py<CircuitInstruction>> {
            let index = self_.convert_py_index(index)?;
            if let Some(inst) = self_.data.get(index) {
                self_.unpack(py, inst)
            } else {
                Err(PyIndexError::new_err(format!(
                    "No element at index {:?} in circuit data",
                    index
                )))
            }
        }

        if index.is_exact_instance_of::<PySlice>() {
            let slice = self.convert_py_slice(index.downcast_exact::<PySlice>()?)?;
            let result = slice
                .into_iter()
                .map(|i| get_at(self, py, i))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(result.into_py(py))
        } else {
            Ok(get_at(self, py, index.extract()?)?.into_py(py))
        }
    }

    pub fn __delitem__(&mut self, index: SliceOrInt) -> PyResult<()> {
        match index {
            SliceOrInt::Slice(slice) => {
                let slice = {
                    let mut s = self.convert_py_slice(slice)?;
                    if s.len() > 1 && s.first().unwrap() < s.last().unwrap() {
                        // Reverse the order so we're sure to delete items
                        // at the back first (avoids messing up indices).
                        s.reverse()
                    }
                    s
                };
                for i in slice.into_iter() {
                    self.__delitem__(SliceOrInt::Int(i))?;
                }
                Ok(())
            }
            SliceOrInt::Int(index) => {
                let index = self.convert_py_index(index)?;
                if self.data.get(index).is_some() {
                    self.data.remove(index);
                    Ok(())
                } else {
                    Err(PyIndexError::new_err(format!(
                        "No element at index {:?} in circuit data",
                        index
                    )))
                }
            }
        }
    }

    pub fn __setitem__(
        &mut self,
        py: Python<'_>,
        index: SliceOrInt,
        value: &PyAny,
    ) -> PyResult<()> {
        match index {
            SliceOrInt::Slice(slice) => {
                let indices = slice.indices(self.data.len().try_into().unwrap())?;
                let slice = self.convert_py_slice(slice)?;
                let values = value.iter()?.collect::<PyResult<Vec<&PyAny>>>()?;
                if indices.step != 1 && slice.len() != values.len() {
                    // A replacement of a different length when step isn't exactly '1'
                    // would result in holes.
                    return Err(PyValueError::new_err(format!(
                        "attempt to assign sequence of size {:?} to extended slice of size {:?}",
                        values.len(),
                        slice.len(),
                    )));
                }

                for (i, v) in slice.iter().zip(values.iter()) {
                    self.__setitem__(py, SliceOrInt::Int(*i), v)?;
                }

                if slice.len() > values.len() {
                    // Delete any extras.
                    let slice = PySlice::new(
                        py,
                        indices.start + values.len() as isize,
                        indices.stop,
                        1isize,
                    );
                    self.__delitem__(SliceOrInt::Slice(slice))?;
                } else {
                    // Insert any extra values.
                    for v in values.iter().skip(slice.len()).rev() {
                        let v: PyRef<CircuitInstruction> = v.extract()?;
                        self.insert(py, indices.stop, v)?;
                    }
                }

                Ok(())
            }
            SliceOrInt::Int(index) => {
                let index = self.convert_py_index(index)?;
                let value: PyRef<CircuitInstruction> = value.extract()?;
                let mut packed = self.pack(py, value)?;
                std::mem::swap(&mut packed, &mut self.data[index]);
                Ok(())
            }
        }
    }

    pub fn insert(
        &mut self,
        py: Python<'_>,
        index: isize,
        value: PyRef<CircuitInstruction>,
    ) -> PyResult<()> {
        let index = self.convert_py_index_clamped(index);
        let packed = self.pack(py, value)?;
        self.data.insert(index, packed);
        Ok(())
    }

    pub fn pop(&mut self, py: Python<'_>, index: Option<PyObject>) -> PyResult<PyObject> {
        let index =
            index.unwrap_or_else(|| std::cmp::max(0, self.data.len() as isize - 1).into_py(py));
        let item = self.__getitem__(py, index.as_ref(py))?;
        self.__delitem__(index.as_ref(py).extract()?)?;
        Ok(item)
    }

    pub fn append(&mut self, py: Python<'_>, value: PyRef<CircuitInstruction>) -> PyResult<()> {
        let packed = self.pack(py, value)?;
        self.data.push(packed);
        Ok(())
    }

    // To prevent the entire iterator from being loaded into memory,
    // we create a `GILPool` for each iteration of the loop, which
    // ensures that the `CircuitInstruction` returned by the call
    // to `next` is dropped before the next iteration.
    pub fn extend(&mut self, py: Python<'_>, itr: &PyAny) -> PyResult<()> {
        // To ensure proper lifetime management, we explicitly store
        // the result of calling `iter(itr)` as a GIL-independent
        // reference that we access only with the most recent GILPool.
        // It would be dangerous to access the original `itr` or any
        // GIL-dependent derivatives of it after creating the new pool.
        let itr: Py<PyIterator> = itr.iter()?.into_py(py);
        loop {
            // Create a new pool, so that PyO3 can clear memory at
            // the end of the loop.
            let pool = unsafe { py.new_pool() };

            // It is recommended to *always* immediately set py to the pool's
            // Python, to help avoid creating references with invalid lifetimes.
            let py = pool.python();

            // Access the iterator using the new pool.
            match itr.as_ref(py).next() {
                None => {
                    break;
                }
                Some(v) => {
                    self.append(py, v?.extract()?)?;
                }
            }
            // The GILPool is dropped here, which cleans up the ref
            // returned from `next` as well as any resources used by
            // `self.append`.
        }
        Ok(())
    }

    pub fn clear(&mut self, _py: Python<'_>) -> PyResult<()> {
        std::mem::take(&mut self.data);
        Ok(())
    }

    // Marks this pyclass as NOT hashable.
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __eq__(slf: &PyCell<Self>, other: &PyAny) -> PyResult<bool> {
        let slf: &PyAny = slf;
        if slf.is(other) {
            return Ok(true);
        }
        if slf.len()? != other.len()? {
            return Ok(false);
        }
        // Implemented using generic iterators on both sides
        // for simplicity.
        let mut ours_itr = slf.iter()?;
        let mut theirs_itr = other.iter()?;
        loop {
            match (ours_itr.next(), theirs_itr.next()) {
                (Some(ours), Some(theirs)) => {
                    if !ours?.eq(theirs?)? {
                        return Ok(false);
                    }
                }
                (None, None) => {
                    return Ok(true);
                }
                _ => {
                    return Ok(false);
                }
            }
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for packed in self.data.iter() {
            visit.call(&packed.op)?;
        }
        for bit in self.qubits_native.iter().chain(self.clbits_native.iter()) {
            visit.call(bit)?;
        }

        // Note:
        //   There's no need to visit the native Rust data
        //   structures used for internal tracking: the only Python
        //   references they contain are to the bits in these lists!
        visit.call(&self.qubits)?;
        visit.call(&self.clbits)?;
        Ok(())
    }

    fn __clear__(&mut self) {
        // Clear anything that could have a reference cycle.
        self.data.clear();
        self.qubits_native.clear();
        self.clbits_native.clear();
        self.qubit_indices_native.clear();
        self.clbit_indices_native.clear();
    }
}

impl CircuitData {
    /// Converts a Python slice to a `Vec` of indices into
    /// the instruction listing, [CircuitData.data].
    fn convert_py_slice(&self, slice: &PySlice) -> PyResult<Vec<isize>> {
        let indices = slice.indices(self.data.len().try_into().unwrap())?;
        if indices.step > 0 {
            Ok((indices.start..indices.stop)
                .step_by(indices.step as usize)
                .collect())
        } else {
            let mut out = Vec::with_capacity(indices.slicelength as usize);
            let mut x = indices.start;
            while x > indices.stop {
                out.push(x);
                x += indices.step;
            }
            Ok(out)
        }
    }

    /// Converts a Python index to an index into the instruction listing,
    /// or one past its end.
    /// If the resulting index would be < 0, clamps to 0.
    /// If the resulting index would be > len(data), clamps to len(data).
    fn convert_py_index_clamped(&self, index: isize) -> usize {
        let index = if index < 0 {
            index + self.data.len() as isize
        } else {
            index
        };
        std::cmp::min(std::cmp::max(0, index), self.data.len() as isize) as usize
    }

    /// Converts a Python index to an index into the instruction listing.
    fn convert_py_index(&self, index: isize) -> PyResult<usize> {
        let index = if index < 0 {
            index + self.data.len() as isize
        } else {
            index
        };

        if index < 0 || index >= self.data.len() as isize {
            return Err(PyIndexError::new_err(format!(
                "Index {:?} is out of bounds.",
                index,
            )));
        }
        Ok(index as usize)
    }

    /// Returns a [PackedInstruction] containing the original operation
    /// of `elem` and [InternContext] indices of its `qubits` and `clbits`
    /// fields.
    fn pack(
        &mut self,
        py: Python<'_>,
        inst: PyRef<CircuitInstruction>,
    ) -> PyResult<PackedInstruction> {
        let mut interned_bits =
            |indices: &HashMap<BitAsKey, BitType>, bits: &PyTuple| -> PyResult<IndexType> {
                let args = bits
                    .into_iter()
                    .map(|b| {
                        let key = BitAsKey::new(b)?;
                        indices.get(&key).copied().ok_or_else(|| {
                            PyKeyError::new_err(format!(
                                "Bit {:?} has not been added to this circuit.",
                                b
                            ))
                        })
                    })
                    .collect::<PyResult<Vec<BitType>>>()?;
                self.intern_context.intern(args)
            };
        Ok(PackedInstruction {
            op: inst.operation.clone_ref(py),
            qubits_id: interned_bits(&self.qubit_indices_native, inst.qubits.as_ref(py))?,
            clbits_id: interned_bits(&self.clbit_indices_native, inst.clbits.as_ref(py))?,
        })
    }

    fn unpack(&self, py: Python<'_>, inst: &PackedInstruction) -> PyResult<Py<CircuitInstruction>> {
        Py::new(
            py,
            CircuitInstruction {
                operation: inst.op.clone_ref(py),
                qubits: py_ext::tuple_new(
                    py,
                    self.intern_context
                        .lookup(inst.qubits_id)
                        .iter()
                        .map(|i| self.qubits_native[*i as usize].clone_ref(py))
                        .collect(),
                ),
                clbits: py_ext::tuple_new(
                    py,
                    self.intern_context
                        .lookup(inst.clbits_id)
                        .iter()
                        .map(|i| self.clbits_native[*i as usize].clone_ref(py))
                        .collect(),
                ),
            },
        )
    }
}
