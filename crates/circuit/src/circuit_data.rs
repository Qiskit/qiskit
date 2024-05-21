// This code is part of Qiskit.
//
// (C) Copyright IBM 2023, 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::bit_data::{BitData, BitNotFoundError};
use crate::circuit_instruction::CircuitInstruction;
use crate::interner::{CacheFullError, IndexedInterner, Interner, InternerKey};
use crate::packed_instruction::PackedInstruction;
use crate::{Clbit, Qubit, SliceOrInt};

use pyo3::exceptions::{PyIndexError, PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySet, PySlice, PyTuple, PyType};
use pyo3::{PyObject, PyResult, PyTraverseError, PyVisit};
use std::mem;

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
#[pyclass(sequence, module = "qiskit._accelerate.circuit")]
#[derive(Clone, Debug)]
pub struct CircuitData {
    /// The packed instruction listing.
    data: Vec<PackedInstruction>,
    /// The cache used to intern instruction bits.
    qargs_interner: IndexedInterner<Vec<Qubit>>,
    /// The cache used to intern instruction bits.
    cargs_interner: IndexedInterner<Vec<Clbit>>,
    /// Qubits registered in the circuit.
    qubits: BitData<Qubit>,
    /// Clbits registered in the circuit.
    clbits: BitData<Clbit>,
}

impl<'py> From<BitNotFoundError<'py>> for PyErr {
    fn from(error: BitNotFoundError) -> Self {
        PyKeyError::new_err(format!(
            "Bit {:?} has not been added to this circuit.",
            error.0
        ))
    }
}

impl From<CacheFullError> for PyErr {
    fn from(_: CacheFullError) -> Self {
        PyRuntimeError::new_err("The bit operands cache is full!")
    }
}

#[pymethods]
impl CircuitData {
    #[new]
    #[pyo3(signature = (qubits=None, clbits=None, data=None, reserve=0))]
    pub fn new(
        py: Python<'_>,
        qubits: Option<&Bound<PyAny>>,
        clbits: Option<&Bound<PyAny>>,
        data: Option<&Bound<PyAny>>,
        reserve: usize,
    ) -> PyResult<Self> {
        let mut self_ = CircuitData {
            data: Vec::new(),
            qargs_interner: IndexedInterner::new(),
            cargs_interner: IndexedInterner::new(),
            qubits: BitData::new(py, "qubits".to_string()),
            clbits: BitData::new(py, "clbits".to_string()),
        };
        if let Some(qubits) = qubits {
            for bit in qubits.iter()? {
                self_.add_qubit(py, &bit?, true)?;
            }
        }
        if let Some(clbits) = clbits {
            for bit in clbits.iter()? {
                self_.add_clbit(py, &bit?, true)?;
            }
        }
        if let Some(data) = data {
            self_.reserve(py, reserve);
            self_.extend(py, data)?;
        }
        Ok(self_)
    }

    pub fn __reduce__(self_: &Bound<CircuitData>, py: Python<'_>) -> PyResult<PyObject> {
        let ty: Bound<PyType> = self_.get_type();
        let args = {
            let self_ = self_.borrow();
            (
                self_.qubits.cached().clone_ref(py),
                self_.clbits.cached().clone_ref(py),
                None::<()>,
                self_.data.len(),
            )
        };
        Ok((ty, args, None::<()>, self_.iter()?).into_py(py))
    }

    /// Returns the current sequence of registered :class:`.Qubit` instances as a list.
    ///
    /// .. warning::
    ///
    ///     Do not modify this list yourself.  It will invalidate the :class:`CircuitData` data
    ///     structures.
    ///
    /// Returns:
    ///     list(:class:`.Qubit`): The current sequence of registered qubits.
    #[getter]
    pub fn qubits(&self, py: Python<'_>) -> Py<PyList> {
        self.qubits.cached().clone_ref(py)
    }

    /// Returns the current sequence of registered :class:`.Clbit`
    /// instances as a list.
    ///
    /// .. warning::
    ///
    ///     Do not modify this list yourself.  It will invalidate the :class:`CircuitData` data
    ///     structures.
    ///
    /// Returns:
    ///     list(:class:`.Clbit`): The current sequence of registered clbits.
    #[getter]
    pub fn clbits(&self, py: Python<'_>) -> Py<PyList> {
        self.clbits.cached().clone_ref(py)
    }

    /// Registers a :class:`.Qubit` instance.
    ///
    /// Args:
    ///     bit (:class:`.Qubit`): The qubit to register.
    ///     strict (bool): When set, raises an error if ``bit`` is already present.
    ///
    /// Raises:
    ///     ValueError: The specified ``bit`` is already present and flag ``strict``
    ///         was provided.
    #[pyo3(signature = (bit, *, strict=true))]
    pub fn add_qubit(&mut self, py: Python, bit: &Bound<PyAny>, strict: bool) -> PyResult<()> {
        self.qubits.add(py, bit, strict)
    }

    /// Registers a :class:`.Clbit` instance.
    ///
    /// Args:
    ///     bit (:class:`.Clbit`): The clbit to register.
    ///     strict (bool): When set, raises an error if ``bit`` is already present.
    ///
    /// Raises:
    ///     ValueError: The specified ``bit`` is already present and flag ``strict``
    ///         was provided.
    #[pyo3(signature = (bit, *, strict=true))]
    pub fn add_clbit(&mut self, py: Python, bit: &Bound<PyAny>, strict: bool) -> PyResult<()> {
        self.clbits.add(py, bit, strict)
    }

    /// Performs a shallow copy.
    ///
    /// Returns:
    ///     CircuitData: The shallow copy.
    pub fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        let mut res = CircuitData::new(
            py,
            Some(self.qubits.cached().bind(py)),
            Some(self.clbits.cached().bind(py)),
            None,
            0,
        )?;
        res.qargs_interner = self.qargs_interner.clone();
        res.cargs_interner = self.cargs_interner.clone();
        res.data.clone_from(&self.data);
        Ok(res)
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

    /// Returns a tuple of the sets of :class:`.Qubit` and :class:`.Clbit` instances
    /// that appear in at least one instruction's bit lists.
    ///
    /// Returns:
    ///     tuple[set[:class:`.Qubit`], set[:class:`.Clbit`]]: The active qubits and clbits.
    pub fn active_bits(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let qubits = PySet::empty_bound(py)?;
        let clbits = PySet::empty_bound(py)?;
        for inst in self.data.iter() {
            for b in self.qargs_interner.intern(inst.qubits_id).value.iter() {
                qubits.add(self.qubits.get(*b).unwrap().clone_ref(py))?;
            }
            for b in self.cargs_interner.intern(inst.clbits_id).value.iter() {
                clbits.add(self.clbits.get(*b).unwrap().clone_ref(py))?;
            }
        }

        Ok((qubits, clbits).into_py(py))
    }

    /// Invokes callable ``func`` with each instruction's operation.
    ///
    /// Args:
    ///     func (Callable[[:class:`~.Operation`], None]):
    ///         The callable to invoke.
    #[pyo3(signature = (func))]
    pub fn foreach_op(&self, py: Python<'_>, func: &Bound<PyAny>) -> PyResult<()> {
        for inst in self.data.iter() {
            func.call1((inst.op.bind(py),))?;
        }
        Ok(())
    }

    /// Invokes callable ``func`` with the positional index and operation
    /// of each instruction.
    ///
    /// Args:
    ///     func (Callable[[int, :class:`~.Operation`], None]):
    ///         The callable to invoke.
    #[pyo3(signature = (func))]
    pub fn foreach_op_indexed(&self, py: Python<'_>, func: &Bound<PyAny>) -> PyResult<()> {
        for (index, inst) in self.data.iter().enumerate() {
            func.call1((index, inst.op.bind(py)))?;
        }
        Ok(())
    }

    /// Invokes callable ``func`` with each instruction's operation,
    /// replacing the operation with the result.
    ///
    /// Args:
    ///     func (Callable[[:class:`~.Operation`], :class:`~.Operation`]):
    ///         A callable used to map original operation to their
    ///         replacements.
    #[pyo3(signature = (func))]
    pub fn map_ops(&mut self, py: Python<'_>, func: &Bound<PyAny>) -> PyResult<()> {
        for inst in self.data.iter_mut() {
            inst.op = func.call1((inst.op.bind(py),))?.into_py(py);
        }
        Ok(())
    }

    /// Replaces the bits of this container with the given ``qubits``
    /// and/or ``clbits``.
    ///
    /// The `:attr:`~.CircuitInstruction.qubits` and
    /// :attr:`~.CircuitInstruction.clbits` of existing instructions are
    /// reinterpreted using the new bit sequences on access.
    /// As such, the primary use-case for this method is to remap a circuit to
    /// a different set of bits in constant time relative to the number of
    /// instructions in the circuit.
    ///
    /// Args:
    ///     qubits (Iterable[:class:`.Qubit] | None):
    ///         The qubit sequence which should replace the container's
    ///         existing qubits, or ``None`` to skip replacement.
    ///     clbits (Iterable[:class:`.Clbit] | None):
    ///         The clbit sequence which should replace the container's
    ///         existing qubits, or ``None`` to skip replacement.
    ///
    /// Raises:
    ///     ValueError: A replacement sequence is smaller than the bit list
    ///         its contents would replace.
    ///
    /// .. note::
    ///
    ///     Instruction operations themselves are NOT adjusted.
    ///     To modify bits referenced by an operation, use
    ///     :meth:`~.CircuitData.foreach_op` or
    ///     :meth:`~.CircuitData.foreach_op_indexed` or
    ///     :meth:`~.CircuitData.map_ops` to adjust the operations manually
    ///     after calling this method.
    ///
    /// Examples:
    ///
    ///     The following :class:`.CircuitData` is reinterpreted as if its bits
    ///     were originally added in reverse.
    ///
    ///     .. code-block::
    ///
    ///         qr = QuantumRegister(3)
    ///         data = CircuitData(qubits=qr, data=[
    ///             CircuitInstruction(XGate(), [qr[0]], []),
    ///             CircuitInstruction(XGate(), [qr[1]], []),
    ///             CircuitInstruction(XGate(), [qr[2]], []),
    ///         ])
    ///
    ///         data.replace_bits(qubits=reversed(qr))
    ///         assert(data == [
    ///             CircuitInstruction(XGate(), [qr[2]], []),
    ///             CircuitInstruction(XGate(), [qr[1]], []),
    ///             CircuitInstruction(XGate(), [qr[0]], []),
    ///         ])
    #[pyo3(signature = (qubits=None, clbits=None))]
    pub fn replace_bits(
        &mut self,
        py: Python<'_>,
        qubits: Option<&Bound<PyAny>>,
        clbits: Option<&Bound<PyAny>>,
    ) -> PyResult<()> {
        let mut temp = CircuitData::new(py, qubits, clbits, None, 0)?;
        if qubits.is_some() {
            if temp.qubits.len() < self.qubits.len() {
                return Err(PyValueError::new_err(format!(
                    "Replacement 'qubits' of size {:?} must contain at least {:?} bits.",
                    temp.qubits.len(),
                    self.qubits.len(),
                )));
            }
            mem::swap(&mut temp.qubits, &mut self.qubits);
        }
        if clbits.is_some() {
            if temp.clbits.len() < self.clbits.len() {
                return Err(PyValueError::new_err(format!(
                    "Replacement 'clbits' of size {:?} must contain at least {:?} bits.",
                    temp.clbits.len(),
                    self.clbits.len(),
                )));
            }
            mem::swap(&mut temp.clbits, &mut self.clbits);
        }
        Ok(())
    }

    pub fn __len__(&self) -> usize {
        self.data.len()
    }

    // Note: we also rely on this to make us iterable!
    pub fn __getitem__(&self, py: Python, index: &Bound<PyAny>) -> PyResult<PyObject> {
        // Internal helper function to get a specific
        // instruction by index.
        fn get_at(
            self_: &CircuitData,
            py: Python<'_>,
            index: isize,
        ) -> PyResult<Py<CircuitInstruction>> {
            let index = self_.convert_py_index(index)?;
            if let Some(inst) = self_.data.get(index) {
                let qubits = self_.qargs_interner.intern(inst.qubits_id);
                let clbits = self_.cargs_interner.intern(inst.clbits_id);
                CircuitInstruction::new(
                    py,
                    inst.op.clone_ref(py),
                    self_.qubits.map_indices(qubits.value),
                    self_.clbits.map_indices(clbits.value),
                )
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
                    let mut s = self.convert_py_slice(&slice)?;
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
        value: &Bound<PyAny>,
    ) -> PyResult<()> {
        match index {
            SliceOrInt::Slice(slice) => {
                let indices = slice.indices(self.data.len().try_into().unwrap())?;
                let slice = self.convert_py_slice(&slice)?;
                let values = value.iter()?.collect::<PyResult<Vec<Bound<PyAny>>>>()?;
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
                    let slice = PySlice::new_bound(
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
                mem::swap(&mut packed, &mut self.data[index]);
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
        let item = self.__getitem__(py, index.bind(py))?;
        self.__delitem__(index.bind(py).extract()?)?;
        Ok(item)
    }

    pub fn append(&mut self, py: Python<'_>, value: PyRef<CircuitInstruction>) -> PyResult<()> {
        let packed = self.pack(py, value)?;
        self.data.push(packed);
        Ok(())
    }

    pub fn extend(&mut self, py: Python<'_>, itr: &Bound<PyAny>) -> PyResult<()> {
        if let Ok(other) = itr.extract::<PyRef<CircuitData>>() {
            // Fast path to avoid unnecessary construction of
            // CircuitInstruction instances.
            self.data.reserve(other.data.len());
            for inst in other.data.iter() {
                let qubits = other
                    .qargs_interner
                    .intern(inst.qubits_id)
                    .value
                    .iter()
                    .map(|b| {
                        Ok(self
                            .qubits
                            .find(other.qubits.get(*b).unwrap().bind(py))
                            .unwrap())
                    })
                    .collect::<PyResult<Vec<Qubit>>>()?;
                let clbits = other
                    .cargs_interner
                    .intern(inst.clbits_id)
                    .value
                    .iter()
                    .map(|b| {
                        Ok(self
                            .clbits
                            .find(other.clbits.get(*b).unwrap().bind(py))
                            .unwrap())
                    })
                    .collect::<PyResult<Vec<Clbit>>>()?;

                let qubits_id =
                    Interner::intern(&mut self.qargs_interner, InternerKey::Value(qubits))?;
                let clbits_id =
                    Interner::intern(&mut self.cargs_interner, InternerKey::Value(clbits))?;
                self.data.push(PackedInstruction {
                    op: inst.op.clone_ref(py),
                    qubits_id: qubits_id.index,
                    clbits_id: clbits_id.index,
                });
            }
            return Ok(());
        }

        for v in itr.iter()? {
            self.append(py, v?.extract()?)?;
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

    fn __eq__(slf: &Bound<Self>, other: &Bound<PyAny>) -> PyResult<bool> {
        let slf = slf.as_any();
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
        for bit in self.qubits.bits().iter().chain(self.clbits.bits().iter()) {
            visit.call(bit)?;
        }

        // Note:
        //   There's no need to visit the native Rust data
        //   structures used for internal tracking: the only Python
        //   references they contain are to the bits in these lists!
        visit.call(self.qubits.cached())?;
        visit.call(self.clbits.cached())?;
        Ok(())
    }

    fn __clear__(&mut self) {
        // Clear anything that could have a reference cycle.
        self.data.clear();
        self.qubits.dispose();
        self.clbits.dispose();
    }
}

impl CircuitData {
    /// Converts a Python slice to a `Vec` of indices into
    /// the instruction listing, [CircuitData.data].
    fn convert_py_slice(&self, slice: &Bound<PySlice>) -> PyResult<Vec<isize>> {
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

    fn pack(
        &mut self,
        py: Python,
        value: PyRef<CircuitInstruction>,
    ) -> PyResult<PackedInstruction> {
        let qubits = Interner::intern(
            &mut self.qargs_interner,
            InternerKey::Value(self.qubits.map_bits(value.qubits.bind(py))?.collect()),
        )?;
        let clbits = Interner::intern(
            &mut self.cargs_interner,
            InternerKey::Value(self.clbits.map_bits(value.clbits.bind(py))?.collect()),
        )?;
        Ok(PackedInstruction {
            op: value.operation.clone_ref(py),
            qubits_id: qubits.index,
            clbits_id: clbits.index,
        })
    }
}
