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

#[cfg(feature = "cache_pygates")]
use std::cell::RefCell;

use crate::bit_data::BitData;
use crate::circuit_instruction::{CircuitInstruction, OperationFromPython};
use crate::imports::{BUILTIN_LIST, QUBIT};
use crate::interner::{IndexedInterner, Interner, InternerKey};
use crate::operations::{Operation, Param, StandardGate};
use crate::packed_instruction::PackedInstruction;
use crate::parameter_table::{ParamEntry, ParamTable, GLOBAL_PHASE_INDEX};
use crate::slice::{PySequenceIndex, SequenceIndex};
use crate::{Clbit, Qubit};

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple, PyType};
use pyo3::{intern, PyTraverseError, PyVisit};

use hashbrown::{HashMap, HashSet};
use smallvec::SmallVec;

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
    param_table: ParamTable,
    #[pyo3(get)]
    global_phase: Param,
}

impl CircuitData {
    /// An alternate constructor to build a new `CircuitData` from an iterator
    /// of standard gates. This can be used to build a circuit from a sequence
    /// of standard gates, such as for a `StandardGate` definition or circuit
    /// synthesis without needing to involve Python.
    ///
    /// This can be connected with the Python space
    /// QuantumCircuit.from_circuit_data() constructor to build a full
    /// QuantumCircuit from Rust.
    ///
    /// # Arguments
    ///
    /// * py: A GIL handle this is needed to instantiate Qubits in Python space
    /// * num_qubits: The number of qubits in the circuit. These will be created
    ///     in Python as loose bits without a register.
    /// * instructions: An iterator of the standard gate params and qubits to
    ///     add to the circuit
    /// * global_phase: The global phase to use for the circuit
    pub fn from_standard_gates<I>(
        py: Python,
        num_qubits: u32,
        instructions: I,
        global_phase: Param,
    ) -> PyResult<Self>
    where
        I: IntoIterator<Item = (StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>)>,
    {
        let instruction_iter = instructions.into_iter();
        let mut res = CircuitData {
            data: Vec::with_capacity(instruction_iter.size_hint().0),
            qargs_interner: IndexedInterner::new(),
            cargs_interner: IndexedInterner::new(),
            qubits: BitData::new(py, "qubits".to_string()),
            clbits: BitData::new(py, "clbits".to_string()),
            param_table: ParamTable::new(),
            global_phase,
        };
        if num_qubits > 0 {
            let qubit_cls = QUBIT.get_bound(py);
            for _i in 0..num_qubits {
                let bit = qubit_cls.call0()?;
                res.add_qubit(py, &bit, true)?;
            }
        }
        let no_clbit_index = (&mut res.cargs_interner)
            .intern(InternerKey::Value(Vec::new()))?
            .index;
        for (operation, params, qargs) in instruction_iter {
            let qubits = (&mut res.qargs_interner)
                .intern(InternerKey::Value(qargs.to_vec()))?
                .index;
            let params = (!params.is_empty()).then(|| Box::new(params));
            res.data.push(PackedInstruction {
                op: operation.into(),
                qubits,
                clbits: no_clbit_index,
                params,
                extra_attrs: None,
                #[cfg(feature = "cache_pygates")]
                py_op: RefCell::new(None),
            });
        }
        Ok(res)
    }

    fn handle_manual_params(
        &mut self,
        py: Python,
        inst_index: usize,
        params: &[(usize, Vec<PyObject>)],
    ) -> PyResult<bool> {
        let mut new_param = false;
        let mut atomic_parameters: HashMap<u128, PyObject> = HashMap::new();
        for (param_index, raw_param_objs) in params {
            raw_param_objs.iter().for_each(|x| {
                atomic_parameters.insert(
                    x.getattr(py, intern!(py, "_uuid"))
                        .expect("Not a parameter")
                        .getattr(py, intern!(py, "int"))
                        .expect("Not a uuid")
                        .extract::<u128>(py)
                        .unwrap(),
                    x.clone_ref(py),
                );
            });
            for (param_uuid, param_obj) in atomic_parameters.iter() {
                match self.param_table.table.get_mut(param_uuid) {
                    Some(entry) => entry.add(inst_index, *param_index),
                    None => {
                        new_param = true;
                        let new_entry = ParamEntry::new(inst_index, *param_index);
                        self.param_table
                            .insert(py, param_obj.clone_ref(py), new_entry)?;
                    }
                };
            }
            atomic_parameters.clear()
        }
        Ok(new_param)
    }

    /// Add an instruction's entries to the parameter table
    fn update_param_table(
        &mut self,
        py: Python,
        inst_index: usize,
        params: Option<Vec<(usize, Vec<PyObject>)>>,
    ) -> PyResult<bool> {
        if let Some(params) = params {
            return self.handle_manual_params(py, inst_index, &params);
        }
        // Update the parameter table
        let mut new_param = false;
        let inst_params = self.data[inst_index].params_view();
        if !inst_params.is_empty() {
            let params: Vec<(usize, PyObject)> = inst_params
                .iter()
                .enumerate()
                .filter_map(|(idx, x)| match x {
                    Param::ParameterExpression(param_obj) => Some((idx, param_obj.clone_ref(py))),
                    _ => None,
                })
                .collect();
            if !params.is_empty() {
                let list_builtin = BUILTIN_LIST.get_bound(py);
                let mut atomic_parameters: HashMap<u128, PyObject> = HashMap::new();
                for (param_index, param) in &params {
                    let temp: PyObject = param.getattr(py, intern!(py, "parameters"))?;
                    let raw_param_objs: Vec<PyObject> = list_builtin.call1((temp,))?.extract()?;
                    raw_param_objs.iter().for_each(|x| {
                        atomic_parameters.insert(
                            x.getattr(py, intern!(py, "_uuid"))
                                .expect("Not a parameter")
                                .getattr(py, intern!(py, "int"))
                                .expect("Not a uuid")
                                .extract(py)
                                .unwrap(),
                            x.clone_ref(py),
                        );
                    });
                    for (param_uuid, param_obj) in &atomic_parameters {
                        match self.param_table.table.get_mut(param_uuid) {
                            Some(entry) => entry.add(inst_index, *param_index),
                            None => {
                                new_param = true;
                                let new_entry = ParamEntry::new(inst_index, *param_index);
                                self.param_table
                                    .insert(py, param_obj.clone_ref(py), new_entry)?;
                            }
                        };
                    }
                    atomic_parameters.clear();
                }
            }
        }
        Ok(new_param)
    }

    /// Remove an index's entries from the parameter table.
    fn remove_from_parameter_table(&mut self, py: Python, inst_index: usize) -> PyResult<()> {
        let list_builtin = BUILTIN_LIST.get_bound(py);
        if inst_index == GLOBAL_PHASE_INDEX {
            if let Param::ParameterExpression(global_phase) = &self.global_phase {
                let temp: PyObject = global_phase.getattr(py, intern!(py, "parameters"))?;
                let raw_param_objs: Vec<PyObject> = list_builtin.call1((temp,))?.extract()?;
                for (param_index, param_obj) in raw_param_objs.iter().enumerate() {
                    let uuid: u128 = param_obj
                        .getattr(py, intern!(py, "_uuid"))?
                        .getattr(py, intern!(py, "int"))?
                        .extract(py)?;
                    let name: String = param_obj.getattr(py, intern!(py, "name"))?.extract(py)?;
                    self.param_table
                        .discard_references(uuid, inst_index, param_index, name);
                }
            }
        } else if !self.data[inst_index].params_view().is_empty() {
            let params: Vec<(usize, PyObject)> = self.data[inst_index]
                .params_view()
                .iter()
                .enumerate()
                .filter_map(|(idx, x)| match x {
                    Param::ParameterExpression(param_obj) => Some((idx, param_obj.clone_ref(py))),
                    _ => None,
                })
                .collect();
            if !params.is_empty() {
                for (param_index, param) in &params {
                    let temp: PyObject = param.getattr(py, intern!(py, "parameters"))?;
                    let raw_param_objs: Vec<PyObject> = list_builtin.call1((temp,))?.extract()?;
                    let mut atomic_parameters: HashSet<(u128, String)> =
                        HashSet::with_capacity(params.len());
                    for x in raw_param_objs {
                        let uuid = x
                            .getattr(py, intern!(py, "_uuid"))?
                            .getattr(py, intern!(py, "int"))?
                            .extract(py)?;
                        let name = x.getattr(py, intern!(py, "name"))?.extract(py)?;
                        atomic_parameters.insert((uuid, name));
                    }
                    for (uuid, name) in atomic_parameters {
                        self.param_table
                            .discard_references(uuid, inst_index, *param_index, name);
                    }
                }
            }
        }
        Ok(())
    }

    fn reindex_parameter_table(&mut self, py: Python) -> PyResult<()> {
        self.param_table.clear();

        for inst_index in 0..self.data.len() {
            self.update_param_table(py, inst_index, None)?;
        }
        // Technically we could keep the global phase entry directly if it exists, but we're
        // the incremental cost is minimal after reindexing everything.
        self.global_phase(py, self.global_phase.clone())?;
        Ok(())
    }

    pub fn append_inner(&mut self, py: Python, value: &CircuitInstruction) -> PyResult<bool> {
        let packed = self.pack(py, value)?;
        let new_index = self.data.len();
        self.data.push(packed);
        self.update_param_table(py, new_index, None)
    }
}

#[pymethods]
impl CircuitData {
    #[new]
    #[pyo3(signature = (qubits=None, clbits=None, data=None, reserve=0, global_phase=Param::Float(0.0)))]
    pub fn new(
        py: Python<'_>,
        qubits: Option<&Bound<PyAny>>,
        clbits: Option<&Bound<PyAny>>,
        data: Option<&Bound<PyAny>>,
        reserve: usize,
        global_phase: Param,
    ) -> PyResult<Self> {
        let mut self_ = CircuitData {
            data: Vec::new(),
            qargs_interner: IndexedInterner::new(),
            cargs_interner: IndexedInterner::new(),
            qubits: BitData::new(py, "qubits".to_string()),
            clbits: BitData::new(py, "clbits".to_string()),
            param_table: ParamTable::new(),
            global_phase: Param::Float(0.),
        };
        self_.global_phase(py, global_phase)?;
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

    /// Return the number of qubits. This is equivalent to the length of the list returned by
    /// :meth:`.CircuitData.qubits`
    ///
    /// Returns:
    ///     int: The number of qubits.
    #[getter]
    pub fn num_qubits(&self) -> usize {
        self.qubits.len()
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

    /// Return the number of clbits. This is equivalent to the length of the list returned by
    /// :meth:`.CircuitData.clbits`.
    ///
    /// Returns:
    ///     int: The number of clbits.
    #[getter]
    pub fn num_clbits(&self) -> usize {
        self.clbits.len()
    }

    /// Return the width of the circuit. This is the number of qubits plus the
    /// number of clbits.
    ///
    /// Returns:
    ///     int: The width of the circuit.
    pub fn width(&self) -> usize {
        self.num_qubits() + self.num_clbits()
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
    #[pyo3(signature = (copy_instructions=true, deepcopy=false))]
    pub fn copy(&self, py: Python<'_>, copy_instructions: bool, deepcopy: bool) -> PyResult<Self> {
        let mut res = CircuitData::new(
            py,
            Some(self.qubits.cached().bind(py)),
            Some(self.clbits.cached().bind(py)),
            None,
            self.data.len(),
            self.global_phase.clone(),
        )?;
        res.qargs_interner = self.qargs_interner.clone();
        res.cargs_interner = self.cargs_interner.clone();
        res.param_table.clone_from(&self.param_table);

        if deepcopy {
            let memo = PyDict::new_bound(py);
            for inst in &self.data {
                res.data.push(PackedInstruction {
                    op: inst.op.py_deepcopy(py, Some(&memo))?,
                    qubits: inst.qubits,
                    clbits: inst.clbits,
                    params: inst.params.clone(),
                    extra_attrs: inst.extra_attrs.clone(),
                    #[cfg(feature = "cache_pygates")]
                    py_op: RefCell::new(None),
                });
            }
        } else if copy_instructions {
            for inst in &self.data {
                res.data.push(PackedInstruction {
                    op: inst.op.py_copy(py)?,
                    qubits: inst.qubits,
                    clbits: inst.clbits,
                    params: inst.params.clone(),
                    extra_attrs: inst.extra_attrs.clone(),
                    #[cfg(feature = "cache_pygates")]
                    py_op: RefCell::new(None),
                });
            }
        } else {
            res.data.extend(self.data.iter().cloned());
        }
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
            for b in self.qargs_interner.intern(inst.qubits).value.iter() {
                qubits.add(self.qubits.get(*b).unwrap().clone_ref(py))?;
            }
            for b in self.cargs_interner.intern(inst.clbits).value.iter() {
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
            func.call1((inst.unpack_py_op(py)?,))?;
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
            func.call1((index, inst.unpack_py_op(py)?))?;
        }
        Ok(())
    }

    /// Invokes callable ``func`` with each instruction's operation, replacing the operation with
    /// the result, if the operation is not a standard gate without a condition.
    ///
    /// .. warning::
    ///
    ///     This is a shim for while there are still important components of the circuit still
    ///     implemented in Python space.  This method **skips** any instruction that contains an
    ///     non-conditional standard gate (which is likely to be most instructions).
    ///
    /// Args:
    ///     func (Callable[[:class:`~.Operation`], :class:`~.Operation`]):
    ///         A callable used to map original operations to their replacements.
    #[pyo3(signature = (func))]
    pub fn map_nonstandard_ops(&mut self, py: Python<'_>, func: &Bound<PyAny>) -> PyResult<()> {
        for inst in self.data.iter_mut() {
            if inst.op.try_standard_gate().is_some()
                && !inst
                    .extra_attrs
                    .as_ref()
                    .is_some_and(|attrs| attrs.condition.is_some())
            {
                continue;
            }
            let py_op = func.call1((inst.unpack_py_op(py)?,))?;
            let result = py_op.extract::<OperationFromPython>()?;
            inst.op = result.operation;
            inst.params = (!result.params.is_empty()).then(|| Box::new(result.params));
            inst.extra_attrs = result.extra_attrs;
            #[cfg(feature = "cache_pygates")]
            {
                *inst.py_op.borrow_mut() = Some(py_op.unbind());
            }
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
    ///     :meth:`~.CircuitData.map_nonstandard_ops` to adjust the operations manually
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
        let mut temp = CircuitData::new(py, qubits, clbits, None, 0, self.global_phase.clone())?;
        if qubits.is_some() {
            if temp.num_qubits() < self.num_qubits() {
                return Err(PyValueError::new_err(format!(
                    "Replacement 'qubits' of size {:?} must contain at least {:?} bits.",
                    temp.num_qubits(),
                    self.num_qubits(),
                )));
            }
            std::mem::swap(&mut temp.qubits, &mut self.qubits);
        }
        if clbits.is_some() {
            if temp.num_clbits() < self.num_clbits() {
                return Err(PyValueError::new_err(format!(
                    "Replacement 'clbits' of size {:?} must contain at least {:?} bits.",
                    temp.num_clbits(),
                    self.num_clbits(),
                )));
            }
            std::mem::swap(&mut temp.clbits, &mut self.clbits);
        }
        Ok(())
    }

    pub fn __len__(&self) -> usize {
        self.data.len()
    }

    // Note: we also rely on this to make us iterable!
    pub fn __getitem__(&self, py: Python, index: PySequenceIndex) -> PyResult<PyObject> {
        // Get a single item, assuming the index is validated as in bounds.
        let get_single = |index: usize| {
            let inst = &self.data[index];
            let qubits = self.qargs_interner.intern(inst.qubits);
            let clbits = self.cargs_interner.intern(inst.clbits);
            CircuitInstruction {
                operation: inst.op.clone(),
                qubits: PyTuple::new_bound(py, self.qubits.map_indices(qubits.value)).unbind(),
                clbits: PyTuple::new_bound(py, self.clbits.map_indices(clbits.value)).unbind(),
                params: inst.params_view().iter().cloned().collect(),
                extra_attrs: inst.extra_attrs.clone(),
                #[cfg(feature = "cache_pygates")]
                py_op: inst.py_op.clone(),
            }
            .into_py(py)
        };
        match index.with_len(self.data.len())? {
            SequenceIndex::Int(index) => Ok(get_single(index)),
            indices => Ok(PyList::new_bound(py, indices.iter().map(get_single)).into_py(py)),
        }
    }

    pub fn __delitem__(&mut self, py: Python, index: PySequenceIndex) -> PyResult<()> {
        self.delitem(py, index.with_len(self.data.len())?)
    }

    pub fn setitem_no_param_table_update(
        &mut self,
        py: Python,
        index: usize,
        value: &CircuitInstruction,
    ) -> PyResult<()> {
        self.data[index] = self.pack(py, value)?;
        Ok(())
    }

    pub fn __setitem__(&mut self, index: PySequenceIndex, value: &Bound<PyAny>) -> PyResult<()> {
        fn set_single(slf: &mut CircuitData, index: usize, value: &Bound<PyAny>) -> PyResult<()> {
            let py = value.py();
            slf.data[index] = slf.pack(py, &value.downcast::<CircuitInstruction>()?.borrow())?;
            slf.remove_from_parameter_table(py, index)?;
            slf.update_param_table(py, index, None)?;
            Ok(())
        }

        let py = value.py();
        match index.with_len(self.data.len())? {
            SequenceIndex::Int(index) => set_single(self, index, value),
            indices @ SequenceIndex::PosRange {
                start,
                stop,
                step: 1,
            } => {
                // `list` allows setting a slice with step +1 to an arbitrary length.
                let values = value.iter()?.collect::<PyResult<Vec<_>>>()?;
                for (index, value) in indices.iter().zip(values.iter()) {
                    set_single(self, index, value)?;
                }
                if indices.len() > values.len() {
                    self.delitem(
                        py,
                        SequenceIndex::PosRange {
                            start: start + values.len(),
                            stop,
                            step: 1,
                        },
                    )?
                } else {
                    for value in values[indices.len()..].iter().rev() {
                        self.insert(stop as isize, value.downcast()?.borrow())?;
                    }
                }
                Ok(())
            }
            indices => {
                let values = value.iter()?.collect::<PyResult<Vec<_>>>()?;
                if indices.len() == values.len() {
                    for (index, value) in indices.iter().zip(values.iter()) {
                        set_single(self, index, value)?;
                    }
                    Ok(())
                } else {
                    Err(PyValueError::new_err(format!(
                        "attempt to assign sequence of size {:?} to extended slice of size {:?}",
                        values.len(),
                        indices.len(),
                    )))
                }
            }
        }
    }

    pub fn insert(&mut self, mut index: isize, value: PyRef<CircuitInstruction>) -> PyResult<()> {
        // `list.insert` has special-case extra clamping logic for its index argument.
        let index = {
            if index < 0 {
                // This can't exceed `isize::MAX` because `self.data[0]` is larger than a byte.
                index += self.data.len() as isize;
            }
            if index < 0 {
                0
            } else if index as usize > self.data.len() {
                self.data.len()
            } else {
                index as usize
            }
        };
        let py = value.py();
        let packed = self.pack(py, &value)?;
        self.data.insert(index, packed);
        if index == self.data.len() - 1 {
            self.update_param_table(py, index, None)?;
        } else {
            self.reindex_parameter_table(py)?;
        }
        Ok(())
    }

    pub fn pop(&mut self, py: Python<'_>, index: Option<PySequenceIndex>) -> PyResult<PyObject> {
        let index = index.unwrap_or(PySequenceIndex::Int(-1));
        let native_index = index.with_len(self.data.len())?;
        let item = self.__getitem__(py, index)?;
        self.delitem(py, native_index)?;
        Ok(item)
    }

    pub fn append(
        &mut self,
        py: Python<'_>,
        value: &Bound<CircuitInstruction>,
        params: Option<Vec<(usize, Vec<PyObject>)>>,
    ) -> PyResult<bool> {
        let new_index = self.data.len();
        let packed = self.pack(py, &value.borrow())?;
        self.data.push(packed);
        self.update_param_table(py, new_index, params)
    }

    pub fn extend(&mut self, py: Python<'_>, itr: &Bound<PyAny>) -> PyResult<()> {
        if let Ok(other) = itr.downcast::<CircuitData>() {
            let other = other.borrow();
            // Fast path to avoid unnecessary construction of CircuitInstruction instances.
            self.data.reserve(other.data.len());
            for inst in other.data.iter() {
                let qubits = other
                    .qargs_interner
                    .intern(inst.qubits)
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
                    .intern(inst.clbits)
                    .value
                    .iter()
                    .map(|b| {
                        Ok(self
                            .clbits
                            .find(other.clbits.get(*b).unwrap().bind(py))
                            .unwrap())
                    })
                    .collect::<PyResult<Vec<Clbit>>>()?;
                let new_index = self.data.len();
                let qubits_id =
                    Interner::intern(&mut self.qargs_interner, InternerKey::Value(qubits))?;
                let clbits_id =
                    Interner::intern(&mut self.cargs_interner, InternerKey::Value(clbits))?;
                self.data.push(PackedInstruction {
                    op: inst.op.clone(),
                    qubits: qubits_id.index,
                    clbits: clbits_id.index,
                    params: inst.params.clone(),
                    extra_attrs: inst.extra_attrs.clone(),
                    #[cfg(feature = "cache_pygates")]
                    py_op: inst.py_op.clone(),
                });
                self.update_param_table(py, new_index, None)?;
            }
            return Ok(());
        }
        for v in itr.iter()? {
            self.append_inner(py, &v?.downcast()?.borrow())?;
        }
        Ok(())
    }

    pub fn clear(&mut self, _py: Python<'_>) -> PyResult<()> {
        std::mem::take(&mut self.data);
        self.param_table.clear();
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

    #[setter]
    pub fn global_phase(&mut self, py: Python, angle: Param) -> PyResult<()> {
        let list_builtin = BUILTIN_LIST.get_bound(py);
        self.remove_from_parameter_table(py, GLOBAL_PHASE_INDEX)?;
        match angle {
            Param::Float(angle) => {
                self.global_phase = Param::Float(angle.rem_euclid(2. * std::f64::consts::PI));
            }
            Param::ParameterExpression(angle) => {
                let temp: PyObject = angle.getattr(py, intern!(py, "parameters"))?;
                let raw_param_objs: Vec<PyObject> = list_builtin.call1((temp,))?.extract()?;

                for (param_index, param_obj) in raw_param_objs.into_iter().enumerate() {
                    let param_uuid: u128 = param_obj
                        .getattr(py, intern!(py, "_uuid"))?
                        .getattr(py, intern!(py, "int"))?
                        .extract(py)?;
                    match self.param_table.table.get_mut(&param_uuid) {
                        Some(entry) => entry.add(GLOBAL_PHASE_INDEX, param_index),
                        None => {
                            let new_entry = ParamEntry::new(GLOBAL_PHASE_INDEX, param_index);
                            self.param_table.insert(py, param_obj, new_entry)?;
                        }
                    };
                }
                self.global_phase = Param::ParameterExpression(angle);
            }
            Param::Obj(_) => return Err(PyValueError::new_err("Invalid type for global phase")),
        };
        Ok(())
    }

    /// Get the global_phase sentinel value
    #[classattr]
    pub const fn global_phase_param_index() -> usize {
        GLOBAL_PHASE_INDEX
    }

    // Below are functions to interact with the parameter table. These methods
    // are done to avoid needing to deal with shared references and provide
    // an entry point via python through an owned CircuitData object.
    pub fn num_params(&self) -> usize {
        self.param_table.table.len()
    }

    pub fn get_param_from_name(&self, py: Python, name: String) -> Option<PyObject> {
        self.param_table.get_param_from_name(py, name)
    }

    pub fn get_params_unsorted(&self, py: Python) -> PyResult<Py<PySet>> {
        Ok(PySet::new_bound(py, self.param_table.uuid_map.values())?.unbind())
    }

    pub fn pop_param(&mut self, py: Python, uuid: u128, name: &str, default: PyObject) -> PyObject {
        match self.param_table.pop(uuid, name) {
            Some(res) => res.into_py(py),
            None => default.clone_ref(py),
        }
    }

    pub fn _get_param(&self, py: Python, uuid: u128) -> PyObject {
        self.param_table.table[&uuid].clone().into_py(py)
    }

    pub fn contains_param(&self, uuid: u128) -> bool {
        self.param_table.table.contains_key(&uuid)
    }

    pub fn add_new_parameter(
        &mut self,
        py: Python,
        param: PyObject,
        inst_index: usize,
        param_index: usize,
    ) -> PyResult<()> {
        self.param_table.insert(
            py,
            param.clone_ref(py),
            ParamEntry::new(inst_index, param_index),
        )?;
        Ok(())
    }

    pub fn update_parameter_entry(
        &mut self,
        uuid: u128,
        inst_index: usize,
        param_index: usize,
    ) -> PyResult<()> {
        match self.param_table.table.get_mut(&uuid) {
            Some(entry) => {
                entry.add(inst_index, param_index);
                Ok(())
            }
            None => Err(PyIndexError::new_err(format!(
                "Invalid parameter uuid: {:?}",
                uuid
            ))),
        }
    }

    pub fn _get_entry_count(&self, py: Python, param_obj: PyObject) -> PyResult<usize> {
        let uuid: u128 = param_obj
            .getattr(py, intern!(py, "_uuid"))?
            .getattr(py, intern!(py, "int"))?
            .extract(py)?;
        Ok(self.param_table.table[&uuid].index_ids.len())
    }

    pub fn num_nonlocal_gates(&self) -> usize {
        self.data
            .iter()
            .filter(|inst| inst.op().num_qubits() > 1 && !inst.op().directive())
            .count()
    }
}

impl CircuitData {
    /// Native internal driver of `__delitem__` that uses a Rust-space version of the
    /// `SequenceIndex`.  This assumes that the `SequenceIndex` contains only in-bounds indices, and
    /// panics if not.
    fn delitem(&mut self, py: Python, indices: SequenceIndex) -> PyResult<()> {
        // We need to delete in reverse order so we don't invalidate higher indices with a deletion.
        for index in indices.descending() {
            self.data.remove(index);
        }
        if !indices.is_empty() {
            self.reindex_parameter_table(py)?;
        }
        Ok(())
    }

    fn pack(&mut self, py: Python, inst: &CircuitInstruction) -> PyResult<PackedInstruction> {
        let qubits = Interner::intern(
            &mut self.qargs_interner,
            InternerKey::Value(self.qubits.map_bits(inst.qubits.bind(py))?.collect()),
        )?;
        let clbits = Interner::intern(
            &mut self.cargs_interner,
            InternerKey::Value(self.clbits.map_bits(inst.clbits.bind(py))?.collect()),
        )?;
        Ok(PackedInstruction {
            op: inst.operation.clone(),
            qubits: qubits.index,
            clbits: clbits.index,
            params: (!inst.params.is_empty()).then(|| Box::new(inst.params.clone())),
            extra_attrs: inst.extra_attrs.clone(),
            #[cfg(feature = "cache_pygates")]
            py_op: RefCell::new(inst.py_op.borrow().as_ref().map(|obj| obj.clone_ref(py))),
        })
    }

    /// Returns an iterator over all the instructions present in the circuit.
    pub fn iter(&self) -> impl Iterator<Item = &PackedInstruction> {
        self.data.iter()
    }
}
