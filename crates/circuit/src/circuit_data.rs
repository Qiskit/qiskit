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

use crate::bit_data::BitData;
use crate::circuit_instruction::{
    convert_py_to_operation_type, operation_type_and_data_to_py, CircuitInstruction,
    ExtraInstructionAttributes, OperationInput, PackedInstruction,
};
use crate::imports::{BUILTIN_LIST, QUBIT};
use crate::interner::{IndexedInterner, Interner, InternerKey};
use crate::operations::{Operation, OperationType, Param, StandardGate};
use crate::parameter_table::{ParamEntry, ParamTable};
use crate::{Clbit, Qubit, SliceOrInt};

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySet, PySlice, PyTuple, PyType};
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
        for (operation, params, qargs) in instruction_iter {
            let qubits = PyTuple::new_bound(py, res.qubits.map_indices(&qargs)).unbind();
            let clbits = PyTuple::empty_bound(py).unbind();
            let inst = res.pack_owned(
                py,
                &CircuitInstruction {
                    operation: OperationType::Standard(operation),
                    qubits,
                    clbits,
                    params,
                    extra_attrs: None,
                    #[cfg(feature = "cache_pygates")]
                    py_op: None,
                },
            )?;
            res.data.push(inst);
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
        let inst_params = &self.data[inst_index].params;
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
        if inst_index == usize::MAX {
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
        } else if !self.data[inst_index].params.is_empty() {
            let params: Vec<(usize, PyObject)> = self.data[inst_index]
                .params
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

    pub fn append_inner(&mut self, py: Python, value: PyRef<CircuitInstruction>) -> PyResult<bool> {
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
            0,
            self.global_phase.clone(),
        )?;
        res.qargs_interner = self.qargs_interner.clone();
        res.cargs_interner = self.cargs_interner.clone();
        res.data.clone_from(&self.data);
        res.param_table.clone_from(&self.param_table);

        if deepcopy {
            let deepcopy = py
                .import_bound(intern!(py, "copy"))?
                .getattr(intern!(py, "deepcopy"))?;
            for inst in &mut res.data {
                match &mut inst.op {
                    OperationType::Standard(_) => {
                        #[cfg(feature = "cache_pygates")]
                        {
                            inst.py_op = None;
                        }
                    }
                    OperationType::Gate(ref mut op) => {
                        op.gate = deepcopy.call1((&op.gate,))?.unbind();
                        #[cfg(feature = "cache_pygates")]
                        {
                            inst.py_op = None;
                        }
                    }
                    OperationType::Instruction(ref mut op) => {
                        op.instruction = deepcopy.call1((&op.instruction,))?.unbind();
                        #[cfg(feature = "cache_pygates")]
                        {
                            inst.py_op = None;
                        }
                    }
                    OperationType::Operation(ref mut op) => {
                        op.operation = deepcopy.call1((&op.operation,))?.unbind();
                        #[cfg(feature = "cache_pygates")]
                        {
                            inst.py_op = None;
                        }
                    }
                };
            }
        } else if copy_instructions {
            for inst in &mut res.data {
                match &mut inst.op {
                    OperationType::Standard(_) => {
                        #[cfg(feature = "cache_pygates")]
                        {
                            inst.py_op = None;
                        }
                    }
                    OperationType::Gate(ref mut op) => {
                        op.gate = op.gate.call_method0(py, intern!(py, "copy"))?;
                        #[cfg(feature = "cache_pygates")]
                        {
                            inst.py_op = None;
                        }
                    }
                    OperationType::Instruction(ref mut op) => {
                        op.instruction = op.instruction.call_method0(py, intern!(py, "copy"))?;
                        #[cfg(feature = "cache_pygates")]
                        {
                            inst.py_op = None;
                        }
                    }
                    OperationType::Operation(ref mut op) => {
                        op.operation = op.operation.call_method0(py, intern!(py, "copy"))?;
                        #[cfg(feature = "cache_pygates")]
                        {
                            inst.py_op = None;
                        }
                    }
                };
            }
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
    #[cfg(not(feature = "cache_pygates"))]
    #[pyo3(signature = (func))]
    pub fn foreach_op(&self, py: Python<'_>, func: &Bound<PyAny>) -> PyResult<()> {
        for inst in self.data.iter() {
            let label;
            let duration;
            let unit;
            let condition;
            match &inst.extra_attrs {
                Some(extra_attrs) => {
                    label = &extra_attrs.label;
                    duration = &extra_attrs.duration;
                    unit = &extra_attrs.unit;
                    condition = &extra_attrs.condition;
                }
                None => {
                    label = &None;
                    duration = &None;
                    unit = &None;
                    condition = &None;
                }
            }

            let op = operation_type_and_data_to_py(
                py,
                &inst.op,
                &inst.params,
                label,
                duration,
                unit,
                condition,
            )?;
            func.call1((op,))?;
        }
        Ok(())
    }

    /// Invokes callable ``func`` with each instruction's operation.
    ///
    /// Args:
    ///     func (Callable[[:class:`~.Operation`], None]):
    ///         The callable to invoke.
    #[cfg(feature = "cache_pygates")]
    #[pyo3(signature = (func))]
    pub fn foreach_op(&mut self, py: Python<'_>, func: &Bound<PyAny>) -> PyResult<()> {
        for inst in self.data.iter_mut() {
            let op = match &inst.py_op {
                Some(op) => op.clone_ref(py),
                None => {
                    let label;
                    let duration;
                    let unit;
                    let condition;
                    match &inst.extra_attrs {
                        Some(extra_attrs) => {
                            label = &extra_attrs.label;
                            duration = &extra_attrs.duration;
                            unit = &extra_attrs.unit;
                            condition = &extra_attrs.condition;
                        }
                        None => {
                            label = &None;
                            duration = &None;
                            unit = &None;
                            condition = &None;
                        }
                    }
                    let new_op = operation_type_and_data_to_py(
                        py,
                        &inst.op,
                        &inst.params,
                        label,
                        duration,
                        unit,
                        condition,
                    )?;
                    inst.py_op = Some(new_op.clone_ref(py));
                    new_op
                }
            };
            func.call1((op,))?;
        }
        Ok(())
    }

    /// Invokes callable ``func`` with the positional index and operation
    /// of each instruction.
    ///
    /// Args:
    ///     func (Callable[[int, :class:`~.Operation`], None]):
    ///         The callable to invoke.
    #[cfg(not(feature = "cache_pygates"))]
    #[pyo3(signature = (func))]
    pub fn foreach_op_indexed(&self, py: Python<'_>, func: &Bound<PyAny>) -> PyResult<()> {
        for (index, inst) in self.data.iter().enumerate() {
            let label;
            let duration;
            let unit;
            let condition;
            match &inst.extra_attrs {
                Some(extra_attrs) => {
                    label = &extra_attrs.label;
                    duration = &extra_attrs.duration;
                    unit = &extra_attrs.unit;
                    condition = &extra_attrs.condition;
                }
                None => {
                    label = &None;
                    duration = &None;
                    unit = &None;
                    condition = &None;
                }
            }

            let op = operation_type_and_data_to_py(
                py,
                &inst.op,
                &inst.params,
                label,
                duration,
                unit,
                condition,
            )?;
            func.call1((index, op))?;
        }
        Ok(())
    }

    /// Invokes callable ``func`` with the positional index and operation
    /// of each instruction.
    ///
    /// Args:
    ///     func (Callable[[int, :class:`~.Operation`], None]):
    ///         The callable to invoke.
    #[cfg(feature = "cache_pygates")]
    #[pyo3(signature = (func))]
    pub fn foreach_op_indexed(&mut self, py: Python<'_>, func: &Bound<PyAny>) -> PyResult<()> {
        for (index, inst) in self.data.iter_mut().enumerate() {
            let op = match &inst.py_op {
                Some(op) => op.clone_ref(py),
                None => {
                    let label;
                    let duration;
                    let unit;
                    let condition;
                    match &inst.extra_attrs {
                        Some(extra_attrs) => {
                            label = &extra_attrs.label;
                            duration = &extra_attrs.duration;
                            unit = &extra_attrs.unit;
                            condition = &extra_attrs.condition;
                        }
                        None => {
                            label = &None;
                            duration = &None;
                            unit = &None;
                            condition = &None;
                        }
                    }
                    let new_op = operation_type_and_data_to_py(
                        py,
                        &inst.op,
                        &inst.params,
                        label,
                        duration,
                        unit,
                        condition,
                    )?;
                    inst.py_op = Some(new_op.clone_ref(py));
                    new_op
                }
            };
            func.call1((index, op))?;
        }
        Ok(())
    }

    /// Invokes callable ``func`` with each instruction's operation,
    /// replacing the operation with the result.
    ///
    /// .. note::
    ///
    ///     This is only to be used by map_vars() in quantumcircuit.py it
    ///     assumes that a full Python instruction will only be returned from
    ///     standard gates iff a condition is set.
    ///
    /// Args:
    ///     func (Callable[[:class:`~.Operation`], :class:`~.Operation`]):
    ///         A callable used to map original operation to their
    ///         replacements.
    #[cfg(not(feature = "cache_pygates"))]
    #[pyo3(signature = (func))]
    pub fn map_ops(&mut self, py: Python<'_>, func: &Bound<PyAny>) -> PyResult<()> {
        for inst in self.data.iter_mut() {
            let old_op = match &inst.op {
                OperationType::Standard(op) => {
                    let label;
                    let duration;
                    let unit;
                    let condition;
                    match &inst.extra_attrs {
                        Some(extra_attrs) => {
                            label = &extra_attrs.label;
                            duration = &extra_attrs.duration;
                            unit = &extra_attrs.unit;
                            condition = &extra_attrs.condition;
                        }
                        None => {
                            label = &None;
                            duration = &None;
                            unit = &None;
                            condition = &None;
                        }
                    }
                    if condition.is_some() {
                        operation_type_and_data_to_py(
                            py,
                            &inst.op,
                            &inst.params,
                            label,
                            duration,
                            unit,
                            condition,
                        )?
                    } else {
                        op.into_py(py)
                    }
                }
                OperationType::Gate(op) => op.gate.clone_ref(py),
                OperationType::Instruction(op) => op.instruction.clone_ref(py),
                OperationType::Operation(op) => op.operation.clone_ref(py),
            };
            let result: OperationInput = func.call1((old_op,))?.extract()?;
            match result {
                OperationInput::Standard(op) => {
                    inst.op = OperationType::Standard(op);
                }
                OperationInput::Gate(op) => {
                    inst.op = OperationType::Gate(op);
                }
                OperationInput::Instruction(op) => {
                    inst.op = OperationType::Instruction(op);
                }
                OperationInput::Operation(op) => {
                    inst.op = OperationType::Operation(op);
                }
                OperationInput::Object(new_op) => {
                    let new_inst_details = convert_py_to_operation_type(py, new_op)?;
                    inst.op = new_inst_details.operation;
                    inst.params = new_inst_details.params;
                    if new_inst_details.label.is_some()
                        || new_inst_details.duration.is_some()
                        || new_inst_details.unit.is_some()
                        || new_inst_details.condition.is_some()
                    {
                        inst.extra_attrs = Some(Box::new(ExtraInstructionAttributes {
                            label: new_inst_details.label,
                            duration: new_inst_details.duration,
                            unit: new_inst_details.unit,
                            condition: new_inst_details.condition,
                        }))
                    }
                }
            }
        }
        Ok(())
    }

    /// Invokes callable ``func`` with each instruction's operation,
    /// replacing the operation with the result.
    ///
    /// .. note::
    ///
    ///     This is only to be used by map_vars() in quantumcircuit.py it
    ///     assumes that a full Python instruction will only be returned from
    ///     standard gates iff a condition is set.
    ///
    /// Args:
    ///     func (Callable[[:class:`~.Operation`], :class:`~.Operation`]):
    ///         A callable used to map original operation to their
    ///         replacements.
    #[cfg(feature = "cache_pygates")]
    #[pyo3(signature = (func))]
    pub fn map_ops(&mut self, py: Python<'_>, func: &Bound<PyAny>) -> PyResult<()> {
        for inst in self.data.iter_mut() {
            let old_op = match &inst.py_op {
                Some(op) => op.clone_ref(py),
                None => match &inst.op {
                    OperationType::Standard(op) => {
                        let label;
                        let duration;
                        let unit;
                        let condition;
                        match &inst.extra_attrs {
                            Some(extra_attrs) => {
                                label = &extra_attrs.label;
                                duration = &extra_attrs.duration;
                                unit = &extra_attrs.unit;
                                condition = &extra_attrs.condition;
                            }
                            None => {
                                label = &None;
                                duration = &None;
                                unit = &None;
                                condition = &None;
                            }
                        }
                        if condition.is_some() {
                            let new_op = operation_type_and_data_to_py(
                                py,
                                &inst.op,
                                &inst.params,
                                label,
                                duration,
                                unit,
                                condition,
                            )?;
                            inst.py_op = Some(new_op.clone_ref(py));
                            new_op
                        } else {
                            op.into_py(py)
                        }
                    }
                    OperationType::Gate(op) => op.gate.clone_ref(py),
                    OperationType::Instruction(op) => op.instruction.clone_ref(py),
                    OperationType::Operation(op) => op.operation.clone_ref(py),
                },
            };
            let result: OperationInput = func.call1((old_op,))?.extract()?;
            match result {
                OperationInput::Standard(op) => {
                    inst.op = OperationType::Standard(op);
                }
                OperationInput::Gate(op) => {
                    inst.op = OperationType::Gate(op);
                }
                OperationInput::Instruction(op) => {
                    inst.op = OperationType::Instruction(op);
                }
                OperationInput::Operation(op) => {
                    inst.op = OperationType::Operation(op);
                }
                OperationInput::Object(new_op) => {
                    let new_inst_details = convert_py_to_operation_type(py, new_op.clone_ref(py))?;
                    inst.op = new_inst_details.operation;
                    inst.params = new_inst_details.params;
                    if new_inst_details.label.is_some()
                        || new_inst_details.duration.is_some()
                        || new_inst_details.unit.is_some()
                        || new_inst_details.condition.is_some()
                    {
                        inst.extra_attrs = Some(Box::new(ExtraInstructionAttributes {
                            label: new_inst_details.label,
                            duration: new_inst_details.duration,
                            unit: new_inst_details.unit,
                            condition: new_inst_details.condition,
                        }))
                    }
                    inst.py_op = Some(new_op);
                }
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
                Py::new(
                    py,
                    CircuitInstruction::new(
                        py,
                        inst.op.clone(),
                        self_.qubits.map_indices(qubits.value),
                        self_.clbits.map_indices(clbits.value),
                        inst.params.clone(),
                        inst.extra_attrs.clone(),
                    ),
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

    pub fn __delitem__(&mut self, py: Python, index: SliceOrInt) -> PyResult<()> {
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
                    self.__delitem__(py, SliceOrInt::Int(i))?;
                }
                self.reindex_parameter_table(py)?;
                Ok(())
            }
            SliceOrInt::Int(index) => {
                let index = self.convert_py_index(index)?;
                if self.data.get(index).is_some() {
                    if index == self.data.len() {
                        // For individual removal from param table before
                        // deletion
                        self.remove_from_parameter_table(py, index)?;
                        self.data.remove(index);
                    } else {
                        // For delete in the middle delete before reindexing
                        self.data.remove(index);
                        self.reindex_parameter_table(py)?;
                    }
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

    pub fn setitem_no_param_table_update(
        &mut self,
        py: Python<'_>,
        index: isize,
        value: &Bound<PyAny>,
    ) -> PyResult<()> {
        let index = self.convert_py_index(index)?;
        let value: PyRef<CircuitInstruction> = value.downcast()?.borrow();
        let mut packed = self.pack(py, value)?;
        std::mem::swap(&mut packed, &mut self.data[index]);
        Ok(())
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
                    self.__delitem__(py, SliceOrInt::Slice(slice))?;
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
                self.remove_from_parameter_table(py, index)?;
                std::mem::swap(&mut packed, &mut self.data[index]);
                self.update_param_table(py, index, None)?;
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
        let old_len = self.data.len();
        let packed = self.pack(py, value)?;
        self.data.insert(index, packed);
        if index == old_len {
            self.update_param_table(py, old_len, None)?;
        } else {
            self.reindex_parameter_table(py)?;
        }
        Ok(())
    }

    pub fn pop(&mut self, py: Python<'_>, index: Option<PyObject>) -> PyResult<PyObject> {
        let index =
            index.unwrap_or_else(|| std::cmp::max(0, self.data.len() as isize - 1).into_py(py));
        let item = self.__getitem__(py, index.bind(py))?;

        self.__delitem__(py, index.bind(py).extract()?)?;
        Ok(item)
    }

    pub fn append(
        &mut self,
        py: Python<'_>,
        value: &Bound<CircuitInstruction>,
        params: Option<Vec<(usize, Vec<PyObject>)>>,
    ) -> PyResult<bool> {
        let packed = self.pack(py, value.try_borrow()?)?;
        let new_index = self.data.len();
        self.data.push(packed);
        self.update_param_table(py, new_index, params)
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
                let new_index = self.data.len();
                let qubits_id =
                    Interner::intern(&mut self.qargs_interner, InternerKey::Value(qubits))?;
                let clbits_id =
                    Interner::intern(&mut self.cargs_interner, InternerKey::Value(clbits))?;
                self.data.push(PackedInstruction {
                    op: inst.op.clone(),
                    qubits_id: qubits_id.index,
                    clbits_id: clbits_id.index,
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
            self.append_inner(py, v?.extract()?)?;
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
        self.remove_from_parameter_table(py, usize::MAX)?;
        match angle {
            Param::Float(angle) => {
                self.global_phase = Param::Float(angle.rem_euclid(2. * std::f64::consts::PI));
            }
            Param::ParameterExpression(angle) => {
                // usize::MAX is the global phase sentinel value for the inst index
                let inst_index = usize::MAX;
                let temp: PyObject = angle.getattr(py, intern!(py, "parameters"))?;
                let raw_param_objs: Vec<PyObject> = list_builtin.call1((temp,))?.extract()?;

                for (param_index, param_obj) in raw_param_objs.into_iter().enumerate() {
                    let param_uuid: u128 = param_obj
                        .getattr(py, intern!(py, "_uuid"))?
                        .getattr(py, intern!(py, "int"))?
                        .extract(py)?;
                    match self.param_table.table.get_mut(&param_uuid) {
                        Some(entry) => entry.add(inst_index, param_index),
                        None => {
                            let new_entry = ParamEntry::new(inst_index, param_index);
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
        usize::MAX
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

    pub fn pop_param(
        &mut self,
        py: Python,
        uuid: u128,
        name: String,
        default: PyObject,
    ) -> PyObject {
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
            .filter(|inst| inst.op.num_qubits() > 1 && !inst.op.directive())
            .count()
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

    fn pack(&mut self, py: Python, inst: PyRef<CircuitInstruction>) -> PyResult<PackedInstruction> {
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
            qubits_id: qubits.index,
            clbits_id: clbits.index,
            params: inst.params.clone(),
            extra_attrs: inst.extra_attrs.clone(),
            #[cfg(feature = "cache_pygates")]
            py_op: inst.py_op.clone(),
        })
    }

    fn pack_owned(&mut self, py: Python, inst: &CircuitInstruction) -> PyResult<PackedInstruction> {
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
            qubits_id: qubits.index,
            clbits_id: clbits.index,
            params: inst.params.clone(),
            extra_attrs: inst.extra_attrs.clone(),
            #[cfg(feature = "cache_pygates")]
            py_op: inst.py_op.clone(),
        })
    }
}
