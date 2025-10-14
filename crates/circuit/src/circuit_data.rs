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

use std::fmt::Debug;
use std::hash::Hash;
#[cfg(feature = "cache_pygates")]
use std::sync::OnceLock;

use crate::bit::{
    BitLocations, ClassicalRegister, PyBit, QuantumRegister, Register, ShareableClbit,
    ShareableQubit,
};
use crate::bit_locator::BitLocator;
use crate::circuit_instruction::{CircuitInstruction, OperationFromPython};
use crate::classical::expr;
use crate::dag_circuit::add_global_phase;
use crate::imports::{ANNOTATED_OPERATION, QUANTUM_CIRCUIT};
use crate::interner::{Interned, InternedMap, Interner};
use crate::object_registry::ObjectRegistry;
use crate::operations::{Operation, OperationRef, Param, PythonOperation, StandardGate};
use crate::packed_instruction::{PackedInstruction, PackedOperation};
use crate::parameter::parameter_expression::ParameterExpression;
use crate::parameter::symbol_expr::{Symbol, Value};
use crate::parameter_table::{ParameterTable, ParameterTableError, ParameterUse, ParameterUuid};
use crate::register_data::RegisterData;
use crate::slice::{PySequenceIndex, SequenceIndex};
use crate::var_stretch_container::{StretchType, VarStretchContainer, VarType};
use crate::{Clbit, Qubit, VarsMode};

use num_complex::Complex64;
use numpy::PyReadonlyArray1;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyList, PySet, PyTuple, PyType};
use pyo3::{PyTraverseError, PyVisit, import_exception, intern};

use hashbrown::{HashMap, HashSet};
use indexmap::IndexMap;
use smallvec::SmallVec;

import_exception!(qiskit.circuit.exceptions, CircuitError);

/// A tuple of a `CircuitData`'s internal state used for pickle's `__setstate__()` method.
type CircuitDataState<'py> = (
    Vec<QuantumRegister>,
    Vec<ClassicalRegister>,
    Bound<'py, PyDict>,
    Bound<'py, PyDict>,
    Vec<(String, Py<PyAny>)>,
    Vec<expr::Var>,
    Vec<expr::Stretch>,
);

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
/// .. plot::
///    :include-source:
///    :no-figs:
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
    qargs_interner: Interner<[Qubit]>,
    /// The cache used to intern instruction bits.
    cargs_interner: Interner<[Clbit]>,
    /// Qubits registered in the circuit.
    qubits: ObjectRegistry<Qubit, ShareableQubit>,
    /// Clbits registered in the circuit.
    clbits: ObjectRegistry<Clbit, ShareableClbit>,
    /// QuantumRegisters stored in the circuit
    qregs: RegisterData<QuantumRegister>,
    /// ClassicalRegisters stored in the circuit
    cregs: RegisterData<ClassicalRegister>,
    /// Mapping between [ShareableQubit] and its locations in
    /// the circuit
    qubit_indices: BitLocator<ShareableQubit, QuantumRegister>,
    /// Mapping between [ShareableClbit] and its locations in
    /// the circuit
    clbit_indices: BitLocator<ShareableClbit, ClassicalRegister>,
    /// Variables and stretches registered in the circuit
    vars_stretches: VarStretchContainer,

    param_table: ParameterTable,
    #[pyo3(get)]
    global_phase: Param,
}

#[pymethods]
impl CircuitData {
    #[new]
    #[pyo3(signature = (qubits=None, clbits=None, data=None, reserve=0, global_phase=Param::Float(0.0)))]
    pub fn new(
        qubits: Option<Vec<ShareableQubit>>,
        clbits: Option<Vec<ShareableClbit>>,
        data: Option<&Bound<PyAny>>,
        reserve: usize,
        global_phase: Param,
    ) -> PyResult<Self> {
        let qubit_size = qubits.as_ref().map_or(0, |bits| bits.len());
        let clbit_size = clbits.as_ref().map_or(0, |bits| bits.len());
        let qubits_registry = ObjectRegistry::with_capacity(qubit_size);
        let clbits_registry = ObjectRegistry::with_capacity(clbit_size);
        let qubit_indices = BitLocator::with_capacity(qubit_size);
        let clbit_indices = BitLocator::with_capacity(clbit_size);

        let mut self_ = CircuitData {
            data: Vec::new(),
            qargs_interner: Interner::new(),
            cargs_interner: Interner::new(),
            qubits: qubits_registry,
            clbits: clbits_registry,
            param_table: ParameterTable::new(),
            global_phase: Param::Float(0.),
            qregs: RegisterData::new(),
            cregs: RegisterData::new(),
            qubit_indices,
            clbit_indices,
            vars_stretches: VarStretchContainer::new(),
        };
        self_.set_global_phase(global_phase)?;
        if let Some(qubits) = qubits {
            for bit in qubits.into_iter() {
                self_.add_qubit(bit, true)?;
            }
        }
        if let Some(clbits) = clbits {
            for bit in clbits.into_iter() {
                self_.add_clbit(bit, true)?;
            }
        }
        if let Some(data) = data {
            self_.reserve(reserve);
            self_.extend(data)?;
        }
        Ok(self_)
    }

    /// Put ``self`` into the canonical physical form, with the given number of qubits.
    ///
    /// This acts in place, and does not need to traverse the circuit.  It is intended for use when
    /// the circuit is known to already represent a physical circuit, and we just need to assert
    /// that it is canonical physical form.
    ///
    /// This erases any information about virtual qubits in the :class:`CircuitData`.  Effectively,
    /// this applies the "trivial" layout mapping virtual qubit 0 to physical qubit 0, and so on.
    ///
    /// Args:
    ///     num_qubits: if given, the total number of physical qubits in the output; it must be at
    ///         least as large as the number of qubits in the circuit.  If not given, the number of
    ///         qubits is unchanged.
    #[pyo3(name = "make_physical", signature = (num_qubits=None))]
    pub fn py_make_physical(&mut self, num_qubits: Option<u32>) -> PyResult<()> {
        let num_qubits = match num_qubits {
            Some(num_qubits) => {
                if (num_qubits as usize) < self.num_qubits() {
                    return Err(PyValueError::new_err(format!(
                        "cannot have fewer physical qubits ({}) than virtual ({})",
                        num_qubits,
                        self.num_qubits()
                    )));
                }
                num_qubits
            }
            None => self
                .num_qubits()
                .try_into()
                .expect("qubits are stored in 32-bit integers"),
        };
        self.make_physical(num_qubits);
        Ok(())
    }

    pub fn __reduce__(self_: &Bound<CircuitData>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let ty: Bound<PyType> = self_.get_type();
        let args = {
            let self_ = self_.borrow();
            (
                (!self_.qubits.is_empty()).then_some(self_.qubits.objects().clone()),
                (!self_.clbits.is_empty()).then_some(self_.clbits.objects().clone()),
                None::<()>,
                self_.data.len(),
                self_.global_phase.clone(),
            )
        };
        let state = {
            let borrowed = self_.borrow();
            let vars_stretches_state = borrowed.vars_stretches.to_pickle(py);

            (
                borrowed.qregs.registers().to_vec(),
                borrowed.cregs.registers().to_vec(),
                borrowed.qubit_indices.cached(py).clone_ref(py),
                borrowed.clbit_indices.cached(py).clone_ref(py),
                vars_stretches_state.0, // identifiers vector
                vars_stretches_state.1, // variables objects
                vars_stretches_state.2, // stretch objects
            )
        };
        (ty, args, state, self_.try_iter()?).into_py_any(py)
    }

    pub fn __setstate__(slf: &Bound<CircuitData>, state: CircuitDataState) -> PyResult<()> {
        let mut borrowed_mut = slf.borrow_mut();
        // Add the registers directly to the `RegisterData` struct
        // to not modify the bit indices.
        for qreg in state.0.into_iter() {
            borrowed_mut.qregs.add_register(qreg, false)?;
        }
        for creg in state.1.into_iter() {
            borrowed_mut.cregs.add_register(creg, false)?;
        }

        // After the registers are added, reset bit locations.
        borrowed_mut.qubit_indices = BitLocator::from_py_dict(&state.2)?;
        borrowed_mut.clbit_indices = BitLocator::from_py_dict(&state.3)?;

        // borrowed_mut.identifier_info =
        //     IndexMap::with_capacity_and_hasher(state.4.len(), RandomState::default());
        borrowed_mut.vars_stretches =
            VarStretchContainer::from_pickle(slf.py(), (state.4, state.5, state.6))?;

        Ok(())
    }

    /// The list of registered :class:`.QuantumRegister` instances.
    ///
    /// .. warning::
    ///
    ///     Do not modify this list yourself.  It will invalidate/corrupt :attr:`.data` for this circuit.
    ///
    /// Returns:
    ///     list[:class:`.QuantumRegister`]: The current sequence of registered qubits.
    #[getter("qregs")]
    pub fn py_qregs<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyList> {
        self.qregs.cached_list(py)
    }

    /// A dict mapping Qubit instances to tuple comprised of 0) the corresponding index in
    /// circuit.qubits and 1) a list of Register-int pairs for each Register containing the Bit and
    /// its index within that register.
    #[getter("_qubit_indices")]
    pub fn get_qubit_indices(&self, py: Python) -> &Py<PyDict> {
        self.qubit_indices.cached(py)
    }

    #[setter("qregs")]
    fn set_qregs(&mut self, other: Vec<QuantumRegister>) -> PyResult<()> {
        self.qregs.dispose();
        for register in other {
            self.add_qreg(register, true)?;
        }

        for (index, qubit) in self.qubits.objects().iter().enumerate() {
            if !self.qubit_indices.contains_key(qubit) {
                self.qubit_indices
                    .insert(qubit.clone(), BitLocations::new(index as u32, []));
            }
        }
        Ok(())
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
    #[getter("qubits")]
    pub fn py_qubits(&self, py: Python<'_>) -> Py<PyList> {
        self.qubits.cached(py).clone_ref(py)
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

    /// The list of registered :class:`.ClassicalRegisters` instances.
    ///
    /// .. warning::
    ///
    ///     Do not modify this list yourself.  It will invalidate/corrupt :attr:`.data` for this circuit.
    ///
    /// Returns:
    ///     list[:class:`.ClassicalRegister`]: The current sequence of registered qubits.
    #[getter("cregs")]
    pub fn py_cregs<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyList> {
        self.cregs.cached_list(py)
    }

    /// A dict mapping Clbit instances to tuple comprised of 0) the corresponding index in
    /// circuit.clbits and 1) a list of Register-int pairs for each Register containing the Bit and
    /// its index within that register.
    #[getter("_clbit_indices")]
    pub fn get_clbit_indices(&self, py: Python) -> &Py<PyDict> {
        self.clbit_indices.cached(py)
    }

    #[setter("cregs")]
    fn set_cregs(&mut self, other: Vec<ClassicalRegister>) -> PyResult<()> {
        self.cregs.dispose();
        for register in other {
            self.add_creg(register, true)?;
        }

        for (index, clbit) in self.clbits.objects().iter().enumerate() {
            if !self.clbit_indices.contains_key(clbit) {
                self.clbit_indices
                    .insert(clbit.clone(), BitLocations::new(index as u32, []));
            }
        }
        Ok(())
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
    #[getter("clbits")]
    pub fn py_clbits(&self, py: Python<'_>) -> Py<PyList> {
        self.clbits.cached(py).clone_ref(py)
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

    /// Return the number of unbound compile-time symbolic parameters tracked by the circuit.
    pub fn num_parameters(&self) -> usize {
        self.param_table.num_parameters()
    }

    /// Get a (cached) sorted list of the Python-space `Parameter` instances tracked by this circuit
    /// data's parameter table.
    #[getter]
    pub fn get_parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.parameters().iter().cloned())
    }

    pub fn unsorted_parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        PySet::new(py, self.param_table.iter_symbols().cloned())
    }

    fn _raw_parameter_table_entry(&self, param: Bound<PyAny>) -> PyResult<Py<PySet>> {
        self.param_table._py_raw_entry(param)
    }

    pub fn get_parameter_by_name(&self, py: Python, name: &str) -> Option<Py<PyAny>> {
        self.param_table
            .parameter_by_name(name)
            .map(|ob| ob.clone().into_py_any(py).unwrap())
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
    pub fn add_qubit(&mut self, bit: ShareableQubit, strict: bool) -> PyResult<()> {
        let index = self.qubits.add(bit.clone(), strict)?;
        self.qubit_indices
            .insert(bit, BitLocations::new(index.0, []));
        Ok(())
    }

    /// Registers a :class:`.QuantumRegister` instance.
    ///
    /// Args:
    ///     bit (:class:`.QuantumRegister`): The register to add.
    #[pyo3(signature = (register, *,strict = true))]
    pub fn add_qreg(&mut self, register: QuantumRegister, strict: bool) -> PyResult<()> {
        self.qregs.add_register(register.clone(), strict)?;

        for (index, bit) in register.bits().enumerate() {
            if let Some(entry) = self.qubit_indices.get_mut(&bit) {
                entry.add_register(register.clone(), index);
            } else if let Some(bit_idx) = self.qubits.find(&bit) {
                self.qubit_indices.insert(
                    bit,
                    BitLocations::new(bit_idx.0, [(register.clone(), index)]),
                );
            } else {
                let bit_idx = self.qubits.len();
                self.add_qubit(bit.clone(), true)?;
                self.qubit_indices.insert(
                    bit,
                    BitLocations::new(
                        bit_idx.try_into().map_err(|_| {
                            CircuitError::new_err(format!(
                                "Qubit at index {bit_idx} exceeds circuit capacity."
                            ))
                        })?,
                        [(register.clone(), index)],
                    ),
                );
            }
        }
        Ok(())
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
    pub fn add_clbit(&mut self, bit: ShareableClbit, strict: bool) -> PyResult<()> {
        let index = self.clbits.add(bit.clone(), strict)?;
        self.clbit_indices
            .insert(bit, BitLocations::new(index.0, []));
        Ok(())
    }

    /// Registers a :class:`.QuantumRegister` instance.
    ///
    /// Args:
    ///     bit (:class:`.QuantumRegister`): The register to add.
    #[pyo3(signature = (register, *,strict = true))]
    pub fn add_creg(&mut self, register: ClassicalRegister, strict: bool) -> PyResult<()> {
        self.cregs.add_register(register.clone(), strict)?;

        for (index, bit) in register.bits().enumerate() {
            if let Some(entry) = self.clbit_indices.get_mut(&bit) {
                entry.add_register(register.clone(), index);
            } else if let Some(bit_idx) = self.clbits.find(&bit) {
                self.clbit_indices.insert(
                    bit,
                    BitLocations::new(bit_idx.0, [(register.clone(), index)]),
                );
            } else {
                let bit_idx = self.clbits.len();
                self.add_clbit(bit.clone(), true)?;
                self.clbit_indices.insert(
                    bit,
                    BitLocations::new(
                        bit_idx.try_into().map_err(|_| {
                            CircuitError::new_err(format!(
                                "Clbit at index {bit_idx} exceeds circuit capacity."
                            ))
                        })?,
                        [(register.clone(), index)],
                    ),
                );
            }
        }
        Ok(())
    }

    /// Performs a shallow copy.
    ///
    /// Returns:
    ///     CircuitData: The shallow copy.
    #[pyo3(signature = (copy_instructions=true, deepcopy=false))]
    pub fn copy(&self, py: Python<'_>, copy_instructions: bool, deepcopy: bool) -> PyResult<Self> {
        let mut res = self.copy_empty_like(VarsMode::Alike)?;
        res.qargs_interner = self.qargs_interner.clone();
        res.cargs_interner = self.cargs_interner.clone();
        res.reserve(self.data().len());
        res.param_table.clone_from(&self.param_table);

        if deepcopy {
            let memo = PyDict::new(py);
            for inst in &self.data {
                let new_op = match inst.op.view() {
                    OperationRef::Gate(gate) => gate.py_deepcopy(py, Some(&memo))?.into(),
                    OperationRef::Instruction(instruction) => {
                        instruction.py_deepcopy(py, Some(&memo))?.into()
                    }
                    OperationRef::Operation(operation) => {
                        operation.py_deepcopy(py, Some(&memo))?.into()
                    }
                    OperationRef::StandardGate(gate) => gate.into(),
                    OperationRef::StandardInstruction(instruction) => instruction.into(),
                    OperationRef::Unitary(unitary) => unitary.clone().into(),
                };
                res.data.push(PackedInstruction {
                    op: new_op,
                    qubits: inst.qubits,
                    clbits: inst.clbits,
                    params: inst.params.clone(),
                    label: inst.label.clone(),
                    #[cfg(feature = "cache_pygates")]
                    py_op: OnceLock::new(),
                });
            }
        } else if copy_instructions {
            for inst in &self.data {
                let new_op = match inst.op.view() {
                    OperationRef::Gate(gate) => gate.py_copy(py)?.into(),
                    OperationRef::Instruction(instruction) => instruction.py_copy(py)?.into(),
                    OperationRef::Operation(operation) => operation.py_copy(py)?.into(),
                    OperationRef::StandardGate(gate) => gate.into(),
                    OperationRef::StandardInstruction(instruction) => instruction.into(),
                    OperationRef::Unitary(unitary) => unitary.clone().into(),
                };
                res.data.push(PackedInstruction {
                    op: new_op,
                    qubits: inst.qubits,
                    clbits: inst.clbits,
                    params: inst.params.clone(),
                    label: inst.label.clone(),
                    #[cfg(feature = "cache_pygates")]
                    py_op: OnceLock::new(),
                });
            }
        } else {
            res.data.extend(self.data.iter().cloned());
        }

        Ok(res)
    }

    /// Performs a copy with no instructions.
    ///
    /// # Arguments:
    ///
    /// * vars_mode: specifies realtime variables copy mode.
    ///     * VarsMode::Alike: variables will be copied following declaration semantics in self.
    ///     * VarsMode::Captures: variables will be copied as captured variables.
    ///     * VarsMode::Drop: variables will not be copied.
    ///
    /// # Returns:
    ///
    /// CircuitData: The empty copy like self.
    #[pyo3(signature = (*, vars_mode=VarsMode::Alike))]
    pub fn copy_empty_like(&self, vars_mode: VarsMode) -> PyResult<Self> {
        self.copy_empty_like_with_capacity(0, vars_mode)
    }

    /// Performs a copy with no instructions, but reserving the specified capacity for instructions
    ///
    /// # Arguments:
    ///
    /// * capacity: initial capacity of the instructions vector.
    /// * vars_mode: specifies realtime variables copy mode.
    ///     * VarsMode::Alike: variables will be copied following declaration semantics in self.
    ///     * VarsMode::Captures: variables will be copied as captured variables.
    ///     * VarsMode::Drop: variables will not be copied.
    ///
    /// # Returns:
    ///
    /// CircuitData: The empty copy like self.
    pub fn copy_empty_like_with_capacity(
        &self,
        capacity: usize,
        vars_mode: VarsMode,
    ) -> PyResult<Self> {
        let mut res = CircuitData::new(
            Some(self.qubits.objects().clone()),
            Some(self.clbits.objects().clone()),
            None,
            0,
            self.global_phase.clone(),
        )?;

        res.reserve(capacity);
        res.qargs_interner = self.qargs_interner.clone();
        res.cargs_interner = self.cargs_interner.clone();

        // After initialization, copy register info.
        res.qregs = self.qregs.clone();
        res.cregs = self.cregs.clone();
        res.qubit_indices = self.qubit_indices.clone();
        res.clbit_indices = self.clbit_indices.clone();

        match vars_mode {
            VarsMode::Alike => {
                res.vars_stretches = self.vars_stretches.clone();
            }
            VarsMode::Captures => {
                res.vars_stretches = self
                    .vars_stretches
                    .clone_as_captures()
                    .map_err(CircuitError::new_err)?;
            }
            VarsMode::Drop => {}
        };

        Ok(res)
    }

    /// Reserves capacity for at least ``additional`` more
    /// :class:`.CircuitInstruction` instances to be added to this container.
    ///
    /// Args:
    ///     additional (int): The additional capacity to reserve. If the
    ///         capacity is already sufficient, does nothing.
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
    }

    /// Returns a tuple of the sets of :class:`.Qubit` and :class:`.Clbit` instances
    /// that appear in at least one instruction's bit lists.
    ///
    /// Returns:
    ///     tuple[set[:class:`.Qubit`], set[:class:`.Clbit`]]: The active qubits and clbits.
    pub fn active_bits(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let qubits = PySet::empty(py)?;
        let clbits = PySet::empty(py)?;
        for inst in self.data.iter() {
            for b in self.qargs_interner.get(inst.qubits) {
                qubits.add(self.qubits.get(*b).unwrap())?;
            }
            for b in self.cargs_interner.get(inst.clbits) {
                clbits.add(self.clbits.get(*b).unwrap())?;
            }
        }

        Ok((qubits, clbits).into_pyobject(py)?.unbind())
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
            if inst.op.try_standard_gate().is_some() {
                continue;
            }
            let py_op = func.call1((inst.unpack_py_op(py)?,))?;
            let result = py_op.extract::<OperationFromPython>()?;
            inst.op = result.operation;
            inst.params = (!result.params.is_empty()).then(|| Box::new(result.params));
            inst.label = result.label;
            #[cfg(feature = "cache_pygates")]
            {
                inst.py_op = py_op.unbind().into();
            }
        }
        Ok(())
    }

    /// Checks whether the circuit has an instance of :class:`.ControlFlowOp`
    /// present amongst its operations.
    pub fn has_control_flow_op(&self) -> bool {
        self.data.iter().any(|inst| inst.op.control_flow())
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
    #[pyo3(signature = (qubits=None, clbits=None, qregs=None, cregs=None))]
    pub fn replace_bits(
        &mut self,
        qubits: Option<Vec<ShareableQubit>>,
        clbits: Option<Vec<ShareableClbit>>,
        qregs: Option<Vec<QuantumRegister>>,
        cregs: Option<Vec<ClassicalRegister>>,
    ) -> PyResult<()> {
        let qubits_is_some = qubits.is_some();
        let clbits_is_some = clbits.is_some();
        let mut temp = CircuitData::new(qubits, clbits, None, 0, self.global_phase.clone())?;

        // Add qregs if provided.
        if let Some(qregs) = qregs {
            for qreg in qregs {
                temp.add_qreg(qreg, true)?;
            }
        }
        // Add cregs if provided.
        if let Some(cregs) = cregs {
            for creg in cregs {
                temp.add_creg(creg, true)?;
            }
        }

        if qubits_is_some {
            if temp.num_qubits() < self.num_qubits() {
                return Err(PyValueError::new_err(format!(
                    "Replacement 'qubits' of size {:?} must contain at least {:?} bits.",
                    temp.num_qubits(),
                    self.num_qubits(),
                )));
            }
            std::mem::swap(&mut temp.qubits, &mut self.qubits);
            std::mem::swap(&mut temp.qregs, &mut self.qregs);
            std::mem::swap(&mut temp.qubit_indices, &mut self.qubit_indices);
        }
        if clbits_is_some {
            if temp.num_clbits() < self.num_clbits() {
                return Err(PyValueError::new_err(format!(
                    "Replacement 'clbits' of size {:?} must contain at least {:?} bits.",
                    temp.num_clbits(),
                    self.num_clbits(),
                )));
            }
            std::mem::swap(&mut temp.clbits, &mut self.clbits);
            std::mem::swap(&mut temp.cregs, &mut self.cregs);
            std::mem::swap(&mut temp.clbit_indices, &mut self.clbit_indices);
        }
        Ok(())
    }

    pub fn __len__(&self) -> usize {
        self.data.len()
    }

    // Note: we also rely on this to make us iterable!
    pub fn __getitem__(&self, py: Python, index: PySequenceIndex) -> PyResult<Py<PyAny>> {
        // Get a single item, assuming the index is validated as in bounds.
        let get_single = |index: usize| {
            let inst = &self.data[index];
            let qubits = self.qargs_interner.get(inst.qubits);
            let clbits = self.cargs_interner.get(inst.clbits);
            CircuitInstruction {
                operation: inst.op.clone(),
                qubits: PyTuple::new(py, self.qubits.map_indices(qubits))
                    .unwrap()
                    .unbind(),
                clbits: PyTuple::new(py, self.clbits.map_indices(clbits))
                    .unwrap()
                    .unbind(),
                params: inst.params_view().iter().cloned().collect(),
                label: inst.label.clone(),
                #[cfg(feature = "cache_pygates")]
                py_op: inst.py_op.clone(),
            }
            .into_py_any(py)
            .unwrap()
        };
        match index.with_len(self.data.len())? {
            SequenceIndex::Int(index) => Ok(get_single(index)),
            indices => PyList::new(py, indices.iter().map(get_single))?.into_py_any(py),
        }
    }

    pub fn __delitem__(&mut self, index: PySequenceIndex) -> PyResult<()> {
        self.delitem(index.with_len(self.data.len())?)
    }

    pub fn __setitem__(&mut self, index: PySequenceIndex, value: &Bound<PyAny>) -> PyResult<()> {
        fn set_single(slf: &mut CircuitData, index: usize, value: &Bound<PyAny>) -> PyResult<()> {
            let py = value.py();
            slf.untrack_instruction_parameters(index)?;
            slf.data[index] = slf.pack(py, &value.downcast::<CircuitInstruction>()?.borrow())?;
            slf.track_instruction_parameters(index)?;
            Ok(())
        }

        match index.with_len(self.data.len())? {
            SequenceIndex::Int(index) => set_single(self, index, value),
            indices @ SequenceIndex::PosRange {
                start,
                stop,
                step: 1,
            } => {
                // `list` allows setting a slice with step +1 to an arbitrary length.
                let values = value.try_iter()?.collect::<PyResult<Vec<_>>>()?;
                for (index, value) in indices.iter().zip(values.iter()) {
                    set_single(self, index, value)?;
                }
                if indices.len() > values.len() {
                    self.delitem(SequenceIndex::PosRange {
                        start: start + values.len(),
                        stop,
                        step: 1,
                    })?
                } else {
                    for value in values[indices.len()..].iter().rev() {
                        self.insert(stop as isize, value.downcast()?.borrow())?;
                    }
                }
                Ok(())
            }
            indices => {
                let values = value.try_iter()?.collect::<PyResult<Vec<_>>>()?;
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
            self.track_instruction_parameters(index)?;
        } else {
            self.reindex_parameter_table()?;
        }
        Ok(())
    }

    #[pyo3(signature = (index=None))]
    pub fn pop(&mut self, py: Python<'_>, index: Option<PySequenceIndex>) -> PyResult<Py<PyAny>> {
        let index = index.unwrap_or(PySequenceIndex::Int(-1));
        let native_index = index.with_len(self.data.len())?;
        let item = self.__getitem__(py, index)?;
        self.delitem(native_index)?;
        Ok(item)
    }

    /// Primary entry point for appending an instruction from Python space.
    pub fn append(&mut self, value: &Bound<CircuitInstruction>) -> PyResult<()> {
        let py = value.py();
        let new_index = self.data.len();
        let packed = self.pack(py, &value.borrow())?;
        self.data.push(packed);
        self.track_instruction_parameters(new_index)
    }

    /// Backup entry point for appending an instruction from Python space, in the unusual case that
    /// one of the instruction parameters contains a cyclical reference to the circuit itself.
    ///
    /// In this case, the `params` field should be a list of `(index, parameters)` tuples, where the
    /// index is into the instruction's `params` attribute, and `parameters` is a Python iterable
    /// of `Parameter` objects.
    pub fn append_manual_params(
        &mut self,
        value: &Bound<CircuitInstruction>,
        params: &Bound<PyList>,
    ) -> PyResult<()> {
        let instruction_index = self.data.len();
        let packed = self.pack(value.py(), &value.borrow())?;
        self.data.push(packed);
        for item in params.iter() {
            let (parameter_index, parameters) = item.extract::<(u32, Bound<PyAny>)>()?;
            let usage = ParameterUse::Index {
                instruction: instruction_index,
                parameter: parameter_index,
            };
            for param in parameters.try_iter()? {
                let symbol = param?.extract::<Symbol>()?;
                self.param_table.track(&symbol, Some(usage))?;
            }
        }
        Ok(())
    }

    pub fn extend(&mut self, itr: &Bound<PyAny>) -> PyResult<()> {
        if let Ok(other) = itr.downcast::<CircuitData>() {
            let other = other.borrow();
            // Fast path to avoid unnecessary construction of CircuitInstruction instances.
            self.data.reserve(other.data.len());
            for inst in other.data.iter() {
                let qubits = other
                    .qargs_interner
                    .get(inst.qubits)
                    .iter()
                    .map(|b| Ok(self.qubits.find(other.qubits.get(*b).unwrap()).unwrap()))
                    .collect::<PyResult<Vec<Qubit>>>()?;
                let clbits = other
                    .cargs_interner
                    .get(inst.clbits)
                    .iter()
                    .map(|b| Ok(self.clbits.find(other.clbits.get(*b).unwrap()).unwrap()))
                    .collect::<PyResult<Vec<Clbit>>>()?;
                let new_index = self.data.len();
                let qubits_id = self.qargs_interner.insert_owned(qubits);
                let clbits_id = self.cargs_interner.insert_owned(clbits);
                self.data.push(PackedInstruction {
                    op: inst.op.clone(),
                    qubits: qubits_id,
                    clbits: clbits_id,
                    params: inst.params.clone(),
                    label: inst.label.clone(),
                    #[cfg(feature = "cache_pygates")]
                    py_op: inst.py_op.clone(),
                });
                self.track_instruction_parameters(new_index)?;
            }
            return Ok(());
        }
        for v in itr.try_iter()? {
            self.append(v?.downcast()?)?;
        }
        Ok(())
    }

    /// Assign all the circuit parameters, given an iterable input of `Param` instances.
    fn assign_parameters_iterable(&mut self, sequence: Bound<PyAny>) -> PyResult<()> {
        if let Ok(readonly) = sequence.extract::<PyReadonlyArray1<f64>>() {
            // Fast path for Numpy arrays; in this case we can easily handle them without copying
            // the data across into a Rust-space `Vec` first.
            let array = readonly.as_array();
            if array.len() != self.param_table.num_parameters() {
                return Err(PyValueError::new_err(concat!(
                    "Mismatching number of values and parameters. For partial binding ",
                    "please pass a dictionary of {parameter: value} pairs."
                )));
            }
            let mut old_table = std::mem::take(&mut self.param_table);
            self.assign_parameters_inner(
                array
                    .iter()
                    .map(|value| Param::Float(*value))
                    .zip(old_table.drain_ordered())
                    .map(|(value, (obj, uses))| (obj, value, uses)),
            )
        } else {
            let values = sequence
                .try_iter()?
                .map(|ob| Param::extract_no_coerce(&ob?))
                .collect::<PyResult<Vec<_>>>()?;
            self.assign_parameters_from_slice(&values)
        }
    }

    /// Assign all uses of the circuit parameters as keys `mapping` to their corresponding values.
    ///
    /// Any items in the mapping that are not present in the circuit are skipped; it's up to Python
    /// space to turn extra bindings into an error, if they choose to do it.
    fn assign_parameters_mapping(&mut self, mapping: Bound<PyAny>) -> PyResult<()> {
        let mut items = Vec::new();
        for item in mapping.call_method0("items")?.try_iter()? {
            let (symbol, value) = item?.extract::<(Symbol, AssignParam)>()?;
            let uuid = ParameterUuid::from_symbol(&symbol);
            // It's fine if the mapping contains parameters that we don't have - just skip those.
            if let Ok(uses) = self.param_table.pop(uuid) {
                items.push((symbol, value.0, uses));
            }
        }
        self.assign_parameters_inner(items)
    }

    pub fn clear(&mut self) {
        std::mem::take(&mut self.data);
        self.param_table.clear();
    }

    /// Counts the number of times each operation is used in the circuit.
    ///
    /// # Parameters
    /// - `self` - A mutable reference to the CircuitData struct.
    ///
    /// # Returns
    /// An IndexMap containing the operation names as keys and their respective counts as values.
    pub fn count_ops(&self) -> IndexMap<&str, usize, ::ahash::RandomState> {
        let mut ops_count: IndexMap<&str, usize, ::ahash::RandomState> = IndexMap::default();
        for instruction in &self.data {
            *ops_count.entry(instruction.op.name()).or_insert(0) += 1;
        }
        ops_count.par_sort_by(|_k1, v1, _k2, v2| v2.cmp(v1));
        ops_count
    }

    // Marks this pyclass as NOT hashable.
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __eq__(slf: &Bound<Self>, other: &Bound<PyAny>) -> PyResult<bool> {
        let self_cd = slf.borrow();
        let slf = slf.as_any();
        if slf.is(other) {
            return Ok(true);
        }
        if slf.len()? != other.len()? {
            return Ok(false);
        }

        if let Ok(other_cd) = other.downcast::<CircuitData>() {
            if !slf
                .getattr("global_phase")?
                .eq(other_cd.getattr("global_phase")?)?
            {
                return Ok(false);
            }

            if self_cd.vars_stretches != other_cd.borrow().vars_stretches {
                return Ok(false);
            }
        }

        // Implemented using generic iterators on both sides
        // for simplicity.
        let mut ours_itr = slf.try_iter()?;
        let mut theirs_itr = other.try_iter()?;
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
        // Note:
        //   There's no need to visit the native Rust data
        //   structures used for internal tracking: the only Python
        //   references they contain are to the bits in these lists!
        if let Some(bits) = self.qubits.cached_raw() {
            visit.call(bits)?;
        }
        if let Some(bits) = self.clbits.cached_raw() {
            visit.call(bits)?;
        }
        if let Some(regs) = self.qregs.cached_raw() {
            visit.call(regs)?;
        }
        if let Some(regs) = self.cregs.cached_raw() {
            visit.call(regs)?;
        }
        if let Some(locations) = self.qubit_indices.cached_raw() {
            visit.call(locations)?;
        }
        if let Some(locations) = self.clbit_indices.cached_raw() {
            visit.call(locations)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        // Clear anything that could have a reference cycle.
        self.data.clear();
        self.qubits.dispose();
        self.clbits.dispose();
        self.qregs.dispose();
        self.cregs.dispose();
        self.clbit_indices.dispose();
        self.qubit_indices.dispose();
        self.param_table.clear();
    }

    /// Set the global phase of the circuit.
    ///
    /// This method assumes that the parameter table is either fully consistent, or contains zero
    /// entries for the global phase, regardless of what value is currently stored there.  It's not
    /// uncommon for subclasses and other parts of Qiskit to have filled in the global phase field
    /// by copies or other means, before making the parameter table consistent.
    #[setter]
    pub fn set_global_phase(&mut self, angle: Param) -> PyResult<()> {
        if let Param::ParameterExpression(expr) = &self.global_phase {
            for symbol in expr.iter_symbols() {
                match self.param_table.remove_use(
                    ParameterUuid::from_symbol(symbol),
                    ParameterUse::GlobalPhase,
                ) {
                    Ok(_)
                    | Err(ParameterTableError::ParameterNotTracked(_))
                    | Err(ParameterTableError::UsageNotTracked(_)) => (),
                    // Any errors added later might want propagating.
                }
            }
        };
        match &angle {
            Param::Float(angle) => {
                self.global_phase = Param::Float(angle.rem_euclid(2. * std::f64::consts::PI));
                Ok(())
            }
            Param::ParameterExpression(expr) => {
                for symbol in expr.iter_symbols() {
                    self.param_table
                        .track(symbol, Some(ParameterUse::GlobalPhase))?;
                }
                self.global_phase = angle;
                Ok(())
            }
            Param::Obj(_) => Err(PyTypeError::new_err("invalid type for global phase")),
        }
    }

    pub fn num_nonlocal_gates(&self) -> usize {
        self.data
            .iter()
            .filter(|inst| inst.op.num_qubits() > 1 && !inst.op.directive())
            .count()
    }

    /// Converts several qubit representations (such as indexes, range, etc.)
    /// into a list of qubits.
    ///
    /// Args:
    ///     qubit_representation: Representation to expand.
    ///
    /// Returns:
    ///     The resolved instances of the qubits.
    fn _qbit_argument_conversion(
        &self,
        qubit_representation: Bound<PyAny>,
    ) -> PyResult<Vec<ShareableQubit>> {
        bit_argument_conversion(
            &qubit_representation,
            self.qubits.objects(),
            &self.qubit_indices,
        )
    }

    /// Converts several clbit representations (such as indexes, range, etc.)
    /// into a list of qubits.
    ///
    /// Args:
    ///     clbit_representation: Representation to expand.
    ///
    /// Returns:
    ///     The resolved instances of the qubits.
    fn _cbit_argument_conversion(
        &self,
        clbit_representation: Bound<PyAny>,
    ) -> PyResult<Vec<ShareableClbit>> {
        bit_argument_conversion(
            &clbit_representation,
            self.clbits.objects(),
            &self.clbit_indices,
        )
    }

    /// Raise exception if list of qubits contains duplicates.
    #[staticmethod]
    fn _check_dups(qubits: Vec<ShareableQubit>) -> PyResult<()> {
        let qubit_set: HashSet<&ShareableQubit> = qubits.iter().collect();
        if qubits.len() != qubit_set.len() {
            return Err(CircuitError::new_err("duplicate qubit arguments"));
        }
        Ok(())
    }

    /// Adds an input variable to the circuit.
    ///
    /// Args:
    ///     var: the variable to add.
    #[pyo3(name = "add_input_var")]
    fn py_add_input_var(&mut self, var: expr::Var) -> PyResult<()> {
        self.vars_stretches
            .add_var(var, VarType::Input)
            .map_err(CircuitError::new_err)?;
        Ok(())
    }

    /// Adds a captured variable to the circuit.
    ///
    /// Args:
    ///     var: the variable to add.
    #[pyo3(name = "add_captured_var")]
    fn py_add_captured_var(&mut self, var: expr::Var) -> PyResult<()> {
        self.vars_stretches
            .add_var(var, VarType::Capture)
            .map_err(CircuitError::new_err)?;
        Ok(())
    }

    /// Adds a declared variable to the circuit.
    ///
    /// Args:
    ///     var: the variable to add.
    #[pyo3(name = "add_declared_var")]
    fn py_add_declared_var(&mut self, var: expr::Var) -> PyResult<()> {
        self.vars_stretches
            .add_var(var, VarType::Declare)
            .map_err(CircuitError::new_err)?;
        Ok(())
    }

    /// Checks if this variable is in the circuit.
    ///
    /// Args:
    ///     var: the variable or name to check.
    #[pyo3(name = "has_var")]
    fn py_has_var(&self, var: &Bound<PyAny>) -> PyResult<bool> {
        if let Ok(name) = var.extract::<String>() {
            Ok(self.vars_stretches.has_var(&name))
        } else {
            let var = var.extract::<expr::Var>()?;
            Ok(self.vars_stretches.vars().contains(&var))
        }
    }

    /// Checks if the circuit contains an input variable with the specified name.
    #[pyo3(name = "has_input_var")]
    fn py_has_input_var(&self, name: &str) -> PyResult<bool> {
        Ok(self.vars_stretches.has_var_by_type(name, VarType::Input))
    }

    /// Checks if the circuit contains a local variable with the specified name.
    #[pyo3(name = "has_declared_var")]
    fn py_has_declared_var(&self, name: &str) -> PyResult<bool> {
        Ok(self.vars_stretches.has_var_by_type(name, VarType::Declare))
    }

    /// Checks if the circuit contains a capture variable with the specified name.
    #[pyo3(name = "has_captured_var")]
    fn py_has_captured_var(&self, name: &str) -> PyResult<bool> {
        Ok(self.vars_stretches.has_var_by_type(name, VarType::Capture))
    }

    /// Returns a list of the captured variables tracked in this circuit.
    #[pyo3(name = "get_captured_vars")]
    fn py_get_captured_vars(&self, py: Python) -> PyResult<Py<PyList>> {
        Ok(PyList::new(
            py,
            self.vars_stretches
                .iter_vars(VarType::Capture)
                .map(|var| var.clone().into_pyobject(py).unwrap()),
        )?
        .unbind())
    }

    /// Returns a list of the local variables tracked in this circuit.
    #[pyo3(name = "get_declared_vars")]
    fn py_get_declared_vars(&self, py: Python) -> PyResult<Py<PyList>> {
        Ok(PyList::new(
            py,
            self.vars_stretches
                .iter_vars(VarType::Declare)
                .map(|var| var.clone().into_pyobject(py).unwrap()),
        )?
        .unbind())
    }

    /// Returns the variable in the circuit corresponding to the specified name, or None if no such variable.                                                                                                                                                                                                                                                                  Return the variable in the circuit corresponding to the given name, or None if no such variable.
    #[pyo3(name = "get_var")]
    fn py_get_var(&self, py: Python, name: &str) -> PyResult<Py<PyAny>> {
        if let Some(var) = self.vars_stretches.get_var(name) {
            var.clone().into_py_any(py)
        } else {
            Ok(py.None())
        }
    }

    /// Returns a list of the input variables tracked in this circuit
    #[pyo3(name = "get_input_vars")]
    fn py_get_input_vars(&self, py: Python) -> PyResult<Py<PyList>> {
        Ok(PyList::new(
            py,
            self.vars_stretches
                .iter_vars(VarType::Input)
                .map(|var| var.clone().into_pyobject(py).unwrap()),
        )?
        .unbind())
    }

    /// Returns the number of classical input variables in the circuit.
    #[getter]
    pub fn num_input_vars(&self) -> usize {
        self.vars_stretches.num_vars(VarType::Input)
    }

    /// Returns the number of captured variables in the circuit.
    #[getter]
    pub fn num_captured_vars(&self) -> usize {
        self.vars_stretches.num_vars(VarType::Capture)
    }

    /// Returns the number of local variables in the circuit.
    #[getter]
    pub fn num_declared_vars(&self) -> usize {
        self.vars_stretches.num_vars(VarType::Declare)
    }

    /// Adds a captured stretch to the circuit.
    ///
    /// Args:
    ///     stretch: the stretch variable to add.
    #[pyo3(name = "add_captured_stretch")]
    fn py_add_captured_stretch(&mut self, stretch: expr::Stretch) -> PyResult<()> {
        self.vars_stretches
            .add_stretch(stretch, StretchType::Capture)
            .map_err(CircuitError::new_err)?;
        Ok(())
    }

    /// Adds a local stretch to the circuit.
    ///
    /// Args:
    ///     stretch: the stretch variable to add.
    #[pyo3(name = "add_declared_stretch")]
    fn py_add_declared_stretch(&mut self, stretch: expr::Stretch) -> PyResult<()> {
        self.vars_stretches
            .add_stretch(stretch, StretchType::Declare)
            .map_err(CircuitError::new_err)?;
        Ok(())
    }

    /// Checkss if this stretch variable is in the circuit.
    ///
    /// Args:
    ///     stretch: the stretch or name to check.
    #[pyo3(name = "has_stretch")]
    fn py_has_stretch(&self, stretch: &Bound<PyAny>) -> PyResult<bool> {
        if let Ok(name) = stretch.extract::<String>() {
            Ok(self.vars_stretches.has_stretch(&name))
        } else {
            let stretch = stretch.extract::<expr::Stretch>()?;
            Ok(self.vars_stretches.stretches().contains(&stretch))
        }
    }

    /// Check if the circuit contains a captured stretch with the specified name.
    #[pyo3(name = "has_captured_stretch")]
    fn py_has_captured_stretch(&self, name: &str) -> PyResult<bool> {
        Ok(self
            .vars_stretches
            .has_stretch_by_type(name, StretchType::Capture))
    }

    /// Checks if the circuit contains a local stretch with the specified name.
    #[pyo3(name = "has_declared_stretch")]
    fn py_has_declared_stretch(&self, name: &str) -> PyResult<bool> {
        Ok(self
            .vars_stretches
            .has_stretch_by_type(name, StretchType::Declare))
    }

    // Returns the stretch  in the circuit corresponding to the specified name, or None if no such variable.
    #[pyo3(name = "get_stretch")]
    fn py_get_stretch(&self, py: Python, name: &str) -> PyResult<Py<PyAny>> {
        if let Some(stretch) = self.vars_stretches.get_stretch(name) {
            stretch.clone().into_py_any(py)
        } else {
            Ok(py.None())
        }
    }

    /// Returns a list of the captured stretch variables tracked in this circuit.
    #[pyo3(name = "get_captured_stretches")]
    fn py_get_captured_stretches(&self, py: Python) -> PyResult<Py<PyList>> {
        Ok(PyList::new(
            py,
            self.vars_stretches
                .iter_stretches(StretchType::Capture)
                .map(|stretch| stretch.clone().into_pyobject(py).unwrap()),
        )?
        .unbind())
    }

    /// Returns a list of the local stretch variables tracked in this circuit.
    #[pyo3(name = "get_declared_stretches")]
    fn py_get_declared_stretches(&self, py: Python) -> PyResult<Py<PyList>> {
        Ok(PyList::new(
            py,
            self.vars_stretches
                .iter_stretches(StretchType::Declare)
                .map(|stretch| stretch.clone().into_pyobject(py).unwrap()),
        )?
        .unbind())
    }

    /// Returns the number of local stretch variables in the circuit.
    #[getter]
    pub fn num_declared_stretches(&self) -> usize {
        self.vars_stretches.num_stretches(StretchType::Declare)
    }

    /// Returns the number of captured stretch variables in the circuit.
    #[getter]
    pub fn num_captured_stretches(&self) -> usize {
        self.vars_stretches.num_stretches(StretchType::Capture)
    }
}

impl CircuitData {
    /// An alternate constructor to build a new `CircuitData` from an iterator
    /// of packed operations. This can be used to build a circuit from a sequence
    /// of `PackedOperation` without needing to involve Python.
    ///
    /// This can be connected with the Python space
    /// QuantumCircuit.from_circuit_data() constructor to build a full
    /// QuantumCircuit from Rust.
    ///
    /// # Arguments
    ///
    /// * py: A GIL handle this is needed to instantiate Qubits in Python space
    /// * num_qubits: The number of qubits in the circuit. These will be created
    ///   in Python as loose bits without a register.
    /// * num_clbits: The number of classical bits in the circuit. These will be created
    ///   in Python as loose bits without a register.
    /// * instructions: An iterator of the (packed operation, params, qubits, clbits) to
    ///   add to the circuit
    /// * global_phase: The global phase to use for the circuit
    pub fn from_packed_operations<I>(
        num_qubits: u32,
        num_clbits: u32,
        instructions: I,
        global_phase: Param,
    ) -> PyResult<Self>
    where
        I: IntoIterator<
            Item = PyResult<(
                PackedOperation,
                SmallVec<[Param; 3]>,
                Vec<Qubit>,
                Vec<Clbit>,
            )>,
        >,
    {
        let instruction_iter = instructions.into_iter();
        let mut res = Self::with_capacity(
            num_qubits,
            num_clbits,
            instruction_iter.size_hint().0,
            global_phase,
        )?;

        for item in instruction_iter {
            let (operation, params, qargs, cargs) = item?;
            let qubits = res.qargs_interner.insert_owned(qargs);
            let clbits = res.cargs_interner.insert_owned(cargs);
            let params = (!params.is_empty()).then(|| Box::new(params));
            res.data.push(PackedInstruction {
                op: operation,
                qubits,
                clbits,
                params,
                label: None,
                #[cfg(feature = "cache_pygates")]
                py_op: OnceLock::new(),
            });
            res.track_instruction_parameters(res.data.len() - 1)?;
        }
        Ok(res)
    }

    /// A constructor for CircuitData from an iterator of PackedInstruction objects
    ///
    /// This is typically useful when iterating over a CircuitData or DAGCircuit
    /// to construct a new CircuitData from the iterator of PackedInstructions. As
    /// such it requires that you have `BitData` and `Interner` objects to run. If
    /// you just wish to build a circuit data from an iterator of instructions
    /// the `from_packed_operations` or `from_standard_gates` constructor methods
    /// are a better choice
    ///
    /// # Args
    ///
    /// * py: A GIL handle this is needed to instantiate Qubits in Python space
    /// * qubits: The BitData to use for the new circuit's qubits
    /// * clbits: The BitData to use for the new circuit's clbits
    /// * qargs_interner: The interner for Qubit objects in the circuit. This must
    ///   contain all the Interned<Qubit> indices stored in the
    ///   PackedInstructions from `instructions`
    /// * cargs_interner: The interner for Clbit objects in the circuit. This must
    ///   contain all the Interned<Clbit> indices stored in the
    ///   PackedInstructions from `instructions`
    /// * qregs: The internal QuantumRegister data stored within the circuit.
    /// * cregs: The internal ClassicalRegister data stored within the circuit.
    /// * qubit_indices: The Mapping between qubit instances and their locations within
    ///   registers in the circuit.
    /// * clbit_indices: The Mapping between clbit instances and their locations within
    ///   registers in the circuit.
    /// * Instructions: An iterator with items of type: `PyResult<PackedInstruction>`
    ///   that contains the instructions to insert in iterator order to the new
    ///   CircuitData. This returns a `PyResult` to facilitate the case where
    ///   you need to make a python copy (such as with `PackedOperation::py_deepcopy()`)
    ///   of the operation while iterating for constructing the new `CircuitData`. An
    ///   example of this use case is in `qiskit_circuit::converters::dag_to_circuit`.
    /// * global_phase: The global phase value to use for the new circuit.
    /// * vars_stretches: variables and stretches to add to the new circuit.
    #[allow(clippy::too_many_arguments)]
    pub fn from_packed_instructions<I>(
        qubits: ObjectRegistry<Qubit, ShareableQubit>,
        clbits: ObjectRegistry<Clbit, ShareableClbit>,
        qargs_interner: Interner<[Qubit]>,
        cargs_interner: Interner<[Clbit]>,
        qregs: RegisterData<QuantumRegister>,
        cregs: RegisterData<ClassicalRegister>,
        qubit_indices: BitLocator<ShareableQubit, QuantumRegister>,
        clbit_indices: BitLocator<ShareableClbit, ClassicalRegister>,
        instructions: I,
        global_phase: Param,
        vars_stretches: Option<&VarStretchContainer>,
    ) -> PyResult<Self>
    where
        I: IntoIterator<Item = PyResult<PackedInstruction>>,
    {
        let instruction_iter = instructions.into_iter();
        let mut res = CircuitData {
            data: Vec::with_capacity(instruction_iter.size_hint().0),
            qargs_interner,
            cargs_interner,
            qubits,
            clbits,
            param_table: ParameterTable::new(),
            global_phase: Param::Float(0.0),
            qregs,
            cregs,
            qubit_indices,
            clbit_indices,
            vars_stretches: vars_stretches
                .cloned()
                .unwrap_or_else(VarStretchContainer::new),
        };

        // use the global phase setter to ensure parameters are registered
        // in the parameter table
        res.set_global_phase(global_phase)?;

        for inst in instruction_iter {
            res.data.push(inst?);
            res.track_instruction_parameters(res.data.len() - 1)?;
        }

        Ok(res)
    }

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
    ///   in Python as loose bits without a register.
    /// * instructions: An iterator of the standard gate params and qubits to
    ///   add to the circuit
    /// * global_phase: The global phase to use for the circuit
    pub fn from_standard_gates<I>(
        num_qubits: u32,
        instructions: I,
        global_phase: Param,
    ) -> PyResult<Self>
    where
        I: IntoIterator<Item = (StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>)>,
    {
        let instruction_iter = instructions.into_iter();
        let mut res =
            Self::with_capacity(num_qubits, 0, instruction_iter.size_hint().0, global_phase)?;

        for (operation, params, qargs) in instruction_iter {
            let qubits = res.qargs_interner.insert(&qargs);
            let params = (!params.is_empty()).then(|| Box::new(params));
            res.data.push(PackedInstruction::from_standard_gate(
                operation, params, qubits,
            ));
            res.track_instruction_parameters(res.data.len() - 1)?;
        }
        Ok(res)
    }

    /// Build an empty CircuitData object with an initially allocated instruction capacity
    pub fn with_capacity(
        num_qubits: u32,
        num_clbits: u32,
        instruction_capacity: usize,
        global_phase: Param,
    ) -> PyResult<Self> {
        let mut res = CircuitData {
            data: Vec::with_capacity(instruction_capacity),
            qargs_interner: Interner::new(),
            cargs_interner: Interner::new(),
            qubits: ObjectRegistry::with_capacity(num_qubits as usize),
            clbits: ObjectRegistry::with_capacity(num_clbits as usize),
            param_table: ParameterTable::new(),
            global_phase: Param::Float(0.0),
            qregs: RegisterData::new(),
            cregs: RegisterData::new(),
            qubit_indices: BitLocator::with_capacity(num_qubits as usize),
            clbit_indices: BitLocator::with_capacity(num_clbits as usize),
            vars_stretches: VarStretchContainer::new(),
        };

        // use the global phase setter to ensure parameters are registered
        // in the parameter table
        res.set_global_phase(global_phase)?;

        if num_qubits > 0 {
            for _i in 0..num_qubits {
                let bit = ShareableQubit::new_anonymous();
                res.add_qubit(bit, true)?;
            }
        }
        if num_clbits > 0 {
            for _i in 0..num_clbits {
                let bit = ShareableClbit::new_anonymous();
                res.add_clbit(bit, true)?;
            }
        }
        Ok(res)
    }

    /// Modify `self` to mark its qubits as physical.
    ///
    /// This deletes the information about the virtual registers, and replaces it with the single
    /// (implicitly) physical register.  This method does not need to traverse the circuit.
    ///
    /// The qubit indices all stay the same; effectively, this is the application of the "trivial"
    /// layout.  If the incoming circuit is supposed to be considered physical, this method can be
    /// used to ensure it is in the canonical physical form.
    ///
    /// # Panics
    ///
    /// If `num_qubits` is less than the number of qubits in the circuit already.
    pub fn make_physical(&mut self, num_qubits: u32) {
        // If this method needs updating, `DAGCircuit::make_physical` probably does too.
        assert!(
            num_qubits as usize >= self.num_qubits(),
            "number of qubits {num_qubits} too small for circuit"
        );
        // The strategy here is just to modify the qubit and quantum register objects entirely
        // inplace; we maintain all relative indices, so we don't need to modify any interner keys.
        let register = QuantumRegister::new_owning("q", num_qubits);
        let mut registry = ObjectRegistry::with_capacity(num_qubits as usize);
        let mut locator = BitLocator::with_capacity(num_qubits as usize);
        for (index, bit) in register.iter().enumerate() {
            registry
                .add(bit.clone(), false)
                .expect("no duplicates, and in-bounds check already performed");
            locator.insert(
                bit,
                BitLocations::new(index as u32, [(register.clone(), index)]),
            );
        }
        let mut register_data = RegisterData::with_capacity(1);
        register_data
            .add_register(register, false)
            .expect("infallible when 'strict=false'");
        self.qubits = registry;
        self.qregs = register_data;
        self.qubit_indices = locator;
    }

    /// Append a standard gate to this CircuitData
    pub fn push_standard_gate(
        &mut self,
        operation: StandardGate,
        params: &[Param],
        qargs: &[Qubit],
    ) -> PyResult<()> {
        let params = (!params.is_empty()).then(|| Box::new(params.iter().cloned().collect()));
        let qubits = self.qargs_interner.insert(qargs);
        self.push(PackedInstruction::from_standard_gate(
            operation, params, qubits,
        ))
    }

    /// Append a packed operation to this CircuitData
    pub fn push_packed_operation(
        &mut self,
        operation: PackedOperation,
        params: &[Param],
        qargs: &[Qubit],
        cargs: &[Clbit],
    ) -> PyResult<()> {
        let params = (!params.is_empty()).then(|| Box::new(params.iter().cloned().collect()));
        let qubits = self.qargs_interner.insert(qargs);
        let clbits = self.cargs_interner.insert(cargs);
        self.push(PackedInstruction {
            op: operation,
            qubits,
            clbits,
            params,
            label: None,
            #[cfg(feature = "cache_pygates")]
            py_op: OnceLock::new(),
        })
    }

    /// Add the entries from the `PackedInstruction` at the given index to the internal parameter
    /// table.
    fn track_instruction_parameters(&mut self, instruction_index: usize) -> PyResult<()> {
        for (index, param) in self.data[instruction_index]
            .params_view()
            .iter()
            .enumerate()
        {
            if matches!(param, Param::Float(_)) {
                continue;
            }
            let usage = ParameterUse::Index {
                instruction: instruction_index,
                parameter: index as u32,
            };
            for symbol in param.iter_parameters()? {
                self.param_table.track(&symbol, Some(usage))?;
            }
        }
        Ok(())
    }

    /// Remove the entries from the `PackedInstruction` at the given index from the internal
    /// parameter table.
    fn untrack_instruction_parameters(&mut self, instruction_index: usize) -> PyResult<()> {
        for (index, param) in self.data[instruction_index]
            .params_view()
            .iter()
            .enumerate()
        {
            if matches!(param, Param::Float(_)) {
                continue;
            }
            let usage = ParameterUse::Index {
                instruction: instruction_index,
                parameter: index as u32,
            };
            for symbol in param.iter_parameters()? {
                self.param_table.untrack(&symbol, usage)?;
            }
        }
        Ok(())
    }

    /// Retrack the entire `ParameterTable`.
    ///
    /// This is necessary each time an insertion or removal occurs on `self.data` other than in the
    /// last position.
    fn reindex_parameter_table(&mut self) -> PyResult<()> {
        self.param_table.clear();

        for inst_index in 0..self.data.len() {
            self.track_instruction_parameters(inst_index)?;
        }
        if matches!(self.global_phase, Param::Float(_)) {
            return Ok(());
        }
        for symbol in self.global_phase.iter_parameters()? {
            self.param_table
                .track(&symbol, Some(ParameterUse::GlobalPhase))?;
        }
        Ok(())
    }

    /// Native internal driver of `__delitem__` that uses a Rust-space version of the
    /// `SequenceIndex`.  This assumes that the `SequenceIndex` contains only in-bounds indices, and
    /// panics if not.
    fn delitem(&mut self, indices: SequenceIndex) -> PyResult<()> {
        // We need to delete in reverse order so we don't invalidate higher indices with a deletion.
        for index in indices.descending() {
            self.data.remove(index);
        }
        if !indices.is_empty() {
            self.reindex_parameter_table()?;
        }
        Ok(())
    }

    fn pack(&mut self, py: Python, inst: &CircuitInstruction) -> PyResult<PackedInstruction> {
        let qubits = self.qargs_interner.insert_owned(
            self.qubits
                .map_objects(inst.qubits.extract::<Vec<ShareableQubit>>(py)?.into_iter())?
                .collect(),
        );
        let clbits = self.cargs_interner.insert_owned(
            self.clbits
                .map_objects(inst.clbits.extract::<Vec<ShareableClbit>>(py)?.into_iter())?
                .collect(),
        );
        Ok(PackedInstruction {
            op: inst.operation.clone(),
            qubits,
            clbits,
            params: (!inst.params.is_empty()).then(|| Box::new(inst.params.clone())),
            label: inst.label.clone(),
            #[cfg(feature = "cache_pygates")]
            py_op: inst.py_op.clone(),
        })
    }

    /// Returns an iterator over all the instructions present in the circuit.
    pub fn iter(&self) -> impl Iterator<Item = &PackedInstruction> {
        self.data.iter()
    }

    /// Get the sorted symbols in this circuit.
    pub fn parameters(&self) -> &[Symbol] {
        self.param_table.symbols()
    }

    /// Get the unsorted symbols in this circuit.
    pub fn iter_parameters(&self) -> impl Iterator<Item = &Symbol> {
        self.param_table.iter_symbols()
    }

    /// Assigns parameters to circuit data based on a slice of `Param`.
    pub fn assign_parameters_from_slice(&mut self, slice: &[Param]) -> PyResult<()> {
        if slice.len() != self.param_table.num_parameters() {
            return Err(PyValueError::new_err(concat!(
                "Mismatching number of values and parameters. For partial binding ",
                "please pass a mapping of {parameter: value} pairs."
            )));
        }
        let mut old_table = std::mem::take(&mut self.param_table);
        self.assign_parameters_inner(
            slice
                .iter()
                .zip(old_table.drain_ordered())
                .map(|(value, (symbol, uses))| (symbol, value.clone(), uses)),
        )
    }

    /// Assigns parameters to circuit data based on a mapping of `ParameterUuid` : `Param`.
    /// This mapping assumes that the provided `ParameterUuid` keys are instances
    /// of `ParameterExpression`.
    pub fn assign_parameters_from_mapping<I, T>(&mut self, iter: I) -> PyResult<()>
    where
        I: IntoIterator<Item = (ParameterUuid, T)>,
        T: AsRef<Param>,
    {
        let mut items = Vec::new();
        for (param_uuid, value) in iter {
            // Assume all the Parameters are already in the circuit
            let symbol = self.get_parameter_by_uuid(param_uuid);
            if let Some(symbol) = symbol {
                items.push((
                    symbol.clone(),
                    value.as_ref().clone(),
                    self.param_table.pop(param_uuid)?,
                ));
            } else {
                return Err(PyValueError::new_err("An invalid parameter was provided."));
            }
        }
        self.assign_parameters_inner(items)
    }

    /// Returns an immutable view of the Interner used for Qargs
    pub fn qargs_interner(&self) -> &Interner<[Qubit]> {
        &self.qargs_interner
    }

    /// Merge the `qargs` in a different [Interner] into this Circuit, remapping the qubits.
    ///
    /// This is useful for simplifying the direct mapping of [PackedInstruction]s from one circuit to
    /// another, like when composing two circuits. See [Interner::merge_map_slice] for more
    /// information on the mapping function.
    ///
    /// The input [InternedMap] is cleared of its previous entries by this method, and then we
    /// re-use the allocation.
    pub fn merge_qargs_using(
        &mut self,
        other: &Interner<[Qubit]>,
        map_fn: impl FnMut(&Qubit) -> Option<Qubit>,
        map: &mut InternedMap<[Qubit]>,
    ) {
        // 4 is an arbitrary guess for the amount of stack space to allocate for mapping the
        // `qargs`, but it doesn't matter if it's too short because it'll safely spill to the heap.
        self.qargs_interner
            .merge_map_slice_using::<4>(other, map_fn, map);
    }

    /// Merge the `qargs` in a different [Interner] into this circuit, remapping the qubits.
    ///
    /// This is useful for simplifying the direct mapping of [PackedInstruction]s from one circuit to
    /// another, like when composing two circuits. See [Interner::merge_map_slice] for more
    /// information on the mapping function.
    pub fn merge_qargs(
        &mut self,
        other: &Interner<[Qubit]>,
        map_fn: impl FnMut(&Qubit) -> Option<Qubit>,
    ) -> InternedMap<[Qubit]> {
        let mut out = InternedMap::new();
        self.merge_qargs_using(other, map_fn, &mut out);
        out
    }

    /// Returns an immutable view of the Interner used for Cargs
    pub fn cargs_interner(&self) -> &Interner<[Clbit]> {
        &self.cargs_interner
    }

    /// Returns an immutable view of the Global Phase `Param` of the circuit
    pub fn global_phase(&self) -> &Param {
        &self.global_phase
    }

    /// Returns an immutable view of the Qubits registered in the circuit
    pub fn qubits(&self) -> &ObjectRegistry<Qubit, ShareableQubit> {
        &self.qubits
    }

    /// Returns an immutable view of the Classical bits registered in the circuit
    pub fn clbits(&self) -> &ObjectRegistry<Clbit, ShareableClbit> {
        &self.clbits
    }

    /// Returns an immutable view of the [QuantumRegister] instances in the circuit.
    pub fn qregs(&self) -> &[QuantumRegister] {
        self.qregs.registers()
    }

    /// Returns an immutable view of the [ClassicalRegister] instances in the circuit.
    pub fn cregs(&self) -> &[ClassicalRegister] {
        self.cregs.registers()
    }

    /// Returns an immutable view of the [QuantumRegister] data struct in the circuit.
    #[inline(always)]
    pub fn qregs_data(&self) -> &RegisterData<QuantumRegister> {
        &self.qregs
    }

    /// Returns an immutable view of the [ClassicalRegister] data struct in the circuit.
    #[inline(always)]
    pub fn cregs_data(&self) -> &RegisterData<ClassicalRegister> {
        &self.cregs
    }

    /// Returns an immutable view of the qubit locations of the [DAGCircuit]
    #[inline(always)]
    pub fn qubit_indices(&self) -> &BitLocator<ShareableQubit, QuantumRegister> {
        &self.qubit_indices
    }

    /// Returns an immutable view of the clbit locations of the [DAGCircuit]
    #[inline(always)]
    pub fn clbit_indices(&self) -> &BitLocator<ShareableClbit, ClassicalRegister> {
        &self.clbit_indices
    }

    /// Unpacks from interned value to `[Qubit]`
    pub fn get_qargs(&self, index: Interned<[Qubit]>) -> &[Qubit] {
        self.qargs_interner().get(index)
    }

    /// Insert qargs into the interner and return the interned value
    pub fn add_qargs(&mut self, qubits: &[Qubit]) -> Interned<[Qubit]> {
        self.qargs_interner.insert(qubits)
    }

    /// Unpacks from InternerIndex to `[Clbit]`
    pub fn get_cargs(&self, index: Interned<[Clbit]>) -> &[Clbit] {
        self.cargs_interner().get(index)
    }

    fn assign_parameters_inner<I, T>(&mut self, iter: I) -> PyResult<()>
    where
        I: IntoIterator<Item = (Symbol, T, HashSet<ParameterUse>)>,
        T: AsRef<Param> + Clone,
    {
        let inconsistent =
            || PyRuntimeError::new_err("internal error: circuit parameter table is inconsistent");

        // Bind a single `Parameter` into a `ParameterExpression`.
        let bind_expr = |expr: &ParameterExpression,
                         symbol: &Symbol,
                         value: &Param,
                         coerce: bool|
         -> PyResult<Param> {
            let new_expr = match value {
                Param::Float(f) => {
                    let map: HashMap<&Symbol, Value> = HashMap::from([(symbol, Value::Real(*f))]);
                    expr.bind(&map, false)?
                }
                Param::ParameterExpression(e) => {
                    let map: HashMap<Symbol, ParameterExpression> =
                        HashMap::from([(symbol.clone(), e.as_ref().clone())]);
                    expr.subs(&map, false)?
                }
                Param::Obj(ob) => {
                    Python::attach(|py| {
                        // The integer handling is only needed to support the case where an int is
                        // passed in directly instead of a float. This will be handled when we add
                        // int to the param enum to support dt target.
                        if let Ok(int) = ob.extract::<i64>(py) {
                            let map: HashMap<&Symbol, Value> =
                                HashMap::from([(symbol, Value::Int(int))]);
                            expr.bind(&map, false).map_err(|x| x.into())
                        } else if let Ok(c) = ob.extract::<Complex64>(py) {
                            let map: HashMap<&Symbol, Value> =
                                HashMap::from([(symbol, Value::Complex(c))]);
                            expr.bind(&map, false).map_err(|x| x.into())
                        } else {
                            Err(PyTypeError::new_err(format!(
                                "Cannot assign object ({ob}) object to parameter."
                            )))
                        }
                    })?
                }
            };
            Param::from_expr(new_expr, coerce)
        };

        let mut user_operations = HashMap::new();
        let mut uuids = Vec::new();
        for (symbol, value, uses) in iter {
            debug_assert!(!uses.is_empty());
            uuids.clear();
            for inner_symbol in value.as_ref().iter_parameters()? {
                uuids.push(self.param_table.track(&inner_symbol, None)?)
            }
            for usage in uses {
                match usage {
                    ParameterUse::GlobalPhase => {
                        let Param::ParameterExpression(expr) = &self.global_phase else {
                            return Err(inconsistent());
                        };
                        self.set_global_phase(bind_expr(expr, &symbol, value.as_ref(), true)?)?;
                    }
                    ParameterUse::Index {
                        instruction,
                        parameter,
                    } => {
                        let parameter = parameter as usize;
                        let previous = &mut self.data[instruction];
                        if let Some(standard) = previous.standard_gate() {
                            let params = previous.params_mut();
                            let Param::ParameterExpression(expr) = &params[parameter] else {
                                return Err(inconsistent());
                            };
                            let new_param = bind_expr(expr, &symbol, value.as_ref(), true)?;

                            // standard gates don't allow for complex parameters
                            if let Param::Obj(expr) = &new_param {
                                return Err(CircuitError::new_err(format!(
                                    "bad type after binding for gate '{}': '{:?}'",
                                    standard.name(),
                                    expr,
                                )));
                            }
                            params[parameter] = new_param.clone();
                            for uuid in uuids.iter() {
                                self.param_table.add_use(*uuid, usage)?
                            }
                            #[cfg(feature = "cache_pygates")]
                            {
                                // Standard gates can all rebuild their definitions, so if the
                                // cached py_op exists, discard it to prompt the instruction
                                // to rebuild its cached python gate upon request later on. This is
                                // done to avoid an unintentional duplicated reference to the same gate
                                // instance in python. For more information, see
                                // https://github.com/Qiskit/qiskit/issues/13504
                                previous.py_op.take();
                            }
                        } else {
                            // Track user operations we've seen so we can rebind their definitions.
                            // Strictly this can add the same binding pair more than once, if an
                            // instruction has the same `Parameter` in several of its `params`, but
                            // we're going to turn that into a `dict` anyway, so it doesn't matter.
                            user_operations
                                .entry(instruction)
                                .or_insert_with(Vec::new)
                                .push((symbol.clone(), value.as_ref().clone()));

                            // This is a Python-only path, since we don't have any operations in
                            // Rust that accept a `Param::ParameterExpression` which aren't standard
                            // gates. Technically `StandardInstruction::Delay` could, but in
                            // practice that's not a common path, and it's only supported for
                            // backwards compatability from before Stretch was introduced. If we did
                            // it in rust without Python that's a mistake and this attach() call
                            // will panic and point out the error of your ways when this comment is
                            // read.
                            Python::attach(|py| {
                                let validate_parameter_attr = intern!(py, "validate_parameter");
                                let assign_parameters_attr = intern!(py, "assign_parameters");

                                let op = previous.unpack_py_op(py)?.into_bound(py);
                                let previous_param = &previous.params_view()[parameter];
                                let new_param = match previous_param {
                                    Param::Float(_) => return Err(inconsistent()),
                                    Param::ParameterExpression(expr) => {
                                        let new_param =
                                            bind_expr(expr, &symbol, value.as_ref(), false)?;
                                        // Historically, `assign_parameters` called `validate_parameter`
                                        // only when a `ParameterExpression` became fully bound.  Some
                                        // "generalised" (or user) gates fail without this, though
                                        // arguably, that's them indicating they shouldn't be allowed to
                                        // be parametric.
                                        //
                                        // Our `bind_expr` coercion means that a non-parametric
                                        // `ParameterExperssion` after binding would have been coerced
                                        // to a numeric quantity already, so the match here is
                                        // definitely parameterized.
                                        match &new_param {
                                            Param::ParameterExpression(expr) => match expr
                                                .try_to_value(true)
                                            {
                                                Ok(_) => {
                                                    // fully bound, validate parameters
                                                    Param::extract_no_coerce(&op.call_method1(
                                                        validate_parameter_attr,
                                                        (new_param,),
                                                    )?)?
                                                }
                                                Err(_) => new_param, // not bound yet, cannot validate
                                            },
                                            new_param => {
                                                Param::extract_no_coerce(&op.call_method1(
                                                    validate_parameter_attr,
                                                    (new_param,),
                                                )?)?
                                            }
                                        }
                                    }
                                    Param::Obj(obj) => {
                                        let obj = obj.bind_borrowed(py);
                                        if !obj.is_instance(QUANTUM_CIRCUIT.get_bound(py))? {
                                            return Err(inconsistent());
                                        }
                                        Param::extract_no_coerce(
                                            &obj.call_method(
                                                assign_parameters_attr,
                                                ([(symbol.clone(), value.as_ref())]
                                                    .into_py_dict(py)?,),
                                                Some(
                                                    &[("inplace", false), ("flat_input", true)]
                                                        .into_py_dict(py)?,
                                                ),
                                            )?,
                                        )?
                                    }
                                };
                                op.getattr(intern!(py, "params"))?
                                    .set_item(parameter, new_param)?;
                                let mut new_op = op.extract::<OperationFromPython>()?;
                                previous.op = new_op.operation;
                                previous.params_mut().swap_with_slice(&mut new_op.params);
                                previous.label = new_op.label;
                                #[cfg(feature = "cache_pygates")]
                                {
                                    previous.py_op = op.unbind().into();
                                }
                                for uuid in uuids.iter() {
                                    self.param_table.add_use(*uuid, usage)?
                                }
                                Ok(())
                            })?;
                        }
                    }
                }
            }
        }

        // handle custom gates, this can only happen in Py-space
        if !user_operations.is_empty() {
            Python::attach(|py| -> PyResult<()> {
                let _definition_attr = intern!(py, "_definition");
                let assign_parameters_attr = intern!(py, "assign_parameters");

                let assign_kwargs = [("inplace", true), ("flat_input", true), ("strict", false)]
                    .into_py_dict(py)
                    .unwrap();
                for (instruction, bindings) in user_operations {
                    // We only put non-standard gates in `user_operations`, so we're not risking creating a
                    // previously non-existent Python object.
                    let instruction = &self.data[instruction];
                    let definition_cache =
                        if matches!(instruction.op.view(), OperationRef::Operation(_)) {
                            // `Operation` instances don't have a `definition` as part of their interfaces, but
                            // they might be an `AnnotatedOperation`, which is one of our special built-ins.
                            // This should be handled more completely in the user-customisation interface by a
                            // delegating method, but that's not the data model we currently have.
                            let py_op = instruction.unpack_py_op(py)?;
                            let py_op = py_op.bind(py);
                            if !py_op.is_instance(ANNOTATED_OPERATION.get_bound(py))? {
                                continue;
                            }
                            py_op
                                .getattr(intern!(py, "base_op"))?
                                .getattr(_definition_attr)?
                        } else {
                            instruction
                                .unpack_py_op(py)?
                                .bind(py)
                                .getattr(_definition_attr)?
                        };
                    if !definition_cache.is_none() {
                        definition_cache.call_method(
                            assign_parameters_attr,
                            (bindings.into_py_dict(py)?.into_any().unbind(),),
                            Some(&assign_kwargs),
                        )?;
                    }
                }
                Ok(())
            })?;
        }
        Ok(())
    }

    /// Retrieves the python `Param` object based on its `ParameterUuid`.
    pub fn get_parameter_by_uuid(&self, uuid: ParameterUuid) -> Option<Symbol> {
        self.param_table.parameter_by_uuid(uuid).cloned()
    }

    /// Get an immutable view of the instructions in the circuit data
    pub fn data(&self) -> &[PackedInstruction] {
        &self.data
    }

    /// Returns an immutable view of the vars and stretches in the circuit
    pub fn get_var_stretch_container(&self) -> &VarStretchContainer {
        &self.vars_stretches
    }

    /// Consume the CircuitData and create an iterator of the [`PackedInstruction`] objects in the
    /// circuit.
    pub fn into_data_iter(self) -> impl Iterator<Item = PackedInstruction> {
        self.data.into_iter()
    }

    /// Remove the label for an instruction in the circuit
    ///
    /// This modifies the circuit in place and sets the label
    /// field of an instruction to ``None``.
    ///
    /// # Arguments
    ///
    /// * index: The index of the instruction in the circuit to remove the label of.
    pub fn invalidate_label(&mut self, index: usize) {
        self.data[index].label = None;
    }

    /// Append a PackedInstruction to the circuit data.
    ///
    /// # Arguments
    ///
    /// * packed: The new packed instruction to insert to the end of the CircuitData
    ///   The qubits and clbits **must** already be present in the interner for this
    ///   function to work. If they are not this will corrupt the circuit.
    pub fn push(&mut self, packed: PackedInstruction) -> PyResult<()> {
        let new_index = self.data.len();
        self.data.push(packed);
        self.track_instruction_parameters(new_index)
    }

    /// Add a param to the current global phase of the circuit
    pub fn add_global_phase(&mut self, value: &Param) -> PyResult<()> {
        match value {
            Param::Obj(_) => Err(PyTypeError::new_err(
                "Invalid parameter type, only float and parameter expression are supported",
            )),
            _ => self.set_global_phase(add_global_phase(&self.global_phase, value)?),
        }
    }

    /// Return a copy of the circuit with instructions in reverse order
    pub fn reverse(&self) -> PyResult<Self> {
        let mut out = self.copy_empty_like_with_capacity(self.data().len(), VarsMode::Alike)?;
        for inst in self.data().iter().rev() {
            out.push(inst.clone())?;
        }
        Ok(out)
    }
}

/// Helper struct for `assign_parameters` to allow use of `Param::extract_no_coerce` in
/// PyO3-provided `FromPyObject` implementations on containers.
#[repr(transparent)]
struct AssignParam(Param);
impl<'py> FromPyObject<'py> for AssignParam {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(Self(Param::extract_no_coerce(ob)?))
    }
}

/// Get the `Vec` of bits referred to by the specifier `specifier`.
///
/// Valid types for `specifier` are integers, bits of the correct type (as given in `type_`), or
/// iterables of one of those two scalar types.  Integers are interpreted as indices into the
/// sequence `bit_sequence`.  All allowed bits must be in `bit_set` (which should implement
/// fast lookup), which is assumed to contain the same bits as `bit_sequence`.
///
/// # Args
///   `specifier` - the Python object specifier to look up.
///   `bit_sequence` - The sequence of bit objects assumed to be the same bits as `bit_set`
///   ` bit_set` - The bit locator, contains the same bits as `bit_sequence`
///
/// # Returns
///     A list of the specified bits from `bits`. The `PyResult` error will contain a Python
///     `CircuitError` if an incorrect type or index is encountered, if the same bit is specified
///     more than once, or if the specifier is to a bit not in the `bit_set`.
fn bit_argument_conversion<B, R>(
    specifier: &Bound<PyAny>,
    bit_sequence: &[B],
    bit_set: &BitLocator<B, R>,
) -> PyResult<Vec<B>>
where
    B: Debug + Clone + Hash + Eq + for<'py> FromPyObject<'py>,
    R: Register + Debug + Clone + Hash + for<'py> FromPyObject<'py>,
{
    // The duplication between this function and `_bit_argument_conversion_scalar` is so that fast
    // paths return as quickly as possible, and all valid specifiers will resolve without needing to
    // try/catch exceptions (which is too slow for inner-loop code).
    if let Ok(bit) = specifier.extract() {
        if bit_set.contains_key(&bit) {
            return Ok(vec![bit]);
        }
        Err(CircuitError::new_err(format!(
            "Bit '{specifier}' is not in the circuit."
        )))
    } else if let Ok(sequence) = specifier.extract::<PySequenceIndex>() {
        match sequence {
            PySequenceIndex::Int(index) => {
                if let Ok(index) = PySequenceIndex::convert_idx(index, bit_sequence.len()) {
                    if let Some(bit) = bit_sequence.get(index).cloned() {
                        return Ok(vec![bit]);
                    }
                }
                Err(CircuitError::new_err(format!(
                    "Index {specifier} out of range for size {}.",
                    bit_sequence.len()
                )))
            }
            _ => {
                let Ok(sequence) = sequence.with_len(bit_sequence.len()) else {
                    return Ok(vec![]);
                };
                Ok(sequence
                    .iter()
                    .map(|index| &bit_sequence[index])
                    .cloned()
                    .collect())
            }
        }
    } else {
        if let Ok(iter) = specifier.try_iter() {
            return iter
                .map(|spec| -> PyResult<B> {
                    bit_argument_conversion_scalar(&spec?, bit_sequence, bit_set)
                })
                .collect::<PyResult<_>>();
        }
        let err_message = if let Ok(bit) = specifier.downcast::<PyBit>() {
            format!(
                "Incorrect bit type: expected '{}' but got '{}'",
                stringify!(B),
                bit.get_type().name()?
            )
        } else {
            format!(
                "Invalid bit index: '{specifier}' of type '{}'",
                specifier.get_type().name()?
            )
        };
        Err(CircuitError::new_err(err_message))
    }
}

fn bit_argument_conversion_scalar<B, R>(
    specifier: &Bound<PyAny>,
    bit_sequence: &[B],
    bit_set: &BitLocator<B, R>,
) -> PyResult<B>
where
    B: Debug + Clone + Hash + Eq + for<'py> FromPyObject<'py>,
    R: Register + Debug + Clone + Hash + for<'py> FromPyObject<'py>,
{
    if let Ok(bit) = specifier.extract() {
        if bit_set.contains_key(&bit) {
            return Ok(bit);
        }
        Err(CircuitError::new_err(format!(
            "Bit '{specifier}' is not in the circuit."
        )))
    } else if let Ok(index) = specifier.extract::<isize>() {
        if let Some(bit) = PySequenceIndex::convert_idx(index, bit_sequence.len())
            .map(|index| bit_sequence.get(index).cloned())
            .map_err(|_| {
                CircuitError::new_err(format!(
                    "Index {specifier} out of range for size {}.",
                    bit_sequence.len()
                ))
            })?
        {
            Ok(bit)
        } else {
            Err(CircuitError::new_err(format!(
                "Index {specifier} out of range for size {}.",
                bit_sequence.len()
            )))
        }
    } else {
        let err_message = if let Ok(bit) = specifier.downcast::<PyBit>() {
            format!(
                "Incorrect bit type: expected '{}' but got '{}'",
                stringify!(B),
                bit.get_type().name()?
            )
        } else {
            format!(
                "Invalid bit index: '{specifier}' of type '{}'",
                specifier.get_type().name()?
            )
        };
        Err(CircuitError::new_err(err_message))
    }
}
