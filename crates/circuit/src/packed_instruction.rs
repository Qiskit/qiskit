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

use crate::circuit_instruction::CircuitInstruction;
use crate::slotted_cache::CacheSlot;
use crate::Interner;
use crate::{Clbit, PyNativeMapper, Qubit};
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Private type used to store instructions with interned arg lists.
#[derive(Clone, Debug)]
pub(crate) struct PackedInstruction {
    /// The Python-side operation instance.
    pub op: PyObject,
    /// The index under which the interner has stored `qubits`.
    pub qubits_id: CacheSlot,
    /// The index under which the interner has stored `clbits`.
    pub clbits_id: CacheSlot,
}

/// This trait exists to provide a shared implementation of
/// instruction packing to Rust `pyclass` types that deal in
/// terms of Python CircuitInstruction instances in their
/// public API but need to store [PackedInstruction]
/// instances, internally.
pub(crate) trait InstructionPacker {
    /// Packs the provided instruction.
    fn pack(
        &mut self,
        py: Python,
        instruction: PyRef<CircuitInstruction>,
    ) -> PyResult<PackedInstruction>;
    /// Packs the provided list of qubits via interning.
    fn pack_qubits(&mut self, bits: &Bound<PyTuple>) -> PyResult<CacheSlot>;
    /// Packs the provided list of clbits via interning.
    fn pack_clbits(&mut self, bits: &Bound<PyTuple>) -> PyResult<CacheSlot>;
    /// Unpacks the provided packed instruction.
    fn unpack(&self, py: Python, packed: &PackedInstruction) -> PyResult<CircuitInstruction>;
    /// Unpacks qubits by retrieving the provided slot from the interner.
    fn unpack_qubits<'py>(&self, py: Python<'py>, bits_id: &CacheSlot) -> Bound<'py, PyTuple>;
    /// Unpacks clbits by retrieving the provided slot from the interner.
    fn unpack_clbits<'py>(&self, py: Python<'py>, bits_id: &CacheSlot) -> Bound<'py, PyTuple>;
}

impl<T> InstructionPacker for T
where
    T: Interner<Vec<Qubit>, InternedType = CacheSlot, Error = PyErr>
        + Interner<Vec<Clbit>, InternedType = CacheSlot, Error = PyErr>
        + PyNativeMapper<Qubit>
        + PyNativeMapper<Clbit>,
{
    fn pack(
        &mut self,
        py: Python,
        instruction: PyRef<CircuitInstruction>,
    ) -> PyResult<PackedInstruction> {
        Ok(PackedInstruction {
            op: instruction.operation.clone_ref(py),
            qubits_id: self.pack_qubits(instruction.qubits.bind(py))?,
            clbits_id: self.pack_clbits(instruction.clbits.bind(py))?,
        })
    }

    fn pack_qubits(&mut self, bits: &Bound<PyTuple>) -> PyResult<CacheSlot> {
        self.intern(
            bits.into_iter()
                .map(|b| {
                    self.map_to_native(&b).ok_or_else(|| {
                        PyKeyError::new_err(format!(
                            "Bit {:?} has not been added to this circuit.",
                            b
                        ))
                    })
                })
                .collect::<PyResult<Vec<Qubit>>>()?,
        )
    }

    fn pack_clbits(&mut self, bits: &Bound<PyTuple>) -> PyResult<CacheSlot> {
        self.intern(
            bits.into_iter()
                .map(|b| {
                    self.map_to_native(&b).ok_or_else(|| {
                        PyKeyError::new_err(format!(
                            "Bit {:?} has not been added to this circuit.",
                            b
                        ))
                    })
                })
                .collect::<PyResult<Vec<Clbit>>>()?,
        )
    }

    fn unpack(&self, py: Python, packed: &PackedInstruction) -> PyResult<CircuitInstruction> {
        Ok(CircuitInstruction {
            operation: packed.op.clone_ref(py),
            qubits: self.unpack_qubits(py, &packed.qubits_id).unbind(),
            clbits: self.unpack_clbits(py, &packed.clbits_id).unbind(),
        })
    }

    fn unpack_qubits<'py>(&self, py: Python<'py>, bits_id: &CacheSlot) -> Bound<'py, PyTuple> {
        let bits: Vec<Qubit> = self.get_interned(bits_id);
        PyTuple::new_bound(
            py,
            bits.into_iter()
                .map(|i| self.map_to_py(i).unwrap().clone_ref(py))
                .collect::<Vec<_>>(),
        )
    }

    fn unpack_clbits<'py>(&self, py: Python<'py>, bits_id: &CacheSlot) -> Bound<'py, PyTuple> {
        let bits: Vec<Clbit> = self.get_interned(bits_id);
        PyTuple::new_bound(
            py,
            bits.into_iter()
                .map(|i| self.map_to_py(i).unwrap().clone_ref(py))
                .collect::<Vec<_>>(),
        )
    }
}
