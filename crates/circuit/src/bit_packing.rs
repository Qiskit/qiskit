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

use crate::circuit_instruction::CircuitInstruction;
use crate::intern_context::{BitType, IndexType, InternContext};
use hashbrown::HashMap;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::hash::{Hash, Hasher};

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
pub(crate) struct BitAsKey {
    /// Python's `hash()` of the wrapped instance.
    hash: isize,
    /// The wrapped instance.
    bit: PyObject,
}

/// Private type used to store instructions with interned arg lists.
#[derive(Clone, Debug)]
pub(crate) struct PackedInstruction {
    /// The Python-side operation instance.
    pub op: PyObject,
    /// The index under which the interner has stored `qubits`.
    pub qubits_id: IndexType,
    /// The index under which the interner has stored `clbits`.
    pub clbits_id: IndexType,
}

impl BitAsKey {
    pub fn new(bit: &Bound<PyAny>) -> PyResult<Self> {
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
                    .bind(py)
                    .repr()
                    .unwrap()
                    .eq(other.bit.bind(py).repr().unwrap())
                    .unwrap()
            })
    }
}

impl Eq for BitAsKey {}

/// Returns a [PackedInstruction] created by packing a circuit
/// instruction, which involves interning its bit lists.
pub(crate) fn pack(
    py: Python<'_>,
    intern_context: &mut InternContext,
    qubit_indices_native: &HashMap<BitAsKey, BitType>,
    clbit_indices_native: &HashMap<BitAsKey, BitType>,
    inst: PyRef<CircuitInstruction>,
) -> PyResult<PackedInstruction> {
    let mut interned_bits = |indices: &HashMap<BitAsKey, BitType>,
                             bits: &Bound<PyTuple>|
     -> PyResult<IndexType> {
        let args = bits
            .into_iter()
            .map(|b| {
                let key = BitAsKey::new(&b)?;
                indices.get(&key).copied().ok_or_else(|| {
                    PyKeyError::new_err(format!("Bit {:?} has not been added to this circuit.", b))
                })
            })
            .collect::<PyResult<Vec<BitType>>>()?;
        intern_context.intern(args)
    };
    Ok(PackedInstruction {
        op: inst.operation.clone_ref(py),
        qubits_id: interned_bits(qubit_indices_native, inst.qubits.bind(py))?,
        clbits_id: interned_bits(clbit_indices_native, inst.clbits.bind(py))?,
    })
}

pub(crate) fn unpack_qubits(
    py: Python<'_>,
    intern_context: &InternContext,
    qubits_native: &Vec<PyObject>,
    inst: &PackedInstruction,
) -> Vec<PyObject> {
    intern_context
        .lookup(inst.qubits_id)
        .iter()
        .map(|i| qubits_native[*i as usize].clone_ref(py))
        .collect::<Vec<_>>()
}

/// Returns a [CircuitInstruction] created by unpacking the packed
/// instruction.
pub(crate) fn unpack(
    py: Python<'_>,
    intern_context: &InternContext,
    qubits_native: &Vec<PyObject>,
    clbits_native: &Vec<PyObject>,
    inst: &PackedInstruction,
) -> PyResult<Py<CircuitInstruction>> {
    Py::new(
        py,
        CircuitInstruction {
            operation: inst.op.clone_ref(py),
            qubits: PyTuple::new_bound(
                py,
                intern_context
                    .lookup(inst.qubits_id)
                    .iter()
                    .map(|i| qubits_native[*i as usize].clone_ref(py))
                    .collect::<Vec<_>>(),
            )
            .unbind(),
            clbits: PyTuple::new_bound(
                py,
                intern_context
                    .lookup(inst.clbits_id)
                    .iter()
                    .map(|i| clbits_native[*i as usize].clone_ref(py))
                    .collect::<Vec<_>>(),
            )
            .unbind(),
        },
    )
}
