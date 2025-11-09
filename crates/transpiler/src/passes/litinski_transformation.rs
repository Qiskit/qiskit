// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;
use std::fmt;

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::imports::PAULI_EVOLUTION_GATE;
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{
    Operation, OperationRef, Param, PauliProductMeasurement, PyGate, StandardGate,
    StandardInstruction, multiply_param,
};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::{BlocksMode, Qubit, VarsMode};

use crate::TranspilerError;
use num_complex::Complex64;
use qiskit_quantum_info::sparse_observable::{BitTerm, SparseObservable};

use fixedbitset::FixedBitSet;
use smallvec::smallvec;
use std::f64::consts::PI;

// List of gate/instruction names supported by the pass: the pass raises an error if the circuit
// contains instruction with names outside of this list.
static SUPPORTED_INSTRUCTION_NAMES: [&str; 20] = [
    "id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "cx", "cz", "cy", "swap", "iswap", "ecr",
    "dcx", "t", "tdg", "rz", "measure",
];

// List of instruction names which are modified by the pass: the pass is skipped if the circuit
// contains no instructions with names in this list.
static HANDLED_INSTRUCTION_NAMES: [&str; 4] = ["t", "tdg", "rz", "measure"];

#[pyfunction]
#[pyo3(signature = (dag, fix_clifford=true))]
pub fn run_litinski_transformation(
    py: Python,
    dag: &DAGCircuit,
    fix_clifford: bool,
) -> PyResult<Option<DAGCircuit>> {
    let op_counts = dag.get_op_counts();

    // Skip the pass if there are no rotation gates.
    if op_counts
        .keys()
        .all(|k| !HANDLED_INSTRUCTION_NAMES.contains(&k.as_str()))
    {
        return Ok(None);
    }

    // Skip the pass if there are unsupported gates.
    if !op_counts
        .keys()
        .all(|k| SUPPORTED_INSTRUCTION_NAMES.contains(&k.as_str()))
    {
        let unsupported: Vec<_> = op_counts
            .keys()
            .filter(|k| !SUPPORTED_INSTRUCTION_NAMES.contains(&k.as_str()))
            .collect();

        return Err(TranspilerError::new_err(format!(
            "Unable to run Litinski tranformation as the circuit contains instructions not supported by the pass: {:?}",
            unsupported
        )));
    }
    let non_clifford_handled_count: usize = op_counts
        .iter()
        .filter_map(|(k, v)| {
            if HANDLED_INSTRUCTION_NAMES.contains(&k.as_str()) {
                Some(v)
            } else {
                None
            }
        })
        .sum();
    let clifford_count = dag.size(false)? - non_clifford_handled_count;

    let new_dag = dag.copy_empty_like_with_same_capacity(VarsMode::Alike, BlocksMode::Keep)?;
    let mut new_dag = new_dag.into_builder();

    let py_evo_cls = PAULI_EVOLUTION_GATE.get_bound(py);

    let num_qubits = dag.num_qubits();
    let mut clifford = Clifford::identity(num_qubits);

    // Keep track of the update to the global phase (produced when converting T/Tdg gates
    // to RZ-rotations).
    let mut global_phase_update = 0.;

    // Keep track of the clifford operations in the circuit.
    let mut clifford_ops: Vec<&PackedInstruction> = Vec::with_capacity(clifford_count);
    // Apply the Litinski transformation: that is, express a given circuit as a sequence of Pauli
    // product rotations and Pauli product measurements, followed by a final Clifford operator.
    for node_index in dag.topological_op_nodes(false)? {
        // Convert T and Tdg gates to RZ rotations.
        if let NodeType::Operation(inst) = &dag[node_index] {
            let name = inst.op.name();

            match inst.op.view() {
                OperationRef::StandardGate(StandardGate::I) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                }
                OperationRef::StandardGate(StandardGate::X) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_x(dag.get_qargs(inst.qubits)[0].index())
                }
                OperationRef::StandardGate(StandardGate::Y) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_y(dag.get_qargs(inst.qubits)[0].index())
                }
                OperationRef::StandardGate(StandardGate::Z) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_z(dag.get_qargs(inst.qubits)[0].index())
                }
                OperationRef::StandardGate(StandardGate::H) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_h(dag.get_qargs(inst.qubits)[0].index())
                }
                OperationRef::StandardGate(StandardGate::S) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_s(dag.get_qargs(inst.qubits)[0].index())
                }
                OperationRef::StandardGate(StandardGate::Sdg) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_sdg(dag.get_qargs(inst.qubits)[0].index())
                }
                OperationRef::StandardGate(StandardGate::SX) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_sx(dag.get_qargs(inst.qubits)[0].index())
                }
                OperationRef::StandardGate(StandardGate::SXdg) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_sxdg(dag.get_qargs(inst.qubits)[0].index())
                }
                OperationRef::StandardGate(StandardGate::CX) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_cx(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    )
                }
                OperationRef::StandardGate(StandardGate::CZ) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_cz(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    )
                }
                OperationRef::StandardGate(StandardGate::CY) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_cy(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    )
                }
                OperationRef::StandardGate(StandardGate::Swap) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_swap(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    )
                }
                OperationRef::StandardGate(StandardGate::ISwap) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_iswap(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    )
                }
                OperationRef::StandardGate(StandardGate::ECR) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_ecr(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    )
                }
                OperationRef::StandardGate(StandardGate::DCX) => {
                    if fix_clifford {
                        clifford_ops.push(inst);
                    }
                    clifford.append_dcx(
                        dag.get_qargs(inst.qubits)[0].index(),
                        dag.get_qargs(inst.qubits)[1].index(),
                    )
                }
                OperationRef::StandardGate(StandardGate::T)
                | OperationRef::StandardGate(StandardGate::Tdg)
                | OperationRef::StandardGate(StandardGate::RZ) => {
                    // Convert T and Tdg gates to RZ rotations
                    let (angle, phase_update) = match inst.op.view() {
                        OperationRef::StandardGate(StandardGate::T) => {
                            (Param::Float(PI / 4.), PI / 8.)
                        }
                        OperationRef::StandardGate(StandardGate::Tdg) => {
                            (Param::Float(-PI / 4.0), -PI / 8.)
                        }
                        OperationRef::StandardGate(StandardGate::RZ) => {
                            let param = &inst.params_view()[0];
                            (param.clone(), 0.)
                        }
                        _ => {
                            unreachable!("We cannot have gates other than T/Tdg/RZ at this point.");
                        }
                    };
                    global_phase_update += phase_update;

                    // Evolve the single-qubit Pauli-Z with Z on the given qubit.
                    // Returns the evolved Pauli in the sparse format: (sign, pauli z, pauli x, indices),
                    // where signs `true` and `false` correspond to coefficients `-1` and `+1` respectively.
                    let (sign, terms, indices) =
                        clifford.get_inverse_z(dag.get_qargs(inst.qubits)[0].index());
                    let coeffs = vec![Complex64::new(1., 0.)];
                    let terms_len = terms.len() as u32;
                    let boundaries = vec![0, terms_len as usize];
                    // SAFETY: This is computed from the clifford and has a known size based on
                    // the returned terms that is always valid.
                    let obs = unsafe {
                        SparseObservable::new_unchecked(
                            terms_len,
                            coeffs,
                            terms,
                            (0..terms_len).collect(),
                            boundaries,
                        )
                    };

                    let time = if sign {
                        multiply_param(&angle, -0.5)
                    } else {
                        multiply_param(&angle, 0.5)
                    };
                    let py_evo = py_evo_cls.call1((obs, time.clone()))?;
                    let py_gate = PyGate {
                        qubits: indices.len() as u32,
                        clbits: 0,
                        params: 1,
                        op_name: "PauliEvolution".to_string(),
                        gate: py_evo.into(),
                    };

                    new_dag.apply_operation_back(
                        py_gate.into(),
                        &indices,
                        &[],
                        Some(Parameters::Params(smallvec![time])),
                        None,
                        #[cfg(feature = "cache_pygates")]
                        None,
                    )?;
                }
                OperationRef::StandardInstruction(StandardInstruction::Measure) => {
                    // Returns the evolved Pauli in the sparse format: (sign, pauli z, pauli x, indices),
                    // where signs `true` and `false` correspond to coefficients `-1` and `+1` respectively.
                    let (sign, z, x, indices) = clifford
                        .get_inverse_z_for_measurement(dag.get_qargs(inst.qubits)[0].index());
                    let ppm = PauliProductMeasurement { z, x, neg: sign };

                    let ppm_clbits = dag.get_cargs(inst.clbits);

                    new_dag.apply_operation_back(
                        ppm.into(),
                        &indices,
                        ppm_clbits,
                        None,
                        None,
                        #[cfg(feature = "cache_pygates")]
                        None,
                    )?;
                }
                _ => unreachable!(
                    "We cannot have unsupported names at this step of Litinski Transformation: {}",
                    name
                ),
            }
        }
    }

    new_dag.add_global_phase(&Param::Float(global_phase_update))?;

    // Add Clifford gates to the Qiskit circuit (when required).
    // Since we aim to preserve the global phase of the circuit, we add the Clifford operations from
    // the original circuit (and not the final Clifford operator).
    if fix_clifford {
        for inst in clifford_ops.into_iter() {
            new_dag.push_back(inst.clone())?;
        }
    }

    Ok(Some(new_dag.build()))
}

pub fn litinski_transformation_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_litinski_transformation))?;
    Ok(())
}

/// SIMD accelerated Clifford.
struct Clifford {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Matrix with dimensions (2 * num_qubits) x (2 * num_qubits + 1).
    pub tableau: Vec<FixedBitSet>,
}

impl Clifford {
    /// Creates the identity Clifford on num_qubits
    fn identity(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            tableau: (0..2 * num_qubits + 1)
                .map(|i| {
                    let mut row = FixedBitSet::with_capacity(2 * num_qubits);
                    // SAFETY: We know row is large enough since it's larger than the range
                    // i is from
                    unsafe {
                        row.insert_unchecked(i);
                    }
                    row
                })
                .collect(),
        }
    }

    fn get_phase_mut(&mut self) -> &mut FixedBitSet {
        self.tableau.get_mut(2 * self.num_qubits).unwrap()
    }

    fn get_phase(&self) -> &FixedBitSet {
        self.tableau.get(2 * self.num_qubits).unwrap()
    }

    fn get_z(&self, qubit: usize) -> &FixedBitSet {
        self.tableau.get(self.num_qubits + qubit).unwrap()
    }

    fn get_z_mut(&mut self, qubit: usize) -> &mut FixedBitSet {
        self.tableau.get_mut(self.num_qubits + qubit).unwrap()
    }

    /// Modifies the tableau in-place by appending S-gate
    fn append_s(&mut self, qubit: usize) {
        let x_and_z = &self.tableau[qubit] & self.get_z(qubit);
        *self.get_phase_mut() ^= x_and_z;
        let xor = self.get_z(qubit) ^ &self.tableau[qubit];
        *self.get_z_mut(qubit) = xor;
    }

    /// Modifies the tableau in-place by appending Sdg-gate
    #[allow(dead_code)]
    fn append_sdg(&mut self, qubit: usize) {
        let x_and_not_z = if let Some(x) = self.tableau.get(qubit) {
            let mut not_z = self.get_z(qubit).clone();
            not_z.toggle_range(..);
            x & &not_z
        } else {
            unreachable!();
        };
        *self.get_phase_mut() ^= x_and_not_z;
        let xor = &self.tableau[qubit] ^ self.get_z(qubit);
        *self.get_z_mut(qubit) = xor;
    }

    /// Modifies the tableau in-place by appending SX-gate
    fn append_sx(&mut self, qubit: usize) {
        let not_x_and_z = if let Some(x) = self.tableau.get(qubit) {
            let z = self.get_z(qubit);
            let mut not_x = x.clone();
            not_x.toggle_range(..);
            &not_x & z
        } else {
            unreachable!();
        };
        *self.get_phase_mut() ^= not_x_and_z;
        let xor = &self.tableau[qubit] ^ self.get_z(qubit);
        self.tableau[qubit] = xor;
    }

    /// Modifies the tableau in-place by appending SXDG-gate
    fn append_sxdg(&mut self, qubit: usize) {
        let x_and_z = &self.tableau[qubit] & self.get_z(qubit);
        *self.get_phase_mut() ^= x_and_z;
        let xor = &self.tableau[qubit] ^ self.get_z(qubit);
        self.tableau[qubit] = xor;
    }

    /// Modifies the tableau in-place by appending H-gate
    fn append_h(&mut self, qubit: usize) {
        let x_and_z = if let Some(x) = self.tableau.get(qubit) {
            let z = self.get_z(qubit);
            x & z
        } else {
            unreachable!();
        };
        *self.get_phase_mut() ^= x_and_z;
        self.tableau.swap(qubit, self.num_qubits + qubit);
    }

    /// Modifies the tableau in-place by appending SWAP-gate
    fn append_swap(&mut self, qubit0: usize, qubit1: usize) {
        self.tableau.swap(qubit0, qubit1);
        self.tableau
            .swap(self.num_qubits + qubit0, self.num_qubits + qubit1);
    }

    /// Modifies the tableau in-place by appending CX-gate
    fn append_cx(&mut self, qubit0: usize, qubit1: usize) {
        let val = if let Some(x0) = self.tableau.get(qubit0) {
            let z0 = self.get_z(qubit0);
            let x1 = &self.tableau[qubit1];
            let z1 = self.get_z(qubit1);

            let mut x1_xor_z0 = x1 ^ z0;
            x1_xor_z0.toggle_range(..);
            let tmp = &x1_xor_z0 & z1;
            &tmp & x0
        } else {
            unreachable!();
        };
        *self.get_phase_mut() ^= val;
        let xor_x = &self.tableau[qubit1] ^ &self.tableau[qubit0];
        let xor_z = self.get_z(qubit0) ^ self.get_z(qubit1);
        self.tableau[qubit1] = xor_x;
        *self.get_z_mut(qubit0) = xor_z;
    }

    /// Modifies the tableau in-place by appending CZ-gate
    fn append_cz(&mut self, qubit0: usize, qubit1: usize) {
        let val = if let Some(x0) = self.tableau.get(qubit0) {
            let z0 = self.get_z(qubit0);
            let x1 = &self.tableau[qubit1];
            let z1 = self.get_z(qubit1);
            let z0_xor_z1 = z0 ^ z1;
            &(x0 & x1) & &z0_xor_z1
        } else {
            unreachable!();
        };
        *self.get_phase_mut() ^= val;
        let xor_z1_x0 = self.get_z(qubit1) ^ &self.tableau[qubit0];
        let xor_z0_x1 = self.get_z(qubit0) ^ &self.tableau[qubit1];
        *self.get_z_mut(qubit1) = xor_z1_x0;
        *self.get_z_mut(qubit0) = xor_z0_x1;
    }

    /// Modifies the tableau in-place by appending CY-gate
    /// (todo: rewrite using native tableau manipulations)
    fn append_cy(&mut self, qubit0: usize, qubit1: usize) {
        self.append_sdg(qubit1);
        self.append_cx(qubit0, qubit1);
        self.append_s(qubit1);
    }

    /// Modifies the tableau in-place by appending X-gate
    fn append_x(&mut self, qubit: usize) {
        let xor = self.get_phase() ^ self.get_z(qubit);
        *self.get_phase_mut() = xor;
    }

    /// Modifies the tableau in-place by appending Z-gate
    fn append_z(&mut self, qubit: usize) {
        let xor = self.get_phase() ^ &self.tableau[qubit];
        *self.get_phase_mut() = xor;
    }

    /// Modifies the tableau in-place by appending Y-gate
    fn append_y(&mut self, qubit: usize) {
        let xor = &self.tableau[qubit] ^ self.get_z(qubit);
        *self.get_phase_mut() ^= xor;
    }

    /// Modifies the tableau in-place by appending iSWAP-gate
    /// (todo: rewrite using native tableau manipulations)
    fn append_iswap(&mut self, qubit0: usize, qubit1: usize) {
        self.append_s(qubit0);
        self.append_s(qubit1);
        self.append_h(qubit0);
        self.append_cx(qubit0, qubit1);
        self.append_cx(qubit1, qubit0);
        self.append_h(qubit1);
    }

    /// Modifies the tableau in-place by appending ECR-gate
    /// (todo: rewrite using native tableau manipulations)
    fn append_ecr(&mut self, qubit0: usize, qubit1: usize) {
        self.append_s(qubit0);
        self.append_sx(qubit1);
        self.append_cx(qubit0, qubit1);
        self.append_x(qubit0);
    }

    /// Modifies the tableau in-place by appending DCX-gate
    /// (todo: rewrite using native tableau manipulations)
    fn append_dcx(&mut self, qubit0: usize, qubit1: usize) {
        self.append_cx(qubit0, qubit1);
        self.append_cx(qubit1, qubit0);
    }

    /// Modifies the tableau in-place by appending V-gate.
    /// This is equivalent to an Sdg gate followed by an H gate.
    #[allow(dead_code)]
    fn append_v(&mut self, qubit: usize) {
        let xor = &self.tableau[qubit] & self.get_z(qubit);
        self.tableau.swap(qubit, self.num_qubits + qubit);
        self.tableau[qubit] = xor;
    }

    /// Modifies the tableau in-place by appending W-gate.
    /// This is equivalent to two V gates.
    #[allow(dead_code)]
    fn append_w(&mut self, qubit: usize) {
        let xor = &self.tableau[qubit] & self.get_z(qubit);
        self.tableau.swap(qubit, self.num_qubits + qubit);
        *self.get_z_mut(qubit) = xor;
    }

    /// Evolving the single-qubit Pauli-Z with Z on qubit qbit.
    /// Returns the evolved Pauli in the sparse format: (sign, paulis, indices).
    fn get_inverse_z(&self, qbit: usize) -> (bool, Vec<BitTerm>, Vec<Qubit>) {
        // Potentially overallocated, but this is temporary in the only use from litinski transform.
        let mut bit_terms = Vec::with_capacity(self.num_qubits);
        let mut pauli_indices = Vec::<usize>::with_capacity(2 * self.num_qubits);
        // Compute the y-count to avoid recomputing it later
        let mut pauli_y_count: u32 = 0;

        let indices = (0..self.num_qubits)
            .filter_map(|i| {
                let x_bit = self.tableau[qbit][i + self.num_qubits];
                let z_bit = self.tableau[qbit][i];
                match [z_bit, x_bit] {
                    [true, true] => {
                        pauli_y_count += 1;
                        bit_terms.push(BitTerm::Y);
                        pauli_indices.push(i);
                        pauli_indices.push(i + self.num_qubits);
                        Some(Qubit::new(i))
                    }
                    [false, true] => {
                        bit_terms.push(BitTerm::X);
                        pauli_indices.push(i);
                        Some(Qubit::new(i))
                    }
                    [true, false] => {
                        bit_terms.push(BitTerm::Z);
                        pauli_indices.push(i + self.num_qubits);
                        Some(Qubit::new(i))
                    }
                    [false, false] => None,
                }
            })
            .collect();

        let phase = compute_phase_product_pauli(self, &pauli_indices, pauli_y_count);
        (phase, bit_terms, indices)
    }
    pub fn get_inverse_z_for_measurement(
        &self,
        qbit: usize,
    ) -> (bool, Vec<bool>, Vec<bool>, Vec<Qubit>) {
        let mut z = Vec::with_capacity(self.num_qubits);
        let mut x = Vec::with_capacity(self.num_qubits);
        let mut indices = Vec::with_capacity(self.num_qubits);
        let mut pauli_indices = Vec::<usize>::with_capacity(2 * self.num_qubits);
        // Compute the y-count to avoid recomputing it later
        let mut pauli_y_count: u32 = 0;
        for i in 0..self.num_qubits {
            let z_bit = self.tableau[qbit][i];
            let x_bit = self.tableau[qbit][i + self.num_qubits];
            if z_bit || x_bit {
                z.push(z_bit);
                x.push(x_bit);
                indices.push(Qubit::new(i));
                if x_bit {
                    pauli_indices.push(i);
                }
                if z_bit {
                    pauli_indices.push(i + self.num_qubits);
                }
                pauli_y_count += (x_bit && z_bit) as u32;
            }
        }
        let phase = compute_phase_product_pauli(self, &pauli_indices, pauli_y_count);

        (phase, z, x, indices)
    }
}

/// Computes the sign (either +1 or -1) when conjugating a Pauli by a Clifford
fn compute_phase_product_pauli(
    clifford: &Clifford,
    pauli_indices: &[usize],
    pauli_y_count: u32,
) -> bool {
    let phase = pauli_indices.iter().fold(false, |acc, &pauli_index| {
        acc ^ (clifford.tableau[2 * clifford.num_qubits][pauli_index])
    });

    let mut ifact: u8 = pauli_y_count as u8 % 4;

    for j in 0..clifford.num_qubits {
        let mut x = false;
        let mut z = false;
        for &pauli_index in pauli_indices.iter() {
            let x1: bool = clifford.tableau[j][pauli_index];
            let z1: bool = clifford.tableau[j + clifford.num_qubits][pauli_index];

            match (x1, z1, x, z) {
                (false, true, true, true)
                | (true, false, false, true)
                | (true, true, true, false) => {
                    ifact += 1;
                }
                (false, true, true, false)
                | (true, false, true, true)
                | (true, true, false, true) => {
                    ifact += 3;
                }
                _ => {}
            };
            x ^= x1;
            z ^= z1;
            ifact %= 4;
        }
    }
    (((ifact % 4) >> 1) != 0) ^ phase
}

impl fmt::Debug for Clifford {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        writeln!(f, "Tableau:")?;
        for i in 0..2 * self.num_qubits {
            for j in 0..2 * self.num_qubits + 1 {
                write!(f, "{} ", self.tableau[j][i] as u8)?;
            }
            writeln!(f)?;
        }
        writeln!(f)?;
        Ok(())
    }
}
