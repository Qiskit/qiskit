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

use std::fmt::rt::v1::Count::Param;
use itertools::iproduct;
use ndarray::{Array2, ArrayView2};
use smallvec::smallvec;
use qiskit_circuit::operations::StandardGate;
use qiskit_circuit::Qubit;
use crate::synthesis::clifford::utils::{Clifford, CliffordGatesVec};

pub struct BmCliffordSynthesis<'a> {
    /// The Clifford tableau to be synthesized.
    tableau: ArrayView2<'a, bool>,

    /// The total number of qubits.
    num_qubits: usize,
}


/// Return the number of CX gates required for Clifford decomposition.
fn cx_cost(clifford: &Clifford) -> usize {
    if clifford.num_qubits == 2 {
        cx_cost2(&clifford)
    }
    else {
        cx_cost3(&clifford)
    }
}

/// Return CX cost of a 2-qubit clifford.
fn cx_cost2(clifford: &Clifford) -> usize {
    let r00 = rank2(clifford.tableau[[0, 0]], clifford.tableau[[0, 2]], clifford.tableau[[2, 0]], clifford.tableau[[2, 2]]);
    let r01 = rank2(clifford.tableau[[0, 1]], clifford.tableau[[0, 3]], clifford.tableau[[2, 1]], clifford.tableau[[2, 3]]);
    if r00 == 2 {
        r01
    }
    else {
        r01 + 1 - r00
    }
}

/// Return rank of 2x2 boolean matrix.
fn rank2(a: bool, b: bool, c: bool, d: bool) -> usize {
    if (a & d) ^ (b & c) {
        2
    }
    else if a | b | c | d {
        1
    }
    else {
        0
    }
}


/// Return CX cost of a 3-qubit clifford.
fn cx_cost3(clifford: &Clifford) -> usize {
    let n = 3;

    /// create information transfer matrices R1, R2


    0 /// todo!
}


/// Two-qubit cost reduction step.
/// Modifies in-place clifford, gates and cost.
/// Returns
fn reduce_cost(cliff: &mut Clifford, gates: &mut CliffordGatesVec, cost: &mut usize) -> Result<(), String>{
    for qubit0 in 0..cliff.num_qubits {
        for qubit1 in qubit0 + 1 ..cliff.num_qubits {
            for (n0,n1) in iproduct!(0..3, 0..3) {
                // Apply a 2-qubit block
                // todo: this code is obtained by direct porting
                // todo: see if it can be simplified
                if n0 == 1 {
                    cliff.append_v(qubit0);
                }
                else if n0 == 2 {
                    cliff.append_w(qubit0);
                }
                if n1 == 1 {
                    cliff.append_v(qubit1);
                }
                else if n1 == 2 {
                    cliff.append_w(qubit1);
                }
                cliff.append_cx(qubit0, qubit1);

                let new_cost = cx_cost(&cliff);

                if new_cost == cost - 1 {
                    // Add gates in the decomposition.
                    if n0 == 1 {
                        gates.push((StandardGate::SdgGate, smallvec![], smallvec![Qubit(n0)]));
                        gates.push((StandardGate::HGate, smallvec![], smallvec![Qubit(n0)]));
                    }
                    else if n0 == 2 {
                        gates.push((StandardGate::HGate, smallvec![], smallvec![Qubit(n0)]));
                        gates.push((StandardGate::SGate, smallvec![], smallvec![Qubit(n0)]));
                    }
                    if n1 == 1 {
                        gates.push((StandardGate::SdgGate, smallvec![], smallvec![Qubit(n1)]));
                        gates.push((StandardGate::HGate, smallvec![], smallvec![Qubit(n1)]));
                    }
                    else if n1 == 2 {
                        gates.push((StandardGate::HGate, smallvec![], smallvec![Qubit(n1)]));
                        gates.push((StandardGate::SGate, smallvec![], smallvec![Qubit(n1)]));
                    }
                    gates.push((StandardGate::CXGate, smallvec![], smallvec![Qubit(n0), Qubit(n1)]));
                    *cost = new_cost;
                    return Ok(())
                }
            }
        }
    }

    // If we didn't reduce cost.
    Err("Failed to reduce Clifford CX_cost.".to_string())
}


/// Decompose a single-qubit clifford.
/// todo: EXPLAIN ARGS
fn decompose_clifford_1q(cliff: &Clifford, gates: &mut CliffordGatesVec, output_qubit: usize)  {
    let mut clifford_gates = CliffordGatesVec::new();
    let destab_phase = cliff.tableau[[0, 2]];
    let stab_phase = cliff.tableau[[1, 2]];

    if destab_phase && !stab_phase {
        clifford_gates.push((StandardGate::ZGate, smallvec![], smallvec![Qubit(output_qubit)]));
    }
    else if !destab_phase && stab_phase {
        clifford_gates.push((StandardGate::XGate, smallvec![], smallvec![Qubit(output_qubit)]));
    }
    else if destab_phase && stab_phase {
        clifford_gates.push((StandardGate::YGate, smallvec![], smallvec![Qubit(output_qubit)]));
    }

    let destab_x = cliff.tableau[[0, 0]];
    let destab_z = cliff.tableau[[0, 1]];
    let stab_x = cliff.tableau[[1, 0]];
    let stab_z = cliff.tableau[[1, 1]];

    if stab_z && !stab_x {
        if destab_z {
            clifford_gates.push((StandardGate::SGate, smallvec![], smallvec![Qubit(output_qubit)]));
        }
    }
    else if !stab_z && stab_x {
        if destab_x {
            clifford_gates.push((StandardGate::SdgGate, smallvec![], smallvec![Qubit(output_qubit)]));
        }
        clifford_gates.push((StandardGate::HGate, smallvec![], smallvec![Qubit(output_qubit)]));
    }
    else {
        if !destab_z {
            clifford_gates.push((StandardGate::SGate, smallvec![], smallvec![Qubit(output_qubit)]));
        }
        clifford_gates.push((StandardGate::HGate, smallvec![], smallvec![Qubit(output_qubit)]));
        clifford_gates.push((StandardGate::SGate, smallvec![], smallvec![Qubit(output_qubit)]));
    }
}




fn synth_clifford_bm_inner(tableau: ArrayView2<bool>) -> Result<(usize, CliffordGatesVec), String> {
    let tableau_shape = tableau.shape();
    let num_qubits = tableau_shape[0] / 2;

    if num_qubits > 3 {
        return Err("Can only decompose Cliffords up to 3-qubits.".to_string());
    }

    // This is the Clifford that we will reduce to a product of single-qubit Cliffords
    let mut clifford = Clifford { num_qubits: num_qubits, tableau: tableau.to_owned() };

    if num_qubits == 1 {
        let mut gates = CliffordGatesVec::new();
        decompose_clifford_1q(&clifford, &mut gates, 0);
        return Ok((1, gates));
    }

    // After reducing Clifford, we will need to reverse the order of gates.
    let mut reversed_gates = CliffordGatesVec::new();

    // CNOT cost of Clifford.
    let mut cost = cx_cost();
    println!("original cost is {:?}", cost);

    // Find composition of circuits with CX and (H.S)^a gates to reduce CNOT count,
    while cost > 0 {
        println!("current cost is {:?}", cost);
        reduce_cost(&mut clifford, &mut reversed_gates, &mut cost)?;
        println!("reduced cost is {:?}", cost);
    }

    let mut all_gates = CliffordGatesVec::new();

    // Decompose the remaining product of 1-qubit cliffords.
    for qubit in 0..num_qubits {
        let arr = [
            [
                clifford.tableau[[qubit, qubit]],
                clifford.tableau[[qubit, qubit + num_qubits]],
                clifford.tableau[[qubit, 2 * num_qubits]],
            ],
            [
                clifford.tableau[[qubit + num_qubits, qubit]],
                clifford.tableau[[qubit + num_qubits, qubit + num_qubits]],
                clifford.tableau[[qubit + num_qubits, 2 * num_qubits]],
            ]
        ];

        let clifford1q = Clifford { num_qubits: 1, tableau: Array2::from_shape_fn((2, 3), |i, j| arr[[i, j]]) }

        decompose_clifford_1q(&clifford1q, &mut all_gates, qubit);
    }
    all_gates.extend(reversed_gates.iter().rev());

    Ok((num_qubits, all_gates))
}

