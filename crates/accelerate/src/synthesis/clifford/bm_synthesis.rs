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

use crate::synthesis::clifford::utils::{Clifford, CliffordGatesVec};
use itertools::iproduct;
use ndarray::{arr1, arr2, s, ArrayView2};
use qiskit_circuit::operations::StandardGate;
use qiskit_circuit::Qubit;
use smallvec::smallvec;

/// Return the number of CX-gates required for Clifford decomposition,
/// for either a 2-qubit or a 3-qubit Clifford.
fn cx_cost(clifford: &Clifford) -> usize {
    if clifford.num_qubits == 2 {
        cx_cost2(clifford)
    } else {
        cx_cost3(clifford)
    }
}

/// Return the number of CX gates required for Clifford decomposition
/// for a 2-qubit Clifford.
fn cx_cost2(clifford: &Clifford) -> usize {
    let r00 = rank2(
        clifford.tableau[[0, 0]],
        clifford.tableau[[0, 2]],
        clifford.tableau[[2, 0]],
        clifford.tableau[[2, 2]],
    );
    let r01 = rank2(
        clifford.tableau[[0, 1]],
        clifford.tableau[[0, 3]],
        clifford.tableau[[2, 1]],
        clifford.tableau[[2, 3]],
    );
    if r00 == 2 {
        r01
    } else {
        r01 + 1 - r00
    }
}

/// Return the rank of a 2x2 boolean matrix.
fn rank2(a: bool, b: bool, c: bool, d: bool) -> usize {
    if (a & d) ^ (b & c) {
        2
    } else if a | b | c | d {
        1
    } else {
        0
    }
}

/// Return the number of CX gates required for Clifford decomposition
/// for a 3-qubit Clifford.
fn cx_cost3(clifford: &Clifford) -> usize {
    // create information transfer matrices r1, r2
    let mut r1 = arr2(&[[0, 0, 0], [0, 0, 0], [0, 0, 0]]);
    let mut r2 = arr2(&[[0, 0, 0], [0, 0, 0], [0, 0, 0]]);
    for (q1, q2) in iproduct!(0..3, 0..3) {
        r2[[q1, q2]] = rank2(
            clifford.tableau[[q1, q2]],
            clifford.tableau[[q1, q2 + 3]],
            clifford.tableau[[q1 + 3, q2]],
            clifford.tableau[[q1 + 3, q2 + 3]],
        );
        let mut mask = arr1(&[false, false, false, false, false, false]);
        mask[q2] = true;
        mask[q2 + 3] = true;
        let xs = clifford.tableau.slice(s![q1, 0..6]);
        let zs = clifford.tableau.slice(s![q1 + 3, 0..6]);
        let ys = &xs ^ &zs;
        let is_loc_x = (&xs & &mask) == xs;
        let is_loc_z = (&zs & &mask) == zs;
        let is_loc_y = (&ys & &mask) == ys;
        r1[[q1, q2]] = if is_loc_x || is_loc_z || is_loc_y {
            1
        } else {
            0
        } + if is_loc_x && is_loc_z && is_loc_y {
            1
        } else {
            0
        };
    }

    let mut diag1 = r1.diag().to_vec();
    diag1.sort();
    let mut diag2 = r2.diag().to_vec();
    diag2.sort();

    let nz1 = r1.iter().filter(|v| **v != 0).count();
    let nz2 = r2.iter().filter(|v| **v != 0).count();

    if diag1 == [2, 2, 2] {
        0
    } else if diag1 == [1, 1, 2] {
        1
    } else if (diag1 == [0, 1, 1])
        || (diag1 == [1, 1, 1] && nz2 < 9)
        || (diag1 == [0, 0, 2] && diag2 == [1, 1, 2])
    {
        2
    } else if (diag1 == [1, 1, 1] && nz2 == 9)
        || (diag1 == [0, 0, 1]
            && (nz1 == 1 || diag2 == [2, 2, 2] || (diag2 == [1, 1, 2] && nz2 < 9)))
        || (diag1 == [0, 0, 2] && diag2 == [0, 0, 2])
        || (diag2 == [1, 2, 2] && nz1 == 0)
    {
        3
    } else if diag2 == [0, 0, 1]
        || (diag1 == [0, 0, 0]
            && ((diag2 == [1, 1, 1] && nz2 == 9 && nz1 == 3)
                || (diag2 == [0, 1, 1] && nz2 == 8 && nz1 == 2)))
    {
        5
    } else if nz1 == 3 && nz2 == 3 {
        6
    } else {
        4
    }
}

/// Cost-reduction step. Given a Clifford over 2 or 3 qubits, finds a 2-qubit block
/// (consisting of a single CX-gate, and two single-qubit I/V/W-gates applied to its
/// control and target qubits) that reduces the two-qubit CX-cost of the Clifford
/// (by definition, such a block must always exist).
/// Returns the modified Clifford, the new cost (equal to the original cost - 1),
/// and appends the corresponding gates to the ``gates`` vector. Slightly different
/// from the Python implementation, the gates are already inverted (in practice this
/// means applying an SGate instead of an SdgGate and vice versa), and at the end
/// of the algorithm the vector of computed gates only needs to be reversed (but not
/// also inverted).
fn reduce_cost(
    cliff: &Clifford,
    cost: usize,
    gates: &mut CliffordGatesVec,
) -> Result<(Clifford, usize), String> {
    // All choices for a 2-qubit block
    for qubit0 in 0..cliff.num_qubits {
        for qubit1 in qubit0 + 1..cliff.num_qubits {
            for (n0, n1) in iproduct!(0..3, 0..3) {
                let mut reduced_cliff = cliff.clone();

                // Apply the 2-qubit block and compute the new cost
                if n0 == 1 {
                    reduced_cliff.append_v(qubit0);
                } else if n0 == 2 {
                    reduced_cliff.append_w(qubit0);
                }
                if n1 == 1 {
                    reduced_cliff.append_v(qubit1);
                } else if n1 == 2 {
                    reduced_cliff.append_w(qubit1);
                }
                reduced_cliff.append_cx(qubit0, qubit1);

                let new_cost = cx_cost(&reduced_cliff);

                // If the cost is reduced, we are done.
                // We append the gates from this decomposition.
                if new_cost == cost - 1 {
                    if n0 == 1 {
                        gates.push((
                            StandardGate::SGate,
                            smallvec![],
                            smallvec![Qubit(qubit0 as u32)],
                        ));
                        gates.push((
                            StandardGate::HGate,
                            smallvec![],
                            smallvec![Qubit(qubit0 as u32)],
                        ));
                    } else if n0 == 2 {
                        gates.push((
                            StandardGate::HGate,
                            smallvec![],
                            smallvec![Qubit(qubit0 as u32)],
                        ));
                        gates.push((
                            StandardGate::SdgGate,
                            smallvec![],
                            smallvec![Qubit(qubit0 as u32)],
                        ));
                    }
                    if n1 == 1 {
                        gates.push((
                            StandardGate::SGate,
                            smallvec![],
                            smallvec![Qubit(qubit1 as u32)],
                        ));
                        gates.push((
                            StandardGate::HGate,
                            smallvec![],
                            smallvec![Qubit(qubit1 as u32)],
                        ));
                    } else if n1 == 2 {
                        gates.push((
                            StandardGate::HGate,
                            smallvec![],
                            smallvec![Qubit(qubit1 as u32)],
                        ));
                        gates.push((
                            StandardGate::SdgGate,
                            smallvec![],
                            smallvec![Qubit(qubit1 as u32)],
                        ));
                    }
                    gates.push((
                        StandardGate::CXGate,
                        smallvec![],
                        smallvec![Qubit(qubit0 as u32), Qubit(qubit1 as u32)],
                    ));

                    return Ok((reduced_cliff, new_cost));
                }
            }
        }
    }

    // If all the cost computations are correct, we should never get here.
    Err("Failed to reduce Clifford CX_cost.".to_string())
}

/// Decomposes a single-qubit clifford and appends the corresponding gates to the
/// ``gates`` vector. The ``output_qubit`` specifies for which qubit (in a possibly
/// larger circuit) the decomposition corresponds.
fn decompose_clifford_1q(cliff: &Clifford, gates: &mut CliffordGatesVec, output_qubit: usize) {
    let destab_phase = cliff.tableau[[0, 2]];
    let stab_phase = cliff.tableau[[1, 2]];

    if destab_phase && !stab_phase {
        gates.push((
            StandardGate::ZGate,
            smallvec![],
            smallvec![Qubit(output_qubit as u32)],
        ));
    } else if !destab_phase && stab_phase {
        gates.push((
            StandardGate::XGate,
            smallvec![],
            smallvec![Qubit(output_qubit as u32)],
        ));
    } else if destab_phase && stab_phase {
        gates.push((
            StandardGate::YGate,
            smallvec![],
            smallvec![Qubit(output_qubit as u32)],
        ));
    }

    let destab_x = cliff.tableau[[0, 0]];
    let destab_z = cliff.tableau[[0, 1]];
    let stab_x = cliff.tableau[[1, 0]];
    let stab_z = cliff.tableau[[1, 1]];

    if stab_z && !stab_x {
        if destab_z {
            gates.push((
                StandardGate::SGate,
                smallvec![],
                smallvec![Qubit(output_qubit as u32)],
            ));
        }
    } else if !stab_z && stab_x {
        if destab_x {
            gates.push((
                StandardGate::SdgGate,
                smallvec![],
                smallvec![Qubit(output_qubit as u32)],
            ));
        }
        gates.push((
            StandardGate::HGate,
            smallvec![],
            smallvec![Qubit(output_qubit as u32)],
        ));
    } else {
        if !destab_z {
            gates.push((
                StandardGate::SGate,
                smallvec![],
                smallvec![Qubit(output_qubit as u32)],
            ));
        }
        gates.push((
            StandardGate::HGate,
            smallvec![],
            smallvec![Qubit(output_qubit as u32)],
        ));
        gates.push((
            StandardGate::SGate,
            smallvec![],
            smallvec![Qubit(output_qubit as u32)],
        ));
    }
}

/// Optimal CX-cost decomposition of a Clifford object (represented by ``tableau``)
/// for Cliffords up to 3 qubits.
///
/// This implementation follows the paper "Hadamard-free circuits expose the structure
/// of the Clifford group" by S. Bravyi, D. Maslov (2020), `<https://arxiv.org/abs/2003.09412>`__.
pub fn synth_clifford_bm_inner(
    tableau: ArrayView2<bool>,
) -> Result<(usize, CliffordGatesVec), String> {
    let tableau_shape = tableau.shape();
    let num_qubits = tableau_shape[0] / 2;

    if num_qubits > 3 {
        return Err("Can only decompose Cliffords up to 3-qubits.".to_string());
    }

    // This is the Clifford that we will reduce to a product of single-qubit Cliffords
    let mut clifford = Clifford {
        num_qubits,
        tableau: tableau.to_owned(),
    };

    if num_qubits == 1 {
        let mut gates = CliffordGatesVec::new();
        decompose_clifford_1q(&clifford, &mut gates, 0);
        return Ok((1, gates));
    }

    // After reducing Clifford, we will need to reverse the order of gates.
    let mut reversed_gates = CliffordGatesVec::new();

    // Original CNOT cost of the Clifford.
    let mut cost = cx_cost(&clifford);

    // Iteratively reduce cost by appending 2-qubit blocks consisting of a CX-gate
    // and I/V/W-gates on its control and target qubits.
    while cost > 0 {
        let (reduced_clifford, reduced_cost) = reduce_cost(&clifford, cost, &mut reversed_gates)?;
        clifford = reduced_clifford;
        cost = reduced_cost;
    }

    let mut all_gates = CliffordGatesVec::new();

    // Decompose the remaining cost-0 Clifford into a product of 1-qubit Cliffords.
    for qubit in 0..num_qubits {
        let arr = arr2(&[
            [
                clifford.tableau[[qubit, qubit]],
                clifford.tableau[[qubit, qubit + num_qubits]],
                clifford.tableau[[qubit, 2 * num_qubits]],
            ],
            [
                clifford.tableau[[qubit + num_qubits, qubit]],
                clifford.tableau[[qubit + num_qubits, qubit + num_qubits]],
                clifford.tableau[[qubit + num_qubits, 2 * num_qubits]],
            ],
        ]);

        let clifford1q = Clifford {
            num_qubits: 1,
            tableau: arr,
        };

        decompose_clifford_1q(&clifford1q, &mut all_gates, qubit);
    }

    all_gates.extend(reversed_gates.into_iter().rev());

    Ok((num_qubits, all_gates))
}
