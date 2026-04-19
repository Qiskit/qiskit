// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// use crate::clifford::Clifford;

/// A dense Pauli class
#[derive(Clone, PartialEq)]
pub struct Pauli {
    pub pauli_z: Vec<bool>,
    pub pauli_x: Vec<bool>,
    pub pauli_phase: u8,
}

/// Helper function for counting Y terms
fn _count_y(x: &[bool], z: &[bool]) -> u8 {
    let out = x.iter().zip(z.iter()).filter(|&(&x, &y)| x && y).count();
    out as u8
}

/// Compose the Paulis p1 and p2.
/// Returns the output Pauli in the a sparse ZX format: (z, x, phase).
pub fn pauli_compose(p1: Pauli, p2: Pauli) -> (Vec<bool>, Vec<bool>, u8) {
    let x1 = p1.pauli_x;
    let z1 = p1.pauli_z;
    let x2 = p2.pauli_x;
    let z2 = p2.pauli_z;

    let mut pauli_phase = p1.pauli_phase + p2.pauli_phase;
    pauli_phase += 2 * _count_y(&x1, &z2);

    let cnt_y1 = _count_y(&x1, &z1);
    let cnt_y2 = _count_y(&x2, &z2);
    let pauli_x: Vec<bool> = x1.iter().zip(x2.iter()).map(|(x, y)| x ^ y).collect(); // x = x1 ^ x2
    let pauli_z: Vec<bool> = z1.iter().zip(z2.iter()).map(|(x, y)| x ^ y).collect(); // z = z1 ^ z2
    let cnt_y = _count_y(&pauli_x, &pauli_z);
    pauli_phase = (4 + pauli_phase + cnt_y - cnt_y1 - cnt_y2) % 4;

    (pauli_z, pauli_x, pauli_phase)
}

#[test]
fn test_pauli_compose() {
    let px = Pauli {
        pauli_z: vec![false],
        pauli_x: vec![true],
        pauli_phase: 0,
    }; // X
    let py = Pauli {
        pauli_z: vec![true],
        pauli_x: vec![true],
        pauli_phase: 0,
    }; // Y
    let pz = Pauli {
        pauli_z: vec![true],
        pauli_x: vec![false],
        pauli_phase: 0,
    }; // Z

    let (out_z, out_x, out_phase) = pauli_compose(px.clone(), py.clone());
    let (expect_p_z, expect_p_x, expect_phase) = (vec![true], vec![false], 1); // -iZ
    assert_eq!(out_z, expect_p_z);
    assert_eq!(out_x, expect_p_x);
    assert_eq!(out_phase, expect_phase);

    let (out_z, out_x, out_phase) = pauli_compose(py.clone(), px.clone());
    let (expect_p_z, expect_p_x, expect_phase) = (vec![true], vec![false], 3); // iZ
    assert_eq!(out_z, expect_p_z);
    assert_eq!(out_x, expect_p_x);
    assert_eq!(out_phase, expect_phase);

    let (out_z, out_x, out_phase) = pauli_compose(px.clone(), pz.clone());
    let (expect_p_z, expect_p_x, expect_phase) = (vec![true], vec![true], 3); // iY
    assert_eq!(out_z, expect_p_z);
    assert_eq!(out_x, expect_p_x);
    assert_eq!(out_phase, expect_phase);

    let (out_z, out_x, out_phase) = pauli_compose(pz.clone(), px.clone());
    let (expect_p_z, expect_p_x, expect_phase) = (vec![true], vec![true], 1); // -iY
    assert_eq!(out_z, expect_p_z);
    assert_eq!(out_x, expect_p_x);
    assert_eq!(out_phase, expect_phase);

    let (out_z, out_x, out_phase) = pauli_compose(py.clone(), pz.clone());
    let (expect_p_z, expect_p_x, expect_phase) = (vec![false], vec![true], 1); // -iX
    assert_eq!(out_z, expect_p_z);
    assert_eq!(out_x, expect_p_x);
    assert_eq!(out_phase, expect_phase);

    let (out_z, out_x, out_phase) = pauli_compose(pz.clone(), py.clone());
    let (expect_p_z, expect_p_x, expect_phase) = (vec![false], vec![true], 3); // iX
    assert_eq!(out_z, expect_p_z);
    assert_eq!(out_x, expect_p_x);
    assert_eq!(out_phase, expect_phase);

    let pxx = Pauli {
        pauli_z: vec![false, false],
        pauli_x: vec![true, true],
        pauli_phase: 0,
    }; // XX
    let pxy = Pauli {
        pauli_z: vec![true, false],
        pauli_x: vec![true, true],
        pauli_phase: 0,
    }; // XY

    let (out_z, out_x, out_phase) = pauli_compose(pxx.clone(), pxy.clone());
    let (expect_p_z, expect_p_x, expect_phase) = (vec![true, false], vec![false, false], 1); // -iIZ
    assert_eq!(out_z, expect_p_z);
    assert_eq!(out_x, expect_p_x);
    assert_eq!(out_phase, expect_phase);

    let (out_z, out_x, out_phase) = pauli_compose(pxy.clone(), pxx.clone());
    let (expect_p_z, expect_p_x, expect_phase) = (vec![true, false], vec![false, false], 3); // iIZ
    assert_eq!(out_z, expect_p_z);
    assert_eq!(out_x, expect_p_x);
    assert_eq!(out_phase, expect_phase);
}
