// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::clifford::Clifford;

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

/// Evolve a Pauli P by a Clifford C.
/// According to the Heisenberg picture, namely computing Cdag.P.C .
/// Returns the output Pauli in the a sparse ZX format: (z, x, phase).
pub fn evolve_pauli_by_clifford(p: Pauli, cliff: Clifford) -> (Vec<bool>, Vec<bool>, u8) {
    let pauli_z = p.pauli_z;
    let pauli_x = p.pauli_x;
    let pauli_phase = p.pauli_phase;
    let pauli_num_qubits = pauli_z.len();

    let mut out_z = vec![false; pauli_num_qubits];
    let mut out_x = vec![false; pauli_num_qubits];
    let mut out_phase = pauli_phase;

    // decompose p into single qubit paulis on each of the qubits
    for qbit in 0..(pauli_num_qubits) {
        let out_pauli = Pauli {
            pauli_z: out_z,
            pauli_x: out_x,
            pauli_phase: out_phase,
        };

        // evolve the singe qubit pauli by cliff
        let pz = pauli_z[qbit];
        let px = pauli_x[qbit];
        let (ev_p_sign, ev_p_z, ev_p_x, ev_p_indices) =
            cliff.evolve_single_qubit_pauli(pz, px, qbit);

        let evolved_pauli_phase = if ev_p_sign { 2 as u8 } else { 0 as u8 };
        // transform the evolved pauli to a dense format
        let mut evolved_pauli_z = vec![false; pauli_num_qubits];
        let mut evolved_pauli_x = vec![false; pauli_num_qubits];
        for (&i, (&zv, &xv)) in ev_p_indices.iter().zip(ev_p_z.iter().zip(ev_p_x.iter())) {
            evolved_pauli_z[i as usize] = zv;
            evolved_pauli_x[i as usize] = xv;
        }
        // compose the ouput evolved dense paulies
        let evolved_pauli = Pauli {
            pauli_z: evolved_pauli_z,
            pauli_x: evolved_pauli_x,
            pauli_phase: evolved_pauli_phase,
        };

        (out_z, out_x, out_phase) = pauli_compose(out_pauli, evolved_pauli);
    }

    (out_z, out_x, out_phase)
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

#[test]
fn test_evolve_1_qubit() {
    use ndarray::Array2;

    let cliff_s =
        Clifford::from_array(Array2::from(vec![[true, true, false], [false, true, false]]).view()); // S
    let cliff_h =
        Clifford::from_array(Array2::from(vec![[false, true, false], [true, false, false]]).view()); // H
    let cliff_sdg =
        Clifford::from_array(Array2::from(vec![[true, true, true], [false, true, false]]).view()); // Sdg
    let cliff_sx =
        Clifford::from_array(Array2::from(vec![[true, false, false], [true, true, true]]).view()); // SX
    let cliff_sxdg =
        Clifford::from_array(Array2::from(vec![[true, false, false], [true, true, false]]).view()); // SXdg

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
    let px_min = Pauli {
        pauli_z: vec![false],
        pauli_x: vec![true],
        pauli_phase: 2,
    }; // X
    let py_min = Pauli {
        pauli_z: vec![true],
        pauli_x: vec![true],
        pauli_phase: 2,
    }; // Y
    let pz_min = Pauli {
        pauli_z: vec![true],
        pauli_x: vec![false],
        pauli_phase: 2,
    }; // Z

    let (out_z, out_x, out_phase) = evolve_pauli_by_clifford(px.clone(), cliff_s.clone()); // -Y
    assert_eq!(out_z, (py_min.pauli_z).clone());
    assert_eq!(out_x, (py_min.pauli_x).clone());
    assert_eq!(out_phase, (py_min.pauli_phase).clone());

    let (out_z, out_x, out_phase) = evolve_pauli_by_clifford(py.clone(), cliff_s.clone()); // X
    assert_eq!(out_z, (px.pauli_z).clone());
    assert_eq!(out_x, (px.pauli_x).clone());
    assert_eq!(out_phase, (px.pauli_phase).clone());

    let (out_z, out_x, out_phase) = evolve_pauli_by_clifford(pz.clone(), cliff_s.clone()); // Z
    assert_eq!(out_z, (pz.pauli_z).clone());
    assert_eq!(out_x, (pz.pauli_x).clone());
    assert_eq!(out_phase, (pz.pauli_phase).clone());

    let (out_z, out_x, out_phase) = evolve_pauli_by_clifford(px.clone(), cliff_sdg.clone()); // Y
    assert_eq!(out_z, (py.pauli_z).clone());
    assert_eq!(out_x, (py.pauli_x).clone());
    assert_eq!(out_phase, (py.pauli_phase).clone());

    let (out_z, out_x, out_phase) = evolve_pauli_by_clifford(py.clone(), cliff_sdg.clone()); // -X
    assert_eq!(out_z, (px_min.pauli_z).clone());
    assert_eq!(out_x, (px_min.pauli_x).clone());
    assert_eq!(out_phase, (px_min.pauli_phase).clone());

    let (out_z, out_x, out_phase) = evolve_pauli_by_clifford(pz.clone(), cliff_sdg.clone()); // Z
    assert_eq!(out_z, (pz.pauli_z).clone());
    assert_eq!(out_x, (pz.pauli_x).clone());
    assert_eq!(out_phase, (pz.pauli_phase).clone());

    let (out_z, out_x, out_phase) = evolve_pauli_by_clifford(px.clone(), cliff_h.clone()); // Z
    assert_eq!(out_z, (pz.pauli_z).clone());
    assert_eq!(out_x, (pz.pauli_x).clone());
    assert_eq!(out_phase, (pz.pauli_phase).clone());

    let (out_z, out_x, out_phase) = evolve_pauli_by_clifford(py.clone(), cliff_h.clone()); // -Y
    assert_eq!(out_z, (py_min.pauli_z).clone());
    assert_eq!(out_x, (py_min.pauli_x).clone());
    assert_eq!(out_phase, (py_min.pauli_phase).clone());

    let (out_z, out_x, out_phase) = evolve_pauli_by_clifford(pz.clone(), cliff_h.clone()); // X
    assert_eq!(out_z, (px.pauli_z).clone());
    assert_eq!(out_x, (px.pauli_x).clone());
    assert_eq!(out_phase, (px.pauli_phase).clone());

    let (out_z, out_x, out_phase) = evolve_pauli_by_clifford(px.clone(), cliff_sx.clone()); // X
    assert_eq!(out_z, (px.pauli_z).clone());
    assert_eq!(out_x, (px.pauli_x).clone());
    assert_eq!(out_phase, (px.pauli_phase).clone());

    let (out_z, out_x, out_phase) = evolve_pauli_by_clifford(py.clone(), cliff_sx.clone()); // -Z
    assert_eq!(out_z, (pz_min.pauli_z).clone());
    assert_eq!(out_x, (pz_min.pauli_x).clone());
    assert_eq!(out_phase, (pz_min.pauli_phase).clone());

    let (out_z, out_x, out_phase) = evolve_pauli_by_clifford(pz.clone(), cliff_sx.clone()); // Y
    assert_eq!(out_z, (py.pauli_z).clone());
    assert_eq!(out_x, (py.pauli_x).clone());
    assert_eq!(out_phase, (py.pauli_phase).clone());

    let (out_z, out_x, out_phase) = evolve_pauli_by_clifford(px.clone(), cliff_sxdg.clone()); // X
    assert_eq!(out_z, (px.pauli_z).clone());
    assert_eq!(out_x, (px.pauli_x).clone());
    assert_eq!(out_phase, (px.pauli_phase).clone());

    let (out_z, out_x, out_phase) = evolve_pauli_by_clifford(py.clone(), cliff_sxdg.clone()); // Z
    assert_eq!(out_z, (pz.pauli_z).clone());
    assert_eq!(out_x, (pz.pauli_x).clone());
    assert_eq!(out_phase, (pz.pauli_phase).clone());

    let (out_z, out_x, out_phase) = evolve_pauli_by_clifford(pz.clone(), cliff_sxdg.clone()); // -Y
    assert_eq!(out_z, (py_min.pauli_z).clone());
    assert_eq!(out_x, (py_min.pauli_x).clone());
    assert_eq!(out_phase, (py_min.pauli_phase).clone());
}

#[test]
fn test_evolve_2_qubits() {
    use ndarray::Array2;

    // random clifford, seed=1234
    let cliff = Clifford::from_array(
        Array2::from(vec![
            [false, true, false, true, false],
            [false, true, true, true, true],
            [true, false, true, true, false],
            [true, false, true, false, true],
        ])
        .view(),
    );

    let pxx = Pauli {
        pauli_z: vec![false, false],
        pauli_x: vec![true, true],
        pauli_phase: 0,
    }; // XX
    let pxz_min = Pauli {
        pauli_z: vec![true, false],
        pauli_x: vec![false, true],
        pauli_phase: 2,
    }; // -XZ
    let pxy = Pauli {
        pauli_z: vec![true, false],
        pauli_x: vec![true, true],
        pauli_phase: 0,
    }; // XY
    let piy_min = Pauli {
        pauli_z: vec![true, false],
        pauli_x: vec![true, false],
        pauli_phase: 2,
    }; // -IY
    let pyy = Pauli {
        pauli_z: vec![true, true],
        pauli_x: vec![true, true],
        pauli_phase: 0,
    }; // YY
    let pzx_min = Pauli {
        pauli_z: vec![false, true],
        pauli_x: vec![true, false],
        pauli_phase: 2,
    }; // -ZX
    let pyz_min = Pauli {
        pauli_z: vec![true, true],
        pauli_x: vec![false, true],
        pauli_phase: 2,
    }; // -YZ
    let pxi = Pauli {
        pauli_z: vec![false, false],
        pauli_x: vec![false, true],
        pauli_phase: 0,
    }; // -XI
    let pjyx = Pauli {
        pauli_z: vec![false, true],
        pauli_x: vec![true, true],
        pauli_phase: 3,
    }; // iYX
    let pjyi = Pauli {
        pauli_z: vec![false, true],
        pauli_x: vec![false, true],
        pauli_phase: 3,
    }; // iYX

    let (out_z, out_x, out_phase) = evolve_pauli_by_clifford(pxx.clone(), cliff.clone());
    assert_eq!(out_z, (pxz_min.pauli_z).clone());
    assert_eq!(out_x, (pxz_min.pauli_x).clone());
    assert_eq!(out_phase, (pxz_min.pauli_phase).clone());

    let (out_z, out_x, out_phase) = evolve_pauli_by_clifford(pyy.clone(), cliff.clone());
    assert_eq!(out_z, (pzx_min.pauli_z).clone());
    assert_eq!(out_x, (pzx_min.pauli_x).clone());
    assert_eq!(out_phase, (pzx_min.pauli_phase).clone());

    let (out_z, out_x, out_phase) = evolve_pauli_by_clifford(pxy.clone(), cliff.clone());
    assert_eq!(out_z, (piy_min.pauli_z).clone());
    assert_eq!(out_x, (piy_min.pauli_x).clone());
    assert_eq!(out_phase, (piy_min.pauli_phase).clone());

    let (out_z, out_x, out_phase) = evolve_pauli_by_clifford(pyz_min.clone(), cliff.clone());
    assert_eq!(out_z, (pxi.pauli_z).clone());
    assert_eq!(out_x, (pxi.pauli_x).clone());
    assert_eq!(out_phase, (pxi.pauli_phase).clone());

    let (out_z, out_x, out_phase) = evolve_pauli_by_clifford(pjyx.clone(), cliff.clone());
    assert_eq!(out_z, (pjyi.pauli_z).clone());
    assert_eq!(out_x, (pjyi.pauli_x).clone());
    assert_eq!(out_phase, (pjyi.pauli_phase).clone());
}
