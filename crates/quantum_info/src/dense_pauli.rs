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
#[derive(Clone, Debug, PartialEq)]
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
/// Returns the output Pauli.
pub fn pauli_compose(p1: &Pauli, p2: &Pauli) -> Pauli {
    let x1 = &p1.pauli_x;
    let z1 = &p1.pauli_z;
    let x2 = &p2.pauli_x;
    let z2 = &p2.pauli_z;

    let mut pauli_phase = p1.pauli_phase + p2.pauli_phase;
    pauli_phase += 2 * _count_y(x1, z2);

    let cnt_y1 = _count_y(x1, z1);
    let cnt_y2 = _count_y(x2, z2);
    let pauli_x: Vec<bool> = x1.iter().zip(x2.iter()).map(|(x, y)| x ^ y).collect(); // x = x1 ^ x2
    let pauli_z: Vec<bool> = z1.iter().zip(z2.iter()).map(|(x, y)| x ^ y).collect(); // z = z1 ^ z2
    let cnt_y = _count_y(&pauli_x, &pauli_z);
    pauli_phase = (4 + pauli_phase + cnt_y - cnt_y1 - cnt_y2) % 4;

    Pauli {
        pauli_z,
        pauli_x,
        pauli_phase,
    }
}

/// Evolve a Pauli P by a Clifford C.
/// According to the Heisenberg picture, namely compute Cdag.P.C .
/// Returns the output Pauli.
pub fn evolve_pauli_by_clifford(p: &Pauli, cliff: &Clifford) -> Pauli {
    let pauli_z = &p.pauli_z;
    let pauli_x = &p.pauli_x;
    let pauli_phase = p.pauli_phase;
    let pauli_num_qubits = pauli_z.len();

    let mut out_pauli = Pauli {
        pauli_z: vec![false; pauli_num_qubits],
        pauli_x: vec![false; pauli_num_qubits],
        pauli_phase,
    };

    // decompose p into single qubit paulis on each of the qubits
    for qbit in 0..(pauli_num_qubits) {
        // evolve the singe qubit pauli by cliff
        let pz = pauli_z[qbit];
        let px = pauli_x[qbit];

        if [pz, px] != [false, false] {
            // single qubit pauli is not I (only X, Y, Z)
            let (ev_p_sign, ev_p_z, ev_p_x, ev_p_indices) =
                cliff.evolve_single_qubit_pauli(pz, px, qbit);

            let evolved_pauli_phase = if ev_p_sign { 2u8 } else { 0u8 };
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

            out_pauli = pauli_compose(&out_pauli, &evolved_pauli);
        }
    }

    out_pauli
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
    let pi = Pauli {
        pauli_z: vec![false],
        pauli_x: vec![false],
        pauli_phase: 0,
    }; // I

    let pauli_out = pauli_compose(&px, &py);
    let pauli_exp = Pauli {
        pauli_z: vec![true],
        pauli_x: vec![false],
        pauli_phase: 1,
    }; // -iZ
    assert_eq!(pauli_out, pauli_exp);

    let pauli_out = pauli_compose(&py, &px);
    let pauli_exp = Pauli {
        pauli_z: vec![true],
        pauli_x: vec![false],
        pauli_phase: 3,
    }; // iZ
    assert_eq!(pauli_out, pauli_exp);

    let pauli_out = pauli_compose(&px, &pz);
    let pauli_exp = Pauli {
        pauli_z: vec![true],
        pauli_x: vec![true],
        pauli_phase: 3,
    }; // iY
    assert_eq!(pauli_out, pauli_exp);

    let pauli_out = pauli_compose(&pz, &px);
    let pauli_exp = Pauli {
        pauli_z: vec![true],
        pauli_x: vec![true],
        pauli_phase: 1,
    }; // -iY
    assert_eq!(pauli_out, pauli_exp);

    let pauli_out = pauli_compose(&py, &pz);
    let pauli_exp = Pauli {
        pauli_z: vec![false],
        pauli_x: vec![true],
        pauli_phase: 1,
    }; // -iX
    assert_eq!(pauli_out, pauli_exp);

    let pauli_out = pauli_compose(&pz, &py);
    let pauli_exp = Pauli {
        pauli_z: vec![false],
        pauli_x: vec![true],
        pauli_phase: 3,
    }; // iX
    assert_eq!(pauli_out, pauli_exp);

    let pauli_out = pauli_compose(&pi, &px);
    assert_eq!(pauli_out, px);

    let pauli_out = pauli_compose(&py, &pi);
    assert_eq!(pauli_out, py);

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

    let pauli_out = pauli_compose(&pxx, &pxy);
    let pauli_exp = Pauli {
        pauli_z: vec![true, false],
        pauli_x: vec![false, false],
        pauli_phase: 1,
    }; // -iIZ
    assert_eq!(pauli_out, pauli_exp);

    let pauli_out = pauli_compose(&pxy, &pxx);
    let pauli_exp = Pauli {
        pauli_z: vec![true, false],
        pauli_x: vec![false, false],
        pauli_phase: 3,
    }; // iIZ
    assert_eq!(pauli_out, pauli_exp);
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
    }; // -Z
    let pi = Pauli {
        pauli_z: vec![false],
        pauli_x: vec![false],
        pauli_phase: 0,
    }; // I
    let pi_min = Pauli {
        pauli_z: vec![false],
        pauli_x: vec![false],
        pauli_phase: 2,
    }; // -I

    let pauli_out = evolve_pauli_by_clifford(&px, &cliff_s); // -Y
    assert_eq!(pauli_out, py_min);

    let pauli_out = evolve_pauli_by_clifford(&py, &cliff_s); // X
    assert_eq!(pauli_out, px);

    let pauli_out = evolve_pauli_by_clifford(&pz, &cliff_s); // Z
    assert_eq!(pauli_out, pz);

    let pauli_out = evolve_pauli_by_clifford(&px, &cliff_sdg); // Y
    assert_eq!(pauli_out, py);

    let pauli_out = evolve_pauli_by_clifford(&py, &cliff_sdg); // -X
    assert_eq!(pauli_out, px_min);

    let pauli_out = evolve_pauli_by_clifford(&pz, &cliff_sdg); // Z
    assert_eq!(pauli_out, pz);

    let pauli_out = evolve_pauli_by_clifford(&px, &cliff_h); // Z
    assert_eq!(pauli_out, pz);

    let pauli_out = evolve_pauli_by_clifford(&py, &cliff_h); // -Y
    assert_eq!(pauli_out, py_min);

    let pauli_out = evolve_pauli_by_clifford(&pz, &cliff_h); // X
    assert_eq!(pauli_out, px);

    let pauli_out = evolve_pauli_by_clifford(&pi, &cliff_h); // I
    assert_eq!(pauli_out, pi);

    let pauli_out = evolve_pauli_by_clifford(&pi_min, &cliff_h); // -I
    assert_eq!(pauli_out, pi_min);

    let pauli_out = evolve_pauli_by_clifford(&px, &cliff_sx); // X
    assert_eq!(pauli_out, px);

    let pauli_out = evolve_pauli_by_clifford(&py, &cliff_sx); // -Z
    assert_eq!(pauli_out, pz_min);

    let pauli_out = evolve_pauli_by_clifford(&pz, &cliff_sx); // Y
    assert_eq!(pauli_out, py);

    let pauli_out = evolve_pauli_by_clifford(&px, &cliff_sxdg); // X
    assert_eq!(pauli_out, px);

    let pauli_out = evolve_pauli_by_clifford(&py, &cliff_sxdg); // Z
    assert_eq!(pauli_out, pz);

    let pauli_out = evolve_pauli_by_clifford(&pz, &cliff_sxdg); // -Y
    assert_eq!(pauli_out, py_min);
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
    let pzi = Pauli {
        pauli_z: vec![false, true],
        pauli_x: vec![false, false],
        pauli_phase: 0,
    }; // ZZ
    let pzz_min = Pauli {
        pauli_z: vec![true, true],
        pauli_x: vec![false, false],
        pauli_phase: 2,
    }; // ZI
    let pyx_im = Pauli {
        pauli_z: vec![false, true],
        pauli_x: vec![true, true],
        pauli_phase: 3,
    }; // iYX
    let pyi_im = Pauli {
        pauli_z: vec![false, true],
        pauli_x: vec![false, true],
        pauli_phase: 3,
    }; // iYX
    let pii = Pauli {
        pauli_z: vec![false, false],
        pauli_x: vec![false, false],
        pauli_phase: 0,
    }; // II

    let pauli_out = evolve_pauli_by_clifford(&pxx, &cliff);
    assert_eq!(pauli_out, pxz_min);

    let pauli_out = evolve_pauli_by_clifford(&pyy, &cliff);
    assert_eq!(pauli_out, pzx_min);

    let pauli_out = evolve_pauli_by_clifford(&pxy, &cliff);
    assert_eq!(pauli_out, piy_min);

    let pauli_out = evolve_pauli_by_clifford(&piy_min, &cliff);
    assert_eq!(pauli_out, pzi);

    let pauli_out = evolve_pauli_by_clifford(&pzi, &cliff);
    assert_eq!(pauli_out, pzz_min);

    let pauli_out = evolve_pauli_by_clifford(&pyz_min, &cliff);
    assert_eq!(pauli_out, pxi);

    let pauli_out = evolve_pauli_by_clifford(&pyx_im, &cliff);
    assert_eq!(pauli_out, pyi_im);

    let pauli_out = evolve_pauli_by_clifford(&pii, &cliff);
    assert_eq!(pauli_out, pii);
}
