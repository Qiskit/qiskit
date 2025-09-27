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

use ndarray::{s, Array1, Array2, ArrayView1};

/// Clifford.
#[derive(Clone)]
pub struct Clifford {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Matrix with dimensions (2 * num_qubits) x (2 * num_qubits).
    pub tableau: Array2<bool>,
    /// Phases.
    pub phase: Array1<bool>,
}

/// Pauli.
#[derive(Clone, PartialEq)]
pub struct Pauli {
    pub z: Array1<bool>,
    pub x: Array1<bool>,
    pub phase: u8,
}

impl Pauli {
    /// Composition of two Paulis: p1 and p2
    fn compose(p1: Pauli, p2: Pauli) -> Pauli {
        let x1 = p1.x;
        let z1 = p1.z;
        let x2 = p2.x;
        let z2 = p2.z;

        let mut phase = p1.phase + p2.phase;
        //if front {
        phase += 2 * _count_y(x1.view(), z2.view());
        //}
        //else {
        // phase += 2 * _count_y(x2.view(), z1.view());
        //}

        let cnt_y1 = _count_y(x1.view(), z1.view());
        let cnt_y2 = _count_y(x2.view(), z2.view());
        let x = x1 ^ x2;
        let z = z1 ^ z2;
        let cnt_y = _count_y(x.view(), z.view());
        phase = (4 + phase + cnt_y - cnt_y1 - cnt_y2) % 4;
        // println!("{} {} {} {}", cnt_y, cnt_y1, cnt_y2, phase);
        let ret = Pauli { x, z, phase };
        ret
    }

    /// Evolve a Pauli p by a Clifford cliff, with frame="s".
    /// For frame="h" we should calculate the adjoint of the Clifford.
    fn evolve(p: Pauli, cliff: Clifford) -> Pauli {
        let n = cliff.num_qubits;
        let cliff_phase = cliff.phase;
        let mut ret = Pauli {
            x: Array1::from(vec![false; n]),
            z: Array1::from(vec![false; n]),
            phase: 0,
        };

        for (id_z, &z) in (p.z).iter().enumerate() {
            if z {
                // out = out.compose(Pauli((cliff.stab_z[id_z], cliff.stab_x[id_z], 2 * cliff.stab_phase[id_z])))
                let p_cliff = Pauli {
                    x: cliff.tableau.slice(s![n.., id_z]).to_owned(),
                    z: cliff.tableau.slice(s![n.., id_z + n]).to_owned(),
                    phase: 2 * (cliff_phase[id_z + n] as u8),
                };
                let ret1 = ret.clone();
                ret = Pauli::compose(ret1, p_cliff);
            }
        }
        for (id_x, &x) in (p.x).iter().enumerate() {
            // out = out.compose(Pauli((cliff.destab_z[id_x], cliff.destab_x[id_x], 2 * cliff.destab_phase[id_x])))
            if x {
                let p_cliff = Pauli {
                    x: cliff.tableau.slice(s![..n, id_x]).to_owned(),
                    z: cliff.tableau.slice(s![..n, id_x + n]).to_owned(),
                    phase: 2 * (cliff_phase[id_x] as u8),
                };
                let ret1 = ret.clone();
                ret = Pauli::compose(ret1, p_cliff);
            }
        }
        let cnt_y = _count_y((p.x).view(), (p.z).view());
        let phase = (4 + ret.phase - cnt_y) % 4;
        let x = ret.x;
        let z = ret.z;

        let out = Pauli {
            x: x,
            z: z,
            phase: phase,
        };
        // println!("{},{}, {},{},{}", out.x, out.z, out.phase, ret.phase, cnt_y);
        out
    }
}

fn _count_y(x: ArrayView1<bool>, z: ArrayView1<bool>) -> u8 {
    let out = x.iter().zip(z.iter()).filter(|(&x, &y)| x && y).count();
    // println!("{}", out);
    out as u8
}

#[cfg(test)]
mod test {
    use crate::evolve::{Clifford, Pauli};
    use ndarray::{Array1, Array2};

    #[test]
    fn test_compose() {
        let px = Pauli {
            x: Array1::from(vec![true]),
            z: Array1::from(vec![false]),
            phase: 0,
        }; // X
        let py = Pauli {
            x: Array1::from(vec![true]),
            z: Array1::from(vec![true]),
            phase: 0,
        }; // Y
        let pz = Pauli {
            x: Array1::from(vec![false]),
            z: Array1::from(vec![true]),
            phase: 0,
        }; // Z

        let p = Pauli::compose(px.clone(), py.clone());
        let expect_p = Pauli {
            x: Array1::from(vec![false]),
            z: Array1::from(vec![true]),
            phase: 1,
        }; // -iZ
        assert_eq!(p.x, expect_p.x);
        assert_eq!(p.z, expect_p.z);
        assert_eq!(p.phase, expect_p.phase);

        let p = Pauli::compose(px.clone(), pz.clone());
        let expect_p = Pauli {
            x: Array1::from(vec![true]),
            z: Array1::from(vec![true]),
            phase: 3,
        }; // iY
        assert_eq!(p.x, expect_p.x);
        assert_eq!(p.z, expect_p.z);
        assert_eq!(p.phase, expect_p.phase);

        let p = Pauli::compose(py.clone(), pz.clone());
        let expect_p = Pauli {
            x: Array1::from(vec![true]),
            z: Array1::from(vec![false]),
            phase: 1,
        }; // -iX
        assert_eq!(p.x, expect_p.x);
        assert_eq!(p.z, expect_p.z);
        assert_eq!(p.phase, expect_p.phase);

        let pxx = Pauli {
            x: Array1::from(vec![true, true]),
            z: Array1::from(vec![false, false]),
            phase: 0,
        }; // XX
        let pxy = Pauli {
            x: Array1::from(vec![true, true]),
            z: Array1::from(vec![true, false]),
            phase: 0,
        }; // XY
        let p = Pauli::compose(pxx, pxy);
        let expect_p = Pauli {
            x: Array1::from(vec![false, false]),
            z: Array1::from(vec![true, false]),
            phase: 1,
        }; // -iIZ
        assert_eq!(p.x, expect_p.x);
        assert_eq!(p.z, expect_p.z);
        assert_eq!(p.phase, expect_p.phase);
    }

    #[test]
    fn test_evolve() {
        let cliff_s = Clifford {
            num_qubits: 1,
            tableau: Array2::from(vec![[true, true], [false, true]]),
            phase: Array1::from(vec![false, false]),
        }; // S
        let cliff_h = Clifford {
            num_qubits: 1,
            tableau: Array2::from(vec![[false, true], [true, false]]),
            phase: Array1::from(vec![false, false]),
        }; // H
        let cliff_sdg = Clifford {
            num_qubits: 1,
            tableau: Array2::from(vec![[true, true], [false, true]]),
            phase: Array1::from(vec![true, false]),
        }; // Sdg

        let px = Pauli {
            x: Array1::from(vec![true]),
            z: Array1::from(vec![false]),
            phase: 0,
        }; // X
        let py = Pauli {
            x: Array1::from(vec![true]),
            z: Array1::from(vec![true]),
            phase: 0,
        }; // Y
        let pz = Pauli {
            x: Array1::from(vec![false]),
            z: Array1::from(vec![true]),
            phase: 0,
        }; // Z
        let px_min = Pauli {
            x: Array1::from(vec![true]),
            z: Array1::from(vec![false]),
            phase: 2,
        }; // -X
        let py_min = Pauli {
            x: Array1::from(vec![true]),
            z: Array1::from(vec![true]),
            phase: 2,
        }; // -Y

        let p = Pauli::evolve(px.clone(), cliff_s.clone()); // Y
        assert_eq!(p.x, (py.x).clone());
        assert_eq!(p.z, (py.z).clone());
        assert_eq!(p.phase, (py.phase).clone());

        let p = Pauli::evolve(py.clone(), cliff_s.clone()); // -X
        assert_eq!(p.x, (px_min.x).clone());
        assert_eq!(p.z, (px_min.z).clone());
        assert_eq!(p.phase, (px_min.phase).clone());

        let p = Pauli::evolve(pz.clone(), cliff_s.clone()); // Z
        assert_eq!(p.x, (pz.x).clone());
        assert_eq!(p.z, (pz.z).clone());
        assert_eq!(p.phase, (pz.phase).clone());

        let p = Pauli::evolve(px.clone(), cliff_h.clone()); // Z
        assert_eq!(p.x, (pz.x).clone());
        assert_eq!(p.z, (pz.z).clone());
        assert_eq!(p.phase, (pz.phase).clone());

        let p = Pauli::evolve(py.clone(), cliff_h.clone()); // -Y
        assert_eq!(p.x, (py_min.x).clone());
        assert_eq!(p.z, (py_min.z).clone());
        assert_eq!(p.phase, (py_min.phase).clone());

        let p = Pauli::evolve(pz.clone(), cliff_h.clone()); // X
        assert_eq!(p.x, (px.x).clone());
        assert_eq!(p.z, (px.z).clone());
        assert_eq!(p.phase, (px.phase).clone());

        let p = Pauli::evolve(px.clone(), cliff_sdg.clone()); // -Y
        assert_eq!(p.x, (py_min.x).clone());
        assert_eq!(p.z, (py_min.z).clone());
        assert_eq!(p.phase, (py_min.phase).clone());

        let p = Pauli::evolve(py.clone(), cliff_sdg.clone()); // X
        assert_eq!(p.x, (px.x).clone());
        assert_eq!(p.z, (px.z).clone());
        assert_eq!(p.phase, (px.phase).clone());

        let p = Pauli::evolve(pz.clone(), cliff_sdg.clone()); // Z
        assert_eq!(p.x, (pz.x).clone());
        assert_eq!(p.z, (pz.z).clone());
        assert_eq!(p.phase, (pz.phase).clone());

        let cliff = Clifford {
            num_qubits: 2,
            tableau: Array2::from(vec![
                [true, false, false, false],
                [false, true, false, false],
                [false, true, true, false],
                [true, true, false, true],
            ]),
            phase: Array1::from(vec![false, false, false, false]),
        };
        let pxx = Pauli {
            x: Array1::from(vec![true, true]),
            z: Array1::from(vec![false, false]),
            phase: 0,
        }; // XX
        let pxy = Pauli {
            x: Array1::from(vec![true, true]),
            z: Array1::from(vec![true, false]),
            phase: 0,
        }; // XY
        let pyy = Pauli {
            x: Array1::from(vec![true, true]),
            z: Array1::from(vec![true, true]),
            phase: 0,
        }; // YY
        let pzz = Pauli {
            x: Array1::from(vec![false, false]),
            z: Array1::from(vec![true, true]),
            phase: 0,
        }; // ZZ

        let p = Pauli::evolve(pxx.clone(), cliff.clone()); // XX
        assert_eq!(p.x, (pxx.x).clone());
        assert_eq!(p.z, (pxx.z).clone());
        assert_eq!(p.phase, (pxx.phase).clone());

        let p = Pauli::evolve(pxy, cliff.clone()); // IY
        let expect_p = Pauli {
            x: Array1::from(vec![true, false]),
            z: Array1::from(vec![true, false]),
            phase: 0,
        }; // IY
        assert_eq!(p.x, expect_p.x);
        assert_eq!(p.z, expect_p.z);
        assert_eq!(p.phase, expect_p.phase);

        let p = Pauli::evolve(pyy, cliff.clone()); // YZ
        let expect_p = Pauli {
            x: Array1::from(vec![false, true]),
            z: Array1::from(vec![true, true]),
            phase: 0,
        }; // YZ
        assert_eq!(p.x, expect_p.x);
        assert_eq!(p.z, expect_p.z);
        assert_eq!(p.phase, expect_p.phase);

        let p = Pauli::evolve(pzz, cliff.clone());
        let expect_p = Pauli {
            x: Array1::from(vec![true, false]),
            z: Array1::from(vec![true, true]),
            phase: 2,
        }; // -ZY
        assert_eq!(p.x, expect_p.x);
        assert_eq!(p.z, expect_p.z);
        assert_eq!(p.phase, expect_p.phase);
    }
}
