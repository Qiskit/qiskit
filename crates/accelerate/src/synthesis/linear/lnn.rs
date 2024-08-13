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

use hashbrown::HashMap;
use ndarray::{Array1, ArrayView2};

use qiskit_circuit::{
    operations::{Param, StandardGate},
    Qubit,
};
use smallvec::{smallvec, SmallVec};

type LnnGatesVec = Vec<(StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>)>;

fn _odd_pattern1(n: isize) -> Vec<isize> {
    let mut pat = Vec::new();

    pat.push(n - 2);
    for i in 0..((n - 3) / 2) {
        pat.push(n - 2 * i - 4);
        pat.push(n - 2 * i - 4)
    }

    for i in 0..((n - 1) / 2) {
        pat.push(2 * i);
        pat.push(2 * i);
    }

    pat
}

fn _odd_pattern2(n: isize) -> Vec<isize> {
    let mut pat = Vec::new();

    for i in 0..((n - 1) / 2) {
        pat.push(2 * i + 2);
        pat.push(2 * i + 2);
    }

    for i in 0..((n - 3) / 2) {
        pat.push(n - 2 * i - 2);
        pat.push(n - 2 * i - 2);
    }

    pat.push(1);
    pat
}

fn _even_pattern1(n: isize) -> Vec<isize> {
    let mut pat = Vec::new();

    pat.push(n - 1);

    for i in 0..((n - 2) / 2) {
        pat.push(n - 2 * i - 3);
        pat.push(n - 2 * i - 3);
    }

    for i in 0..((n - 2) / 2) {
        pat.push(2 * i);
        pat.push(2 * i);
    }

    pat.push(n - 2);
    pat
}

fn _even_pattern2(n: isize) -> Vec<isize> {
    let mut pat = Vec::new();

    for i in 0..((n - 2) / 2) {
        pat.push(2 * (i + 1));
        pat.push(2 * (i + 1));
    }
    for i in 0..(n / 2) {
        pat.push(n - 2 * i - 1);
        pat.push(n - 2 * i - 1);
    }

    pat
}

fn _create_patterns(n: isize) -> HashMap<(isize, isize), (isize, isize)> {
    let (pat1, pat2) = if n % 2 == 0 {
        (_even_pattern1(n), _even_pattern2(n))
    } else {
        (_odd_pattern1(n), _odd_pattern2(n))
    };

    let mut pats = HashMap::from_iter((0..n).map(|i| ((0, i), (i, i))));

    let mut ind2 = 0;

    let mut ind1 = if n % 2 == 0 {
        (2 * n - 4) / 2
    } else {
        (2 * n - 4) / 2 - 1
    };

    for layer in 0..(n / 2) {
        for i in 0..n {
            pats.insert(
                (layer + 1, i),
                (pat1[(ind1 + i) as usize], pat2[(ind2 + i) as usize]),
            );
        }
        ind1 -= 2;
        ind2 += 2;
    }

    pats
}

fn _append_cx_stage1(gates: &mut LnnGatesVec, n: isize) {
    for i in 0..(n / 2) {
        gates.push((
            StandardGate::CXGate,
            smallvec![],
            smallvec![Qubit((2 * i) as u32), Qubit((2 * i + 1) as u32)],
        ))
    }

    for i in 0..((n + 1) / 2 - 1) {
        gates.push((
            StandardGate::CXGate,
            smallvec![],
            smallvec![Qubit((2 * i + 2) as u32), Qubit((2 * i + 1) as u32)],
        ))
    }
}

fn _append_cx_stage2(gates: &mut LnnGatesVec, n: isize) {
    for i in 0..(n / 2) {
        gates.push((
            StandardGate::CXGate,
            smallvec![],
            smallvec![Qubit((2 * i + 1) as u32), Qubit((2 * i) as u32)],
        ))
    }

    for i in 0..((n + 1) / 2 - 1) {
        gates.push((
            StandardGate::CXGate,
            smallvec![],
            smallvec![Qubit((2 * i + 1) as u32), Qubit((2 * i + 2) as u32)],
        ))
    }
}
pub(super) fn synth_cz_depth_line_mr(matrix: ArrayView2<bool>) -> (usize, LnnGatesVec) {
    let num_qubits = matrix.raw_dim()[0];
    let pats = _create_patterns(num_qubits as isize);

    let mut s_gates = Array1::<isize>::zeros(num_qubits);

    let mut patlist: Vec<(isize, isize)> = Vec::new();

    let mut gates = LnnGatesVec::new();

    for i in 0..num_qubits {
        for j in (i + 1)..num_qubits {
            if matrix[[i, j]] {
                s_gates[[i]] += 2;
                s_gates[[j]] += 2;
                patlist.push((i as isize, j as isize - 1));
                patlist.push((i as isize, j as isize));
                patlist.push((i as isize + 1, j as isize - 1));
                patlist.push((i as isize + 1, j as isize));
            }
        }
    }

    for i in 0..((num_qubits + 1) / 2) {
        for j in 0..num_qubits {
            if patlist.contains(&pats[&(i as isize, j as isize)]) {
                let pacnt = patlist
                    .iter()
                    .filter(|val| **val == pats[&(i as isize, j as isize)])
                    .count();
                for _ in 0..pacnt {
                    s_gates[[j]] += 1;
                }
            }
        }

        for j in 0..num_qubits {
            if s_gates[[j]] % 4 == 1 {
                gates.push((
                    StandardGate::SdgGate,
                    smallvec![],
                    smallvec![Qubit(j as u32)],
                ))
            } else if s_gates[[j]] % 4 == 2 {
                gates.push((StandardGate::ZGate, smallvec![], smallvec![Qubit(j as u32)]))
            } else if s_gates[[j]] % 4 == 3 {
                gates.push((StandardGate::SGate, smallvec![], smallvec![Qubit(j as u32)]))
            }
        }

        _append_cx_stage1(&mut gates, num_qubits as isize);
        _append_cx_stage2(&mut gates, num_qubits as isize);
        s_gates = Array1::<isize>::zeros(num_qubits);
    }

    if num_qubits % 2 == 0 {
        let i = num_qubits / 2;

        for j in 0..num_qubits {
            if patlist.contains(&pats[&(i as isize, j as isize)])
                && pats[&(i as isize, j as isize)].0 != pats[&(i as isize, j as isize)].1
            {
                let pacnt = patlist
                    .iter()
                    .filter(|val| **val == pats[&(i as isize, j as isize)])
                    .count();
                for _ in 0..pacnt {
                    s_gates[[j]] += 1;
                }
            }
        }

        for j in 0..num_qubits {
            if s_gates[[j]] % 4 == 1 {
                gates.push((
                    StandardGate::SdgGate,
                    smallvec![],
                    smallvec![Qubit(j as u32)],
                ))
            } else if s_gates[[j]] % 4 == 2 {
                gates.push((StandardGate::ZGate, smallvec![], smallvec![Qubit(j as u32)]))
            } else if s_gates[[j]] % 4 == 3 {
                gates.push((StandardGate::SGate, smallvec![], smallvec![Qubit(j as u32)]))
            }
        }

        _append_cx_stage1(&mut gates, num_qubits as isize);
    }

    (num_qubits as usize, gates)
}
