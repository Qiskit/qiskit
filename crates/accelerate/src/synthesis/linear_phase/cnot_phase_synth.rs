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

use crate::synthesis::linear::pmh::synth_pmh;
use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::{prelude::*, types::PyList};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Param, StandardGate};
use qiskit_circuit::Qubit;
use smallvec::{smallvec, SmallVec};
use std::f64::consts::PI;
type Instruction = (StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>);

fn get_instr(angle: String, qubit_idx: usize) -> Option<Instruction> {
    Some(match angle.as_str() {
        "t" => (
            StandardGate::TGate,
            smallvec![],
            smallvec![Qubit(qubit_idx as u32)],
        ),
        "tdg" => (
            StandardGate::TdgGate,
            smallvec![],
            smallvec![Qubit(qubit_idx as u32)],
        ),
        "s" => (
            StandardGate::SGate,
            smallvec![],
            smallvec![Qubit(qubit_idx as u32)],
        ),
        "sdg" => (
            StandardGate::SdgGate,
            smallvec![],
            smallvec![Qubit(qubit_idx as u32)],
        ),
        "z" => (
            StandardGate::ZGate,
            smallvec![],
            smallvec![Qubit(qubit_idx as u32)],
        ),
        angles_in_pi => (
            StandardGate::PhaseGate,
            smallvec![Param::Float((angles_in_pi.parse::<f64>().ok()?) % PI)],
            smallvec![Qubit(qubit_idx as u32)],
        ),
    })
}

/// This function is an implementation of the `GraySynth` algorithm which is a heuristic
/// algorithm described in detail in section 4 of [1] for synthesizing small parity networks.
/// It is inspired by Gray codes. Given a set of binary strings :math:`S`
/// (called ``cnots`` bellow), the algorithm synthesizes a parity network for :math:`S` by
/// repeatedly choosing an index :math:`i` to expand and then effectively recursing on
/// the co-factors :math:`S_0` and :math:`S_1`, consisting of the strings :math:`y \in S`,
/// with :math:`y_i = 0` or :math:`1` respectively. As a subset :math:`S` is recursively expanded,
/// ``cx`` gates are applied so that a designated target bit contains the
/// (partial) parity :math:`\chi_y(x)` where :math:`y_i = 1` if and only if :math:`y'_i = 1` for all
/// :math:`y' \in S`. If :math:`S` contains a single element :math:`\{y'\}`, then :math:`y = y'`,
/// and the target bit contains the value :math:`\chi_{y'}(x)` as desired.
/// Notably, rather than uncomputing this sequence of ``cx`` (CNOT) gates when a subset :math:`S`
/// is finished being synthesized, the algorithm maintains the invariant that the remaining
/// parities to be computed are expressed over the current state of bits. This allows the algorithm
/// to avoid the 'backtracking' inherent in uncomputing-based methods.
/// References:
///        1. Matthew Amy, Parsiad Azimzadeh, and Michele Mosca.
///           *On the controlled-NOT complexity of controlled-NOT–phase circuits.*,
///           Quantum Science and Technology 4.1 (2018): 015002.
///           `arXiv:1712.01859 <https://arxiv.org/abs/1712.01859>`_
#[pyfunction]
#[pyo3(signature = (cnots, angles, section_size=2))]
pub fn synth_cnot_phase_aam(
    py: Python,
    cnots: PyReadonlyArray2<u8>,
    angles: &Bound<PyList>,
    section_size: Option<i64>,
) -> PyResult<CircuitData> {
    // converting to Option<usize>
    let section_size: Option<usize> =
        section_size.and_then(|num| if num >= 0 { Some(num as usize) } else { None });

    let cnots: Array2<bool> = cnots.as_array().mapv(|x| x != 0);
    let num_qubits = cnots.nrows();

    let mut angles = angles
        .iter()
        .filter_map(|data| {
            data.extract::<String>()
                .or_else(|_| data.extract::<f64>().map(|f| f.to_string()))
                .ok()
        })
        .collect::<Vec<String>>();

    let mut state: Array2<bool> = Array2::from_shape_fn((num_qubits, num_qubits), |(i, j)| i == j);

    let mut cnots_cpy = cnots.clone();
    let mut q = vec![(
        cnots.clone(),
        (0..num_qubits).collect::<Vec<usize>>(),
        num_qubits,
    )];
    let mut _cnots = cnots;
    let mut _i = (0..num_qubits).collect::<Vec<usize>>();
    let mut _ep = num_qubits;

    // variables keeping the state of iteration
    let mut keep_iterating: bool = true;
    let mut cx_gate_done: bool = false;
    let mut phase_loop_on: bool = false;
    let mut phase_done: bool = false;
    let mut cx_phase_done: bool = false;
    let mut qubit_idx: usize = 0;
    let mut index: usize = 0;
    let mut pmh_init_done: bool = false;
    let mut synth_pmh_iter: Option<Box<dyn Iterator<Item = Instruction>>> = None;

    let cx_phase_iter = std::iter::from_fn(move || {
        let mut gate_instr: Option<Instruction> = None;

        while !phase_done {
            let icnot = cnots_cpy.column(index);
            index += 1;
            let target_state = state.row(qubit_idx);

            if icnot == target_state {
                index -= 1;
                cnots_cpy.remove_index(numpy::ndarray::Axis(1), index);
                let angle = angles.remove(index);
                gate_instr = get_instr(angle, qubit_idx);
                break;
            }
            if index == cnots_cpy.ncols() {
                qubit_idx += 1;
                index = 0;
            }
            if qubit_idx == num_qubits {
                phase_done = true;
                index = 0;
                qubit_idx = 0;
            }
        } // end phase_apply loop

        'outer_loop: while phase_done && !cx_phase_done {
            if !phase_loop_on && q.is_empty() {
                cx_phase_done = true;
                break 'outer_loop;
            }
            if !phase_loop_on {
                (_cnots, _i, _ep) = q.pop().unwrap();
            }
            if !phase_loop_on && _cnots.is_empty() {
                continue 'outer_loop;
            }

            while keep_iterating && (_ep < num_qubits) {
                if !phase_loop_on {
                    keep_iterating = false;
                }
                while qubit_idx < num_qubits {
                    if (qubit_idx != _ep) && !_cnots.row(qubit_idx).iter().any(|&b| !b) {
                        if !cx_gate_done && !phase_loop_on {
                            keep_iterating = true;
                            cx_gate_done = true;
                            gate_instr = Some((
                                StandardGate::CXGate,
                                smallvec![],
                                smallvec![Qubit((qubit_idx) as u32), Qubit(_ep as u32)],
                            ));
                            index = 0;
                            phase_loop_on = true;
                            for _k in 0..state.ncols() {
                                state[(_ep, _k)] ^= state[(qubit_idx, _k)];
                            }
                            break 'outer_loop;
                        }

                        cx_gate_done = false;
                        while phase_loop_on {
                            if index == cnots_cpy.ncols() {
                                phase_loop_on = false;
                                break;
                            }
                            let icnot = cnots_cpy.column(index);
                            index += 1;
                            let target_state = state.row(_ep);
                            if icnot == target_state {
                                index -= 1;
                                cnots_cpy.remove_index(numpy::ndarray::Axis(1), index);
                                let angle = angles.remove(index);
                                gate_instr = get_instr(angle, _ep);
                                break 'outer_loop;
                            }
                        }

                        q.push((_cnots.clone(), _i.clone(), _ep));
                        let mut unique_q = vec![];

                        for data in q.iter() {
                            if !unique_q.contains(data) {
                                let d = (*data).clone();
                                unique_q.push(d);
                            }
                        }
                        q = unique_q;

                        for data in &mut q {
                            let (ref mut _temp_cnots, _, _) = data;
                            if _temp_cnots.is_empty() {
                                continue;
                            }

                            for idx in 0.._temp_cnots.row(qubit_idx).len() {
                                _temp_cnots[(qubit_idx, idx)] ^= _temp_cnots[(_ep, idx)];
                            }
                        }

                        (_cnots, _i, _ep) = q.pop().unwrap();
                    } // end of if _ep < num_qubits ...
                    qubit_idx += 1;
                } // end of while check qubit_idx < num_qubits ...
                qubit_idx = 0;
            } // end of while keep iterating ...
            keep_iterating = true;

            if _i.is_empty() {
                continue 'outer_loop;
            }

            let maxes: Vec<usize> = _cnots
                .axis_iter(numpy::ndarray::Axis(0))
                .map(|row| {
                    std::cmp::max(
                        row.iter().filter(|&&x| !x).count(),
                        row.iter().filter(|&&x| x).count(),
                    )
                })
                .collect();

            let maxes2: Vec<usize> = _i.iter().map(|&_i_idx| maxes[_i_idx]).collect();

            let _temp_argmax = maxes2
                .iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| x.cmp(y))
                .map(|(idx, _)| idx)
                .unwrap();

            let _j = _i[_temp_argmax];

            let mut cnots0_t = vec![];
            let mut cnots1_t = vec![];

            let mut cnots0_t_shape = (0_usize, _cnots.column(0).len());
            let mut cnots1_t_shape = (0_usize, 0_usize);
            cnots1_t_shape.1 = cnots0_t_shape.1;
            for cols in _cnots.columns() {
                if !cols[_j] {
                    cnots0_t_shape.0 += 1;
                    cnots0_t.append(&mut cols.to_vec());
                } else {
                    cnots1_t_shape.0 += 1;
                    cnots1_t.append(&mut cols.to_vec());
                }
            }

            let cnots0 = Array2::from_shape_vec((cnots0_t_shape.0, cnots0_t_shape.1), cnots0_t)
                .unwrap()
                .reversed_axes();
            let cnots1 = Array2::from_shape_vec((cnots1_t_shape.0, cnots1_t_shape.1), cnots1_t)
                .unwrap()
                .reversed_axes();

            if _ep == num_qubits {
                q.push((
                    cnots1,
                    _i.clone().into_iter().filter(|&x| x != _j).collect(),
                    _j,
                ));
            } else {
                q.push((
                    cnots1,
                    _i.clone().into_iter().filter(|&x| x != _j).collect(),
                    _ep,
                ));
            }
            q.push((
                cnots0,
                _i.clone().into_iter().filter(|&x| x != _j).collect(),
                _ep,
            ));
        } // end 'outer_loop

        if phase_done && cx_phase_done && !pmh_init_done {
            synth_pmh_iter = Some(Box::new(synth_pmh(state.clone(), section_size).rev()));
            pmh_init_done = true;
        }

        if pmh_init_done {
            if let Some(ref mut data) = synth_pmh_iter {
                gate_instr = data.next();
            } else {
                gate_instr = None;
            }
        }

        gate_instr
    });

    CircuitData::from_standard_gates(py, num_qubits as u32, cx_phase_iter, Param::Float(0.0))
}
