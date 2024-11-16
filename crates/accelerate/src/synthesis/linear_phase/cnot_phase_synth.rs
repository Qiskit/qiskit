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
use std::cell::RefCell;
use std::f64::consts::PI;
use std::rc::Rc;
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

/// This function implements a Gray-code inspired algorithm of synthesizing a circuit
/// over CNOT and phase-gates with minimal-CNOT for a given phase-polynomial.
/// The algorithm is described as "Gray-Synth" algorithm in Algorithm-1, page 12
/// of paper "https://arxiv.org/abs/1712.01859".
#[pyfunction]
#[pyo3(signature = (cnots, angles, section_size=2))]
pub fn synth_cnot_phase_aam(
    py: Python,
    cnots: PyReadonlyArray2<u8>,
    angles: &Bound<PyList>,
    section_size: Option<i64>,
) -> PyResult<CircuitData> {
    let s = cnots.as_array().to_owned();
    let num_qubits = s.nrows();

    let mut rust_angles = angles
        .iter()
        .filter_map(|data| {
            data.extract::<String>()
                .or_else(|_| data.extract::<f64>().map(|f| f.to_string()))
                .ok()
        })
        .collect::<Vec<String>>();

    let mut state = Array2::<u8>::eye(num_qubits);

    let x: Rc<RefCell<Option<Array2<u8>>>> = Rc::new(RefCell::new(None));
    let x_clone = Rc::clone(&x);

    let mut s_cpy = s.clone();
    let mut q = vec![(
        s.clone(),
        (0..num_qubits).collect::<Vec<usize>>(),
        num_qubits,
    )];
    let mut _s = s;
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

    let cx_phase_iter = std::iter::from_fn(move || {
        let mut gate_instr: Option<Instruction> = None;

        while !phase_done {
            let icnot = s_cpy.column(index);
            index += 1;
            let target_state = state.row(qubit_idx);

            if icnot == target_state {
                index -= 1;
                s_cpy.remove_index(numpy::ndarray::Axis(1), index);
                let angle = rust_angles.remove(index);
                gate_instr = get_instr(angle, qubit_idx);
                break;
            }
            if index == s_cpy.ncols() {
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
                (_s, _i, _ep) = q.pop().unwrap();
            }
            if !phase_loop_on && _s.is_empty() {
                continue 'outer_loop;
            }

            while keep_iterating && (_ep < num_qubits) {
                if !phase_loop_on {
                    keep_iterating = false;
                }
                while qubit_idx < num_qubits {
                    if (qubit_idx != _ep)
                        && (_s.row(qubit_idx).sum() as usize == _s.row(qubit_idx).len())
                    {
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
                            if index == s_cpy.ncols() {
                                phase_loop_on = false;
                                break;
                            }
                            let icnot = s_cpy.column(index);
                            index += 1;
                            let target_state = state.row(_ep);
                            if icnot == target_state {
                                index -= 1;
                                s_cpy.remove_index(numpy::ndarray::Axis(1), index);
                                let angle = rust_angles.remove(index);
                                gate_instr = get_instr(angle, _ep);
                                break 'outer_loop;
                            }
                        }

                        q.push((_s.clone(), _i.clone(), _ep));
                        let mut unique_q = vec![];

                        for data in q.iter() {
                            if !unique_q.contains(data) {
                                let d = (*data).clone();
                                unique_q.push(d);
                            }
                        }
                        q = unique_q;

                        for data in &mut q {
                            let (ref mut _temp_s, _, _) = data;
                            if _temp_s.is_empty() {
                                continue;
                            }

                            for idx in 0.._temp_s.row(qubit_idx).len() {
                                _temp_s[(qubit_idx, idx)] ^= _temp_s[(_ep, idx)];
                            }
                        }

                        (_s, _i, _ep) = q.pop().unwrap();
                    } // end of if _ep < num_qubits ...
                    qubit_idx += 1;
                } // end of while check qubit_idx < num_qubits ...
                qubit_idx = 0;
            } // end of while keep iterating ...
            keep_iterating = true;

            if _i.is_empty() {
                continue 'outer_loop;
            }

            let maxes: Vec<usize> = _s
                .axis_iter(numpy::ndarray::Axis(0))
                .map(|row| {
                    std::cmp::max(
                        row.iter().filter(|&&x| x == 0).count(),
                        row.iter().filter(|&&x| x == 1).count(),
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

            let mut cnots0_t_shape = (0_usize, _s.column(0).len());
            let mut cnots1_t_shape = (0_usize, 0_usize);
            cnots1_t_shape.1 = cnots0_t_shape.1;
            for cols in _s.columns() {
                if cols[_j] == 0 {
                    cnots0_t_shape.0 += 1;
                    cnots0_t.append(&mut cols.to_vec());
                } else if cols[_j] == 1 {
                    cnots1_t_shape.0 += 1;
                    cnots1_t.append(&mut cols.to_vec());
                }
            }

            let cnots0 =
                Array2::from_shape_vec((cnots0_t_shape.0, cnots0_t_shape.1), cnots0_t).unwrap();
            let cnots1 =
                Array2::from_shape_vec((cnots1_t_shape.0, cnots1_t_shape.1), cnots1_t).unwrap();

            let cnots0 = cnots0.reversed_axes().to_owned();
            let cnots1 = cnots1.reversed_axes().to_owned();

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

        if phase_done && cx_phase_done {
            *x_clone.borrow_mut() = Some(state.clone());
            None
        } else {
            gate_instr
        }
    });

    let cx_phase_vec = cx_phase_iter.collect::<Vec<Instruction>>();

    let borrowed_x = x.borrow();
    let residue_state = borrowed_x.as_ref().unwrap();
    let state_bool = residue_state.mapv(|x| x != 0);
    let synth_pmh_iter = synth_pmh(state_bool, section_size).rev();
    let synth_aam_iter = cx_phase_vec.into_iter().chain(synth_pmh_iter);
    CircuitData::from_standard_gates(py, num_qubits as u32, synth_aam_iter, Param::Float(0.0))
}
