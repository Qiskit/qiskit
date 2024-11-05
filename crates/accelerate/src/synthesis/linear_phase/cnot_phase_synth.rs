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

struct PhaseIterator {
    s_cpy: Array2<u8>,
    state: Array2<u8>,
    rust_angles: Vec<String>,
    num_qubits: usize,
    qubit_idx: usize,
    index: usize,
    iter_once: bool,
}

impl PhaseIterator {
    fn new(
        num_qubits: usize,
        s_cpy: Array2<u8>,
        state: Array2<u8>,
        rust_angles: Vec<String>,
    ) -> Self {
        Self {
            s_cpy,
            state,
            rust_angles,
            num_qubits,
            qubit_idx: 0,
            index: 0,
            iter_once: false,
        }
    }
}

impl Iterator for PhaseIterator {
    type Item = Instruction;

    fn next(&mut self) -> Option<Self::Item> {
        if self.qubit_idx >= self.num_qubits {
            return None;
        }

        if self.index < self.s_cpy.ncols() {
            let mut gate_instr: Option<Instruction> = None;
            let icnot = self.s_cpy.column(self.index);
            self.index += 1;
            let target_state = self.state.row(self.qubit_idx);

            if icnot == target_state {
                self.index -= 1;
                self.s_cpy.remove_index(numpy::ndarray::Axis(1), self.index);
                let angle = self.rust_angles.remove(self.index);

                gate_instr = Some(match angle.as_str() {
                    "t" => (
                        StandardGate::TGate,
                        smallvec![],
                        smallvec![Qubit(self.qubit_idx as u32)],
                    ),
                    "tdg" => (
                        StandardGate::TdgGate,
                        smallvec![],
                        smallvec![Qubit(self.qubit_idx as u32)],
                    ),
                    "s" => (
                        StandardGate::SGate,
                        smallvec![],
                        smallvec![Qubit(self.qubit_idx as u32)],
                    ),
                    "sdg" => (
                        StandardGate::SdgGate,
                        smallvec![],
                        smallvec![Qubit(self.qubit_idx as u32)],
                    ),
                    "z" => (
                        StandardGate::ZGate,
                        smallvec![],
                        smallvec![Qubit(self.qubit_idx as u32)],
                    ),
                    angles_in_pi => (
                        StandardGate::PhaseGate,
                        smallvec![Param::Float((angles_in_pi.parse::<f64>().ok()?) % PI)],
                        smallvec![Qubit(self.qubit_idx as u32)],
                    ),
                });
            }
            if gate_instr.is_some() {
                gate_instr
            } else {
                self.next()
            }
        } else {
            if self.iter_once {
                return None;
            }
            self.qubit_idx += 1;
            self.index = 0;
            self.next()
        }
    }
}

struct CXPhaseIterator {
    s_cpy: Array2<u8>,
    state: Array2<u8>,
    rust_angles: Vec<String>,
    q: Vec<(Array2<u8>, Vec<usize>, usize)>,
    num_qubits: usize,
    qubit_idx: usize,
    keep_iterating: bool,
    loop_active: bool,
    _s: Array2<u8>,
    _i: Vec<usize>,
    _ep: usize,
    phase_iter_handle: Option<PhaseIterator>,
}

impl CXPhaseIterator {
    fn new(
        num_qubits: usize,
        s_cpy: Array2<u8>,
        state: Array2<u8>,
        rust_angles: Vec<String>,
        q: Vec<(Array2<u8>, Vec<usize>, usize)>,
    ) -> Self {
        let (init_s, init_i, init_ep) = q.last().unwrap().clone();
        Self {
            s_cpy,
            state,
            rust_angles,
            q,
            num_qubits,
            qubit_idx: 0,
            keep_iterating: false,
            loop_active: false,
            _s: init_s,
            _i: init_i,
            _ep: init_ep,
            phase_iter_handle: None,
        }
    }
}

impl Iterator for CXPhaseIterator {
    type Item = Instruction;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(handle) = self.phase_iter_handle.as_mut() {
            let data = handle.next();
            if data.is_none() {
                self.s_cpy = handle.s_cpy.clone();
                self.rust_angles = handle.rust_angles.clone();
                self.phase_iter_handle = None;
            } else {
                return data;
            }
        }

        if !self.q.is_empty() || self.loop_active || self.keep_iterating {
            if !self.loop_active && !self.keep_iterating {
                (self._s, self._i, self._ep) = self.q.pop().unwrap();
            }

            if !self.loop_active && !self.keep_iterating && self._s.is_empty() {
                return self.next();
            }

            if self._ep < self.num_qubits || self.loop_active || self.keep_iterating {
                if !self.loop_active && !self.keep_iterating {
                    self.keep_iterating = true;
                }

                if self.keep_iterating || self.loop_active {
                    if !self.loop_active {
                        self.keep_iterating = false;
                    }
                    if self.qubit_idx < self.num_qubits {
                        if (self.qubit_idx != self._ep)
                            && (self._s.row(self.qubit_idx).sum() as usize
                                == self._s.row(self.qubit_idx).len())
                        {
                            for _k in 0..self.state.ncols() {
                                self.state[(self._ep, _k)] ^= self.state[(self.qubit_idx, _k)];
                            }

                            let mut phase_iter_h = PhaseIterator::new(
                                self.num_qubits,
                                self.s_cpy.clone(),
                                self.state.clone(),
                                self.rust_angles.clone(),
                            );

                            phase_iter_h.iter_once = true;
                            phase_iter_h.qubit_idx = self._ep;
                            self.phase_iter_handle = Some(phase_iter_h);

                            self.q.push((self._s.clone(), self._i.clone(), self._ep));

                            let mut unique_q = vec![];
                            for data in self.q.iter() {
                                if !unique_q.contains(data) {
                                    let d = (*data).clone();
                                    unique_q.push(d);
                                }
                            }

                            self.q = unique_q;

                            for data in &mut self.q {
                                let (ref mut _temp_s, _, _) = data;
                                if _temp_s.is_empty() {
                                    continue;
                                }
                                for idx in 0.._temp_s.row(self.qubit_idx).len() {
                                    _temp_s[(self.qubit_idx, idx)] ^= _temp_s[(self._ep, idx)];
                                }
                            }

                            (self._s, self._i, self._ep) = self.q.pop().unwrap();

                            self.qubit_idx += 1;
                            self.keep_iterating = true;
                            self.loop_active = true;
                            return Some((
                                StandardGate::CXGate,
                                smallvec![],
                                smallvec![
                                    Qubit((self.qubit_idx - 1) as u32),
                                    Qubit(self._ep as u32)
                                ],
                            ));
                        } else {
                            self.qubit_idx += 1;
                            self.loop_active = true;
                            return self.next();
                        }
                    } else {
                        self.qubit_idx = 0;
                        self.loop_active = false;
                        if self.keep_iterating {
                            return self.next();
                        }
                    }
                }
            }

            if self._i.is_empty() {
                return self.next();
            }

            let maxes: Vec<usize> = self
                ._s
                .axis_iter(numpy::ndarray::Axis(0))
                .map(|row| {
                    std::cmp::max(
                        row.iter().filter(|&&x| x == 0).count(),
                        row.iter().filter(|&&x| x == 1).count(),
                    )
                })
                .collect();

            let maxes2: Vec<usize> = self._i.iter().map(|&_i_idx| maxes[_i_idx]).collect();

            let _temp_argmax = maxes2
                .iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| x.cmp(y))
                .map(|(idx, _)| idx)
                .unwrap();

            let _j = self._i[_temp_argmax];

            let mut cnots0_t = vec![];
            let mut cnots1_t = vec![];

            let mut cnots0_t_shape = (0_usize, self._s.column(0).len());
            let mut cnots1_t_shape = (0_usize, 0_usize);
            cnots1_t_shape.1 = cnots0_t_shape.1;
            for cols in self._s.columns() {
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

            if self._ep == self.num_qubits {
                self.q.push((
                    cnots1,
                    self._i.clone().into_iter().filter(|&x| x != _j).collect(),
                    _j,
                ));
            } else {
                self.q.push((
                    cnots1,
                    self._i.clone().into_iter().filter(|&x| x != _j).collect(),
                    self._ep,
                ));
            }
            self.q.push((
                cnots0,
                self._i.clone().into_iter().filter(|&x| x != _j).collect(),
                self._ep,
            ));

            self.next()
        } else {
            None
        }
    }
}

struct BindingIterator {
    num_qubits: usize,
    q: Vec<(Array2<u8>, Vec<usize>, usize)>,
    phase_iter_handle: PhaseIterator,
    cx_phase_iter_handle: Option<CXPhaseIterator>,
    phase_iterator_done: bool,
}

impl BindingIterator {
    fn new(s: Array2<u8>, angles: Vec<String>, state: Array2<u8>) -> Self {
        let num_qubits = s.nrows();
        let q = vec![(
            s.clone(),
            (0..num_qubits).collect::<Vec<usize>>(),
            num_qubits,
        )];
        Self {
            num_qubits,
            q,
            phase_iter_handle: PhaseIterator::new(num_qubits, s, state, angles),
            cx_phase_iter_handle: None,
            phase_iterator_done: false,
        }
    }
}

impl Iterator for BindingIterator {
    type Item = Instruction;
    fn next(&mut self) -> Option<Self::Item> {
        if !self.phase_iterator_done {
            let data = self.phase_iter_handle.next();
            if data.is_none() {
                self.cx_phase_iter_handle = Some(CXPhaseIterator::new(
                    self.num_qubits,
                    self.phase_iter_handle.s_cpy.clone(),
                    self.phase_iter_handle.state.clone(),
                    self.phase_iter_handle.rust_angles.clone(),
                    self.q.clone(),
                ));
                self.phase_iterator_done = true;
                self.next()
            } else {
                data
            }
        } else if let Some(handle) = self.cx_phase_iter_handle.as_mut() {
            handle.next()
        } else {
            None
        }
    }
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

    let rust_angles = angles
        .iter()
        .filter_map(|data| data.extract::<String>().ok())
        .collect::<Vec<String>>();

    let state = Array2::<u8>::eye(num_qubits);

    let mut binding_iter_handle = BindingIterator::new(s, rust_angles, state);

    // Optimize this one!
    let cx_phase_iter = std::iter::from_fn(|| binding_iter_handle.next());
    let cx_phase_iter_vec = cx_phase_iter.collect::<Vec<Instruction>>();

    let residual_state = binding_iter_handle.cx_phase_iter_handle.unwrap().state;
    let state_bool = residual_state.mapv(|x| x != 0);

    let synth_pmh_iter = synth_pmh(state_bool, section_size).rev();
    let cnot_synth_iter = cx_phase_iter_vec.into_iter().chain(synth_pmh_iter);

    CircuitData::from_standard_gates(py, num_qubits as u32, cnot_synth_iter, Param::Float(0.0))
}
