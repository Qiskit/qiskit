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

use ndarray::{Array2};
use numpy::{PyReadonlyArray2, PyReadonlyArray1};
use smallvec::smallvec;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Param, StandardGate};
use qiskit_circuit::Qubit;
use crate::synthesis::linear::pmh::_synth_cnot_count_full_pmh;
use pyo3::prelude::*;
use std::f64::consts::PI;

#[pyfunction]
#[pyo3(signature = (cnots, angles, section_size=2))]
pub fn synth_cnot_phase_aam(
    py:Python,
    cnots: PyReadonlyArray2<u8>,
    angles: PyReadonlyArray1<PyObject>,
    section_size: Option<i64>,
) -> PyResult<CircuitData> {

    let s = cnots.as_array().to_owned();
    let angles_arr = angles.as_array().to_owned();
    let num_qubits = s.shape()[0];
    let mut s_cpy = s.clone();
    let mut instructions = vec![];

    let mut rust_angles = Vec::new();
    for obj in angles_arr
    {
        rust_angles.push(obj.extract::<String>(py)?);
    }

    let mut state = Array2::<u8>::eye(num_qubits);

    for qubit_idx in 0..num_qubits
    {
        let mut index = 0_usize;
        let mut swtch: bool = true;

        while index < s_cpy.ncols()
        {
            let icnot = s_cpy.column(index).to_vec();
            if icnot == state.row(qubit_idx).to_vec()
            {
                match rust_angles.get(index)
                {
                    Some(gate) if gate == "t" => instructions.push((StandardGate::TGate, smallvec![], smallvec![Qubit(qubit_idx as u32)])),
                    Some(gate) if gate == "tdg" => instructions.push((StandardGate::TdgGate, smallvec![], smallvec![Qubit(qubit_idx as u32)])),
                    Some(gate) if gate == "s" => instructions.push((StandardGate::SGate, smallvec![], smallvec![Qubit(qubit_idx as u32)])),
                    Some(gate) if gate == "sdg" => instructions.push((StandardGate::SdgGate, smallvec![], smallvec![Qubit(qubit_idx as u32)])),
                    Some(gate) if gate == "z" => instructions.push((StandardGate::ZGate, smallvec![], smallvec![Qubit(qubit_idx as u32)])),
                    Some(angles_in_pi) => instructions.push((StandardGate::PhaseGate, smallvec![Param::Float((angles_in_pi.parse::<f64>()?) % PI)], smallvec![Qubit(qubit_idx as u32)])),
                    None => (),
                };
                rust_angles.remove(index);
                s_cpy.remove_index(numpy::ndarray::Axis(1), index);
                if index == s_cpy.shape()[1] {break;}
                if index == 0 {swtch = false;}
                else {index -=1;}
            }
            if swtch {index +=1;}
            else {swtch = true;}
        }
    } 


    let epsilion: usize = num_qubits;
    let mut q = vec![(s, (0..num_qubits).collect::<Vec<usize>>(), epsilion)];


    while !q.is_empty()
        {

        let (_s, _i, _ep) = q.pop().unwrap();

        if _s.is_empty() {continue;}

        if 0 <= _ep as isize &&  _ep < num_qubits
        {
            let mut condition = true;
            while condition
            {
                condition = false;

                for _j in 0..num_qubits
                {
                    if (_j != _ep) && (usize::from(_s.row(_j).sum()) == _s.row(_j).len())
                    {
                        condition = true;
                        instructions.push((StandardGate::CXGate, smallvec![], smallvec![Qubit(_j as u32), Qubit(_ep as u32)]));

                        for _k in 0..state.ncols()
                        {
                            state[(_ep, _k)] ^= state[(_j, _k)];
                        }


                        let mut index = 0_usize;
                        let mut swtch: bool = true;
                        while index < s_cpy.ncols()
                        {
                            let icnot = s_cpy.column(index).to_vec();
                            if icnot == state.row(_ep).to_vec()
                            {
                                match rust_angles.get(index)
                                {
                                    Some(gate) if gate == "t" => instructions.push((StandardGate::TGate, smallvec![], smallvec![Qubit(_ep as u32)])),
                                    Some(gate) if gate == "tdg" => instructions.push((StandardGate::TdgGate, smallvec![], smallvec![Qubit(_ep as u32)])),
                                    Some(gate) if gate == "s" => instructions.push((StandardGate::SGate, smallvec![], smallvec![Qubit(_ep as u32)])),
                                    Some(gate) if gate == "sdg" => instructions.push((StandardGate::SdgGate, smallvec![], smallvec![Qubit(_ep as u32)])),
                                    Some(gate) if gate == "z" => instructions.push((StandardGate::ZGate, smallvec![], smallvec![Qubit(_ep as u32)])),
                                    Some(angles_in_pi) => instructions.push((StandardGate::PhaseGate, smallvec![Param::Float((angles_in_pi.parse::<f64>()?) % PI)], smallvec![Qubit(_ep as u32)])),
                                    None => (),
                                };
                                rust_angles.remove(index);
                                s_cpy.remove_index(numpy::ndarray::Axis(1), index);
                                if index == s_cpy.shape()[1] {break;}
                                if index == 0 {swtch = false;}
                                else {index -=1;}
                            }
                            if swtch {index +=1;}
                            else {swtch = true; }
                        }

                        let temp_var = (_s.clone(), _i.clone(), _ep);
                        if !q.contains(&temp_var)
                        {
                            q.push(temp_var);
                        }

                        for data in &mut q
                        {
                            let (ref mut _temp_s, _temp_i, _temp_ep) = data;

                            if _temp_s.is_empty() {continue;}

                            for idx in 0.._temp_s.row(_j).len()
                            {
                                _temp_s[(_j, idx)] ^= _temp_s[(_ep, idx)];
                            }
                        }
                    }
                }
            }
        }

        if _i.is_empty() {continue;}

         let mut maxes: Vec<usize> = vec![];
        for row in _s.rows()
        {
            maxes.push(std::cmp::max(row.iter().filter(|&&x| x == 0).count(), row.iter().filter(|&&x| x == 1).count()));
        }

        let mut maxes2: Vec<usize> = vec![];
        for _i_idx in _i.clone()
        {
            maxes2.push(maxes[_i_idx]);
        }

        let mut _temp_max = maxes2[0];
        let mut _temp_argmax = 0_usize;

        for (idx, &ele) in maxes2.iter().enumerate()
        {
            if ele > _temp_max
            {
                _temp_max = ele;
                _temp_argmax = idx;
            }
        }

        let _j = _i[_temp_argmax];

        let mut cnots0_t = vec![];
        let mut cnots1_t = vec![];

        let mut cnots0_shape_data = (0_usize, 0_usize);
        let mut cnots1_shape_data = (0_usize, 0_usize);
        for cols in _s.columns()
        {
            if cols[_j] == 0
            {
                cnots0_shape_data.0 += 1;
                cnots0_shape_data.1 = cols.to_vec().len();
                for ele in cols.to_vec()
                {
                    cnots0_t.push(ele);
                }
            }
            else if cols[_j] == 1
            {
                cnots1_shape_data.0 += 1;
                cnots1_shape_data.1 = cols.to_vec().len();
                for ele in cols.to_vec()
                {
                    cnots1_t.push(ele);
                }
            }
        }


        let cnots0 = Array2::from_shape_vec((cnots0_shape_data.0, cnots0_shape_data.1), cnots0_t).unwrap();
        let cnots1 = Array2::from_shape_vec((cnots1_shape_data.0, cnots1_shape_data.1), cnots1_t).unwrap();

        let cnots0 = cnots0.t().to_owned();
        let cnots1 = cnots1.t().to_owned();

        if _ep == num_qubits
        {
            let _temp_data = (cnots1, _i.clone().into_iter().filter(|&x| x != _j).collect(), _j);
            if !q.contains(&_temp_data)
            {
                q.push(_temp_data);
            }
        }
        else
        {
            let _temp_data = (cnots1, _i.clone().into_iter().filter(|&x| x != _j).collect(), _ep);
            if !q.contains(&_temp_data)
            {
                q.push(_temp_data);
            }
        }

        let _temp_data = (cnots0, _i.clone().into_iter().filter(|&x| x != _j).collect(), _ep);
        if !q.contains(&_temp_data)
        {
            q.push(_temp_data);
        }


    }

    let state_bool = state.mapv(|x| x != 0);
    let mut instrs = _synth_cnot_count_full_pmh(state_bool, section_size);
    instrs.reverse();
    for inst in instrs
    {
        instructions.push(inst);
    }
    CircuitData::from_standard_gates(py, num_qubits as u32, instructions, Param::Float(0.0))
}
