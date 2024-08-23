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
use pyo3::prelude::*;
use std::f64::consts::PI;

#[pyfunction]
#[pyo3(signature = (cnots, angles, section_size=2))]
pub fn synth_cnot_phase_aam(
    py:Python,
    cnots: PyReadonlyArray2<u8>,
    angles: PyReadonlyArray1<PyObject>,
    section_size: Option<u8>,
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

        loop
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
                   index -=1;
            }
            index +=1;
        }
    }


    let epsilion = num_qubits;
    let mut q = vec![(s, 0..num_qubits, epsilion)];

    while !q.is_empty()
       {

        let (_s, _i, _ep) = q.pop().unwrap();

        if _s.is_empty() {continue;}

        if 0 <= _ep &&  _ep < num_qubits
        {
            let mut condition = true;
            while condition
            {
                condition = false;

                for _j in 0..num_qubits
                {
                    if (_j != _ep) && (u32::from(_s.row(_j).sum()) == _s.row(_j).len() as u32)
                    {
                        condition = true;
                        instructions.push((StandardGate::CXGate, smallvec![], smallvec![Qubit(_ep as u32), Qubit(_ep as u32)]));

                        for _k in 0..state.ncols()
                        {
                            state[(_ep, _k)] ^= state[(_j, _k)];
                        }

                        let mut index = 0_usize;

                        loop
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
                                if index == s_cpy.shape()[1] { break; }
                                index -=1;
                            }
                            index +=1;
                        }

                        loop
                        {

                        }
                    }
                }
            }
        }
        if _i.is_empty() {continue;}
    }



    CircuitData::from_standard_gates(py, num_qubits as u32, instructions, Param::Float(0.0))
}
