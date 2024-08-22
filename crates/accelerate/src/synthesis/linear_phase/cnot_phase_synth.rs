

#[pyfunction]
#[pyo3(signature = (cnots, angles, section_size=2))]
pub fn synth_cnot_phase_aam(
    py:Python,
    cnots: PyReadonlyArray2<u8>,
    angles: PyReadonlyArray1<PyObject>,
    section_size: Option<u8>,
) -> PyResult<CircuitData> {

    let S = cnots.as_array().to_owned();
    let angles_arr = angles.as_array().to_owned();
    let num_qubits = S.shape()[0];
    let mut S_cpy = S.clone();
    let mut instructions = vec![];

    let mut rust_angles = Vec::new();
    for obj in angles_arr
    {
        rust_angles.push(obj.extract::<String>(py)?);
    }

    let mut state = Array2::<u8>::eye(num_qubits);

    for qubit_idx in 0..num_qubits
    {
        let mut index = 0 as usize;

        loop
        {
            let icnot = S_cpy.column(index).to_vec();

            if icnot == state.row(qubit_idx as usize).to_vec()
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
                   S_cpy.remove_index(numpy::ndarray::Axis(1), index);
                   if index == S_cpy.shape()[1] {break;}
                   index -=1;
            }
            index +=1;
        }
    }


    let epsilion = num_qubits;
    let mut Q = vec![(S, 0..num_qubits, epsilion)];

    while Q.len() != 0
       {

        let (_S, _I, _i) = Q.pop().unwrap();

        if _S.len() == 0 {continue;}

        if 0 <= _i &&  _i < num_qubits
        {
            let condition = true;
            while condition
            {
                let mut condition = false;

                for _j in 0..num_qubits
                {
                    if (_j != _i) && (u32::from(_S.row(_j as usize).sum()) == _S.row(_j).len() as u32)
                    {
                        condition = true;
                        instructions.push((StandardGate::CXGate, smallvec![], smallvec![Qubit(_i as u32), Qubit(_i as u32)]));

                        for _k in 0..state.ncols()
                        {
                            state[(_i, _k)] ^= state[(_j, _k)];
                        }

                        let mut index = 0 as usize;

                        loop
                        {
                            let icnot = S_cpy.column(index).to_vec();

                            if icnot == state.row(_i as usize).to_vec()
                            {

                                match rust_angles.get(index)
                                {
                                    Some(gate) if gate == "t" => instructions.push((StandardGate::TGate, smallvec![], smallvec![Qubit(_i as u32)])),
                                    Some(gate) if gate == "tdg" => instructions.push((StandardGate::TdgGate, smallvec![], smallvec![Qubit(_i as u32)])),
                                    Some(gate) if gate == "s" => instructions.push((StandardGate::SGate, smallvec![], smallvec![Qubit(_i as u32)])),
                                    Some(gate) if gate == "sdg" => instructions.push((StandardGate::SdgGate, smallvec![], smallvec![Qubit(_i as u32)])),
                                    Some(gate) if gate == "z" => instructions.push((StandardGate::ZGate, smallvec![], smallvec![Qubit(_i as u32)])),
                                    Some(angles_in_pi) => instructions.push((StandardGate::PhaseGate, smallvec![Param::Float((angles_in_pi.parse::<f64>()?) % PI)], smallvec![Qubit(_i as u32)])),
                                    None => (),
                                };

                                rust_angles.remove(index);
                                S_cpy.remove_index(numpy::ndarray::Axis(1), index);
                                if index == S_cpy.shape()[1] { break; }
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
        if _I.len() == 0 {continue;}
    }



    CircuitData::from_standard_gates(py, num_qubits as u32, instructions, Param::Float(0.0))
}
