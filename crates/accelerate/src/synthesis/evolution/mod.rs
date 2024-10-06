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

use std::borrow::Borrow;
use pyo3::prelude::*;
use pyo3::pyclass::boolean_struct::True;
use pyo3::types::{PyList, PyTuple, PyString};

use rustiq_core::structures::{CliffordCircuit, CliffordGate, GraphState, Metric, PauliSet};
use rustiq_core::synthesis::pauli_network::{check_circuit, greedy_pauli_network};
use smallvec::{SmallVec, smallvec};
use qiskit_circuit::operations::{Param, StandardGate};
use qiskit_circuit::Qubit;


/// Represents the return type of synthesis algorithms.
pub type QiskitGate = (StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>);
pub type QiskitGatesVec = Vec<QiskitGate>;


// [('zz', [1, 3], 0.334), ... ]

/// Expands the sparse pauli string representation to a full representation
/// e.g. "XY" on "1, 3" with "num_qubits = 5" becomes "IXIYI"
fn expand_pauli(sparse_pauli: String, qubits: Vec<u32>, num_qubits: usize) -> String {
    let mut v: Vec<char> = vec!["I".parse().unwrap(); num_qubits];
    for (i, q) in qubits.iter().enumerate() {
        v[*q as usize] = sparse_pauli.chars().nth(i).unwrap();
    }
    v.into_iter().collect()
}

/// Converts a sequence of Rustiq gates to a sequence of Qiskit gates
fn to_qiskit_gates(rustiq_gates: &Vec<CliffordGate>) -> Vec<QiskitGate> {
    let gates: Vec<QiskitGate> = Vec::with_capacity(rustiq_gates.len());

    for gate in rustiq_circuit.gates {
        match gate {
            CliffordGate::CNOT(i, j) => gates.push((
                StandardGate::CXGate,
                smallvec![],
                smallvec![
                    Qubit::new(i),
                    Qubit::new(j)
                ],
            ));
                ("CNOT".to_owned(), vec![*i, *j]),
            CliffordGate::CZ(i, j) => ("CZ".to_owned(), vec![*i, *j]),
            CliffordGate::H(i) => ("H".to_owned(), vec![*i]),
            CliffordGate::S(i) => ("S".to_owned(), vec![*i]),
            CliffordGate::Sd(i) => ("Sd".to_owned(), vec![*i]),
            CliffordGate::SqrtX(i) => ("SqrtX".to_owned(), vec![*i]),
            CliffordGate::SqrtXd(i) => ("SqrtXd".to_owned(), vec![*i]),
        }
    }

    gates
}



#[pyfunction]
#[pyo3(signature = (pauli_network, num_qubits))]
pub fn pauli_network_synthesis(_py: Python, pauli_network: &Bound<PyList>, num_qubits: usize) -> PyResult<()> {

    let mut operator_sequence: Vec<String> = Vec::new();
    let mut params: Vec<Bound<PyAny>> = Vec::new();

    println!("pauli_network = {:?}", pauli_network);
    for item in pauli_network {
        let tuple = item.downcast::<PyTuple>()?;
        println!("-- item = {:?}", tuple);
        let inner = tuple.borrow();

        let ss: String = inner.get_item(0)?.downcast::<PyString>()?.extract()?;
        // let ff: f64 =  inner.get_item(2)?.extract()?;
        let param = inner.get_item(2)?;
        let vv = inner.get_item(1)?;
        let ww: Vec<u32> = vv.downcast::<PyList>()?.iter().map(|v| v.extract()).collect::<PyResult<_>>()?;
        println!("  ss = {:?}, ww = {:?}, param = {:?}", ss, ww, param);

        // todo: check endianness notation, maybe need to reverse something

        operator_sequence.push(expand_pauli(ss, ww, num_qubits));
        params.push(param);
    }

    // todo: create full op seq

    println!("HERE: operator_sequence = {:?}", operator_sequence);
    println!("HERE: list of params = {:?}", params);

    // todo: maybe we can immediately create pauli set rather than operator_sequence first
    let mut bucket = PauliSet::from_slice(&operator_sequence);
    let circuit = greedy_pauli_network(
        &mut bucket,
        &Metric::COUNT,
         true,
        0,
        false,
        false,
    );

    // todo: what the hell is circuit and can we not go through Python interface?
    println!("circuit: {:?}", circuit);

    //
    let is_ok = check_circuit(&operator_sequence, &circuit);
    println!("is_ok = {:?}", is_ok);

    Ok(())
}

pub fn evolution(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pauli_network_synthesis, m)?)?;
    Ok(())
}
