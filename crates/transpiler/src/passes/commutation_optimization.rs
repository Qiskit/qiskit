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

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::{pyfunction, wrap_pyfunction, Bound, PyResult};

use indexmap::IndexMap;
use smallvec::{smallvec, SmallVec};

use crate::commutation_checker::CommutationChecker;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, Wire};
use qiskit_circuit::operations::{Operation, Param, StandardGate};
use qiskit_circuit::Qubit;
use qiskit_synthesis::QiskitError;

const _CUTOFF_PRECISION: f64 = 1e-5;
static ROTATION_GATES: [&str; 4] = ["p", "u1", "rz", "rx"];

static VAR_Z_MAP: [(&str, StandardGate); 3] = [
    ("rz", StandardGate::RZ),
    ("p", StandardGate::Phase),
    ("u1", StandardGate::U1),
];
static Z_ROTATIONS: [StandardGate; 6] = [
    StandardGate::Phase,
    StandardGate::Z,
    StandardGate::U1,
    StandardGate::RZ,
    StandardGate::T,
    StandardGate::S,
];
static X_ROTATIONS: [StandardGate; 2] = [StandardGate::X, StandardGate::RX];
static SUPPORTED_GATES: [StandardGate; 5] = [
    StandardGate::CX,
    StandardGate::CY,
    StandardGate::CZ,
    StandardGate::H,
    StandardGate::Y,
];

#[derive(Hash, Eq, PartialEq, Debug)]
enum GateOrRotation {
    Gate(StandardGate),
    ZRotation,
    XRotation,
}
#[derive(Hash, Eq, PartialEq, Debug)]
struct CancellationSetKey {
    gate: GateOrRotation,
    qubits: SmallVec<[Qubit; 2]>,
    com_set_index: usize,
    second_index: Option<usize>,
}

#[pyfunction]
#[pyo3(name = "commutation_optimization")]
pub fn run_commutation_optimization(
    dag: &mut DAGCircuit,
    commutation_checker: &mut CommutationChecker
) -> PyResult<()> {

    println!("Running commutation optimization pass!");

    Ok(())
}

pub fn commutation_optimization_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_commutation_optimization))?;
    Ok(())
}
