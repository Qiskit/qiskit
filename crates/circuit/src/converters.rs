// This code is part of Qiskit.
//
// (C) Copyright IBM 2023, 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use ::pyo3::prelude::*;
use hashbrown::HashMap;
use pyo3::types::{PyDict, PyList};

use crate::{circuit_data::CircuitData, dag_circuit::DAGCircuit};

/// An extractable representation of a QuantumCircuit reserved only for
/// conversion purposes.
#[derive(Debug, Clone)]
pub struct QuantumCircuitData<'py> {
    data: CircuitData,
    name: Option<Bound<'py, PyAny>>,
    calibrations: Option<HashMap<String, Py<PyDict>>>,
    metadata: Option<Bound<'py, PyAny>>,
    qregs: Option<Bound<'py, PyList>>,
    cregs: Option<Bound<'py, PyList>>,
    input_vars: Vec<Bound<'py, PyAny>>,
    captured_vars: Vec<Bound<'py, PyAny>>,
    declared_vars: Vec<Bound<'py, PyAny>>,
}

impl<'py> FromPyObject<'py> for QuantumCircuitData<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let circuit_data = ob.getattr("_data")?;
        let data_borrowed = circuit_data.extract::<CircuitData>()?;
        Ok(QuantumCircuitData {
            data: data_borrowed,
            name: ob.getattr("name").ok(),
            calibrations: ob.getattr("calibrations")?.extract().ok(),
            metadata: ob.getattr("metadata").ok(),
            qregs: ob.getattr("qregs").map(|ob| ob.downcast_into())?.ok(),
            cregs: ob.getattr("cregs").map(|ob| ob.downcast_into())?.ok(),
            input_vars: ob
                .call_method0("iter_input_vars")?
                .iter()?
                .collect::<PyResult<Vec<_>>>()?,
            captured_vars: ob
                .call_method0("iter_captured_vars")?
                .iter()?
                .collect::<PyResult<Vec<_>>>()?,
            declared_vars: ob
                .call_method0("iter_declared_vars")?
                .iter()?
                .collect::<PyResult<Vec<_>>>()?,
        })
    }
}

#[pyfunction(signature = (quantum_circuit, copy_operations = true, qubit_order = None, clbit_order = None))]
pub fn circuit_to_dag(
    py: Python,
    quantum_circuit: QuantumCircuitData,
    copy_operations: bool,
    qubit_order: Option<Vec<Bound<PyAny>>>,
    clbit_order: Option<Vec<Bound<PyAny>>>,
) -> PyResult<DAGCircuit> {
    DAGCircuit::from_circuit(
        py,
        &quantum_circuit.data,
        copy_operations,
        Some([
            quantum_circuit.declared_vars,
            quantum_circuit.input_vars,
            quantum_circuit.captured_vars,
        ]),
        quantum_circuit.qregs,
        quantum_circuit.cregs,
        quantum_circuit.metadata,
        quantum_circuit.name,
        quantum_circuit.calibrations,
        qubit_order,
        clbit_order,
    )
}

pub fn converters(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(circuit_to_dag, m)?)?;
    Ok(())
}
