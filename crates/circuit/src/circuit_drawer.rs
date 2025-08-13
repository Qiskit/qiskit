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

use std::fmt::Debug;
use std::hash::{Hash, RandomState};
#[cfg(feature = "cache_pygates")]
use std::sync::OnceLock;

use crate::bit::{
    BitLocations, ClassicalRegister, PyBit, QuantumRegister, Register, ShareableClbit,
    ShareableQubit,
};
use crate::bit_locator::BitLocator;
use crate::circuit_instruction::{CircuitInstruction, OperationFromPython};
use crate::classical::expr;
use crate::dag_circuit::{add_global_phase, DAGStretchType, DAGVarType};
use crate::imports::{ANNOTATED_OPERATION, QUANTUM_CIRCUIT};
use crate::interner::{Interned, Interner};
use crate::object_registry::ObjectRegistry;
use crate::operations::{Operation, OperationRef, Param, PythonOperation, StandardGate};
use crate::packed_instruction::{PackedInstruction, PackedOperation};
use crate::parameter_table::{ParameterTable, ParameterTableError, ParameterUse, ParameterUuid};
use crate::register_data::RegisterData;
use crate::slice::{PySequenceIndex, SequenceIndex};
use crate::{Clbit, Qubit, Stretch, Var, VarsMode};

use numpy::PyReadonlyArray1;
use pyo3::exceptions::*;
use pyo3::prelude::*;
use pyo3::{import_exception};


use crate::converters::circuit_to_dag;
use crate::converters::QuantumCircuitData;
use crate::dag_circuit::NodeType;
use crate::circuit_data::CircuitData;

import_exception!(qiskit.circuit.exceptions, CircuitError);

#[pyclass(sequence, module = "qiskit._accelerate.circuit")]
pub struct CircuitDrawer;

#[pymethods]
impl CircuitDrawer{

    #[staticmethod]
    #[pyo3(name = "draw")]
    fn py_drawer(py: Python, quantum_circuit: &Bound<PyAny>) -> PyResult<()> {
        if !quantum_circuit.is_instance(QUANTUM_CIRCUIT.get_bound(py))? {
            return Err(PyTypeError::new_err(
                "Expected a QuantumCircuit instance"
            ));
        }
        println!("FUNCTION IS BEING CALLED FROM circuit_drawer.rs FILE");
        let circ_data: CircuitData = quantum_circuit.getattr("_data")?.extract()?;
        circuit_draw(&circ_data);
        Ok(())
    }
}


pub fn circuit_draw(circ_data: &CircuitData) {

    let quantum_circuit_data = QuantumCircuitData {
        data: circ_data.clone(),
        name: None,
        metadata: None,
    };

    let dag_circuit = circuit_to_dag(quantum_circuit_data, true, None, None)
        .expect("Failed to convert circuit data to DAGCircuit");

    let mut output = String::new();
    output.push_str("DAG Circuit Operations:\n");
    output.push_str(&format!("Number of qubits: {}\n", dag_circuit.num_qubits()));
    output.push_str(&format!("Number of operations: {}\n", dag_circuit.num_ops()));
    output.push_str("Operations:\n");
    
    //creating representation where each wire is represented by 3 strings
    let mut circuit_rep: Vec<String> = vec![String::new(); (dag_circuit.num_qubits() + 1) * 3];

    // Fill the first column with qubit labels
    for (i, qubit) in dag_circuit.qubits().objects().iter().enumerate() {
        let qubit_index = i * 3 + 1;
        let qubit_name = format!("q_{}: ", i);
        circuit_rep[qubit_index].push_str(&qubit_name);
        circuit_rep[qubit_index - 1].push_str(" ".repeat((&qubit_name).len()).as_str());
        circuit_rep[qubit_index + 1].push_str(" ".repeat((&qubit_name).len()).as_str());
    }

    // Print the circuit representation
    for i in circuit_rep {
        println!("{}", i);
    }
    //getting qubits and clbit information
    for (index, qubit) in dag_circuit.qubits().objects().iter().enumerate() {
        println!("Qubit {}: {:?}", index, qubit);
    };

    // Iterate through clbits with their indices  
    for (index, clbit) in dag_circuit.clbits().objects().iter().enumerate() {
        println!("Clbit {}: {:?}", index, clbit);
    };

    let layer_iterator = dag_circuit.multigraph_layers();
    for (layer_index, layer) in layer_iterator.enumerate() {
        output.push_str(&format!("Layer {}:\n", layer_index));
    
        // Filter for operation nodes only
        let operations: Vec<_> = layer
            .into_iter()
            .filter_map(|node_index| {
                match &dag_circuit.dag()[node_index] {
                    NodeType::Operation(instruction) => Some((node_index, instruction)),
                    _ => None, // Skip input/output nodes
                }
            })
            .collect();
    
        if operations.is_empty() {
            continue; // Skip layers with no operations
        }
    
        for (node_index, instruction) in operations {
            let op_name = instruction.op.name();
            let qubits = dag_circuit.qargs_interner().get(instruction.qubits);
            let clbits = dag_circuit.cargs_interner().get(instruction.clbits);
        
            let qubit_str = qubits.iter()
                .map(|q| q.0.to_string())
                .collect::<Vec<_>>()
                .join(",");
            
            let clbit_str = clbits.iter()
                .map(|c| c.0.to_string())
                .collect::<Vec<_>>()
                .join(",");
        
            output.push_str(&format!(
                "  Node {}: {} qubits=[{}] clbits=[{}]\n",
                node_index.index(), op_name, qubit_str, clbit_str
            ));
        
            // Print parameters if any
            let params = instruction.params_view();
            if !params.is_empty() {
                output.push_str(&format!("    params: {:?}\n", params));
            }
        }
    }

    println!("{}",output);
}

