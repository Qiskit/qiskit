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
use std::thread::current;
use hashbrown::HashSet;
use crate::bit::{
    BitLocations, ClassicalRegister, PyBit, QuantumRegister, Register, ShareableClbit,
    ShareableQubit,
};
use crate::bit_locator::BitLocator;
use crate::circuit_instruction::{CircuitInstruction, OperationFromPython};
use crate::classical::expr;
use crate::dag_circuit::{self, add_global_phase, DAGCircuit, DAGStretchType, DAGVarType};
use crate::imports::{ANNOTATED_OPERATION, QUANTUM_CIRCUIT};
use rustworkx_core::petgraph::stable_graph::{EdgeReference, NodeIndex};
use crate::interner::{Interned, Interner};
use crate::object_registry::ObjectRegistry;
use crate::operations::{Operation, OperationRef, Param, PythonOperation, StandardGate, StandardInstruction};
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


#[pyfunction(name = "draw")]
pub fn py_drawer(py: Python, quantum_circuit: &Bound<PyAny>) -> PyResult<()> {
    if !quantum_circuit.is_instance(QUANTUM_CIRCUIT.get_bound(py))? {
        return Err(PyTypeError::new_err(
            "Expected a QuantumCircuit instance"
        ));
    }
    let circ_data: QuantumCircuitData = quantum_circuit.extract()?;
    let dag_circuit = circuit_to_dag(circ_data, true, None, None)?;
    circuit_draw(&dag_circuit);
    Ok(())
}

pub const q_wire: &str = "─";
pub const c_wire: char = '═';
pub const top_con: char = '┴';
pub const bot_con: char = '┬';
pub const left_con: char = '┤';
pub const right_con: char = '├';
pub const top_left_con: char = '┌';
pub const top_right_con: char = '┐';
pub const bot_left_con: char = '└';
pub const bot_right_con: char = '┘';

#[derive(Clone, Debug, PartialEq, Eq)]
enum wire_type {
    qubit,
    clbit,
}

impl From<&wire_type> for &str {
    fn from(wire_type: &wire_type) -> Self {
        match wire_type {
            wire_type::qubit => "─",
            wire_type::clbit => "═",
        }
    }
}

#[derive(Clone, Debug)]
enum control_type{
    open,
    closed,
}

#[derive(Clone, Debug)]
pub struct input{

}
#[derive(Clone, Debug)]
pub struct wire {
    top: String,
    mid: String,
    bot: String,
    wire_len: u64,
    type_: wire_type,
}

impl wire {
    pub fn new(type_: wire_type) -> Self {
        wire {
            top: String::new(),
            mid: String::new(),
            bot: String::new(),
            wire_len: 0,
            type_: type_,
        }
    }

    pub fn update_wire_len(&mut self) {
        let top_len = self.top.len();
        let mid_len = self.mid.len();
        let bot_len = self.bot.len();
        if top_len == mid_len && mid_len == bot_len {
            self.wire_len = top_len as u64;
        } else {
            panic!("The lengths of the wire components are not equal");
        }
    }
    // setting qubit names
    pub fn qubit_name(&mut self, qubit_name: &str) {
        let name_len = qubit_name.len();
        self.top.push_str(" ".repeat(name_len).as_str());
        self.mid.push_str(&format!("{}", qubit_name));
        self.bot.push_str(" ".repeat(name_len).as_str());
        self.update_wire_len();
    }

    // concatenate full wire representation and send for printing
    pub fn get_wire_rep(&self) -> String {
        let mut wire_rep = String::new();
        wire_rep.push_str(&self.top);
        wire_rep.push('\n');
        wire_rep.push_str(&self.mid);
        wire_rep.push('\n');
        wire_rep.push_str(&self.bot);
        wire_rep.push('\n');
        wire_rep
    }

    pub fn fix_len(&mut self, num: u64) {
        self.wire_len = self.wire_len + num;
        self.top.push_str(" ".repeat(num as usize).as_str());
        self.mid.push_str(<&str>::from(&self.type_).repeat(num as usize).as_str());
        self.bot.push_str(" ".repeat(num as usize).as_str());
    }
}

#[derive(Clone, Debug)]
pub struct enclosed{
    qubits: Vec<u32>,
    clbits: Vec<u32>,
    control: Vec<u32>,
    control_types: Vec<control_type>,
    wire: wire_type,
    name: String,
    label: Option<String>,
}

impl enclosed {
    fn from_instruction(instruction: &PackedInstruction, dag_circ: &DAGCircuit) -> Self {
        let instruction_qubits = dag_circ.qargs_interner().get(instruction.qubits).iter().map(|q| q.0).collect();
        let instruction_clbits = dag_circ.cargs_interner().get(instruction.clbits).iter().map(|c| c.0).collect();
        let instruction_name = instruction.op.name();
        let instruction_label = instruction.label();
        // handle case for when the control state is different

        println!("name: {}", instruction_name);
        println!("label: {:?}", instruction_label);
        println!("qubits: {:?}", instruction_qubits);
        println!("clbits: {:?}", instruction_clbits);

        enclosed {
            qubits: instruction_qubits,
            clbits: instruction_clbits,
            control: vec![],
            control_types: vec![],
            wire: wire_type::qubit,
            name: instruction_name.to_string(),
            label: instruction_label.map(|s| s.to_string()),
        }

    }
}

pub struct circuit_rep {
    q_wires: Vec::<wire>,
    dag_circ: DAGCircuit
}

impl circuit_rep {
    pub fn new(dag_circ: DAGCircuit) -> Self {

        //number of qubits in dag_circuit
        let qubit = dag_circ.num_qubits();

        circuit_rep {
            q_wires: vec!(wire::new(wire_type::qubit); qubit as usize),
            dag_circ: dag_circ
        }
    }

    pub fn circuit_string(&self) -> String {
        let mut output = String::new();
        for wires in self.q_wires.iter() {
            output.push_str(&wires.get_wire_rep());
        }
        output
    }

    pub fn fix_len(&mut self, chr: &str) {
        let mut num = 0;
        for wire in self.q_wires.iter() {
            if wire.wire_len > num {
                num = wire.wire_len;
            }
        }

        for wire in self.q_wires.iter_mut() {
            wire.fix_len(num - wire.wire_len);
        }
    }

    pub fn set_qubit_name(&mut self) {
        for (i, qubit) in self.dag_circ.qubits().objects().iter().enumerate() {
            let qubit_name = if let Some(locations) = self.dag_circ.qubit_locations().get(qubit) {
                if let Some((register, reg_index)) = locations.registers().first() {
                    format!("{}_{}", register.name(), reg_index)
                } else {
                    format!("q_{}", i)
                }
            } else {
                format!("q_{}", i)
            };
            self.q_wires[i].qubit_name(&qubit_name);
        }
        self.fix_len(" ");
    }

    pub fn build_layer(&mut self, layer: Vec<&PackedInstruction>){
        for instruction in layer{
        let enclosed_inst = enclosed::from_instruction(instruction, &self.dag_circ);
        // println!("{:?}", enclosed_inst);
        //     if instruction.op.control_flow() {
        //         panic!("Control flow operations not supported yet: {}", instruction.op.name());
        //     }
    
        //     // Check for standard instructions (barrier, measure, reset, delay)
        //     if let Some(standard_instruction) = instruction.op.try_standard_instruction() {
        //         match standard_instruction {
        //             StandardInstruction::Measure => {
        //                 // This is a measurement operation
        //                 // Handle: MeasureFrom element on qubit, MeasureTo element on clbit
        //                 // Access qubits: instruction.qubits (interned slice)
        //                 // Access clbits: instruction.clbits (interned slice)
        //                 // Create MeasureFrom element for the qubit
        //                 // Create MeasureTo element for the clbit
        //                 // Handle classical register bundling if needed
        //             }
        //             StandardInstruction::Barrier(num_qubits) => {
        //                 // This is a barrier directive
        //                 // Handle: Barrier elements across specified qubits

        //                 //get label of barrier

        //                 // Number of qubits affected: num_qubits
        //                 // Access affected qubits: instruction.qubits
        //                 // Check if barriers should be plotted with plotbarriers flag
        //                 // Place barrier symbol on each affected qubit wire
        //                 // Add label on topmost qubit if present: instruction.label()
        //                 let mut temp:bool = false;
        //                 for wire in self.q_wires.iter_mut() {
        //                     if wire.type_ == wire_type::qubit {
        //                         wire.top.push_str("░");
        //                         wire.mid.push_str("░");
        //                         wire.bot.push_str("░");
        //                     }
        //                 }
        //             }
        //             StandardInstruction::Reset => {
        //                 // PANIC - Reset not supported yet
        //                 panic!("Reset operations not supported yet");
        //             }
        //             StandardInstruction::Delay(_unit) => {
        //                 // PANIC - Delay not supported yet  
        //                 panic!("Delay operations not supported yet");
        //             }
        //         }
        //     }
        
        //     // Check for standard gates
        //     else if let Some(standard_gate) = instruction.op.try_standard_gate() {
        //         // This is a standard gate operation
        //         let num_qubits = instruction.op.num_qubits();
                
        //         if num_qubits == 1 {
        //             // Single qubit standard gate
        //             // Handle: BoxOnQuWire element
        //             // Gate name: instruction.op.name()
        //             // Parameters: instruction.params_view() - format for display
        //             // Label: instruction.label()
        //             // Check for conditional operation (if condition exists)
        //             // Place single qubit box element on the target wire
        //         } else {
        //             // Multi-qubit standard gate
        //             // Access target qubits: instruction.qubits
        //             // Gate name: instruction.op.name()  
        //             // Parameters: instruction.params_view()
        //             // Label: instruction.label()
                    
        //             // Special handling for specific multi-qubit gates:
        //             match standard_gate {
        //                 // SwapGate: Create Ex elements with connections
        //                 // RZZGate: Create Bullet elements with ZZ connection label
        //                 // CXGate, CZGate, etc: Handle as controlled gates
        //                 // Other multi-qubit gates: Use multi-qubit box spanning all qubits
        //                 _ => {
        //                     // Generic multi-qubit gate handling
        //                     // Create box elements spanning all target qubits
        //                     // Handle top/middle/bottom box parts for multi-qubit span
        //                 }
        //             }
        //         }
        //     }
        
        // // Everything else - PANIC for now
        //     else {
        //         match instruction.op.view() {
        //             OperationRef::Gate(_) => {
        //                 panic!("Python gates not supported yet: {}", instruction.op.name());
        //             }
        //             OperationRef::Instruction(_) => {
        //                 panic!("Python instructions not supported yet: {}", instruction.op.name());
        //             }
        //             OperationRef::Operation(_) => {
        //                 panic!("Python operations not supported yet: {}", instruction.op.name());
        //             }
        //             OperationRef::Unitary(_) => {
        //                 panic!("Unitary gates not supported yet: {}", instruction.op.name());
        //             }
        //             _ => {
        //                 panic!("Unknown operation type: {}", instruction.op.name());
        //             }
        //         }
        //     }
        // }
        }
    }

    pub fn build_layers(&mut self) {
        let binding = self.dag_circ.clone();
        let layer_iterator = binding.multigraph_layers();

        let mut final_layers:Vec<Vec<NodeIndex>> = Vec::new();

        let total_qubits = binding.num_qubits() as u32;
        let total_clbits = binding.num_clbits() as u32;

        for (i,layer) in layer_iterator.enumerate(){ 
            let mut sublayers: Vec<Vec<NodeIndex>> = vec![Vec::new()];

            for node_index in layer {

                if let NodeType::Operation(instruction_to_insert) = &binding.dag()[node_index]{

                    if sublayers.is_empty() {
                        sublayers.push(vec![node_index]);
                        continue;
                    }

                    let node_qubits = binding.qargs_interner().get(instruction_to_insert.qubits);
                    let node_clbits = binding.cargs_interner().get(instruction_to_insert.clbits);

                    let node_min_qubit = node_qubits.iter().map(|q| q.0).min();
                    let node_max_qubit = node_qubits.iter().map(|q| q.0).max();
                    let node_min_clbit = node_clbits.iter().map(|c| c.0).min();
                    let node_max_clbit = node_clbits.iter().map(|c| c.0).max();

                    let node_min = match node_min_qubit {
                        Some(val) => val,
                        None => match node_max_qubit {
                            Some(val) => val,
                            None =>  match node_min_clbit {
                                Some(val) => val + total_qubits,
                                None => match node_max_clbit {
                                    Some(val) => val + total_qubits,
                                    None => continue, // No qubits or clbits, skip this node
                                },                                                
                            }, // No qubits or clbits, skip this node
                        },
                    };

                    let node_max = match node_max_clbit {
                        Some(val) => val + total_qubits,
                        None => match node_min_clbit {
                            Some(val) => val + total_qubits,
                            None =>  match node_max_qubit {
                                Some(val) => val,
                                None => match node_min_qubit {
                                    Some(val) => val,
                                    None => continue, // No qubits or clbits, skip this node
                                },                                                
                            }, // No qubits or clbits, skip this node
                        },
                    };

                    let mut sublayer = sublayers.last_mut().unwrap();
                    let mut overlap = false;
                    for &subnode in sublayer.iter() {
                        if let NodeType::Operation(instruction) = &binding.dag()[subnode]{
                            let subnode_qubits = binding.qargs_interner().get(instruction.qubits);
                            let subnode_clbits = binding.cargs_interner().get(instruction.clbits);
                            let subnode_min_qubit = subnode_qubits.iter().map(|q| q.0).min();  
                            let subnode_max_qubit = subnode_qubits.iter().map(|q| q.0).max();
                            let subnode_min_clbit = subnode_clbits.iter().map(|c| c.0).min();
                            let subnode_max_clbit = subnode_clbits.iter().map(|c| c.0).max();
                            let subnode_min = match subnode_min_qubit {
                                Some(val) => val,
                                None => match subnode_max_qubit {
                                    Some(val) => val,
                                    None =>  match subnode_min_clbit {
                                        Some(val) => val + total_qubits,
                                        None => match subnode_max_clbit {
                                            Some(val) => val + total_qubits,
                                            None => continue, // No qubits or clbits, skip this node
                                        },                                                
                                    }, // No qubits or clbits, skip this node
                                },
                            };
                            let subnode_max = match subnode_max_clbit {
                                Some(val) => val + total_qubits,
                                None => match subnode_min_clbit {
                                    Some(val) => val + total_qubits,
                                    None =>  match subnode_max_qubit {
                                        Some(val) => val,
                                        None => match subnode_min_qubit {
                                            Some(val) => val,
                                            None => continue, // No qubits or clbits, skip this node
                                        },                                                
                                    }, // No qubits or clbits, skip this node
                                },
                            };
                            if (subnode_min <= node_min && subnode_max >= node_min) || (subnode_min <= node_max && subnode_max >= node_max) {
                                overlap = true;
                                break;
                            }
                        }
                    }

                    if overlap {
                        sublayers.push(vec![node_index]);
                    } else {
                        sublayer.push(node_index);
                    }
                }
            }
            
            let mut ct = 0;
            for j in sublayers {
                if j.is_empty() {
                    continue;
                } else {
                    final_layers.push(j);
                    ct += 1;
                }
            }
        }


        let mut packedin_layers: Vec<Vec<&PackedInstruction>> = Vec::new();

        for (id,layer) in final_layers.iter().enumerate() {
            let mut packedin_layer: Vec<&PackedInstruction> = Vec::new();
            for nodeind in layer {
                if let NodeType::Operation(instruction) = &binding.dag()[*nodeind] {
                    packedin_layer.push(instruction);
                } 
            }
            println!("Layer {}", id);
            self.build_layer(packedin_layer.clone());
        }
        
    }
}


pub fn circuit_draw(dag_circ: &DAGCircuit) {

    let mut output = String::new();

    // Create a circuit representation
    let mut circuit_rep = circuit_rep::new(dag_circ.clone());
    circuit_rep.set_qubit_name();
    output.push_str(&circuit_rep.circuit_string());
    circuit_rep.build_layers();
    // Print the circuit representation
    println!("{}", output);
}
