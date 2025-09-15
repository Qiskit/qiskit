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

use std::fmt::{format, Debug};
use std::hash::{Hash, RandomState};
use std::sync::Barrier;
#[cfg(feature = "cache_pygates")]
use std::sync::OnceLock;
use std::thread::current;
use hashbrown::HashSet;
use hashbrown::HashMap;
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
pub const barrier_char: char = '░';

#[derive(Clone, Debug, PartialEq, Eq)]
enum Wire_Type {
    qubit,
    clbit,
}

impl From<&Wire_Type> for &str {
    fn from(Wire_Type: &Wire_Type) -> Self {
        match Wire_Type {
            Wire_Type::qubit => "─",
            Wire_Type::clbit => "═",
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
    type_: Wire_Type,
}

impl wire {
    pub fn new(type_: Wire_Type) -> Self {
        wire {
            top: String::new(),
            mid: String::new(),
            bot: String::new(),
            type_: type_,
        }
    }

    pub fn get_len(&mut self) -> usize {
        let top_len = self.top.len();
        let mid_len = self.mid.len();
        let bot_len = self.bot.len();
        if top_len == mid_len && mid_len == bot_len {
            return mid_len;
        } else {
            let max_len = top_len.max(mid_len).max(bot_len);
            self.fix_len(max_len);
        }
        self.mid.len()
    }

    pub fn update_wire_len(&mut self) {
        let top_len = self.top.len();
        let mid_len = self.mid.len();
        let bot_len = self.bot.len();
        if top_len == mid_len && mid_len == bot_len {
            
        } else {
            let max_len = top_len.max(mid_len).max(bot_len);
            self.fix_len(max_len);
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

    pub fn fix_len(&mut self, num: usize) {
        self.top.push_str(" ".repeat(num).as_str());
        self.mid.push_str(<&str>::from(&self.type_).repeat(num).as_str());
        self.bot.push_str(" ".repeat(num).as_str());
    }

    pub fn add_wire_component(&mut self, component: &wire) {
        &self.top.push_str(&component.top);
        &self.mid.push_str(&component.mid);
        &self.bot.push_str(&component.bot);
        &self.update_wire_len();
    }
}


pub trait DrawElement{

    fn get_width(&self) -> u64{
        1
    }

    fn component_dict(&self) -> HashMap<u32, wire>{
        HashMap::<u32, wire>::new()
    }

}

pub trait DirectOnWire{

}

#[derive(Clone, Debug, PartialEq)]
enum VisualType{
    Boxed,
    BetweenWire(String),
    DirectOnWire(String),
    WholeWire,
}

pub struct Enclosed<'a>{
    packedinst: &'a PackedInstruction,
    dag_circ: &'a DAGCircuit,
    wire: Wire_Type,
    name: String,
    label: Option<String>,
    visual_type: VisualType,
}

impl<'a> Enclosed<'a> {
    pub fn new(packedinst: &'a PackedInstruction, dag_circ: &'a DAGCircuit ,wire: Wire_Type, name: String, label: Option<String>, visual_type: VisualType) -> Enclosed<'a>{
        Enclosed {
            packedinst,
            dag_circ,
            wire,
            name,
            label,
            visual_type,
        }
    }

    pub fn get_wire_indices(&self) -> Vec<u32>{
        let mut qubit_indices: Vec<u32> = self.dag_circ.qargs_interner().get(self.packedinst.qubits)
        .iter()
        .map(|qubit| qubit.0)
        .collect();

        let total_qubits:u32 = self.dag_circ.num_qubits() as u32;

        let clbit_indices: Vec<u32> = self.dag_circ.cargs_interner().get(self.packedinst.clbits)
        .iter()
        .map(|clbit| clbit.0 + total_qubits)
        .collect();

        qubit_indices.extend(clbit_indices.into_iter());

        qubit_indices
    }

    pub fn wire_component(&self, ind: u32) -> wire{
        let mut component = wire::new(self.wire.clone());
        let wire_indices = self.get_wire_indices();
        let min_index = *wire_indices.iter().min().unwrap();
        let max_index = *wire_indices.iter().max().unwrap();

        // barrier handling
        if self.visual_type == VisualType::DirectOnWire(barrier_char.to_string()) {
            let mut label = barrier_char.to_string();
            if let Some(l) = &self.label{
                label = l.to_string();
            } else {
                label = barrier_char.to_string();
            }
            let label_len:usize = label.len();

            let mut left_pad_len = 0;
            let mut right_pad_len = 0;
            if label_len == 0 {
                left_pad_len = 0;
                right_pad_len = 0;
            }
            else if label_len % 2 == 0{
                left_pad_len = (label_len / 2) - 1;
                right_pad_len = label_len / 2;
            }
            else {
                left_pad_len = label_len / 2;
                right_pad_len = label_len / 2;
            }
            let component_wire = format!("{}{}{}",
                <&str>::from(&self.wire).repeat(left_pad_len),
                barrier_char,
                <&str>::from(&self.wire).repeat(right_pad_len)
            );

            let between_wire = format!("{}{}{}",
                " ".repeat(left_pad_len),
                barrier_char,
                " ".repeat(right_pad_len)
            );

            if ind == 0 {
                component.top.push_str(&label[..]);
            } else {
                component.top.push_str(&component_wire);
            }
            component.mid.push_str(&between_wire);
            component.bot.push_str( &between_wire);
            return component;
        }
        
        match &self.visual_type {
            VisualType::Boxed => {
                if let Some(label) = &self.label {
                    let label_len = label.len() as usize;
                    component.top.push_str(&format!("{}{}{}", top_left_con, "─".repeat(label_len as usize), top_right_con));
                    component.mid.push_str(&format!("{}{}{}", left_con, label, right_con));
                    component.bot.push_str(&format!("{}{}{}", bot_left_con, "─".repeat(label_len as usize), bot_right_con));
                } else {
                    panic!("Boxed visual has no label");
                }
            }
            VisualType::BetweenWire(label) => {
                let label_len = label.len() as u64;
                component.top.push_str(&format!("{}{}{}", top_con, "─".repeat(label_len as usize), top_con));
                component.mid.push_str(&format!(" {} ", label));
                component.bot.push_str(&format!("{}{}{}", bot_con, "─".repeat(label_len as usize), bot_con));
            }
            VisualType::DirectOnWire(label) => {
                component.top.push_str(" ".repeat(label.len()).as_str());
                component.mid.push_str(label);
                component.bot.push_str(" ".repeat(label.len()).as_str());
            }
            VisualType::WholeWire => {
                let wire_len = 1; // single character for whole wire
                component.top.push_str("░".repeat(wire_len).as_str());
                component.mid.push_str("░".repeat(wire_len).as_str());
                component.bot.push_str("░. ".repeat(wire_len).as_str());
            }
        }
        component.update_wire_len();
        component
    }
}

impl<'a> DrawElement for Enclosed<'a>{
    fn get_width(&self) -> u64 {
        let type_ = match &self.visual_type {
            VisualType::Boxed => {
                if let Some(label) = &self.label {
                    label.len() as u64 + 2 // for the box edges
                } else {
                    panic!("Boxed visual has no label");
                }
            }
            VisualType::BetweenWire(label) => {
                label.len() as u64 + 1 // for the spaces around the label
            }
            VisualType::DirectOnWire(label) => {
                1 as u64// no extra space needed
            }
            VisualType::WholeWire => {
                0 as u64// just a single character for the whole wire
            }
        };

        return type_;
    }
    
    fn component_dict(&self) -> HashMap<u32, wire>{
        // map of qubit index to wire component
        let mut components = HashMap::<u32, wire>::new();
        let indices = self.get_wire_indices();
        // for each qubit in packedinstruction, call wire component
        for idx in indices{
            components.insert(idx, self.wire_component(idx));
        }
        components
    }
}


pub struct circuit_rep {
    q_wires: Vec::<wire>,
    dag_circ: DAGCircuit
}

impl<'a> circuit_rep {
    pub fn from_instruction(&'a self, instruction: &'a PackedInstruction) -> impl DrawElement + 'a {
        // println!("{:?}", instruction.label());
        if let Some(standard_gate) = instruction.op.try_standard_gate() {
            let instruction_name = instruction.op.name();
            let instruction_param =format!("{:?}",instruction.params_view());
            let mut instruction_label: Option<String> = match standard_gate {
                StandardGate::GlobalPhase => {
                    // Global phase gate - affects overall circuit phase
                    
                    None
                }
                StandardGate::H => {
                    // Hadamard gate - creates superposition
                    Some("H".to_string())
                }
                StandardGate::I => {
                    // Identity gate - no operation
                    Some("I".to_string())
                }
                StandardGate::X => {
                    // Pauli-X gate (NOT gate)
                    Some("X".to_string())
                }
                StandardGate::Y => {
                    // Pauli-Y gate
                    Some("Y".to_string())
                }
                StandardGate::Z => {
                    // Pauli-Z gate
                    Some("Z".to_string())
                }
                StandardGate::Phase => {
                    // Phase gate (parameterized)
                    Some(format!("P({})", instruction_param))
                }
                StandardGate::R => {
                    // R gate (rotation about axis in XY plane)
                    Some(format!("R({})", instruction_param))
                }
                StandardGate::RX => {
                    // Rotation about X axis
                    Some(format!("RX({})", instruction_param))
                }
                StandardGate::RY => {
                    // Rotation about Y axis
                    Some(format!("RY({})", instruction_param))
                }
                StandardGate::RZ => {
                    // Rotation about Z axis
                    Some(format!("RZ({})", instruction_param))
                }
                StandardGate::S => {
                    // S gate (phase π/2)
                    Some("S".to_string())
                }
                StandardGate::Sdg => {
                    // S dagger gate (phase -π/2)
                    Some("S†".to_string())
                }
                StandardGate::SX => {
                    // Square root of X gate
                    Some("√X".to_string())
                }
                StandardGate::SXdg => {
                    // Square root of X dagger gate
                    Some("√X†".to_string())
                }
                StandardGate::T => {
                    // T gate (phase π/4)
                    Some("T".to_string())
                }
                StandardGate::Tdg => {
                    // T dagger gate (phase -π/4)
                    Some("T†".to_string())
                }
                StandardGate::U => {
                    // Universal single-qubit gate (3 parameters)
                    Some(format!("U({})", instruction_param))
                }
                StandardGate::U1 => {
                    // U1 gate (1 parameter - phase)
                    Some(format!("U1({})", instruction_param))
                }
                StandardGate::U2 => {
                    // U2 gate (2 parameters)
                    Some(format!("U2({})", instruction_param))
                }
                StandardGate::U3 => {
                    // U3 gate (3 parameters - equivalent to U)
                    Some(format!("U3({})", instruction_param))
                }
                StandardGate::CH => {
                    // Controlled Hadamard gate
                    Some("H".to_string())
                }
                StandardGate::CX => {
                    // Controlled-X gate (CNOT)
                    Some("X".to_string())
                }
                StandardGate::CY => {
                    // Controlled-Y gate
                    Some("Y".to_string())
                }
                StandardGate::CZ => {
                    // Controlled-Z gate
                    Some("Z".to_string())
                }
                StandardGate::DCX => {
                    // Double CNOT gate
                    Some("DCX".to_string())
                }
                StandardGate::ECR => {
                    // Echoed cross-resonance gate
                    Some("ECR".to_string())
                }
                StandardGate::Swap => {
                    // Swap gate
                    None
                }
                StandardGate::ISwap => {
                    // i-Swap gate
                    None
                }
                StandardGate::CPhase => {
                    // Controlled phase gate

                    Some(format!("P({})", instruction_param))
                }
                StandardGate::CRX => {
                    // Controlled rotation about X
                    Some(format!("RX({})", instruction_param))
                }
                StandardGate::CRY => {
                    // Controlled rotation about Y
                    Some(format!("RY({})", instruction_param))
                }
                StandardGate::CRZ => {
                    // Controlled rotation about Z
                    Some(format!("RZ({})", instruction_param))
                }
                StandardGate::CS => {
                    // Controlled S gate
                    Some("S".to_string())
                }
                StandardGate::CSdg => {
                    // Controlled S dagger gate
                    Some("S†".to_string())
                }
                StandardGate::CSX => {
                    // Controlled square root of X gate
                    Some("√X".to_string())
                }
                StandardGate::CU => {
                    // Controlled U gate (4 parameters)
                    Some(format!("U({})", instruction_param))
                }
                StandardGate::CU1 => {
                    // Controlled U1 gate
                    Some(format!("U1({})", instruction_param))
                }
                StandardGate::CU3 => {
                    // Controlled U3 gate
                    Some(format!("U3({})", instruction_param))
                }
                StandardGate::RXX => {
                    // Two-qubit XX rotation
                    Some(format!("RXX({})", instruction_param))
                }
                StandardGate::RYY => {
                    // Two-qubit YY rotation
                    Some(format!("RYY({})", instruction_param))
                }
                StandardGate::RZZ => {
                    // Two-qubit ZZ rotation
                    Some(format!("RZZ({})", instruction_param))
                }
                StandardGate::RZX => {
                    // Two-qubit ZX rotation
                    Some(format!("RZX({})", instruction_param))
                }
                StandardGate::XXMinusYY => {
                    // XX-YY gate
                    Some(format!("XX-YY({})", instruction_param))
                }
                StandardGate::XXPlusYY => {
                    // XX+YY gate
                    Some(format!("XX+YY({})", instruction_param))
                }
                StandardGate::CCX => {
                    // Toffoli gate (controlled-controlled-X)
                    Some("X".to_string())
                }
                StandardGate::CCZ => {
                    // Controlled-controlled-Z gate
                    Some("Z".to_string())
                }
                StandardGate::CSwap => {
                    // Controlled swap gate (Fredkin gate)
                    None
                }
                StandardGate::RCCX => {
                    // Relative-phase Toffoli gate
                    Some(format!("RX({})", instruction_param))
                }
                StandardGate::C3X => {
                    // 3-controlled X gate (4-qubit controlled X)
                    Some("X".to_string())
                }
                StandardGate::C3SX => {
                    // 3-controlled square root of X gate
                    Some("√X".to_string())
                }
                StandardGate::RC3X => {
                    // Relative-phase 3-controlled X gate
                    Some(format!("RX({})", instruction_param))
                }
            };
            
            // let label = instruction.label();
            // instruction_label = match label {
            //     Some(l) => {
            //         if l != "" {
            //             Some(l)
            //         } else {
            //             instruction_label
            //         }
            //     }
            //     None => instruction_label,
            // };
            let visual_type = match standard_gate {
                StandardGate::GlobalPhase => VisualType::BetweenWire(instruction.params_view().get(0).map_or("".to_string(), |p| format!("{:?}", p))),
                StandardGate::Swap => VisualType::DirectOnWire("X".to_string()),
                StandardGate::CPhase => VisualType::BetweenWire(instruction_label.clone().unwrap()),
                _ => VisualType::Boxed,
            };
            // // handle case for when the control state is different
            // println!("name: {}", instruction_name);
            // println!("label: {:?}", instruction_label);
            // println!("params: {:?}", instruction_param);

            // fix so that enclosed only has a refernce to the original packed instruction

            Enclosed {
                packedinst: &instruction,
                wire: Wire_Type::qubit,
                dag_circ: &self.dag_circ,
                name: instruction_name.to_string(),
                label: instruction_label,
                visual_type: visual_type,
            }
        } 
        else if let Some(standard_instruction) = instruction.op.try_standard_instruction() {
            match standard_instruction {
                StandardInstruction::Measure =>{
                    Enclosed{
                        packedinst: &instruction,
                        wire: Wire_Type::qubit,
                        dag_circ: &self.dag_circ,
                        name: "Measure".to_string(),
                        label: Some("M".to_string()),
                        visual_type: VisualType::Boxed,
                    }
                },
                StandardInstruction::Barrier(x) => {
                    
                    let label = instruction.label();
                    let inst_label = match label{
                        None => barrier_char.to_string(),
                        Some("") => barrier_char.to_string(),
                        Some(label) => label.to_string(),
                    };

                    println!("barrier label: {}", inst_label);

                    Enclosed{
                        packedinst: &instruction,
                        wire: Wire_Type::qubit,
                        dag_circ: &self.dag_circ,
                        name: "Barrier".to_string(),
                        label: Some(inst_label.clone()),
                        visual_type: VisualType::DirectOnWire(barrier_char.to_string()),
                    }
                },
                StandardInstruction::Reset => {
                    Enclosed{
                        packedinst: &instruction,
                        wire: Wire_Type::qubit,
                        dag_circ: &self.dag_circ,
                        name: "Reset".to_string(),
                        label: Some("|0>".to_string()),
                        visual_type: VisualType::DirectOnWire("|0>".to_string()),
                    }
                },
                StandardInstruction::Delay(_unit) => {

                    let unit = format!("{:?}", _unit).to_lowercase();

                    let label = if let Some(dur) = instruction.params_view().get(0){
                        format!("Delay({:?}[{:?}])", dur, unit)
                    } else {
                        "Delay".to_string()
                    };

                    Enclosed{
                        packedinst: &instruction,
                        wire: Wire_Type::qubit,
                        dag_circ: &self.dag_circ,
                        name: "Delay".to_string(),
                        label: Some(label.clone()),
                        visual_type: VisualType::Boxed,
                    }
                }
            }
        } 
        else {
            // print all operation details
            println!("Unsupported operation details:");
            //get operator discriminant
            // let PackedOperation(discriminant) = instruction.op;
            // println!("{064b}", discriminant);
            println!("Name: {}", instruction.op.name());
            println!("Parameters: {:?}", instruction.params_view());
            println!("Qubits: {:?}", self.dag_circ.qargs_interner().get(instruction.qubits));
            println!("Clbits: {:?}", self.dag_circ.cargs_interner().get(instruction.clbits));
            panic!("Unsupported operation: {}", instruction.op.name());

        }
    }

    pub fn new(dag_circ: DAGCircuit) -> Self {

        //number of qubits in dag_circuit
        let qubit = dag_circ.num_qubits();

        circuit_rep {
            q_wires: vec!(wire::new(Wire_Type::qubit); qubit as usize),
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

    pub fn fix_len(&mut self) {
        let mut num = 0;
        for wire in self.q_wires.iter_mut() {
            let length = wire.get_len();
            if length > num {
                num = length;
            }
        }

        for wire in self.q_wires.iter_mut() {
            let length = wire.get_len();
            wire.fix_len(num - length);
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
        self.fix_len();
    }

    pub fn build_layer(&mut self, layer: Vec<&PackedInstruction>){
        for instruction in layer{
            let components = {
                let enclosed = self.from_instruction(instruction);
                enclosed.component_dict()
            };
            println!("components: {:?}", components);
            for (idx, component) in components{
                self.q_wires[idx as usize].add_wire_component(&component);
            }
        }
        // normalise all wire widths
        // self.fix_len();
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
            // println!("Layer {}", id);
            self.build_layer(packedin_layer.clone());
        }
        
    }
}


pub fn circuit_draw(dag_circ: &DAGCircuit) {

    let mut output = String::new();

    // Create a circuit representation
    let mut circuit_rep = circuit_rep::new(dag_circ.clone());
    circuit_rep.set_qubit_name();
    circuit_rep.build_layers();
    output.push_str(&circuit_rep.circuit_string());
    // Print the circuit representation
    println!("{}", output);
}
