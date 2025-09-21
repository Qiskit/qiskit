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

use core::panic;
use std::fmt::Debug;
use hashbrown::{HashSet, HashMap};
use crate::dag_circuit::DAGCircuit;
use crate::operations::{Operation, StandardGate, StandardInstruction};
use crate::packed_instruction::PackedInstruction;
use itertools::{Itertools, MinMaxResult};

use pyo3::prelude::*;
use pyo3::{import_exception};


use crate::converters::QuantumCircuitData;
use crate::dag_circuit::NodeType;
use crate::circuit_data::CircuitData;

import_exception!(qiskit.circuit.exceptions, CircuitError);


#[pyfunction(name = "draw")]
pub fn py_drawer(circuit: QuantumCircuitData) -> PyResult<()> {
    draw_circuit(&circuit.data)?;
    Ok(())
}

pub const Q_WIRE: &str = "─";
pub const C_WIRE: char = '═';
pub const TOP_CON: char = '┴';
pub const BOT_CON: char = '┬';
pub const LEFT_CON: char = '┤';
pub const RIGHT_CON: char = '├';
pub const TOP_LEFT_CON: char = '┌';
pub const TOP_RIGHT_CON: char = '┐';
pub const BOT_LEFT_CON: char = '└';
pub const BOT_RIGHT_CON: char = '┘';
pub const BARRIER_CHAR: char = '░';
pub const BULLET:char = '■';


#[derive(Clone, Debug, PartialEq, Eq)]
enum WireType {
    Qubit,
    Clbit,
}

impl From<&WireType> for &str {
    fn from(wire_type: &WireType) -> Self {
        match wire_type {
            WireType::Qubit => "─",
            WireType::Clbit => "═",
        }
    }
}

#[derive(Clone, Debug)]
enum ControlType{
    Open,
    Closed
}

#[derive(Clone, Debug)]
pub struct input{

}
#[derive(Clone, Debug)]
pub struct wire {
    top: String,
    mid: String,
    bot: String,
    type_: WireType,
}

#[derive(Clone, Debug)]
pub struct Enclosed<'a>{
    packedinst: &'a PackedInstruction,
    dag_circ: &'a DAGCircuit,
    wire: WireType,
    name: String,
    label: Option<String>,
    visual_type: VisualType,
}

struct Visualizer<'a>{
    visualization_matrix: VisualizationMatrix<'a>,
    circuit_rep: CircuitRep
}

struct VisualizationMatrix<'a>{
    visualization_layers: Vec<VisualizationLayer<'a>>,
    packedinst_layers: Vec<Vec<&'a PackedInstruction>>,
    circuit_rep: &'a CircuitRep
}

struct VisualizationLayer<'a>{
    elements: Vec<VisualizationElement<'a>>,
    width: u32,
    circuit_rep: &'a CircuitRep
}

#[derive(Clone, Debug)]
pub struct VisualizationElement<'a>{
    element: Enclosed<'a>,
    ascii_string: String,
    ind: u32,
    circuit_rep: &'a CircuitRep
}

// pub struct VisualizationElement<'a>{
//     element: Option<&'a PackedInstruction>,
//     ascii_string: String,
//     ind: u32,
//     circuit_rep: &'a CircuitRep
// }

impl<'a> VisualizationMatrix<'a>{

    pub fn new(circuit_rep: &'a CircuitRep, packedinst_layers: Vec<Vec<&'a PackedInstruction>>) -> Self{
        let mut visualization_layers: Vec<VisualizationLayer> = Vec::with_capacity(packedinst_layers.len());
        for layer in packedinst_layers.iter(){
            let vis_layer = VisualizationLayer::new(circuit_rep, layer.to_vec());
            visualization_layers.push(vis_layer);
        }

        VisualizationMatrix{
            visualization_layers,
            packedinst_layers,
            circuit_rep
        }
    }

    pub fn print(&self){
        let mut output: Vec<String> = Vec::new();
        for layer in &self.visualization_layers{
            let layer_col = layer.get_layer_col();
            if output.is_empty(){
                output = layer_col;
            } else {
                // append strings in ith index to output string in ith index
                for i in 0..output.len(){
                    output[i].push_str(&layer_col[i]);
                }
            }
        }
        for i in output{
            println!("{}", i);
        }
    }
}

impl<'a> VisualizationLayer<'a>{

    pub fn new(circuit_rep: &'a CircuitRep, packedinst_layer: Vec<&'a PackedInstruction>) -> Self{

        let dummy_element: VisualizationElement = VisualizationElement{
            element: Enclosed{
                packedinst: &packedinst_layer[0],
                dag_circ: &circuit_rep.dag_circ,
                wire: WireType::Qubit,
                name: String::new(),
                label: None,
                visual_type: VisualType::Boxed
            },
            ascii_string: String::new(),
            ind: 0,
            circuit_rep
        };


        //println!("{}",packedinst_layer.len());
        let mut vis_layer:Vec<VisualizationElement> = vec![dummy_element;circuit_rep.get_indices() as usize];
        let mut enclosed_elements = {
            let mut enclosed_layer:Vec<Enclosed> = Vec::new();
            for &inst in packedinst_layer.iter(){
                let enclosed = circuit_rep.from_instruction(inst);
                let indices = enclosed.get_wire_indices();
                // println!("enclosed indices: {:?}", indices);
                // println!("{:?}", vis_layer);
                for ind in indices{
                    let vis_element = VisualizationElement::new(enclosed.clone(), circuit_rep, ind);
                    vis_layer[ind as usize] = vis_element;
                }
            }
            enclosed_layer
        };

        let mut visualization_layer = VisualizationLayer{
            elements: vis_layer,
            width: 0,
            circuit_rep
        };
        visualization_layer.width = visualization_layer.set_width();
        visualization_layer

    }

    pub fn get_enclosed(&self, inst: &'a PackedInstruction) -> Enclosed{
        self.circuit_rep.from_instruction(inst)
    }

    pub fn set_width(&mut self) -> u32{
        let mut max_width:u32 = 0;

        for element in self.elements.iter(){
            let temp = element.get_length();
            if temp > max_width {
                max_width = temp;
            }
        }

        max_width
    }

    pub fn get_layer_col(&self) -> Vec<String>{
        let mut layer_col: Vec<String> = Vec::new();
        for element in &self.elements{
            layer_col.push(format!("{}{}",element.get_string(), " ".repeat((self.width - element.get_length()) as usize)));
        }
        layer_col
    }
}



impl<'a> VisualizationElement<'a>{

    pub fn new(element: Enclosed<'a>, circuit_rep: &'a CircuitRep,ind: u32) -> Self{
        let mut vis_element = VisualizationElement{
            element,
            ascii_string: String::new(),
            ind,
            circuit_rep
        };
        vis_element.ascii_string = vis_element.set_string();
        return vis_element;
    }

    pub fn get_string(&self) -> String{
        self.ascii_string.clone()
    }

    pub fn get_length(&self) -> u32{
        self.ascii_string.len() as u32
    }

    pub fn set_string(&mut self) -> String{
        let enclosed = &self.element;
        let mut ret: String = " ".to_string();

        let indices = enclosed.get_wire_indices();

        if self.ind == indices[0] {
            ret = enclosed.get_name();
        }else if indices.contains(&self.ind) {
            ret = format!("{}({})", BULLET,indices[0]);
        }else {
            ret = <&str>::from(&enclosed.wire).to_string();
        }

        format!("{}",ret)
    }
}

impl wire {
    pub fn new(type_: WireType) -> Self {
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
        self.top.push_str(&component.top);
        self.mid.push_str(&component.mid);
        self.bot.push_str(&component.bot);
        self.update_wire_len();
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
    WholeWire
}

impl<'a> Enclosed<'a> {
    pub fn new(packedinst: &'a PackedInstruction, dag_circ: &'a DAGCircuit ,wire: WireType, name: String, label: Option<String>, visual_type: VisualType) -> Enclosed<'a>{
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
        if self.visual_type == VisualType::DirectOnWire(BARRIER_CHAR.to_string()) {
            let mut label = BARRIER_CHAR.to_string();
            if let Some(l) = &self.label{
                label = l.to_string();
            } else {
                label = BARRIER_CHAR.to_string();
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
                BARRIER_CHAR,
                <&str>::from(&self.wire).repeat(right_pad_len)
            );

            let between_wire = format!("{}{}{}",
                " ".repeat(left_pad_len),
                BARRIER_CHAR,
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
                    component.top.push_str(&format!("{}{}{}", TOP_LEFT_CON, "─".repeat(label_len as usize), TOP_RIGHT_CON));
                    component.mid.push_str(&format!("{}{}{}", LEFT_CON, label, RIGHT_CON));
                    component.bot.push_str(&format!("{}{}{}", BOT_LEFT_CON, "─".repeat(label_len as usize), BOT_RIGHT_CON));
                } else {
                    panic!("Boxed visual has no label");
                }
            }
            VisualType::BetweenWire(label) => {
                let label_len = label.len() as u64;
                component.top.push_str(&format!("{}{}{}", TOP_CON, "─".repeat(label_len as usize), TOP_CON));
                component.mid.push_str(&format!(" {} ", label));
                component.bot.push_str(&format!("{}{}{}", BOT_CON, "─".repeat(label_len as usize), BOT_CON));
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

    pub fn get_name(&self) -> String{
        self.name.clone()
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

#[derive(Clone, Debug)]
pub struct CircuitRep {
    q_wires: Vec::<wire>,
    dag_circ: DAGCircuit
}

impl<'a> CircuitRep {
    pub fn from_instruction(&'a self, instruction: &'a PackedInstruction) -> Enclosed<'a>{
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

            let label = instruction.label();
            instruction_label = match label {
                Some(l) => {
                    if l != "" {
                        Some(l.to_string())
                    } else {
                        instruction_label
                    }
                }
                None => instruction_label,
            };
            let visual_type = match standard_gate {
                StandardGate::GlobalPhase => VisualType::BetweenWire(instruction.params_view().get(0).map_or("".to_string(), |p| format!("{:?}", p))),
                StandardGate::Swap => VisualType::DirectOnWire("X".to_string()),
                StandardGate::CPhase => VisualType::BetweenWire(instruction_label.clone().unwrap()),
                _ => VisualType::Boxed,
            };

            Enclosed {
                packedinst: &instruction,
                wire: WireType::Qubit,
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
                        wire: WireType::Qubit,
                        dag_circ: &self.dag_circ,
                        name: "Measure".to_string(),
                        label: Some("M".to_string()),
                        visual_type: VisualType::Boxed,
                    }
                },
                StandardInstruction::Barrier(x) => {

                    let label = instruction.label();
                    let inst_label = match label{
                        None => BARRIER_CHAR.to_string(),
                        Some("") => BARRIER_CHAR.to_string(),
                        Some(label) => label.to_string(),
                    };

                    println!("barrier label: {}", inst_label);

                    Enclosed{
                        packedinst: &instruction,
                        wire: WireType::Qubit,
                        dag_circ: &self.dag_circ,
                        name: "Barrier".to_string(),
                        label: Some(inst_label.clone()),
                        visual_type: VisualType::DirectOnWire(BARRIER_CHAR.to_string()),
                    }
                },
                StandardInstruction::Reset => {
                    Enclosed{
                        packedinst: &instruction,
                        wire: WireType::Qubit,
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
                        wire: WireType::Qubit,
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

        CircuitRep {
            q_wires: vec!(wire::new(WireType::Qubit); qubit as usize),
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

    pub fn get_indices(&self) -> u32{
        let total_qubits = self.dag_circ.num_qubits() as u32;
        let total_clbits = self.dag_circ.num_clbits() as u32;
        total_qubits + total_clbits
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
}

/// Calculate the range (inclusive) of the given instruction qubits/clbits over the wire indices.
/// The assumption is that clbits always appear after the qubits in the visualization, hence the clbit indices
/// are offset by the number of instruction qubits when calculating the range.
 fn get_instruction_range(dag: &DAGCircuit, instruction: &PackedInstruction) -> (usize, usize) {
    let node_qubits = dag.get_qargs(instruction.qubits);
    let node_clbits = dag.get_cargs(instruction.clbits);

    let indices = node_qubits.iter().map(|q| q.index()).chain(
        node_clbits.iter().map(|c| c.index() + dag.num_qubits()));

    match indices.minmax() {
        MinMaxResult::MinMax(min, max) => (min, max),
        MinMaxResult::OneElement(idx) => (idx, idx),
        MinMaxResult::NoElements => panic!("Encountered an instruction without qubits and clbits")
    }
}

/// Return a list of PackedInstruction layers such that each layer contain instruction
/// whose qubits/clbits indices do not overlap. The instructions are packed into each layer
/// as long as there is no qubit/clbit overlap
fn build_layers(dag: &DAGCircuit) -> Vec<Vec<&PackedInstruction>> {
    let mut layers:Vec<Vec<&PackedInstruction>> = Vec::new();
    let mut current_layer: Option<&mut Vec::<&PackedInstruction>> = None;
    let mut used_wires = HashSet::<usize>::new();

    for layer in dag.multigraph_layers() {
        for node_index in layer.into_iter().sorted() {

            if let NodeType::Operation(instruction_to_insert) = &dag.dag()[node_index] {
                let (node_min, node_max) = get_instruction_range(dag, instruction_to_insert);

                // Check for instruction range overlap
                if (node_min..=node_max).any(|idx| used_wires.contains(&idx)) {
                    current_layer = None; // Indication for starting a new layer
                    used_wires.clear();
                }
                used_wires.extend(node_min..=node_max);

                if current_layer.is_none() {
                    layers.push(Vec::new());
                    current_layer = layers.last_mut();
                }

                current_layer.as_mut().unwrap().push(instruction_to_insert);
            }
        }
    }

    layers
}

pub fn draw_circuit(circuit: &CircuitData) -> PyResult<()> {
    let dag = DAGCircuit::from_circuit_data(circuit, false, None, None, None, None)?;

    // let mut circuit_rep = CircuitRep::new(dag.clone());
    let packedinst_layers = build_layers(&dag);
    for (i, layer) in packedinst_layers.iter().enumerate() {
        println!("==== LAYER {} ====", i);
        for inst in layer {
            println!("{:?}", inst.op.name());
        }
    }

    // circuit_rep2.set_qubit_name();
    // let vis_mat:VisualizationMatrix = VisualizationMatrix::new(&circuit_rep, packedinst_layers);

    println!("======================");
    // vis_mat.print();
    Ok(())
}
