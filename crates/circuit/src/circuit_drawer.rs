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
pub const CONNECTING_WIRE:char = '│';
pub const CROSSED_WIRE:char = '┼';

#[derive(Clone, Debug, Copy)]
pub struct CircuitRep {
    q_wires: Vec::<wire>,
    dag_circ: DAGCircuit
}

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
pub struct Drawable<'a>{
    packedinst: &'a PackedInstruction,
    dag_circ: &'a DAGCircuit,
    wire: WireType,
    name: String,
    label: Option<String>,
    visual_type: VisualType,
}
// pub enum DrawElementType<'a>{
//     Controlled(&'a PackedInstruction),
//     Multi(&'a PackedInstruction),
//     Single(&'a PackedInstruction),
//     Custom(&'a Drawable<'a>)
// }

// struct Visualizer<'a>{
//     visualization_matrix: VisualizationMatrix<'a>,
//     circuit_rep: CircuitRep
// }

#[derive(Clone, Debug)]
struct VisualizationMatrix<'a>{
    visualization_layers: Vec<VisualizationLayer<'a>>,
    packedinst_layers: Vec<Vec<&'a PackedInstruction>>,
    circuit_rep: CircuitRep
}

#[derive(Clone, Debug)]
struct VisualizationLayer<'a>{
    elements: Vec<Option<VisualizationElement<'a>>>,
    drawables: Vec<Drawable<'a>>,
    width: u32,
    // circuit_rep: &'a CircuitRep
    //parent: &'a VisualizationMatrix<'a>
}

#[derive(Clone, Debug)]
pub struct VisualizationElement<'a>{
    layer_element_index: usize,
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

    pub fn new(circuit_rep: CircuitRep, packedinst_layers: Vec<Vec<&'a PackedInstruction>>) -> Self {
        let mut temp: Vec<VisualizationLayer> = Vec::with_capacity(packedinst_layers.len());
        for layer in packedinst_layers.iter() {
            let vis_layer = VisualizationLayer::new(&circuit_rep, layer.to_vec());
            temp.push(vis_layer);
        }

        VisualizationMatrix {
            visualization_layers: temp,
            packedinst_layers,
            circuit_rep, // Now we own this, so no borrowing issues
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

    // add wires to the circuit representation and return the full circuit representation
    pub fn draw(&mut self){
        for layer in &self.visualization_layers{
            let wires = layer.draw_layer(&self.circuit_rep);
            for (i, wire) in wires.iter().enumerate(){
                self.circuit_rep.q_wires[i].add_wire_component(wire);
            }
        }

        // self.circuit_rep.fix_len();
        println!("{}", self.circuit_rep.circuit_string());
    }
}

impl<'a> VisualizationLayer<'a>{

    pub fn new(circuit_rep: &'a CircuitRep, packedinst_layer: Vec<&'a PackedInstruction>) -> Self{

        //println!("{}",packedinst_layer.len());
        let mut vis_layer:Vec<Option<VisualizationElement>> = vec![None;circuit_rep.get_indices() as usize];
        let mut drawable_elements = {
            let mut drawable_layer:Vec<Drawable> = Vec::new();
            let mut drawable_ind:usize = 0;
            for &inst in packedinst_layer.iter(){
                let drawable = circuit_rep.from_instruction(inst);
                let indices = drawable.get_wire_indices();
                // println!("drawable indices: {:?}", indices);
                // println!("{:?}", vis_layer);
                for ind in indices{
                    let vis_element = VisualizationElement::new(drawable_ind, &circuit_rep, ind);
                    vis_layer[ind as usize] = Option::Some(vis_element);
                }
                drawable_ind += 1;
            }
            drawable_layer
        };

        let mut visualization_layer = VisualizationLayer{
            elements: vis_layer,
            drawables: drawable_elements,
            width: 0
        };
        visualization_layer.width = visualization_layer.set_width();
        visualization_layer

    }

    pub fn draw_layer(&self, circuit_rep: &'a CircuitRep) -> Vec<wire>{
        let mut wires: Vec<wire> = Vec::new();
        let mut ct: usize = 0;
        for element in self.elements.iter(){

            if let Some(vis_element) = element{
                let wire_component = vis_element.draw(&self);
                wires.push(wire_component);
            } else {
                // if index greater than number of qubits then wiretype is clbit
                if ct >= circuit_rep.dag_circ.num_qubits() as usize{
                    let wire_component = wire::new(WireType::Clbit);
                    wires.push(wire_component);
                } else {
                    let wire_component = wire::new(WireType::Qubit);
                    wires.push(wire_component);
                }
            }
        }
        wires
    }

    pub fn set_width(&mut self) -> u32{
        let mut max_width:u32 = 0;

        for element in self.elements.iter(){
            if let Some(vis_element) = element{
                let length = vis_element.get_length(&self);
                if length > max_width{
                    max_width = length;
                }
            } else {
                if 1 > max_width{
                    max_width = 1;
                }
            }
        }

        max_width
    }

    pub fn get_layer_col(&self) -> Vec<String>{
        let mut layer_col: Vec<String> = Vec::new();
        for element in &self.elements{
            if let Some(vis_element) = element{
                layer_col.push(format!("{}{}",vis_element.get_string(&self), " ".repeat((self.width - vis_element.get_length(&self)) as usize)));
            } else {
                layer_col.push(" ".repeat(self.width as usize));
            }
        }
        layer_col
    }
}

impl<'a> VisualizationElement<'a>{

    pub fn new(layer_element_index: usize, circuit_rep: &'a CircuitRep,ind: u32) -> Self{
        let vis_element = VisualizationElement{
            layer_element_index,
            ind,
            circuit_rep
        };
        return vis_element;
    }

    // pub fn get_string(&self, layer:&VisualizationLayer) -> String{     
    //     layer.drawables[self.layer_element_index].get_name()
    // }

    pub fn get_length(&self, layer:&VisualizationLayer) -> u32{
        self.get_string(layer).len() as u32
    }

    pub fn get_string(&self, layer:&VisualizationLayer) -> String{
        let drawable = &layer.drawables[self.layer_element_index];
        let mut ret: String = " ".to_string();

        let indices = drawable.get_wire_indices();

        if self.ind == indices[0] {
            ret = drawable.get_name();
        }else if indices.contains(&self.ind) {
            ret = format!("{}({})", BULLET,indices[0]);
        }else {
            ret = <&str>::from(&drawable.wire).to_string();
        }

        format!("{}",ret)
    }

    pub fn draw(&self, layer:&VisualizationLayer) -> wire{
        let drawable = &layer.drawables[self.layer_element_index];
        drawable.draw(self.ind)
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

}

pub trait DirectOnWire{

}

#[derive(Clone, Debug, PartialEq)]
enum VisualType{
    Boxed,
    MultiBoxed,
    BetweenWire(String),
    DirectOnWire(String),
    WholeWire
}

impl<'a> Drawable<'a> {
    pub fn new(packedinst: &'a PackedInstruction, dag_circ: &'a DAGCircuit ,wire: WireType, name: String, label: Option<String>, visual_type: VisualType) -> Drawable<'a>{
        Drawable {
            packedinst,
            dag_circ,
            wire,
            name,
            label,
            visual_type,
        }
    }

    pub fn draw(&self, ind: u32) -> wire{
        let mut wire_component = wire::new(self.wire.clone());
        let indices = self.get_wire_indices();
        let indices_min = indices.iter().min().unwrap().to_owned();
        let indices_max = indices.iter().max().unwrap().to_owned();
        if indices.contains(&ind){
            match &self.visual_type {
                VisualType::Boxed => {
                    if &ind == &indices[0] {
                        if let Some(label) = &self.label {
                            let box_len = label.len() + 2; // for the box edges
                            wire_component.top.push_str(&format!("{}{}{}", TOP_LEFT_CON, Q_WIRE.repeat(box_len), TOP_RIGHT_CON));
                            wire_component.mid.push_str(&format!("{}{}{}", LEFT_CON, label, RIGHT_CON));
                            wire_component.bot.push_str(&format!("{}{}{}", BOT_LEFT_CON, Q_WIRE.repeat(box_len), BOT_RIGHT_CON));
                        } else {
                            panic!("Boxed visual has no label");
                        }
                    } else if ind == indices_max {
                        wire_component.top.push_str(&CONNECTING_WIRE.to_string());
                        wire_component.mid.push_str(&BULLET.to_string());
                        wire_component.bot.push_str(" ");
                    }
                    else if ind == indices_min {
                        wire_component.top.push_str(" ");
                        wire_component.mid.push_str(&BULLET.to_string());
                        wire_component.bot.push_str(&CONNECTING_WIRE.to_string());
                    } else {
                        wire_component.top.push_str(&CONNECTING_WIRE.to_string());
                        wire_component.mid.push_str(&CROSSED_WIRE.to_string());
                        wire_component.bot.push_str(&CONNECTING_WIRE.to_string());
                    }
                }
                VisualType::MultiBoxed => {
                    let mid = (indices_min + indices_max) / 2;
                    let index_in_array = indices.iter().position(|&r| r == ind);
                    let index = match index_in_array {
                        Some(i) => i.to_string(),
                        None => " ".to_string(),
                    };
                    if let Some(label) = &self.label{
                        let box_len = label.len(); // for the box edges
                        if ind == indices_min {
                            wire_component.top.push_str(&format!("{} {}{}", TOP_LEFT_CON," ".repeat(box_len) , TOP_RIGHT_CON));
                            wire_component.mid.push_str(&format!("{}{}{}{}", LEFT_CON,index,label, RIGHT_CON));
                            wire_component.bot.push_str(&format!("{} {}{}", CONNECTING_WIRE, " ".repeat(box_len), CONNECTING_WIRE));
                        } else if ind == indices_max {
                            wire_component.top.push_str(&format!("{} {}{}", CONNECTING_WIRE," ".repeat(box_len) ,CONNECTING_WIRE));
                            wire_component.mid.push_str(&format!("{}{}{}{}", LEFT_CON, index ," ".repeat(box_len), RIGHT_CON));
                            wire_component.bot.push_str(&format!("{} {}{}", BOT_LEFT_CON, " ".repeat(box_len), BOT_RIGHT_CON));
                        } else if ind == mid{
                            wire_component.top.push_str(&format!("{} {}{}", CONNECTING_WIRE," ".repeat(box_len) , CONNECTING_WIRE));
                            wire_component.mid.push_str(&format!("{}{}{}{}", LEFT_CON, index, label, RIGHT_CON));
                            wire_component.bot.push_str(&format!("{} {}{}",CONNECTING_WIRE, " ".repeat(box_len), CONNECTING_WIRE));
                        } else {
                            wire_component.top.push_str(&format!("{} {}{}", CONNECTING_WIRE," ".repeat(box_len) , CONNECTING_WIRE));
                            wire_component.mid.push_str(&format!("{}{}{}{}", LEFT_CON, index," ".repeat(box_len), RIGHT_CON));
                            wire_component.bot.push_str(&format!("{} {}{}",CONNECTING_WIRE, " ".repeat(box_len), CONNECTING_WIRE));
                        }
                    } else {
                        panic!("Boxed visual has no label");
                    }    
                }
                VisualType::BetweenWire(label) => {
                    if let Some(inst_label) = &self.label{
                        if ind == indices_min {

                        } else if ind == indices_max {

                        } else if indices.contains(&ind){

                        } else {

                        }
                    } else {
                        panic!("BetweenWire visual has no label");
                    }       
                }
                VisualType::DirectOnWire(label) => {
                    if ind == indices_min {

                    } else if ind == indices_max {

                    } else if indices.contains(&ind){

                    } else {

                    }
                }
                VisualType::WholeWire => {

                }
            }
            wire_component.update_wire_len();
            wire_component
        } else {
            panic!("Draw being called on an index which is not affected by the PackedInstruction");
        }
    }
    // this function returns the indices of wires than an instruction is acting upon.
    // Since rust currently only has StandardInstruction and StandardGates, it always returns
    // the qubit indices first, where in usually the first index is of the control qubit and the rest are of 
    // the target. However once ControlFlows are introduced, this handling needs to be more nuanced.
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

    pub fn get_name(&self) -> String{
        self.name.clone()
    }
}

impl<'a> DrawElement for Drawable<'a>{
    fn get_width(&self) -> u64 {
        let type_ = match &self.visual_type {
            VisualType::Boxed | VisualType::MultiBoxed => {
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

}



impl<'a> CircuitRep {
    pub fn from_instruction(&'a self, instruction: &'a PackedInstruction) -> Drawable<'a>{
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

            // for label debugging
            println!("{}{}",instruction_label.clone().unwrap_or("".to_string()), instruction_name);
            Drawable {
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
                    Drawable{
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

                    Drawable{
                        packedinst: &instruction,
                        wire: WireType::Qubit,
                        dag_circ: &self.dag_circ,
                        name: "Barrier".to_string(),
                        label: Some(inst_label.clone()),
                        visual_type: VisualType::DirectOnWire(BARRIER_CHAR.to_string()),
                    }
                },
                StandardInstruction::Reset => {
                    Drawable{
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

                    Drawable{
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

    //using circuit rep to draw circuit
    let mut circuit_rep = CircuitRep::new(dag.clone());
    let vis_mat2:VisualizationMatrix = VisualizationMatrix::new(circuit_rep, packedinst_layers);
    vis_mat2.print();
    println!("{}", circuit_rep.circuit_string());
    println!("======================");

    // vis_mat.print();
    Ok(())
}
