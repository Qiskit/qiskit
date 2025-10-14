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
use std::boxed;
use std::fmt::Debug;
use std::io::ErrorKind;
use hashbrown::HashSet;
use crate::bit::{ShareableClbit, ShareableQubit};
use crate::dag_circuit::{DAGCircuit};
use crate::operations::{Operation, OperationRef, StandardGate, StandardInstruction};
use crate::packed_instruction::{PackedInstruction,PackedOperation};
use itertools::{Itertools, MinMaxResult};
use std::ops::Index;

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

#[derive(Clone, Debug)]
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
    pub visualization_layers: Vec<VisualizationLayer<'a>>,
    packedinst_layers: Vec<Vec<&'a PackedInstruction>>,
    dag_circ: &'a DAGCircuit
}

#[derive(Clone, Debug)]
struct VisualizationLayer<'a>{
    pub elements: Vec<Option<VisualizationElement<'a>>>,
    pub drawables: Vec<Drawable<'a>>,
    width: u32,
    // circuit_rep: &'a CircuitRep
    //parent: &'a VisualizationMatrix<'a>
}

#[derive(Clone, Debug)]
pub struct VisualizationElement<'a>{
    layer_element_index: usize,
    ind: u32,
    dag_circ: &'a DAGCircuit
}

// pub struct VisualizationElement<'a>{
//     element: Option<&'a PackedInstruction>,
//     ascii_string: String,
//     ind: u32,
//     circuit_rep: &'a CircuitRep
// }

impl<'a> VisualizationMatrix<'a>{

    pub fn new(dag_circ: &'a DAGCircuit, packedinst_layers: Vec<Vec<&'a PackedInstruction>>) -> Self {
        // println!("Creating VisualizationMatrix...");
        // let mut temp: Vec<VisualizationLayer> = Vec::with_capacity(packedinst_layers.len());
        // for layer in packedinst_layers.iter() {
        //     let vis_layer = VisualizationLayer::new(dag_circ, layer.to_vec());
        //     temp.push(vis_layer);
        // }
        // println!("Visualization layers created: {:?}", temp);

        // VisualizationMatrix {
        //     visualization_layers: temp,
        //     packedinst_layers,
        //     dag_circ, // Now we own this, so no borrowing issues
        // }
        println!("Creating VisualizationMatrix...");
        let mut temp: Vec<VisualizationLayer> = Vec::with_capacity(packedinst_layers.len());
        for layer in packedinst_layers.iter() {
            let vis_layer = VisualizationLayer::new(dag_circ, layer.clone());
            temp.push(vis_layer);
        }
        VisualizationMatrix {
            visualization_layers: temp,
            packedinst_layers: packedinst_layers,
            dag_circ, // Now we own this, so no borrowing issues
        }
    }

    pub fn print(&self){
        println!("Printing VisualizationMatrix...");
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
    pub fn draw(&self, circuit_rep: &mut CircuitRep){
        println!("Drawing VisualizationMatrix...");
        for layer in &self.visualization_layers{
            let wires = layer.draw_layer(circuit_rep);
            println!("wires {:?}", wires);
            for (i, wire) in wires.iter().enumerate(){
                circuit_rep.q_wires[i].add_wire_component(wire);
            }
        }

        // self.circuit_rep.fix_len();
        println!("{}", circuit_rep.circuit_string());
    }
}

impl<'a> VisualizationLayer<'a>{

    pub fn new(dag_circ: &'a DAGCircuit, packedinst_layer: Vec<&'a PackedInstruction>) -> Self{
        println!("Creating VisualizationLayer...");
        //println!("{}",packedinst_layer.len());
        let mut vis_layer:Vec<Option<VisualizationElement>> = vec![None;get_indices(dag_circ) as usize];
        let mut drawable_elements = {
            let mut drawable_layer:Vec<Drawable> = Vec::new();
            let mut drawable_ind:usize = 0;
            for &inst in packedinst_layer.iter(){
                let drawable = from_instruction(dag_circ,inst);
                let indices = drawable.get_wire_indices();
                // println!("drawable indices: {:?}", indices);
                // println!("{:?}", vis_layer);
                for ind in indices{
                    let vis_element = VisualizationElement::new(drawable_ind, dag_circ, ind);
                    vis_layer[ind as usize] = Option::Some(vis_element);
                }
                drawable_ind += 1;
                drawable_layer.push(drawable);
            }
            drawable_layer
        };

        println!("Drawable elements: {:?}", drawable_elements);
        // println!("Vis layer: {:?}", vis_layer);

        let mut visualization_layer = VisualizationLayer{
            elements: vis_layer,
            drawables: drawable_elements,
            width: 0
        };
        visualization_layer.width = visualization_layer.set_width();
        visualization_layer

    }

    pub fn draw_layer(&self, circuit_rep: &'a CircuitRep) -> Vec<wire>{
        println!("Drawing VisualizationLayer...");
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
            ct += 1;
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

    pub fn new(layer_element_index: usize, dag_circ: &'a DAGCircuit,ind: u32) -> Self{
        let vis_element = VisualizationElement{
            layer_element_index,
            ind,
            dag_circ
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

// pub trait DirectOnWire{

// }

#[derive(Clone, Debug, PartialEq)]
enum VisualType{
    Boxed,
    MultiBoxed,
    BetweenWire(String),
    DirectOnWire(String),
    AllWires(String)
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
                            wire_component.top.push_str(&format!("{}{}", " "," ".repeat(inst_label.len())));
                            wire_component.mid.push_str(&format!("{}{}",label,<&str>::from(&self.wire)));
                            wire_component.bot.push_str(&format!("{}{}", CONNECTING_WIRE, inst_label));
                        } else if ind == indices_max {
                            wire_component.top.push_str(&format!("{}{}", CONNECTING_WIRE," ".repeat(inst_label.len())));
                            wire_component.mid.push_str(&format!("{}{}", label, <&str>::from(&self.wire)));
                            wire_component.bot.push_str(&format!("{}{}", " "  ," ".repeat(inst_label.len())));
                        } else if indices.contains(&ind){
                            wire_component.top.push_str(&format!("{}{}", CONNECTING_WIRE," ".repeat(inst_label.len())));
                            wire_component.mid.push_str(&format!("{}{}", label, <&str>::from(&self.wire)));
                            wire_component.bot.push_str(&format!("{}{}", CONNECTING_WIRE, " ".repeat(inst_label.len())));
                        } else {
                            wire_component.top.push_str(&format!("{}{}", CONNECTING_WIRE," ".repeat(inst_label.len())));
                            wire_component.mid.push_str(&format!("{}{}", CROSSED_WIRE, " ".repeat(inst_label.len())));
                            wire_component.bot.push_str(&format!("{}{}", CONNECTING_WIRE  ," ".repeat(inst_label.len())));
                        }
                    } else {
                        panic!("BetweenWire visual has no label");
                    }
                }
                VisualType::DirectOnWire(label) => {
                    let inst_label = match &self.label {
                        Some(l) => l,
                        None => label,
                    };
                    if ind == indices_min {
                        wire_component.top.push_str(&format!("{}{}", " "," ".repeat(inst_label.len())));
                        wire_component.mid.push_str(&format!("{}{}",label,<&str>::from(&self.wire).repeat(inst_label.len())));
                        wire_component.bot.push_str(&format!("{}{}", CONNECTING_WIRE, inst_label));
                    } else if ind == indices_max {
                        wire_component.top.push_str(&format!("{}{}", CONNECTING_WIRE," ".repeat(inst_label.len())));
                        wire_component.mid.push_str(&format!("{}{}", label, <&str>::from(&self.wire).repeat(inst_label.len())));
                        wire_component.bot.push_str(&format!("{}{}", " "  ," ".repeat(inst_label.len())));
                    } else if indices.contains(&ind){
                        wire_component.top.push_str(&format!("{}{}", CONNECTING_WIRE," ".repeat(inst_label.len())));
                        wire_component.mid.push_str(&format!("{}{}", label, <&str>::from(&self.wire).repeat(inst_label.len())));
                        wire_component.bot.push_str(&format!("{}{}", CONNECTING_WIRE, " ".repeat(inst_label.len())));
                    } else {
                        wire_component.top.push_str(&format!("{}{}", CONNECTING_WIRE," ".repeat(inst_label.len())));
                        wire_component.mid.push_str(&format!("{}{}", CROSSED_WIRE, " ".repeat(inst_label.len())));
                        wire_component.bot.push_str(&format!("{}{}", CONNECTING_WIRE  ," ".repeat(inst_label.len())));
                    }
                }
                VisualType::AllWires(label) => {
                    let mut inst_label = match &self.label {
                        Some(l) => l,
                        None => label,
                    };
                    if ind == 0 {
                        wire_component.top.push_str(&format!("{}{}", inst_label," ".to_string()));
                        wire_component.mid.push_str(&format!("{}{}",label,<&str>::from(&self.wire).repeat(inst_label.len())));
                        wire_component.bot.push_str(&format!("{}{}", label, " ".to_string().repeat(inst_label.len())));
                    } else if indices.contains(&ind) {
                        wire_component.top.push_str(&format!("{}{}", label," ".repeat(inst_label.len())));
                        wire_component.mid.push_str(&format!("{}{}", label,<&str>::from(&self.wire).repeat(inst_label.len())));
                        wire_component.bot.push_str(&format!("{}{}", label," ".repeat(inst_label.len())));
                    }
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
            VisualType::AllWires(label) => {
                0 as u64// just a single character for the whole wire
            }
        };

        return type_;
    }

}



impl<'a> CircuitRep {

    pub fn new(dag_circ: DAGCircuit) -> Self {
        CircuitRep {
            q_wires: vec!(wire::new(WireType::Qubit); dag_circ.num_qubits()),
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

pub fn get_label(instruction: &PackedInstruction) -> Option<String>{
    let label = instruction.label();
    let instruction_param =format!("{:?}",instruction.params_view());
    let instruction_label = match label {
        Some(l) => {
            Some(l.to_string())
        }
        None => {
            if let Some(standard_gate) = instruction.op.try_standard_gate() {
                match standard_gate {
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
                }
            } else {
                Some(instruction.op.name().to_string())
            }
        },
    };
    instruction_label
}

pub fn from_instruction<'a>(dag_circ: &'a DAGCircuit, instruction: &'a PackedInstruction) -> Drawable<'a>{
    // println!("{:?}", instruction.label());
    let instruction_name = instruction.op.name();
    if let Some(standard_gate) = instruction.op.try_standard_gate() {
        let mut instruction_label: Option<String> = get_label(instruction);

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

        // TO DO:🥀
        //
        // assign the correct visual type to all gates and instructions
        //
        //

        let visual_type = match standard_gate {
            StandardGate::GlobalPhase => VisualType::BetweenWire(instruction.params_view().get(0).map_or("".to_string(), |p| format!("{:?}", p))),
            StandardGate::Swap | StandardGate::CSwap => VisualType::DirectOnWire("X".to_string()),
            StandardGate::CPhase => VisualType::BetweenWire(instruction_label.clone().unwrap()),
            _ => VisualType::Boxed,
        };

        // for label debugging
        // println!("{}{}",instruction_label.clone().unwrap_or("".to_string()), instruction_name);
        Drawable {
            packedinst: &instruction,
            wire: WireType::Qubit,
            dag_circ: &dag_circ,
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
                    dag_circ: &dag_circ,
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
                    dag_circ: &dag_circ,
                    name: "Barrier".to_string(),
                    label: Some(inst_label.clone()),
                    visual_type: VisualType::AllWires(BARRIER_CHAR.to_string()),
                }
            },
            StandardInstruction::Reset => {
                Drawable{
                    packedinst: &instruction,
                    wire: WireType::Qubit,
                    dag_circ: &dag_circ,
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
                    dag_circ: &dag_circ,
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
        println!("Qubits: {:?}", dag_circ.qargs_interner().get(instruction.qubits));
        println!("Clbits: {:?}", dag_circ.cargs_interner().get(instruction.clbits));
        panic!("Unsupported operation: {}", instruction.op.name());

    }
}

pub fn get_indices(dag_circ: &DAGCircuit) -> u32{
    let total_qubits = dag_circ.num_qubits() as u32;
    let total_clbits = dag_circ.num_clbits() as u32;
    total_qubits + total_clbits
}



#[derive(Clone)]
enum WireInput<'a> {
    Qubit(&'a ShareableQubit),
    Clbit(&'a ShareableClbit),
}

impl<'a> Debug for WireInput<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            WireInput::Qubit(&ref qubit) => "Qubit",
            WireInput::Clbit(&ref clbit) => "Clbit",
        };

        write!(f, "{}", name)
    }
}

impl<'a> Debug for Boxed<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Boxed::Single(label) => write!(f, "Boxed({})", label),
            Boxed::Multi(inst) => write!(f, "MultiBox({})", inst.op.name()),
        }
    }
}

/// Input Wires. 
/// The Option<String> is for optional labels and can be used when the registers have different names. This allows us to use
/// the same enum for both Input and VerticalLine types.
#[derive(Clone, Debug)]
enum InputType{
    Qubit(Option<String>),
    Clbit(Option<String>)
}

/// Enum for representing elements that can appear directly on a wire.
#[derive(Clone, Debug)]
enum ElementOnWire{
    Control,
    Swap
}

/// Enum for representing elements that appear directly on a wire and how they're connected.
#[derive(Clone, Debug)]
enum OnWire{
    Top(ElementOnWire),
    Mid(ElementOnWire),
    Bot(ElementOnWire),
    Barrier,
    Reset,
}

/// Enum for representing elements that appear in a boxed operation.
#[derive(Clone)]
enum Boxed<'a>{
    Single(String),
    // Multi(MultiBoxElement)
    Multi(&'a PackedInstruction),
}

/// Enum for  representing the elements stored in a visualization matrix. The elements
/// do not directly implement visualization capabilities, but rather carry enough information
/// to enable visualization later on by the actual drawer.
#[derive(Default, Clone, Debug)]
enum VisualizationElement2<'a>{
    #[default]
    Empty, // Marker for no element
    Boxed(Boxed<'a>),
    Input(WireInput<'a>), // TODO: should be enum for qubit/clbits
    DirectOnWire(OnWire), // TODO: should be an enum for the control symbols, e.g. closed, open
    VerticalLine(InputType), // TODO: should be an enum for the various types, e.g. single, double
    Operation, // TODO: should be an enum for the various fine-grained types: standard gates, instruction, etc..
}

/// A representation of a single column (called here a layer) of a visualization matrix
#[derive(Clone, Debug)]
struct VisualizationLayer2<'a>(Vec<VisualizationElement2<'a>>);

impl<'a> VisualizationLayer2<'a> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn add_input(&mut self, input: WireInput<'a>, idx: usize) {
        self.0[idx] = VisualizationElement2::Input(input);
    }

    /// Adds the required visualization elements to represent the given instruction
    fn add_instruction(&mut self, inst: &'a PackedInstruction, dag: &DAGCircuit) {
        match inst.op.view() {
            OperationRef::StandardGate(_gate) => self.add_standard(inst, dag),
            OperationRef::StandardInstruction(_instruction) => self.add_standard(inst, dag),
            _ => unimplemented!("{}", format!("Visualization is not implemented for instruction of type {:?}", inst.op)),
        }
    }


    fn get_controls(&self, inst: &PackedInstruction, dag: &DAGCircuit) -> Vec<usize> {
        let gate_has_control = vec![StandardGate::CX, StandardGate::CCX, StandardGate::CY, StandardGate::CZ,
            StandardGate::CRX, StandardGate::CRY, StandardGate::CRZ,
            StandardGate::CPhase, StandardGate::CS, StandardGate::CSdg,
            StandardGate::CSX, StandardGate::CU, StandardGate::CU1,
            StandardGate::CU3, StandardGate::CH, StandardGate::C3SX, StandardGate::C3X,
            StandardGate::RC3X, StandardGate::RCCX];
    
        let inst_has_controls = vec![StandardInstruction::Measure];

        let mut controls = vec![];

        let std_op = inst.op.view();

        match std_op {
            OperationRef::StandardGate(gate) => {
                if gate_has_control.contains(&gate) {
                    let qargs = dag.get_qargs(inst.qubits).into_iter().map(|q| q.index()).collect_vec();
                    let target = qargs.last().unwrap();
                    let (minima,maxima) = get_instruction_range(dag, inst);

                    for control in minima..=maxima {
                        if control != *target && qargs.contains(&control) {
                            controls.push(control);
                        }
                    }
                controls
                } else {
                    vec![]
                }
            }
            OperationRef::StandardInstruction(instruction) => {
                if inst_has_controls.contains(&instruction) {
                    let qargs = dag.get_qargs(inst.qubits).into_iter().map(|q| q.index()).collect_vec();
                    let target = qargs.last().unwrap();
                    let (minima,maxima) = get_instruction_range(dag, inst);

                    for control in minima..=maxima {
                        if control != *target && qargs.contains(&control) {
                            controls.push(control);
                        }
                    }
                controls
                } else {
                    vec![]
                }
            },
            _ => vec![]
        }
    }

    fn add_controls(&mut self, controls: &Vec<usize>, range: (usize, usize)) {
        for control in controls {
            if *control == range.0 {
                self.0[*control] = VisualizationElement2::DirectOnWire(OnWire::Top(ElementOnWire::Control));
            } else if *control == range.1 {
                self.0[*control] = VisualizationElement2::DirectOnWire(OnWire::Bot(ElementOnWire::Control));
            } else {
                self.0[*control] = VisualizationElement2::DirectOnWire(OnWire::Mid(ElementOnWire::Control));
            }
        }
    }

    fn get_boxed_indices(&mut self, inst: & PackedInstruction, dag: &DAGCircuit) -> Vec<usize>{
        let single_box = vec![StandardGate::H, StandardGate::X, StandardGate::Y, StandardGate::Z,
            StandardGate::RX, StandardGate::RY, StandardGate::RZ, StandardGate::U, StandardGate::U1,
            StandardGate::U2, StandardGate::U3, StandardGate::S, StandardGate::Sdg,
            StandardGate::T, StandardGate::Tdg, StandardGate::Phase, StandardGate::R, StandardGate::SX,
            StandardGate::SXdg, StandardGate::CCX, StandardGate::CCZ, StandardGate::CX, StandardGate::CY,
            StandardGate::CZ, StandardGate::CPhase, StandardGate::CRX, StandardGate::CRY,
            StandardGate::CRZ, StandardGate::CS, StandardGate::CSdg, StandardGate::CSX,
            StandardGate::CU, StandardGate::CU1, StandardGate::CU3, StandardGate::C3X, StandardGate::C3SX, StandardGate::RC3X];
        
        let single_box_instrusctions = vec![StandardInstruction::Measure, StandardInstruction::Reset];
        
        let multi_box = vec![StandardGate::ISwap, StandardGate::DCX,
            StandardGate::ECR, StandardGate::RXX, StandardGate::RYY, StandardGate::RZZ,
            StandardGate::RZX, StandardGate::XXMinusYY, StandardGate::XXPlusYY];

        let direct_on_wire = vec![StandardGate::Swap, StandardGate::CSwap];

        let special_cases = vec![ StandardGate::GlobalPhase, StandardGate::CPhase];

        let qargs = dag.get_qargs(inst.qubits).into_iter().map(|q| q.index()).collect_vec();
        let target = qargs.last().unwrap();
        let range = get_instruction_range(dag, inst);

        if let Some(std_gate) = inst.op.try_standard_gate(){
            if single_box.contains(&std_gate){
                vec![*target]
            } else if multi_box.contains(&std_gate){
                (range.0..=range.1).collect()
            } else {
                vec![]
            }
        } else if let Some(std_instruction) = inst.op.try_standard_instruction(){
            if single_box_instrusctions.contains(&std_instruction){
                vec![*target]
            } else if let StandardInstruction::Barrier(_) = std_instruction {
                (range.0..=range.1).collect()
            } else {
                vec![]
            }

        // Handle special cases and direct on wire 
        } else {
            vec![]
        }
        
    }

    fn add_boxed(&mut self, inst: &'a PackedInstruction, dag: &DAGCircuit, boxed_indices: &Vec<usize>) {
        // The case of delay needs to be handled where it is multiple single qubit gates but shown as a box
        if boxed_indices.len() == 1 {
            let qargs = dag.get_qargs(inst.qubits).into_iter().map(|q| q.index()).collect_vec();
            let target = qargs.last().unwrap();
            let label = get_label(inst).unwrap_or_else(|| inst.op.name().to_string());
            self.0[*target] = VisualizationElement2::Boxed(Boxed::Single(label));
        } else if boxed_indices.len() > 1 {
            for idx in boxed_indices {
                self.0[*idx] = VisualizationElement2::Boxed(Boxed::Multi(inst));
            }
        }
    }

    fn add_vertical_lines(&mut self, inst: &'a PackedInstruction, dag: &DAGCircuit, vertical_lines: &Vec<usize>) {
        let double_lines = vec![StandardInstruction::Measure];
        let input_type: InputType = if let Some(std_instruction) = inst.op.try_standard_instruction(){
            if double_lines.contains(&std_instruction){
                InputType::Qubit(Some("||".to_string()))
            } else {
                InputType::Qubit(None)
            }
        } else {
            InputType::Qubit(None)
        };
        for vline in vertical_lines {
            self.0[*vline] = VisualizationElement2::VerticalLine(input_type.clone() );
        }
    }

    // function to add standard gates and instructions
    fn add_standard(&mut self, inst: &'a PackedInstruction, dag: &DAGCircuit) {
        let (minima,maxima) = get_instruction_range(dag, inst);
        let controls = self.get_controls(inst, dag);
        let boxed_elements = self.get_boxed_indices(inst, dag);
        // println!("Adding gate {:?}\n  on qubits {:?} \n with controls {:?} \n and boxed elements {:?}\n ================", inst.op.name(), qargs, controls, boxed_elements);
        let vert_lines = (minima..=maxima)
            .filter(|idx| !controls.contains(idx) && !boxed_elements.contains(idx))
            .collect_vec();
        self.add_controls(&controls, (minima, maxima));
        self.add_boxed(inst, dag, &boxed_elements);
        self.add_vertical_lines(inst, dag, &vert_lines);
    }
}

impl<'a> Index<usize> for VisualizationLayer2<'a> {
    type Output = VisualizationElement2<'a>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

/// A Plain, logical 2D representation of a circuit.
///
/// A dense representation of the circuit of size N * (M + 1), where the first
/// layer(column) represents the qubits and clbits inputs in the circuits, and
/// M is the number of operation layers.
#[derive(Debug)]
struct VisualizationMatrix2<'a> {
    layers: Vec<VisualizationLayer2<'a>>,
}


impl<'a> VisualizationMatrix2<'a> {

    fn from_circuit(circuit: &'a CircuitData, dag: &'a DAGCircuit) -> PyResult<Self> {
        let inst_layers = build_layers(&dag);

        let num_wires = circuit.num_qubits() + circuit.num_clbits();
        let mut layers = vec![VisualizationLayer2(vec![VisualizationElement2::default(); num_wires]); inst_layers.len() + 1]; // Add 1 to account for the inputs layer

        // TODO: add the qubit/clbit inputs here to layer #0

        let input_layer = layers.first_mut().unwrap();
        let mut input_idx = 0;
        for qubit in circuit.qubits().objects() {
            input_layer.add_input(WireInput::Qubit(qubit), input_idx);
            input_idx += 1;
        }

        for clbit in circuit.clbits().objects() {
            input_layer.add_input(WireInput::Clbit(clbit), input_idx);
            input_idx += 1;
        }

        for (i, layer) in inst_layers.iter().enumerate() {
            for inst in layer {
                layers[i + 1].add_instruction(inst, &dag);
            }
        }

        Ok(VisualizationMatrix2{
            layers,
        })
    }

    fn num_wires(&self) -> usize {
        self.layers.first().map_or(0, |layer| layer.len())
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl<'a> Index<usize> for VisualizationMatrix2<'a> {
    type Output = VisualizationLayer2<'a>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.layers[index]
    }
}

struct TextDrawer{

}

impl TextDrawer {
    fn new() -> Self {
        TextDrawer{

        }
    }

    fn create_vismat(circuit: &CircuitData, dag: &DAGCircuit) -> PyResult<VisualizationMatrix2>{
        VisualizationMatrix2::from_circuit(circuit, dag)
    }

    fn get_element_width(element: &VisualizationElement2) -> u64 {
        
    }
}

pub fn draw_circuit(circuit: &CircuitData) -> PyResult<()> {

    let dag = DAGCircuit::from_circuit_data(circuit, false, None, None, None, None)?;

    let vis_mat2 = VisualizationMatrix2::from_circuit(circuit, &dag)?;

    println!("======================");

    println!("num wires {}, num layers {}", vis_mat2.num_wires(), vis_mat2.num_layers());

    for i in 0..vis_mat2.num_wires() {
        for j in 0..vis_mat2.num_layers() {
            print!("{:^30}", format!("{:?}", vis_mat2[j][i]));
        }
        println!("");
    }
    // circuit_rep2.set_qubit_name();
    // let vis_mat:VisualizationMatrix = VisualizationMatrix::new(&circuit_rep, packedinst_layers);

    println!("======================");

    // for i in 0..vis_mat2.num_wires() {
    //     if let VisualizationElement2::Input(wire_input) = &vis_mat2[0][i] {
    //         match wire_input {
    //             WireInput::Qubit(qubit) => println!("QUBIT: {:?}", qubit),
    //             WireInput::Clbit(clbit) => println!("CLBIT: {:?}", clbit),
    //         }
    //     }
    // }
    // //using circuit rep to draw circuit
    // let mut circuit_rep = CircuitRep::new(dag.clone());
    // // println!("circuit_rep {:?}", circuit_rep);
    // let vis_mat2:VisualizationMatrix = VisualizationMatrix::new(&dag, packedinst_layers);

    // return Ok(());
    // circuit_rep.set_qubit_name();
    // vis_mat2.draw(&mut circuit_rep);
    // // vis_mat2.print();
    // //println!("{}", &circuit_rep.circuit_string());
    // println!("======================");

    // println!("Drawing circuit...");
    // // let mut circuit_rep2 = CircuitRep::new(dag);
    // // vis_mat2.draw(&mut circuit_rep2);

    // println!("finished circuit");
    // // vis_mat.print();
    Ok(())
}
