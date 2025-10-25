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
use std::thread::current;
use hashbrown::HashSet;
use crate::bit::{ShareableClbit, ShareableQubit};
use crate::dag_circuit::{DAGCircuit};
use crate::operations::{Operation, OperationRef, StandardGate, StandardInstruction};
use crate::packed_instruction::{self, PackedInstruction, PackedOperation};
use itertools::{Itertools, MinMaxResult};
use std::ops::Index;

use pyo3::prelude::*;
use pyo3::{import_exception};


use crate::converters::QuantumCircuitData;
use crate::dag_circuit::NodeType;
use crate::circuit_data::CircuitData;

import_exception!(qiskit.circuit.exceptions, CircuitError);

// [ qubit indice]
// int: n 

#[pyfunction(name = "draw")]
pub fn py_drawer(circuit: QuantumCircuitData) -> PyResult<()> {
    draw_circuit(&circuit.data)?;
    Ok(())
}

/// Calculate the range (inclusive) of the given instruction qubits/clbits over the wire indices.
/// The assumption is that clbits always appear after the qubits in the visualization, hence the clbit indices
/// are offset by the number of instruction qubits when calculating the range.
fn get_instruction_range(circuit: &CircuitData, instruction: &PackedInstruction) -> (usize, usize) {
    let node_qubits = circuit.get_qargs(instruction.qubits);
    let node_clbits = circuit.get_cargs(instruction.clbits);


    let indices = node_qubits.iter().map(|q| q.index()).chain(
        node_clbits.iter().map(|c| c.index() + circuit.num_qubits()));

    match indices.minmax() {
        MinMaxResult::MinMax(min, max) => (min, max),
        MinMaxResult::OneElement(idx) => (idx, idx),
        MinMaxResult::NoElements => panic!("Encountered an instruction without qubits and clbits")
    }
}

/// Return a list of PackedInstruction layers such that each layer contain instruction
/// whose qubits/clbits indices do not overlap. The instructions are packed into each layer
/// as long as there is no qubit/clbit overlap
fn build_layers<'a>(dag: &'a DAGCircuit, circuit: &'a CircuitData) -> Vec<Vec<&'a PackedInstruction>> {
    let mut layers:Vec<Vec<&PackedInstruction>> = Vec::new();
    let mut current_layer: Option<&mut Vec::<&PackedInstruction>> = None;
    let mut used_wires = HashSet::<usize>::new();

    for layer in dag.multigraph_layers() {
        for node_index in layer.into_iter().sorted() {

            if let NodeType::Operation(instruction_to_insert) = &dag.dag()[node_index] {
                let (node_min, node_max) = get_instruction_range(circuit, instruction_to_insert);

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

pub fn get_indices(dag_circ: &DAGCircuit) -> u32{
    let total_qubits = dag_circ.num_qubits() as u32;
    let total_clbits = dag_circ.num_clbits() as u32;
    total_qubits + total_clbits
}

#[derive(Clone)]
enum ElementWireInput<'a> {
    Qubit(&'a ShareableQubit),
    Clbit(&'a ShareableClbit),
}

impl<'a> Debug for ElementWireInput<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            ElementWireInput::Qubit(&ref qubit) => "Qubit",
            ElementWireInput::Clbit(&ref clbit) => "Clbit",
        };

        write!(f, "{}", name)
    }
}

/// Input ElementWires. 
/// The Option<String> is for optional labels and can be used when the registers have different names. This allows us to use
/// the same enum for both Input and VerticalLine types.
#[derive(Clone, Debug)]
enum InputType{
    Qubit(Option<String>),
    Clbit(Option<String>)
}

/// Enum for representing elements that can appear directly on a wire.
#[derive(Clone, Debug, Copy)]
enum ElementOnWire{
    Top,
    Mid,
    Bot,
}


/// Enum for representing elements that appear directly on a wire and how they're connected.
#[derive(Clone, Debug, Copy)]
enum OnElementWire{
    Control(ElementOnWire),
    Swap(ElementOnWire),
    Barrier,
    Reset,
}

/// Enum for representing elements that appear in a boxed operation.
#[derive(Clone)]
enum Boxed<'a>{
    Single(&'a PackedInstruction),
    // Multi(MultiBoxElement)
    Multi(&'a PackedInstruction),
}

impl<'a> Debug for Boxed<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Boxed::Single(inst) => write!(f, "Boxed({})", inst.op.name()),
            Boxed::Multi(inst) => write!(f, "MultiBox({})", inst.op.name()),
        }
    }
}


/// Enum for  representing the elements stored in a visualization matrix. The elements
/// do not directly implement visualization capabilities, but rather carry enough information
/// to enable visualization later on by the actual drawer.

struct Op<'a>{
    instruction: &'a PackedInstruction,

}

#[derive(Default, Clone, Debug)]
enum VisualizationElement<'a>{
    #[default]
    Empty, // Marker for no element
    Boxed(Boxed<'a>),
    Input(ElementWireInput<'a>),
    DirectOnElementWire(OnElementWire),
    VerticalLine(InputType),
}


/// A representation of a single column (called here a layer) of a visualization matrix
#[derive(Clone, Debug)]
struct VisualizationLayer<'a>(Vec<VisualizationElement<'a>>);

impl<'a> VisualizationLayer<'a> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn add_input(&mut self, input: ElementWireInput<'a>, idx: usize) {
        self.0[idx] = VisualizationElement::Input(input);
    }

    /// Adds the required visualization elements to represent the given instruction
    fn add_instruction(&mut self, inst: &'a PackedInstruction, circuit: &CircuitData) {
        match inst.op.view() {
            OperationRef::StandardGate(_gate) => self.add_standard(inst, circuit),
            OperationRef::StandardInstruction(_instruction) => {
                if let StandardInstruction::Barrier(_) = _instruction {
                    let barrier_indices = self.get_boxed_indices(inst, circuit);
                    self.add_barrier(inst, circuit, &barrier_indices);
                } else if let StandardInstruction::Reset = _instruction {
                    let reset_indices = self.get_boxed_indices(inst, circuit);
                    self.add_reset(inst, circuit, &reset_indices);
                } else {
                    self.add_standard(inst, circuit)
                }
            },
            _ => unimplemented!("{}", format!("Visualization is not implemented for instruction of type {:?}", inst.op)),
        }
    }


    fn get_controls(&self, inst: &PackedInstruction, circuit: &CircuitData) -> Vec<usize> {
        let gate_has_control = vec![StandardGate::CX, StandardGate::CCX, StandardGate::CY, StandardGate::CZ,
            StandardGate::CRX, StandardGate::CRY, StandardGate::CRZ,
            StandardGate::CPhase, StandardGate::CS, StandardGate::CSdg,
            StandardGate::CSX, StandardGate::CU, StandardGate::CU1,
            StandardGate::CU3, StandardGate::CH, StandardGate::C3SX, StandardGate::C3X,
            StandardGate::RC3X, StandardGate::RCCX, StandardGate::CSwap];
    
        let inst_has_controls = vec![StandardInstruction::Measure];

        let mut controls = vec![];

        let std_op = inst.op.view();

        match std_op {
            OperationRef::StandardGate(gate) => {
                if gate == StandardGate::CSwap {
                    let qargs = circuit.get_qargs(inst.qubits).into_iter().map(|q| q.index()).collect_vec();
                    let control = qargs[0];
                    controls.push(control);
                    controls
                } else if gate_has_control.contains(&gate) {
                    // let qargs = dag.get_qargs(inst.qubits).into_iter().map(|q| q.index()).collect_vec();
                    let qargs = circuit.get_qargs(inst.qubits).into_iter().map(|q| q.index()).collect_vec();

                    let target = qargs.last().unwrap();
                    let (minima,maxima) = get_instruction_range(circuit, inst);

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
                    let qargs = circuit.get_qargs(inst.qubits).into_iter().map(|q| q.index()).collect_vec();
                    let target = qargs.last().unwrap();
                    let cargs = circuit.get_cargs(inst.clbits).into_iter().map(|c| c.index() + circuit.num_qubits()).collect_vec();
                    let (minima,maxima) = get_instruction_range(circuit, inst);

                    for control in minima..=maxima {
                        if control != *target && cargs.contains(&control) {
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
                self.0[*control] = VisualizationElement::DirectOnElementWire(OnElementWire::Control(ElementOnWire::Top));
            } else if *control == range.1 {
                self.0[*control] = VisualizationElement::DirectOnElementWire(OnElementWire::Control(ElementOnWire::Bot));
            } else {
                self.0[*control] = VisualizationElement::DirectOnElementWire(OnElementWire::Control(ElementOnWire::Mid));
            }
        }
    }

    fn get_boxed_indices(&mut self, inst: & PackedInstruction, circuit: &CircuitData) -> Vec<usize>{
        let single_box = vec![StandardGate::H, StandardGate::X, StandardGate::Y, StandardGate::Z,
            StandardGate::RX, StandardGate::RY, StandardGate::RZ, StandardGate::U, StandardGate::U1,
            StandardGate::U2, StandardGate::U3, StandardGate::S, StandardGate::Sdg,
            StandardGate::T, StandardGate::Tdg, StandardGate::Phase, StandardGate::R, StandardGate::SX,
            StandardGate::SXdg, StandardGate::CCX, StandardGate::CCZ, StandardGate::CX, StandardGate::CY,
            StandardGate::CZ, StandardGate::CPhase, StandardGate::CRX, StandardGate::CRY,
            StandardGate::CRZ, StandardGate::CS, StandardGate::CSdg, StandardGate::CSX,
            StandardGate::CU, StandardGate::CU1, StandardGate::CU3, StandardGate::C3X, StandardGate::C3SX, StandardGate::RC3X];
        
        let single_box_instrusctions = vec![StandardInstruction::Measure];

        let multi_box = vec![StandardGate::ISwap, StandardGate::DCX,
            StandardGate::ECR, StandardGate::RXX, StandardGate::RYY, StandardGate::RZZ,
            StandardGate::RZX, StandardGate::XXMinusYY, StandardGate::XXPlusYY];

        let direct_on_wire = vec![StandardGate::Swap, StandardGate::CSwap];

        let special_cases = vec![ StandardGate::GlobalPhase, StandardGate::CPhase];

        let qargs = circuit.get_qargs(inst.qubits).into_iter().map(|q| q.index()).collect_vec();
        let target = qargs.last().unwrap();
        let range = get_instruction_range(circuit, inst);

        if let Some(std_gate) = inst.op.try_standard_gate(){
            if single_box.contains(&std_gate){
                vec![*target]
            } else if multi_box.contains(&std_gate){
                (range.0..=range.1).collect()
            } else if direct_on_wire.contains(&std_gate){
                qargs.iter().rev().take(2).cloned().collect_vec()
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
        } else {
            vec![]
        }
        
    }

    fn add_barrier(&mut self, inst: &'a PackedInstruction, circuit: &CircuitData, barrier_indices: &Vec<usize>) {
        for bline in barrier_indices {
            self.0[*bline] = VisualizationElement::DirectOnElementWire(OnElementWire::Barrier);
        }
    }

    fn add_reset(&mut self, inst: &'a PackedInstruction, circuit: &CircuitData, reset_indices: &Vec<usize>) {
        for rline in reset_indices {
            self.0[*rline] = VisualizationElement::DirectOnElementWire(OnElementWire::Reset);
        }
    }

    // []

    fn add_boxed(&mut self, inst: &'a PackedInstruction, circuit: &CircuitData, boxed_indices: &Vec<usize>) {
        // The case of delay needs to be handled where it is multiple single qubit gates but shown as a box
        // check if packed instruction is swap or cswap
        // let qargs = circuit.get_qargs(inst.qubits).into_iter().map(|q| q.index()).collect_vec();
        let special_cases = vec![StandardGate::Swap, StandardGate::CSwap];
        // if let Some(std_gate) = inst.op.try_standard_gate(){
        //     if special_cases.contains(&std_gate){
        //         let range= get_instruction_range(circuit, inst);
        //         let targets = qargs.iter().rev().take(2).collect_vec();
        //         for target in targets {
        //             if *target == range.0 {
        //                 self.0[*target] = VisualizationElement::DirectOnElementWire(OnElementWire::Swap(ElementOnWire::Top));
        //             } else if *target == range.1 {
        //                 self.0[*target] = VisualizationElement::DirectOnElementWire(OnElementWire::Swap(ElementOnWire::Bot));
        //             } else {
        //                 self.0[*target] = VisualizationElement::DirectOnElementWire(OnElementWire::Swap(ElementOnWire::Mid));
        //             }
        //         }
        //     }
        // }
        if boxed_indices.len() == 1 {
            let qargs = circuit.get_qargs(inst.qubits).into_iter().map(|q| q.index()).collect_vec();
            let target = qargs.last().unwrap();
            self.0[*target] = VisualizationElement::Boxed(Boxed::Single(inst));
        } else if boxed_indices.len() > 1 {
            if let Some(std_gate) = inst.op.try_standard_gate(){
                if special_cases.contains(&std_gate){
                    let range= get_instruction_range(circuit, inst);
                    let targets = boxed_indices.iter().rev().take(2).collect_vec();
                    for target in targets {
                        if *target == range.0 {
                            self.0[*target] = VisualizationElement::DirectOnElementWire(OnElementWire::Swap(ElementOnWire::Top));
                        } else if *target == range.1 {
                            self.0[*target] = VisualizationElement::DirectOnElementWire(OnElementWire::Swap(ElementOnWire::Bot));
                        } else {
                            self.0[*target] = VisualizationElement::DirectOnElementWire(OnElementWire::Swap(ElementOnWire::Mid));
                        }
                    }
                    return;
                } else {
                    for idx in boxed_indices {
                        self.0[*idx] = VisualizationElement::Boxed(Boxed::Multi(inst));
                    }
                }
            } else{
                for idx in boxed_indices {
                    self.0[*idx] = VisualizationElement::Boxed(Boxed::Multi(inst));
                }
            }
        }
    }

    fn add_vertical_lines(&mut self, inst: &'a PackedInstruction, circuit: &CircuitData, vertical_lines: &Vec<usize>) {
        let double_lines = vec![StandardInstruction::Measure];
        let input_type: InputType = if let Some(std_instruction) = inst.op.try_standard_instruction(){
            if double_lines.contains(&std_instruction){
                InputType::Clbit(None)
            } else {
                InputType::Qubit(None)
            }
        } else {
            InputType::Qubit(None)
        };
        for vline in vertical_lines {
            self.0[*vline] = VisualizationElement::VerticalLine(input_type.clone() );
        }
    }

    // function to add standard gates and instructions
    fn add_standard(&mut self, inst: &'a PackedInstruction, circuit: &CircuitData) {
        let (minima,maxima) = get_instruction_range(circuit, inst);
        let controls = self.get_controls(inst, circuit);
        let boxed_elements = self.get_boxed_indices(inst, circuit);
        let vert_lines = (minima..=maxima)
            .filter(|idx| !controls.contains(idx) && !boxed_elements.contains(idx))
            .collect_vec();
        self.add_controls(&controls, (minima, maxima));
        self.add_boxed(inst, circuit, &boxed_elements);
        self.add_vertical_lines(inst, circuit, &vert_lines);

    }
}

impl<'a> Index<usize> for VisualizationLayer<'a> {
    type Output = VisualizationElement<'a>;

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
struct VisualizationMatrix<'a> {
    layers: Vec<VisualizationLayer<'a>>,
    circuit: &'a CircuitData,
}

impl<'a> VisualizationMatrix<'a> {

    fn from_circuit(dag: &'a DAGCircuit,circuit: &'a CircuitData) -> PyResult<Self> {
        // let dag = DAGCircuit::from_circuit_data(circuit, false, None, None, None, None)?;
        let inst_layers = build_layers(&dag, circuit);

        let num_wires = circuit.num_qubits() + circuit.num_clbits();
        let mut layers = vec![VisualizationLayer(vec![VisualizationElement::default(); num_wires]); inst_layers.len() + 1]; // Add 1 to account for the inputs layer

        // TODO: add the qubit/clbit inputs here to layer #0

        let input_layer = layers.first_mut().unwrap();
        let mut input_idx = 0;
        for qubit in circuit.qubits().objects() {
            input_layer.add_input(ElementWireInput::Qubit(qubit), input_idx);
            input_idx += 1;
        }

        for clbit in circuit.clbits().objects() {
            input_layer.add_input(ElementWireInput::Clbit(clbit), input_idx);
            input_idx += 1;
        }

        for (i, layer) in inst_layers.iter().enumerate() {
            for inst in layer {
                layers[i + 1].add_instruction(inst, &circuit);
            }
        }

        Ok(VisualizationMatrix{
            layers,
            circuit,
        })
    }

    fn num_wires(&self) -> usize {
        self.layers.first().map_or(0, |layer| layer.len())
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl<'a> Index<usize> for VisualizationMatrix<'a> {
    type Output = VisualizationLayer<'a>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.layers[index]
    }
}

 // [control control boxed boxed]
 // [swap swap]
 // [control swaps swaps], 

// better name for the struct
#[derive(Clone)]
struct ElementWire{
    top:String,
    mid:String,
    bot:String,
}

impl Debug for ElementWire{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\n{}\n{}", self.top, self.mid, self.bot)
    }
}

impl ElementWire{
    fn width(&self) -> usize{
        // return the max of all strigns
        // self.top.len().max(self.mid.len()).max(self.bot.len())
        let top = self.top.chars().count();
        let mid = self.mid.chars().count();
        let bot = self.bot.chars().count();
        // println!("top:{}, mid:{}, bot:{}", top, mid, bot);
        // println!("{}", top.max(mid).max(bot));
        let max = {
            if top >= mid && top >= bot {
                top
            } else if mid >= top && mid >= bot {
                mid
            } else {
                bot
            }
        };
        // println!("max: {}", max);
        // println!("{}\n{}\n{}", self.top, self.mid, self.bot);
        max
    }

    fn left_pad_string(s: &mut String, pad_char: char, width: usize){
        let current_width = s.len();
        if current_width < width{
            let pad_size = width - current_width;
            let pad_str = pad_char.to_string().repeat(pad_size);
            let new_str = format!("{}{}", pad_str, s);
            *s = new_str;
        }
    }

    fn right_pad_string(s: &mut String, pad_char: char, width: usize){
        let current_width = s.len();
        if current_width < width{
            let pad_size = width - current_width;
            let pad_str = pad_char.to_string().repeat(pad_size);
            let new_str = format!("{}{}", s, pad_str);
            *s = new_str;
        }
    }

    fn pad_string(s: &mut String, pad_char: char, width: usize){
        let current_width = s.chars().count();
        if current_width < width{
            let pad_size = width - current_width;
            let left_pad = pad_size / 2;
            let right_pad = pad_size - left_pad;
            let left_pad_str = pad_char.to_string().repeat(left_pad);
            let right_pad_str = pad_char.to_string().repeat(right_pad);
            let new_str = format!("{}{}{}", left_pad_str, s, right_pad_str);
            *s = new_str;
        }
    }

    fn pad_wire(&mut self, mid_char: char, width: usize){
        let current_width = self.width();
        if current_width < width{
            // let pad_size = width - current_width;
            // let left_pad = pad_size / 2 - 1;
            // let right_pad = pad_size - left_pad - 1;
            // let left_pad_str = mid_char.to_string().repeat(left_pad);
            // let right_pad_str = mid_char.to_string().repeat(right_pad);
            // self.top = format!("{}{}{}", " ".to_string().repeat(left_pad), self.top, " ".to_string().repeat(right_pad));
            // self.mid = format!("{}{}{}", left_pad_str, self.mid, right_pad_str);
            Self::pad_string(&mut self.top, ' ', width);
            Self::pad_string(&mut self.mid, mid_char, width);
            Self::pad_string(&mut self.bot, ' ', width);
        }
        //println!("layer width:{}, object width:{} : \n{}\n{}\n{}", width, current_width, self.top, self.mid, self.bot);
    }
}


pub const Q_WIRE: char = '─';
pub const C_WIRE: char = '═';
pub const TOP_CON: char = '┴';
pub const BOT_CON: char = '┬';
pub const Q_LEFT_CON: char = '┤';
pub const Q_RIGHT_CON: char = '├';
pub const CL_LEFT_CON: char = '╡';
pub const CL_RIGHT_CON: char = '╞';
pub const TOP_LEFT_BOX: char = '┌';
pub const TOP_RIGHT_BOX: char = '┐';
pub const BOT_LEFT_BOX: char = '└';
pub const BOT_RIGHT_BOX: char = '┘';
pub const BARRIER: char = '░';
pub const BULLET:char = '■';
pub const C_BULLET:char = '╩';
pub const CONNECTING_WIRE:char = '│';
pub const CL_CONNECTING_WIRE:char = '║';
pub const Q_Q_CROSSED_WIRE:char = '┼';
pub const Q_CL_CROSSED_WIRE:char = '╪';
pub const CL_CL_CROSSED_WIRE:char = '╬';
pub const CL_Q_CROSSED_WIRE:char = '╫';

struct TextDrawer{
    wires: Vec<Vec<ElementWire>>,
}

impl Index<usize> for TextDrawer{
    type Output = Vec<ElementWire>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.wires[index]
    }
}

impl TextDrawer{

    fn get_label(instruction: &PackedInstruction) -> Option<String>{
        let label = instruction.label();
        let instruction_param =format!("{:?}",instruction.params_view());
        let instruction_label = match label {
            Some(l) => {
                Some(format!("{}{}{}"," ",l.to_string()," "))
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
                            Some(" H ".to_string())
                        }
                        StandardGate::I => {
                            // Identity gate - no operation
                            Some(" I ".to_string())
                        }
                        StandardGate::X => {
                            // Pauli-X gate (NOT gate)
                            Some(" X ".to_string())
                        }
                        StandardGate::Y => {
                            // Pauli-Y gate
                            Some(" Y ".to_string())
                        }
                        StandardGate::Z => {
                            // Pauli-Z gate
                            Some(" Z ".to_string())
                        }
                        StandardGate::Phase => {
                            // Phase gate (parameterized)
                            Some(format!(" P({}) ", instruction_param))
                        }
                        StandardGate::R => {
                            // R gate (rotation about axis in XY plane)
                            Some(format!(" R({}) ", instruction_param))
                        }
                        StandardGate::RX => {
                            // Rotation about X axis
                            Some(format!(" RX({}) ", instruction_param))
                        }
                        StandardGate::RY => {
                            // Rotation about Y axis
                            Some(format!(" RY({}) ", instruction_param))
                        }
                        StandardGate::RZ => {
                            // Rotation about Z axis
                            Some(format!(" RZ({}) ", instruction_param))
                        }
                        StandardGate::S => {
                            // S gate (phase π/2)
                            Some(" S ".to_string())
                        }
                        StandardGate::Sdg => {
                            // S dagger gate (phase -π/2)
                            Some(" Sdg ".to_string())
                        }
                        StandardGate::SX => {
                            // Square root of X gate
                            Some(" √X ".to_string())
                        }
                        StandardGate::SXdg => {
                            // Square root of X dagger gate
                            Some(" √Xdg ".to_string())
                        }
                        StandardGate::T => {
                            // T gate (phase π/4)
                            Some(" T ".to_string())
                        }
                        StandardGate::Tdg => {
                            // T dagger gate (phase -π/4)
                            Some(" T† ".to_string())
                        }
                        StandardGate::U => {
                            // Universal single-qubit gate (3 parameters)
                            Some(format!(" U({}) ", instruction_param))
                        }
                        StandardGate::U1 => {
                            // U1 gate (1 parameter - phase)
                            Some(format!(" U1({}) ", instruction_param))
                        }
                        StandardGate::U2 => {
                            // U2 gate (2 parameters)
                            Some(format!(" U2({}) ", instruction_param))
                        }
                        StandardGate::U3 => {
                            // U3 gate (3 parameters - equivalent to U)
                            Some(format!(" U3({}) ", instruction_param))
                        }
                        StandardGate::CH => {
                            // Controlled Hadamard gate
                            Some(" H ".to_string())
                        }
                        StandardGate::CX => {
                            // Controlled-X gate (CNOT)
                            Some(" X ".to_string())
                        }
                        StandardGate::CY => {
                            // Controlled-Y gate
                            Some(" Y ".to_string())
                        }
                        StandardGate::CZ => {
                            // Controlled-Z gate
                            Some(" Z ".to_string())
                        }
                        StandardGate::DCX => {
                            // Double CNOT gate
                            Some(" DCX ".to_string())
                        }
                        StandardGate::ECR => {
                            // Echoed cross-resonance gate
                            Some(" ECR ".to_string())
                        }
                        StandardGate::Swap => {
                            // Swap gate
                            None
                        }
                        StandardGate::ISwap => {
                            // i-Swap gate
                            Some(" iSWAP ".to_string())
                        }
                        StandardGate::CPhase => {
                            // Controlled phase gate

                            Some(format!(" P({}) ", instruction_param))
                        }
                        StandardGate::CRX => {
                            // Controlled rotation about X
                            Some(format!(" RX({}) ", instruction_param))
                        }
                        StandardGate::CRY => {
                            // Controlled rotation about Y
                            Some(format!(" RY({}) ", instruction_param))
                        }
                        StandardGate::CRZ => {
                            // Controlled rotation about Z
                            Some(format!(" RZ({}) ", instruction_param))
                        }
                        StandardGate::CS => {
                            // Controlled S gate
                            Some(" S ".to_string())
                        }
                        StandardGate::CSdg => {
                            // Controlled S dagger gate
                            Some(" Sdg ".to_string())
                        }
                        StandardGate::CSX => {
                            // Controlled square root of X gate
                            Some(" √X ".to_string())
                        }
                        StandardGate::CU => {
                            // Controlled U gate (4 parameters)
                            Some(format!(" U({}) ", instruction_param))
                        }
                        StandardGate::CU1 => {
                            // Controlled U1 gate
                            Some(format!(" U1({}) ", instruction_param))
                        }
                        StandardGate::CU3 => {
                            // Controlled U3 gate
                            Some(format!(" U3({}) ", instruction_param))
                        }
                        StandardGate::RXX => {
                            // Two-qubit XX rotation
                            Some(format!(" RXX({}) ", instruction_param))
                        }
                        StandardGate::RYY => {
                            // Two-qubit YY rotation
                            Some(format!(" RYY({}) ", instruction_param))
                        }
                        StandardGate::RZZ => {
                            // Two-qubit ZZ rotation
                            Some(format!(" RZZ({}) ", instruction_param))
                        }
                        StandardGate::RZX => {
                            // Two-qubit ZX rotation
                            Some(format!(" RZX({}) ", instruction_param))
                        }
                        StandardGate::XXMinusYY => {
                            // XX-YY gate
                            Some(format!(" XX-YY({}) ", instruction_param))
                        }
                        StandardGate::XXPlusYY => {
                            // XX+YY gate
                            Some(format!(" XX+YY({}) ", instruction_param))
                        }
                        StandardGate::CCX => {
                            // Toffoli gate (controlled-controlled-X)
                            Some(" X ".to_string())
                        }
                        StandardGate::CCZ => {
                            // Controlled-controlled-Z gate
                            Some(" Z ".to_string())
                        }
                        StandardGate::CSwap => {
                            // Controlled swap gate (Fredkin gate)
                            None
                        }
                        StandardGate::RCCX => {
                            // Relative-phase Toffoli gate
                            Some(format!(" RX({}) ", instruction_param))
                        }
                        StandardGate::C3X => {
                            // 3-controlled X gate (4-qubit controlled X)
                            Some(" X ".to_string())
                        }
                        StandardGate::C3SX => {
                            // 3-controlled square root of X gate
                            Some(" √X ".to_string())
                        }
                        StandardGate::RC3X => {
                            // Relative-phase 3-controlled X gate
                            Some(format!(" RX({}) ", instruction_param))
                        }
                    }
                } else if let Some(std_instruction) = instruction.op.try_standard_instruction(){
                    if std_instruction == StandardInstruction::Measure{
                        Some("M".to_string())
                    } else if std_instruction == StandardInstruction::Reset{
                        Some("|0>".to_string())
                    } else if let StandardInstruction::Barrier(_) = std_instruction {
                        Some("░".to_string())
                    } else {
                        // Fallback for non-standard instructions
                        Some(format!("{}{}{}"," ",instruction.op.name().to_string()," "))
                    }
                } else {
                    // Fallback for non-standard operations
                    Some(format!("{}{}{}"," ",instruction.op.name().to_string()," "))
                }
            },
        };
        instruction_label
    }

    fn from_visualization_matrix(vis_mat: &VisualizationMatrix) -> Self{
        let mut wires: Vec<Vec<ElementWire>> = vec![];
        for _ in 0..vis_mat.num_layers(){
            wires.push(vec![]);
        }
        for i in 0..vis_mat.num_wires(){
            wires[i].push(ElementWire{
                top: String::new(),
                mid: String::new(),
                bot: String::new(),
            });
        }

        let mut text_drawer = TextDrawer{
            wires
        };

        let mut ct = 0;
        for layer in &vis_mat.layers{
            let layer_wires = Self::draw_layer(layer, vis_mat.circuit, ct);
            ct += 1;
            for (i, wire) in layer_wires.iter().enumerate(){
                text_drawer.wires[i].push(wire.clone());
            }
        }

        text_drawer
    }

    // fn get_layer_width(&self, layer: &VisualizationLayer, circ_data: &CircuitData) -> usize{
    //     let mut width:usize = 0;
    //     for element in layer.0.iter(){
    //         let ele_width = Self::get_element_width(self, &element, circ_data);
    //         if ele_width > width{
    //             width = ele_width;
    //         }
    //     }
    //     width
    // }

    // fn get_element_width(&self, element: &VisualizationElement, circ_data: &CircuitData) -> usize{
    //     Self::draw_element(element.clone(), circ_data, 0).width()
    // }

    fn draw_layer(layer: &VisualizationLayer, circ_data: &CircuitData, layer_ind: usize) -> Vec<ElementWire>{
        let mut wires: Vec<ElementWire> = vec![];
        for (i,element) in layer.0.iter().enumerate(){
            let wire = Self::draw_element(element.clone(), circ_data,i);
            wires.push(wire);
        }

        let num_qubits = circ_data.num_qubits();

        //let layer_width = wires.iter().map(|w| w.width()).max().unwrap_or(0);
        let mut layer_width = 0;
        for wire in wires.iter(){
            let w = wire.width();
            if w > layer_width{
                layer_width = w;
            }
        }

        for (i,wire) in wires.iter_mut().enumerate(){
            if layer_ind == 0{
                wire.pad_wire(' ', layer_width);
            } else if i < num_qubits{
                wire.pad_wire(Q_WIRE, layer_width);
                // wire.pad_wire('$', layer_width);
            } else {
                wire.pad_wire(C_WIRE, layer_width);
                // wire.pad_wire('$', layer_width);
            }
        }

        wires
    }

    pub fn draw_element(vis_ele: VisualizationElement, circ_data: &CircuitData, ind: usize) -> ElementWire {
        match vis_ele {
            VisualizationElement::Boxed(sub_type) => {
                // implement for cases where the box is on classical wires. The left and right connectors will change
                // from single wired to double wired.

                match sub_type {
                    Boxed::Single(inst) => {
                        let label = Self::get_label(inst).unwrap_or(" ".to_string());
                        ElementWire{
                            top: format!("{}{}{}", TOP_LEFT_BOX, Q_WIRE.to_string().repeat(label.len()), TOP_RIGHT_BOX),
                            mid: format!("{}{}{}", Q_LEFT_CON  , label, Q_RIGHT_CON),
                            bot: format!("{}{}{}", BOT_LEFT_BOX, Q_WIRE.to_string().repeat(label.len()), BOT_RIGHT_BOX),
                        }
                    },
                    Boxed::Multi(inst) => {
                        let label = Self::get_label(inst).unwrap_or(" ".to_string());
                        // get all the indices affected by this multi-box
                        let qargs = circ_data.qargs_interner().get(inst.qubits).into_iter().map(|q| q.index()).collect_vec();
                        let cargs = circ_data.cargs_interner().get(inst.clbits).into_iter().map(|c| c.index() + circ_data.num_qubits()).collect_vec();
                        let minmax = qargs.iter().chain(cargs.iter()).minmax();
                        let range = match minmax {
                            MinMaxResult::MinMax(min, max) => (*min, *max),
                            MinMaxResult::OneElement(idx) => (*idx, *idx),
                            MinMaxResult::NoElements => panic!("Encountered an multi-qubit without qubits and clbits")
                        };
                        let mid = (range.0 + range.1) / 2;
                        
                        let num_affected = {
                            if qargs.contains(&ind) || cargs.contains(&ind) {
                                // Once the packed instruction can handle custom gates, we need to first check the number of controls
                                // and then give an index to the qubit in the multibox an index based on whats left. For example, if the
                                // qubits being affected are [0,1,2,3,4,5] and num controls is 2, then the qubits will be indexed as [C,C,T,T,T,T]
                                // so for qubits [2,3,4,5] the indexes inside the box will be 0,1,2,3 respectively.

                                // get index of ind inside qargs or cargs
                                let temp: String = {
                                    if qargs.contains(&ind) {
                                        let idx = qargs.iter().position(|&x| x == ind).unwrap();
                                        format!("{:^width$}", idx, width = qargs.len())
                                    }
                                    else {
                                        " ".to_string()
                                    }
                                };
                                temp
                            } else {
                                " ".to_string()
                            }          
                        };

                        let mid_section = if ind == mid {
                            format!("{:^total_q$} {:^label_len$}", num_affected, label, total_q = qargs.len(), label_len=label.len())
                        } else {
                            format!("{:^total_q$} {:^label_len$}", num_affected, " ", total_q = qargs.len(), label_len=label.len())
                        };

                        if ind == range.0 {
                            ElementWire{
                                top: format!("{}{}{}", TOP_LEFT_BOX, Q_WIRE.to_string().repeat(mid_section.len()), TOP_RIGHT_BOX),
                                mid: format!("{}{}{}", Q_LEFT_CON, mid_section, Q_RIGHT_CON),
                                bot: format!("{}{}{}", CONNECTING_WIRE, " ".repeat(mid_section.len()), CONNECTING_WIRE),
                            }
                        } else if ind == range.1 {
                            ElementWire{
                                top: format!("{}{}{}", CONNECTING_WIRE," ".to_string().repeat(mid_section.len()), CONNECTING_WIRE),
                                mid: format!("{}{}{}", Q_LEFT_CON, mid_section, Q_RIGHT_CON),
                                bot: format!("{}{}{}", BOT_LEFT_BOX, Q_WIRE.to_string().repeat(mid_section.len()), BOT_RIGHT_BOX),
                            }
                        } else {
                            ElementWire{
                                top: format!("{}{}{}", CONNECTING_WIRE, " ".repeat(mid_section.len()), CONNECTING_WIRE),
                                mid: format!("{}{}{}", Q_LEFT_CON, mid_section, Q_RIGHT_CON),
                                bot: format!("{}{}{}", CONNECTING_WIRE, " ".repeat(mid_section.len()), CONNECTING_WIRE),
                            }
                        }
                    }
                }
            },
            VisualizationElement::DirectOnElementWire(on_wire) => {

                let connecting_wire = if ind < circ_data.num_qubits() {
                    CONNECTING_WIRE
                } else {
                    CL_CONNECTING_WIRE
                };

                let wire_char: String = match on_wire {
                    OnElementWire::Control(position) => {
                        if ind < circ_data.num_qubits() {
                            BULLET.to_string()
                        } else {
                            C_BULLET.to_string()
                        }
                    },
                    OnElementWire::Swap(position) => {
                        "X".to_string()
                    },
                    OnElementWire::Barrier => BARRIER.to_string(),
                    OnElementWire::Reset => "|0>".to_string(),
                };

                let top:String = match on_wire{
                    OnElementWire::Control(position) => {
                        match position {
                            ElementOnWire::Top => " ".to_string(),
                            ElementOnWire::Mid => format!("{}", connecting_wire),
                            ElementOnWire::Bot => format!("{}", connecting_wire),
                        }
                    },
                    OnElementWire::Swap(position) => {
                        match position {
                            ElementOnWire::Top => " ".to_string(),
                            ElementOnWire::Mid => format!("{}", connecting_wire),
                            ElementOnWire::Bot => format!("{}", connecting_wire),
                        }
                    },
                    OnElementWire::Barrier => {
                        format!("{}", BARRIER)
                    },
                    OnElementWire::Reset => {
                        "   ".to_string()
                    },
                };

                let bot: String = match on_wire{
                    OnElementWire::Control(position) => {
                        match position {
                            ElementOnWire::Top => format!("{}", connecting_wire),
                            ElementOnWire::Mid => format!("{}", connecting_wire),
                            ElementOnWire::Bot => " ".to_string(),
                        }
                    },
                    OnElementWire::Swap(position) => {
                        match position {
                            ElementOnWire::Top => format!("{}", connecting_wire),
                            ElementOnWire::Mid => format!("{}", connecting_wire),
                            ElementOnWire::Bot => " ".to_string(),
                        }
                    },
                    OnElementWire::Barrier => {
                        format!("{}", BARRIER)
                    },
                    OnElementWire::Reset => {
                        "   ".to_string()
                    },
                };

                let wire = if ind < circ_data.num_qubits() {
                    Q_WIRE
                } else {
                    C_WIRE
                };

                ElementWire{
                    top: format!("{}{}{}"," ",top," "),
                    mid: format!("{}{}{}", wire, wire_char, wire),
                    bot: format!("{}{}{}"," ",bot," "),
                }

            },
            VisualizationElement::Input(wire_input) => {    
                match wire_input {
                    ElementWireInput::Qubit(qubit) => {
                        let qubit_name = if let Some(bit_info) = circ_data.qubit_indices().get(qubit) {
                            if let Some((register, index)) = bit_info.registers().first() {
                                format!("{}_{}:", register.name(), index)
                            } else {
                                format!("q_{}:", ind)
                            }
                        } else {
                            format!("q_{}:", ind)
                        };
                        ElementWire{
                            top: format!("{}", " ".repeat(qubit_name.len())),
                            mid: format!("{}", qubit_name),
                            bot: format!("{}", " ".repeat(qubit_name.len())),
                        }
                    }
                    ElementWireInput::Clbit(clbit) => {
                        let clbit_name = if let Some(bit_info) = circ_data.clbit_indices().get(clbit) {
                            if let Some((register, index)) = bit_info.registers().first() {
                                format!("{}_{}:", register.name(), index)
                            } else {
                                format!("c_{}:", ind)
                            }
                        } else {
                            format!("c_{}:", ind)
                        };
                        ElementWire{
                            top: format!("{}", " ".repeat(clbit_name.len())),
                            mid: format!("{}", clbit_name),
                            bot: format!("{}", " ".repeat(clbit_name.len())),
                        }
                    },
                }
            },
            VisualizationElement::VerticalLine(input_type) => {
                let crossed = {
                    match &input_type {
                        InputType::Qubit(label) => {
                            if ind < circ_data.num_qubits() {
                                Q_Q_CROSSED_WIRE
                            } else {
                                Q_CL_CROSSED_WIRE
                            }
                        },
                        InputType::Clbit(label) => {
                            if ind < circ_data.num_qubits() {
                                CL_Q_CROSSED_WIRE
                            } else {
                                CL_CL_CROSSED_WIRE
                            }
                        },
                    }
                };
                let connector = match &input_type {
                    InputType::Qubit(_) => CONNECTING_WIRE,
                    InputType::Clbit(_) => CL_CONNECTING_WIRE,
                };
                ElementWire{
                    top: format!("{}", connector),
                    mid: format!("{}", crossed),
                    bot: format!("{}", connector),
                }
            },
            VisualizationElement::Empty => {
                let wire = {
                    if ind < circ_data.num_qubits() {
                        Q_WIRE
                    } else {
                        C_WIRE
                    }
                };
                ElementWire{
                    top: format!(" "),
                    mid: format!("{}", wire),
                    bot: format!(" "),
                }
            },
        }
    }

    fn print(&self){
        let mut output = String::new();
        for i in self.wires.iter(){
            let top_line: String = i.iter().map(|wire| wire.top.clone()).collect::<Vec<String>>().join("");
            let mid_line: String = i.iter().map(|wire| wire.mid.clone()).collect::<Vec<String>>().join("");
            let bot_line: String = i.iter().map(|wire| wire.bot.clone()).collect::<Vec<String>>().join("");
            output.push_str(&format!("{}\n{}\n{}\n", top_line, mid_line, bot_line));
        }
        println!("{}", output);
    }

    // fn add_wire(&mut self, wire: &ElementWire, ind: usize){
    //     self.wires[ind].top.push_str(&wire.top);
    //     self.wires[ind].mid.push_str(&wire.mid);
    //     self.wires[ind].bot.push_str(&wire.bot);
    //     // self.wires[ind].top.push_str(&format!("{}{}",&wire.top,"$"));
    //     // self.wires[ind].mid.push_str(&format!("{}{}",&wire.mid,"$"));
    //     // self.wires[ind].bot.push_str(&format!("{}{}",&wire.bot,"$"));
    // }
}

pub fn draw_circuit(circuit: &CircuitData) -> PyResult<()> {
    let dag = DAGCircuit::from_circuit_data(circuit, false, None, None, None, None)?;

    let vis_mat2 = VisualizationMatrix::from_circuit(&dag,circuit)?;

    println!("======================");

    println!("num wires {}, num layers {}", vis_mat2.num_wires(), vis_mat2.num_layers());

    for i in 0..vis_mat2.num_wires() {
        for j in 0..vis_mat2.num_layers() {
            print!("{:^30}", format!("{:?}", vis_mat2[j][i]));
        }
        println!("");
    }

    let circuit_rep = TextDrawer::from_visualization_matrix(&vis_mat2);
    circuit_rep.print();
    Ok(())
}
