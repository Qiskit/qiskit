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

use crate::bit::{ShareableClbit, ShareableQubit};
use crate::dag_circuit::DAGCircuit;
use crate::operations::{Operation, OperationRef, Param, StandardGate, StandardInstruction};
use crate::packed_instruction::PackedInstruction;
use crate::{Clbit, Qubit};
use core::panic;
use hashbrown::{HashMap, HashSet};
use itertools::{Itertools, MinMaxResult};
use rustworkx_core::petgraph::csr::IndexType;
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::Index;

use pyo3::import_exception;
use pyo3::prelude::*;

use crate::circuit_data::CircuitData;
use crate::converters::QuantumCircuitData;
use crate::dag_circuit::NodeType;

import_exception!(qiskit.circuit.exceptions, CircuitError);

// TODO: remove when dev is done, since this is only for manual testing
#[pyfunction(name = "draw")]
pub fn py_drawer(circuit: QuantumCircuitData) -> PyResult<()> {
    draw_circuit(&circuit.data)?;
    Ok(())
}

pub fn draw_circuit(circuit: &CircuitData) -> PyResult<()> {
    let vis_mat2 = VisualizationMatrix::from_circuit(circuit)?;

    for inst in circuit.data() {
        println!(
            "INST {:?} QARGS: {:?}",
            inst,
            circuit.get_qargs(inst.qubits)
        );
    }

    println!("======================");

    println!(
        "num wires {}, num layers {}",
        vis_mat2.num_wires(),
        vis_mat2.num_layers()
    );

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

/// Return a list of layers such that each layer contains a list of op node indices, representing instructions
/// whose qubits/clbits indices do not overlap. The instruction are packed into each layer as long as there
/// is no qubit/clbit overlap.
fn build_layers(dag: &DAGCircuit) -> Vec<Vec<NodeIndex>> {
    let mut layers: Vec<Vec<NodeIndex>> = Vec::new();
    let mut current_layer: Option<&mut Vec<NodeIndex>> = None;
    let mut used_wires = HashSet::<usize>::new();

    for layer in dag.multigraph_layers() {
        for node_index in layer.into_iter().sorted() {
            if let NodeType::Operation(instruction_to_insert) = &dag.dag()[node_index] {
                let (node_min, node_max) = get_instruction_range(
                    dag.get_qargs(instruction_to_insert.qubits),
                    dag.get_cargs(instruction_to_insert.clbits),
                    dag.num_qubits(),
                );

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

                current_layer.as_mut().unwrap().push(node_index);
            }
        }
    }

    layers
}

/// Calculate the range (inclusive) of the given instruction qubits/clbits over the wire indices.
/// The assumption is that clbits always appear after the qubits in the visualization, hence the clbit indices
/// are offset by the number of instruction qubits when calculating the range.
fn get_instruction_range(
    node_qubits: &[Qubit],
    node_clbits: &[Clbit],
    num_qubits: usize,
) -> (usize, usize) {
    let indices = node_qubits
        .iter()
        .map(|q| q.index())
        .chain(node_clbits.iter().map(|c| c.index() + num_qubits));

    match indices.minmax() {
        MinMaxResult::MinMax(min, max) => (min, max),
        MinMaxResult::OneElement(idx) => (idx, idx),
        MinMaxResult::NoElements => panic!("Encountered an instruction without qubits and clbits"),
    }
}

#[derive(Clone, PartialEq, Eq)]
enum ElementWireInput<'a> {
    Qubit(&'a ShareableQubit),
    Clbit(&'a ShareableClbit),
}

impl PartialOrd for ElementWireInput<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ElementWireInput<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (ElementWireInput::Qubit(q1), ElementWireInput::Qubit(q2)) => Ordering::Equal,
            (ElementWireInput::Clbit(c1), ElementWireInput::Clbit(c2)) => Ordering::Equal,
            (ElementWireInput::Qubit(_), ElementWireInput::Clbit(_)) => Ordering::Less,
            (ElementWireInput::Clbit(_), ElementWireInput::Qubit(_)) => Ordering::Greater,
        }
    }
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
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum InputType {
    Qubit(Option<String>),
    Clbit(Option<String>),
}

/// Enum for representing elements that can appear directly on a wire.
#[derive(Clone, Debug, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ElementOnWire {
    Top,
    Mid,
    Bot,
}

/// Enum for representing elements that appear directly on a wire and how they're connected.
#[derive(Clone, Debug, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum OnWire {
    Control(ElementOnWire),
    Swap(ElementOnWire),
    Barrier,
    Reset,
}

/// Enum for representing elements that appear in a boxed operation.
#[derive(Clone)]
enum Boxed<'a> {
    Single(&'a PackedInstruction),
    // Multi(MultiBoxElement)
    Multi(&'a PackedInstruction),
}

// This is a temporary work around for a derive issue
// however the behaviour of this PartialEq is not ideal as it only checks the variant type
// because PackedInstructions cannot be compared directly

impl PartialOrd for Boxed<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Boxed<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Boxed::Single(inst1), Boxed::Single(inst2)) => true,
            (Boxed::Multi(inst1), Boxed::Multi(inst2)) => true,
            _ => false,
        }
    }
}

impl Eq for Boxed<'_> {}

impl Ord for Boxed<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Boxed::Single(_), Boxed::Single(_)) => std::cmp::Ordering::Equal,
            (Boxed::Multi(_), Boxed::Multi(_)) => std::cmp::Ordering::Equal,
            (Boxed::Single(_), Boxed::Multi(_)) => std::cmp::Ordering::Less,
            (Boxed::Multi(_), Boxed::Single(_)) => std::cmp::Ordering::Greater,
        }
    }
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

struct Op<'a> {
    instruction: &'a PackedInstruction,
}

#[derive(Default, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum VisualizationElement<'a> {
    #[default]
    Empty, // Marker for no element
    VerticalLine(InputType),
    Input(ElementWireInput<'a>),
    DirectOnWire(OnWire),
    Boxed(Boxed<'a>),
}

/// A representation of a single column (called here a layer) of a visualization matrix
#[derive(Clone, Debug)]
struct VisualizationLayer<'a>(Vec<VisualizationElement<'a>>);

impl<'a> Index<usize> for VisualizationLayer<'a> {
    type Output = VisualizationElement<'a>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<'a> VisualizationLayer<'a> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn push(&mut self, element: VisualizationElement<'a>) {
        self.0.push(element);
    }

    fn drain(
        &mut self,
        range: std::ops::Range<usize>,
    ) -> impl Iterator<Item = VisualizationElement<'a>> + '_ {
        self.0.drain(range)
    }

    fn add_input(&mut self, input: ElementWireInput<'a>, idx: usize) {
        self.0[idx] = VisualizationElement::Input(input);
    }

    /// Adds the required visualization elements to represent the given instruction
    fn add_instruction(&mut self, packed_inst: &'a PackedInstruction, circuit: &CircuitData) {
        match packed_inst.op.view() {
            OperationRef::StandardGate(gate) => self.add_standard_gate(gate, packed_inst, circuit),
            OperationRef::StandardInstruction(std_inst) => {
                self.add_standard_instruction(std_inst, packed_inst, circuit)
            }
            _ => unimplemented!(
                "{}",
                format!(
                    "Visualization is not implemented for instruction of type {:?}",
                    packed_inst.op
                )
            ),
        }
    }

    fn add_controls(&mut self, controls: &HashSet<usize>, range: (usize, usize)) {
        for control in controls {
            if *control == range.0 {
                self.0[*control] =
                    VisualizationElement::DirectOnWire(OnWire::Control(ElementOnWire::Top));
            } else if *control == range.1 {
                self.0[*control] =
                    VisualizationElement::DirectOnWire(OnWire::Control(ElementOnWire::Bot));
            } else {
                self.0[*control] =
                    VisualizationElement::DirectOnWire(OnWire::Control(ElementOnWire::Mid));
            }
        }
    }

    fn add_vertical_lines<I>(&mut self, inst: &'a PackedInstruction, vertical_lines: I)
    where
        I: Iterator<Item = usize>,
    {
        let double_lines = vec![StandardInstruction::Measure];
        let input_type: InputType =
            if let Some(std_instruction) = inst.op.try_standard_instruction() {
                if double_lines.contains(&std_instruction) {
                    InputType::Clbit(None)
                } else {
                    InputType::Qubit(None)
                }
            } else {
                InputType::Qubit(None)
            };
        for vline in vertical_lines {
            self.0[vline] = VisualizationElement::VerticalLine(input_type.clone());
        }
    }

    fn add_standard_gate(
        &mut self,
        gate: StandardGate,
        inst: &'a PackedInstruction,
        circuit: &CircuitData,
    ) {
        let qargs = circuit.get_qargs(inst.qubits);
        let (minima, maxima) = get_instruction_range(
            qargs,
            circuit.get_cargs(inst.clbits),
            circuit.num_qubits(),
        );

        match gate {
            StandardGate::ISwap
            | StandardGate::DCX
            | StandardGate::ECR
            | StandardGate::RXX
            | StandardGate::RYY
            | StandardGate::RZZ
            | StandardGate::RZX
            | StandardGate::XXMinusYY
            | StandardGate::XXPlusYY
            | StandardGate::RCCX
            | StandardGate::RC3X => {
                for q in minima..=maxima {
                    self.0[q.index()] = VisualizationElement::Boxed(Boxed::Multi(inst));
                }
            }
            // StandardGate::Swap | StandardGate::CSwap => {
            //     // qargs[qargs.len() - 1..].iter().map(|q| q.index()).collect()
            //     // for target in targets {
            //     //         if *target == range.0 {
            //     //             self.0[*target] = VisualizationElement::DirectOnWire(OnWire::Swap(
            //     //                 ElementOnWire::Top,
            //     //             ));
            //     //         } else if *target == range.1 {
            //     //             self.0[*target] = VisualizationElement::DirectOnWire(OnWire::Swap(
            //     //                 ElementOnWire::Bot,
            //     //             ));
            //     //         } else {
            //     //             self.0[*target] = VisualizationElement::DirectOnWire(OnWire::Swap(
            //     //                 ElementOnWire::Mid,
            //     //             ));
            //     //         }
            // },
            StandardGate::H | StandardGate::RX | StandardGate::RZ => {
                self.0[qargs[0].index()] = VisualizationElement::Boxed(Boxed::Single(inst));
            },
            StandardGate::CX | StandardGate::CCX => {
                self.0[qargs.last().unwrap().index()] = VisualizationElement::Boxed(Boxed::Single(inst));
                let mut control_indices: HashSet<usize> = HashSet::new();
                if gate.num_ctrl_qubits() > 0 {
                    control_indices.extend(qargs.iter().take(qargs.len() - 1).map(|q| q.index()));
                    self.add_controls(&control_indices, (minima, maxima));
                }

            }
            _ => unimplemented!("{}", format!("{:?} is not supported yet", gate))
        }

    //     let vert_lines = (minima..=maxima)
    //         .filter(|idx| !control_indices.contains(idx) && !box_indices.contains(idx));
    //     self.add_vertical_lines(inst, vert_lines);
    }

    fn add_standard_instruction(
        &mut self,
        std_inst: StandardInstruction,
        inst: &'a PackedInstruction,
        circuit: &CircuitData,
    ) {
        let qargs = circuit.get_qargs(inst.qubits);
        let (minima, maxima) =
            get_instruction_range(qargs, circuit.get_cargs(inst.clbits), circuit.num_qubits());

        match std_inst {
            StandardInstruction::Barrier(_) => {
                for q in qargs {
                    self.0[q.index()] = VisualizationElement::DirectOnWire(OnWire::Barrier);
                }
            }
            StandardInstruction::Reset => {
                for q in qargs {
                    self.0[q.index()] = VisualizationElement::DirectOnWire(OnWire::Reset);
                }
            }
            StandardInstruction::Measure => {
                self.0[qargs.last().unwrap().index()] = VisualizationElement::Boxed(Boxed::Single(inst));
                self.add_vertical_lines(inst, minima + 1..=maxima);
            }
            StandardInstruction::Delay(_) => {
                for q in qargs {
                    self.0[q.index()] = VisualizationElement::Boxed(Boxed::Single(inst));
                }
            }
        }
    }
}

/// A Plain, logical 2D representation of a circuit.
///
/// A dense representation of the circuit of size N * (M + 1), where the first
/// layer(column) represents the qubits and clbits inputs in the circuits, and
/// M is the number of operation layers.
#[derive(Debug, Clone)]
struct VisualizationMatrix<'a> {
    layers: Vec<VisualizationLayer<'a>>,
    circuit: &'a CircuitData,
}

impl<'a> VisualizationMatrix<'a> {
    fn from_circuit(circuit: &'a CircuitData) -> PyResult<Self> {
        let dag = DAGCircuit::from_circuit_data(circuit, false, None, None, None, None)?;

        let mut node_index_to_inst: HashMap<NodeIndex, &PackedInstruction> =
            HashMap::with_capacity(circuit.data().len());
        for (idx, node_index) in dag.op_node_indices(true).enumerate() {
            node_index_to_inst.insert(node_index, &circuit.data()[idx]);
        }

        let inst_layers = build_layers(&dag);

        let num_wires = circuit.num_qubits() + circuit.num_clbits();
        let mut layers = vec![
            VisualizationLayer(vec![VisualizationElement::default(); num_wires]);
            inst_layers.len() + 1
        ]; // Add 1 to account for the inputs layer

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
            for node_index in layer {
                layers[i + 1].add_instruction(node_index_to_inst.get(node_index).unwrap(), circuit);
            }
        }

        Ok(VisualizationMatrix { layers, circuit })
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
struct ElementWire {
    top: String,
    mid: String,
    bot: String,
}

impl Debug for ElementWire {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\n{}\n{}", self.top, self.mid, self.bot)
    }
}

impl ElementWire {
    fn width(&self) -> usize {
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

    fn left_pad_string(s: &mut String, pad_char: char, width: usize) {
        let current_width = s.len();
        if current_width < width {
            let pad_size = width - current_width;
            let pad_str = pad_char.to_string().repeat(pad_size);
            let new_str = format!("{}{}", pad_str, s);
            *s = new_str;
        }
    }

    fn right_pad_string(s: &mut String, pad_char: char, width: usize) {
        let current_width = s.len();
        if current_width < width {
            let pad_size = width - current_width;
            let pad_str = pad_char.to_string().repeat(pad_size);
            let new_str = format!("{}{}", s, pad_str);
            *s = new_str;
        }
    }

    fn pad_string(s: &mut String, pad_char: char, width: usize) {
        let current_width = s.chars().count();
        if current_width < width {
            let pad_size = width - current_width;
            let left_pad = pad_size / 2;
            let right_pad = pad_size - left_pad;
            let left_pad_str = pad_char.to_string().repeat(left_pad);
            let right_pad_str = pad_char.to_string().repeat(right_pad);
            let new_str = format!("{}{}{}", left_pad_str, s, right_pad_str);
            *s = new_str;
        }
    }

    fn pad_wire_left(&mut self, mid_char: char, width: usize) {
        let current_width = self.width();
        if current_width < width {
            Self::left_pad_string(&mut self.top, ' ', width);
            Self::left_pad_string(&mut self.mid, mid_char, width);
            Self::left_pad_string(&mut self.bot, ' ', width);
        }
        //println!("layer width:{}, object width:{} : \n{}\n{}\n{}", width, current_width, self.top, self.mid, self.bot);
    }

    fn pad_wire(&mut self, mid_char: char, width: usize) {
        let current_width = self.width();
        if current_width < width {
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
pub const C_WIRE_CON_TOP: char = '╩';
pub const C_BOT_CON: char = '╥';
pub const Q_LEFT_CON: char = '┤';
pub const Q_RIGHT_CON: char = '├';
pub const CL_LEFT_CON: char = '╡';
pub const CL_RIGHT_CON: char = '╞';
pub const TOP_LEFT_BOX: char = '┌';
pub const TOP_RIGHT_BOX: char = '┐';
pub const BOT_LEFT_BOX: char = '└';
pub const BOT_RIGHT_BOX: char = '┘';
pub const BARRIER: char = '░';
pub const BULLET: char = '■';
pub const CONNECTING_WIRE: char = '│';
pub const CL_CONNECTING_WIRE: char = '║';
pub const Q_Q_CROSSED_WIRE: char = '┼';
pub const Q_CL_CROSSED_WIRE: char = '╪';
pub const CL_CL_CROSSED_WIRE: char = '╬';
pub const CL_Q_CROSSED_WIRE: char = '╫';

struct TextDrawer {
    wires: Vec<Vec<ElementWire>>,
}

impl Index<usize> for TextDrawer {
    type Output = Vec<ElementWire>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.wires[index]
    }
}

impl TextDrawer {
    fn get_label(instruction: &PackedInstruction) -> Option<String> {
        let label = instruction.label();
        let mut instruction_param = String::new();
        for param in instruction.params_view() {
            let param_sub_str = match param {
                Param::Float(f) => format!("{:.2}", f),
                _ => format!("{:?}", param),
            };
            if instruction_param.is_empty() {
                instruction_param = param_sub_str;
            } else {
                instruction_param = format!("{}, {}", instruction_param, param_sub_str);
            }
        }
        // let instruction_param =format!("{:?}",instruction.params_view());
        let instruction_label = match label {
            Some(l) => Some(format!("{}{}{}", " ", l.to_string(), " ")),
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
                            Some(" Iswap ".to_string())
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
                } else if let Some(std_instruction) = instruction.op.try_standard_instruction() {
                    if std_instruction == StandardInstruction::Measure {
                        Some("M".to_string())
                    } else if std_instruction == StandardInstruction::Reset {
                        Some("|0>".to_string())
                    } else if let StandardInstruction::Barrier(_) = std_instruction {
                        Some("░".to_string())
                    } else {
                        // Fallback for non-standard instructions
                        Some(format!(
                            "{}{}{}",
                            " ",
                            instruction.op.name().to_string(),
                            " "
                        ))
                    }
                } else {
                    // Fallback for non-standard operations
                    Some(format!(
                        "{}{}{}",
                        " ",
                        instruction.op.name().to_string(),
                        " "
                    ))
                }
            }
        };
        instruction_label
    }

    fn from_visualization_matrix<'a>(vis_mat: &'a VisualizationMatrix) -> Self {
        let mut wires: Vec<Vec<ElementWire>> = vec![];
        for _ in 0..vis_mat.num_wires() {
            wires.push(vec![]);
        }

        let mut text_drawer = TextDrawer { wires };

        let cregbundle = true;

        let post_processed_vis_mat = {
            if !cregbundle {
                None
            } else {
                let classical_bits_range = (
                    vis_mat.circuit.num_qubits(),
                    vis_mat.circuit.num_qubits() + vis_mat.circuit.num_clbits(),
                );
                let mut temp_vis_mat: VisualizationMatrix<'a> = vis_mat.clone();
                // compare the last classical elements and replace with max in the vector
                for layer in temp_vis_mat.layers.iter_mut() {
                    if let Some(max_element) = layer
                        .drain(classical_bits_range.0..classical_bits_range.1)
                        .max()
                    {
                        layer.push(max_element.clone());
                    } else {
                        layer.push(VisualizationElement::default());
                    }
                }
                Some(temp_vis_mat)
            }
        }
        .unwrap_or(vis_mat.clone());

        let mut ct = 0;
        for layer in &post_processed_vis_mat.layers {
            let layer_wires = Self::draw_layer(layer, vis_mat, ct);
            ct += 1;
            for (i, wire) in layer_wires.iter().enumerate() {
                text_drawer.wires[i].push(wire.clone());
            }
        }

        text_drawer
    }

    fn draw_layer(
        layer: &VisualizationLayer,
        vis_mat: &VisualizationMatrix,
        layer_ind: usize,
    ) -> Vec<ElementWire> {
        let mut wires: Vec<ElementWire> = vec![];
        for (i, element) in layer.0.iter().enumerate() {
            let wire = Self::draw_element(element.clone(), layer, vis_mat.circuit, i);
            wires.push(wire);
        }

        let num_qubits = vis_mat.circuit.num_qubits();

        //let layer_width = wires.iter().map(|w| w.width()).max().unwrap_or(0);
        let mut layer_width = 0;
        for wire in wires.iter() {
            let w = wire.width();
            if w > layer_width {
                layer_width = w;
            }
        }

        for (i, wire) in wires.iter_mut().enumerate() {
            if layer_ind == 0 {
                wire.pad_wire_left(' ', layer_width);
            } else if i < num_qubits {
                wire.pad_wire(Q_WIRE, layer_width);
                // wire.pad_wire('$', layer_width);
            } else {
                wire.pad_wire(C_WIRE, layer_width);
                // wire.pad_wire('$', layer_width);
            }
        }

        wires
    }

    pub fn draw_element(
        vis_ele: VisualizationElement,
        vis_layer: &VisualizationLayer,
        circuit: &CircuitData,
        ind: usize,
    ) -> ElementWire {
        match vis_ele {
            VisualizationElement::Boxed(sub_type) => {
                // implement for cases where the box is on classical wires. The left and right connectors will change
                // from single wired to double wired.

                let top_cases: Vec<VisualizationElement> = vec![
                    VisualizationElement::DirectOnWire(OnWire::Control(ElementOnWire::Top)),
                    VisualizationElement::DirectOnWire(OnWire::Swap(ElementOnWire::Top)),
                    VisualizationElement::DirectOnWire(OnWire::Swap(ElementOnWire::Mid)),
                    VisualizationElement::DirectOnWire(OnWire::Swap(ElementOnWire::Mid)),
                    VisualizationElement::VerticalLine(InputType::Qubit(None)),
                    VisualizationElement::VerticalLine(InputType::Clbit(None)),
                ];

                let bot_cases: Vec<VisualizationElement> = vec![
                    VisualizationElement::DirectOnWire(OnWire::Control(ElementOnWire::Bot)),
                    VisualizationElement::DirectOnWire(OnWire::Swap(ElementOnWire::Bot)),
                    VisualizationElement::DirectOnWire(OnWire::Swap(ElementOnWire::Mid)),
                    VisualizationElement::DirectOnWire(OnWire::Swap(ElementOnWire::Mid)),
                    VisualizationElement::VerticalLine(InputType::Qubit(None)),
                    VisualizationElement::VerticalLine(InputType::Clbit(None)),
                ];

                // if subtype is measurement then classical connectors
                let is_measure = match &sub_type {
                    Boxed::Single(inst) => {
                        if let Some(std_instruction) = inst.op.try_standard_instruction() {
                            if std_instruction == StandardInstruction::Measure {
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    }
                    _ => false,
                };

                let top_con = {
                    if ind >= 1 {
                        if top_cases.contains(&vis_layer.0[ind - 1]) {
                            if is_measure {
                                C_WIRE_CON_TOP
                            } else {
                                TOP_CON
                            }
                        } else {
                            Q_WIRE
                        }
                    } else {
                        Q_WIRE
                    }
                };

                let bot_con = {
                    if ind + 1 < vis_layer.0.len() {
                        if bot_cases.contains(&vis_layer.0[ind + 1]) {
                            if is_measure {
                                C_BOT_CON
                            } else {
                                BOT_CON
                            }
                        } else {
                            Q_WIRE
                        }
                    } else {
                        Q_WIRE
                    }
                };

                match sub_type {
                    Boxed::Single(inst) => {
                        let label = Self::get_label(inst).unwrap_or(" ".to_string());
                        let left_len = (label.len() - 1) / 2;
                        let right_len = label.len() - left_len - 1;
                        ElementWire {
                            top: format!(
                                "{}{}{}{}{}",
                                TOP_LEFT_BOX,
                                Q_WIRE.to_string().repeat(left_len),
                                top_con,
                                Q_WIRE.to_string().repeat(right_len),
                                TOP_RIGHT_BOX
                            ),
                            mid: format!("{}{}{}", Q_LEFT_CON, label, Q_RIGHT_CON),
                            bot: format!(
                                "{}{}{}{}{}",
                                BOT_LEFT_BOX,
                                Q_WIRE.to_string().repeat(left_len),
                                bot_con,
                                Q_WIRE.to_string().repeat(right_len),
                                BOT_RIGHT_BOX
                            ),
                        }
                    }
                    Boxed::Multi(inst) => {
                        let label = Self::get_label(inst).unwrap_or(" ".to_string());
                        // get all the indices affected by this multi-box
                        let qargs = circuit
                            .qargs_interner()
                            .get(inst.qubits)
                            .into_iter()
                            .map(|q| q.index())
                            .collect_vec();
                        let cargs = circuit
                            .cargs_interner()
                            .get(inst.clbits)
                            .into_iter()
                            .map(|c| c.index() + circuit.num_qubits())
                            .collect_vec();
                        let minmax = qargs.iter().chain(cargs.iter()).minmax();
                        let range = match minmax {
                            MinMaxResult::MinMax(min, max) => (*min, *max),
                            MinMaxResult::OneElement(idx) => (*idx, *idx),
                            MinMaxResult::NoElements => {
                                panic!("Encountered an multi-qubit without qubits and clbits")
                            }
                        };
                        let mid = (range.0 + range.1) / 2;

                        let num_affected = {
                            if qargs.contains(&ind) || cargs.contains(&ind) {
                                // Once the packed instruction can handle custom gates, we need to first check the number of controls
                                // and then give an index to the qubit in the multibox an index based on whats left. For example, if the
                                // qubits being affected are [0,1,2,3,4,5] and num controls is 2, then the qubits will be indexed as [C,C,T,T,T,T]
                                // so for qubits [2,3,4,5] the indexes inside the box will be 0,1,2,3 respectively.

                                let temp: String = {
                                    if qargs.contains(&ind) {
                                        let idx = qargs.iter().position(|&x| x == ind).unwrap();
                                        format!("{:^width$}", idx, width = qargs.len())
                                    } else {
                                        " ".to_string()
                                    }
                                };
                                temp
                            } else {
                                " ".to_string()
                            }
                        };

                        let mid_section = if ind == mid {
                            format!(
                                "{:^total_q$} {:^label_len$}",
                                num_affected,
                                label,
                                total_q = qargs.len(),
                                label_len = label.len()
                            )
                        } else {
                            format!(
                                "{:^total_q$} {:^label_len$}",
                                num_affected,
                                " ",
                                total_q = qargs.len(),
                                label_len = label.len()
                            )
                        };

                        let left_len = (mid_section.len() - 1) / 2;
                        let right_len = mid_section.len() - left_len - 1;

                        if ind == range.0 {
                            ElementWire {
                                top: format!(
                                    "{}{}{}{}{}",
                                    TOP_LEFT_BOX,
                                    Q_WIRE.to_string().repeat(left_len),
                                    top_con,
                                    Q_WIRE.to_string().repeat(right_len),
                                    TOP_RIGHT_BOX
                                ),
                                mid: format!("{}{}{}", Q_LEFT_CON, mid_section, Q_RIGHT_CON),
                                bot: format!(
                                    "{}{}{}",
                                    CONNECTING_WIRE,
                                    " ".repeat(mid_section.len()),
                                    CONNECTING_WIRE
                                ),
                            }
                        } else if ind == range.1 {
                            ElementWire {
                                top: format!(
                                    "{}{}{}",
                                    CONNECTING_WIRE,
                                    " ".to_string().repeat(mid_section.len()),
                                    CONNECTING_WIRE
                                ),
                                mid: format!("{}{}{}", Q_LEFT_CON, mid_section, Q_RIGHT_CON),
                                bot: format!(
                                    "{}{}{}{}{}",
                                    BOT_LEFT_BOX,
                                    Q_WIRE.to_string().repeat(left_len),
                                    bot_con,
                                    Q_WIRE.to_string().repeat(right_len),
                                    BOT_RIGHT_BOX
                                ),
                            }
                        } else {
                            ElementWire {
                                top: format!(
                                    "{}{}{}",
                                    CONNECTING_WIRE,
                                    " ".repeat(mid_section.len()),
                                    CONNECTING_WIRE
                                ),
                                mid: format!("{}{}{}", Q_LEFT_CON, mid_section, Q_RIGHT_CON),
                                bot: format!(
                                    "{}{}{}",
                                    CONNECTING_WIRE,
                                    " ".repeat(mid_section.len()),
                                    CONNECTING_WIRE
                                ),
                            }
                        }
                    }
                }
            }
            VisualizationElement::DirectOnWire(on_wire) => {
                let connecting_wire = if ind < circuit.num_qubits() {
                    CONNECTING_WIRE
                } else {
                    CL_CONNECTING_WIRE
                };

                let wire_char: String = match on_wire {
                    OnWire::Control(position) => {
                        if ind < circuit.num_qubits() {
                            BULLET.to_string()
                        } else {
                            C_WIRE_CON_TOP.to_string()
                        }
                    }
                    OnWire::Swap(position) => "X".to_string(),
                    OnWire::Barrier => BARRIER.to_string(),
                    OnWire::Reset => "|0>".to_string(),
                };

                let top: String = match on_wire {
                    OnWire::Control(position) => match position {
                        ElementOnWire::Top => " ".to_string(),
                        ElementOnWire::Mid => format!("{}", connecting_wire),
                        ElementOnWire::Bot => format!("{}", connecting_wire),
                    },
                    OnWire::Swap(position) => match position {
                        ElementOnWire::Top => " ".to_string(),
                        ElementOnWire::Mid => format!("{}", connecting_wire),
                        ElementOnWire::Bot => format!("{}", connecting_wire),
                    },
                    OnWire::Barrier => {
                        format!("{}", BARRIER)
                    }
                    OnWire::Reset => "   ".to_string(),
                };

                let bot: String = match on_wire {
                    OnWire::Control(position) => match position {
                        ElementOnWire::Top => format!("{}", connecting_wire),
                        ElementOnWire::Mid => format!("{}", connecting_wire),
                        ElementOnWire::Bot => " ".to_string(),
                    },
                    OnWire::Swap(position) => match position {
                        ElementOnWire::Top => format!("{}", connecting_wire),
                        ElementOnWire::Mid => format!("{}", connecting_wire),
                        ElementOnWire::Bot => " ".to_string(),
                    },
                    OnWire::Barrier => {
                        format!("{}", BARRIER)
                    }
                    OnWire::Reset => "   ".to_string(),
                };

                let wire = if ind < circuit.num_qubits() {
                    Q_WIRE
                } else {
                    C_WIRE
                };

                ElementWire {
                    top: format!("{}{}{}", " ", top, " "),
                    mid: format!("{}{}{}", wire, wire_char, wire),
                    bot: format!("{}{}{}", " ", bot, " "),
                }
            }
            VisualizationElement::Input(wire_input) => match wire_input {
                ElementWireInput::Qubit(qubit) => {
                    let qubit_name = if let Some(bit_info) = circuit.qubit_indices().get(qubit) {
                        if let Some((register, index)) = bit_info.registers().first() {
                            format!("{}_{}:", register.name(), index)
                        } else {
                            format!("q_{}:", ind)
                        }
                    } else {
                        format!("q_{}:", ind)
                    };
                    ElementWire {
                        top: format!("{}", " ".repeat(qubit_name.len())),
                        mid: format!("{}", qubit_name),
                        bot: format!("{}", " ".repeat(qubit_name.len())),
                    }
                }
                ElementWireInput::Clbit(clbit) => {
                    let clbit_name = if let Some(bit_info) = circuit.clbit_indices().get(clbit) {
                        if let Some((register, index)) = bit_info.registers().first() {
                            format!("{}_{}:", register.name(), index)
                        } else {
                            format!("c_{}:", ind)
                        }
                    } else {
                        format!("c_{}:", ind)
                    };
                    ElementWire {
                        top: format!("{}", " ".repeat(clbit_name.len())),
                        mid: format!("{}", clbit_name),
                        bot: format!("{}", " ".repeat(clbit_name.len())),
                    }
                }
            },
            VisualizationElement::VerticalLine(input_type) => {
                let crossed = {
                    match &input_type {
                        InputType::Qubit(label) => {
                            if ind < circuit.num_qubits() {
                                Q_Q_CROSSED_WIRE
                            } else {
                                Q_CL_CROSSED_WIRE
                            }
                        }
                        InputType::Clbit(label) => {
                            if ind < circuit.num_qubits() {
                                CL_Q_CROSSED_WIRE
                            } else {
                                CL_CL_CROSSED_WIRE
                            }
                        }
                    }
                };
                let connector = match &input_type {
                    InputType::Qubit(_) => CONNECTING_WIRE,
                    InputType::Clbit(_) => CL_CONNECTING_WIRE,
                };
                ElementWire {
                    top: format!("{}", connector),
                    mid: format!("{}", crossed),
                    bot: format!("{}", connector),
                }
            }
            VisualizationElement::Empty => {
                let wire = {
                    if ind < circuit.num_qubits() {
                        Q_WIRE
                    } else {
                        C_WIRE
                    }
                };
                ElementWire {
                    top: format!(" "),
                    mid: format!("{}", wire),
                    bot: format!(" "),
                }
            }
        }
    }

    fn print(&self) {
        // let mut output = String::new();
        // for i in self.wires.iter(){
        //     let top_line: String = i.iter().map(|wire| wire.top.clone()).collect::<Vec<String>>().join("");
        //     let mid_line: String = i.iter().map(|wire| wire.mid.clone()).collect::<Vec<String>>().join("");
        //     let bot_line: String = i.iter().map(|wire| wire.bot.clone()).collect::<Vec<String>>().join("");
        //     output.push_str(&format!("{}\n{}\n{}\n", top_line, mid_line, bot_line));
        // }
        // println!("{}", output);

        // print using merge lines
        let num_wires = self.wires.len();
        for i in 0..num_wires - 1 {
            if i == 0 {
                let top_line = self.wires[i]
                    .iter()
                    .map(|wire| wire.top.clone())
                    .collect::<Vec<String>>()
                    .join("");
                let mid_line = self.wires[i]
                    .iter()
                    .map(|wire| wire.mid.clone())
                    .collect::<Vec<String>>()
                    .join("");
                println!("{}", top_line);
                println!("{}", mid_line);
            }
            let bot_line = self.wires[i]
                .iter()
                .map(|wire| wire.bot.clone())
                .collect::<Vec<String>>()
                .join("");
            let top_line_next = self.wires[i + 1]
                .iter()
                .map(|wire| wire.top.clone())
                .collect::<Vec<String>>()
                .join("");
            let merged_line = Self::merge_lines(&bot_line, &top_line_next, "top");
            println!("{}", merged_line);
            let mid_line_next = self.wires[i + 1]
                .iter()
                .map(|wire| wire.mid.clone())
                .collect::<Vec<String>>()
                .join("");
            println!("{}", mid_line_next);
        }
        let last_index = num_wires - 1;
        let bot_line = self.wires[last_index]
            .iter()
            .map(|wire| wire.bot.clone())
            .collect::<Vec<String>>()
            .join("");
        println!("{}", bot_line);
    }

    pub fn merge_lines(top: &str, bot: &str, icod: &str) -> String {
        let mut ret = String::new();

        for (topc, botc) in top.chars().zip(bot.chars()) {
            if topc == botc {
                ret.push(topc);
            } else if "┼╪".contains(topc) && botc == ' ' {
                ret.push('│');
            } else if topc == ' ' {
                ret.push(botc);
            } else if "┬╥".contains(topc) && " ║│".contains(botc) && icod == "top" {
                ret.push(topc);
            } else if topc == '┬' && botc == ' ' && icod == "bot" {
                ret.push('│');
            } else if topc == '╥' && botc == ' ' && icod == "bot" {
                ret.push('║');
            } else if "┬│".contains(topc) && botc == '═' {
                ret.push('╪');
            } else if "┬│".contains(topc) && botc == '─' {
                ret.push('┼');
            } else if "└┘║│░".contains(topc) && botc == ' ' && icod == "top" {
                ret.push(topc);
            } else if "─═".contains(topc) && botc == ' ' && icod == "top" {
                ret.push(topc);
            } else if "─═".contains(topc) && botc == ' ' && icod == "bot" {
                ret.push(botc);
            } else if "║╥".contains(topc) && botc == '═' {
                ret.push('╬');
            } else if "║╥".contains(topc) && botc == '─' {
                ret.push('╫');
            } else if "║╫╬".contains(topc) && botc == ' ' {
                ret.push('║');
            } else if "│┼╪".contains(topc) && botc == ' ' {
                ret.push('│');
            } else if topc == '└' && botc == '┌' && icod == "top" {
                ret.push('├');
            } else if topc == '┘' && botc == '┐' && icod == "top" {
                ret.push('┤');
            } else if "┐┌".contains(botc) && icod == "top" {
                ret.push('┬');
            } else if "┘└".contains(topc) && botc == '─' && icod == "top" {
                ret.push('┴');
            } else if botc == ' ' && icod == "top" {
                ret.push(topc);
            } else {
                ret.push(botc);
            }
        }

        ret
    }
}
