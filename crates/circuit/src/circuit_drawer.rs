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

use crate::bit::{ShareableClbit, ShareableQubit, ClassicalRegister};
use crate::dag_circuit::DAGCircuit;
use crate::operations::{Operation, OperationRef, Param, StandardGate, StandardInstruction};
use crate::packed_instruction::PackedInstruction;
use crate::{Clbit, Qubit};
use core::panic;
use crossterm::terminal::size;
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
#[pyo3(signature = (circuit, cregbundle=true, mergewires=true, fold=None))]
pub fn py_drawer(
    circuit: QuantumCircuitData,
    cregbundle: bool,
    mergewires: bool,
    fold: Option<usize>,
) -> PyResult<String> {
    draw_circuit(&circuit.data, cregbundle, mergewires, fold)
}

pub fn draw_circuit(
    circuit: &CircuitData,
    cregbundle: bool,
    mergewires: bool,
    fold: Option<usize>,
) -> PyResult<String> {
    let vis_mat = VisualizationMatrix::from_circuit(circuit, cregbundle)?;

    println!("{:?}", vis_mat);
    let circuit_rep = TextDrawer::from_visualization_matrix(&vis_mat, cregbundle);

    let fold = match fold {
        Some(f) => f,
        None => {
            let (term_width, _) = size().unwrap_or((80, 24));
            term_width as usize
        }
    };

    let output = circuit_rep.print(mergewires, fold);

    Ok(output)
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

#[derive(Debug, Clone, PartialEq, Eq)]
enum ElementWireInput<'a> {
    Qubit(&'a ShareableQubit),
    Clbit(&'a ShareableClbit),
    Creg(&'a ClassicalRegister)
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
            (ElementWireInput::Creg(_), _) => Ordering::Greater,
            (_, ElementWireInput::Creg(_)) => Ordering::Less,
        }
    }
}

impl<'a> ElementWireInput<'a> {
    fn get_name(&self, circuit: &CircuitData) -> Option<String> {
        match self {
            Self::Qubit(qubit) => {
                if let Some(bit_info) = circuit.qubit_indices().get(qubit) {
                    let (register, index) = bit_info
                        .registers()
                        .first()
                        .expect("Register cannot be empty");
                    Some(format!("{}_{}: ", register.name(), index))
                } else {
                    None
                }
            }
            ElementWireInput::Clbit(clbit) => {
                if let Some(bit_info) = circuit.clbit_indices().get(clbit) {
                    let (register, index) = bit_info
                        .registers()
                        .first()
                        .expect("Register cannot be empty");
                    Some(format!("{}_{}: ", register.name(), index))
                } else {
                    None
                }
            }
            ElementWireInput::Creg(creg) => Some(format!("{}: {}/", creg.name(), creg.len())),
        }
    }
}

/// Enum for representing elements that appear directly on a wire and how they're connected.
#[derive(Clone, Debug, Copy)]
enum OnWire<'a> {
    Control(&'a PackedInstruction),
    Swap(&'a PackedInstruction),
    Barrier,
    Reset,
}

enum PosOnWire {
    Top,
    Mid,
    Bot,
}

impl OnWire<'_> {
    fn get_position(&self, circuit: &CircuitData, ind: usize) -> PosOnWire {
        match self {
            OnWire::Control(inst) | OnWire::Swap(inst) => {
                let node_qubits = circuit.get_qargs(inst.qubits);
                let node_clbits = circuit.get_cargs(inst.clbits);
                let num_qubits = circuit.num_qubits();
                let range = get_instruction_range(node_qubits, node_clbits, num_qubits);
                if ind == range.0 {
                    PosOnWire::Top
                } else if ind == range.1 {
                    PosOnWire::Bot
                } else {
                    PosOnWire::Mid
                }
            }
            OnWire::Reset | OnWire::Barrier => PosOnWire::Mid,
        }
    }
}

/// Represent elements that appear in a boxed operation.
#[derive(Clone)]
enum Boxed<'a> {
    Single(&'a PackedInstruction),
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


/// Enum for representing the elements stored in a visualization matrix. The elements
/// do not directly implement visualization capabilities, but rather carry enough information
/// to enable visualization later on by the actual drawer.

#[derive(Default, Clone, Debug)]
enum VisualizationElement<'a>{
    #[default]
    /// A wire element without any associated information.
    Empty,
    /// Vertical line (e.g for control or measure) element.
    /// The bool indicates whether this is a single (false) or double (true) line
    VerticalLine(bool),
    /// Circuit input element (qubit, clbit, creg).
    Input(ElementWireInput<'a>),
    /// Element which is drawn without a surrounding box. Used only on qubit wires
    DirectOnWire(OnWire<'a>),
    // Boxed element which can span one or more wires. Used only on qubit wires
    Boxed(Boxed<'a>),
}

/// A representation of a single column (called here a layer) of a visualization matrix
#[derive(Clone)]
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
    fn add_instruction(
        &mut self,
        cregbundle: bool,
        inst: &'a PackedInstruction,
        circuit: &CircuitData,
    ) {
        match inst.op.view() {
            OperationRef::StandardGate(gate) => {
                self.add_standard_gate(gate, inst, circuit);
            }
            OperationRef::StandardInstruction(std_inst) => {
                self.add_standard_instruction(cregbundle, std_inst, inst, circuit);
            }
            _ => unimplemented!(
                "{}",
                format!(
                    "Visualization is not implemented for instruction of type {:?}",
                    inst.op
                )
            ),
        }
    }

    fn add_controls(&mut self, inst: &'a PackedInstruction, controls: &Vec<usize>) {
        for control in controls {
            self.0[*control] = VisualizationElement::DirectOnWire(OnWire::Control(inst));
        }
    }

    fn add_vertical_lines<I>(&mut self, vertical_lines: I, double_line: bool)
    where
        I: Iterator<Item = usize>,
    {
        for vline in vertical_lines {
            self.0[vline] = VisualizationElement::VerticalLine(double_line);
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
            StandardGate::H
            | StandardGate::I
            | StandardGate::X
            | StandardGate::Y
            | StandardGate::Z
            | StandardGate::Phase
            | StandardGate::R
            | StandardGate::RX
            | StandardGate::RY
            | StandardGate::RZ
            | StandardGate::S
            | StandardGate::Sdg
            | StandardGate::SX
            | StandardGate::SXdg
            | StandardGate::T
            | StandardGate::Tdg
            | StandardGate::U
            | StandardGate::U1
            | StandardGate::U2
            | StandardGate::U3 => {
                self.0[qargs[0].index()] = VisualizationElement::Boxed(Boxed::Single(inst));
            },
            StandardGate::CX | StandardGate::CCX => {
                self.0[qargs.last().unwrap().index()] = VisualizationElement::Boxed(Boxed::Single(inst));
                let mut control_indices: HashSet<usize> = HashSet::new();
                if gate.num_ctrl_qubits() > 0 {
                    self.add_controls(
                        inst,
                        &qargs
                            .iter()
                            .take(qargs.len() - 1)
                            .map(|q| q.index())
                            .collect(),
                    );
                }

                let vert_lines = (minima..=maxima)
                    .filter(|idx| !(qargs.iter().map(|q| q.0 as usize)).contains(idx));
                self.add_vertical_lines(vert_lines, false);
            }
            StandardGate::GlobalPhase => {}
            StandardGate::Swap | StandardGate::CSwap => {
                // taking the last 2 elements of qargs
                if gate == StandardGate::CSwap {
                    let control = vec![qargs[0].0 as usize];
                    self.add_controls(inst, &control);
                }
                let swap_qubits = qargs.iter().map(|q| q.0 as usize).rev().take(2);
                for qubit in swap_qubits {
                    self.0[qubit] = VisualizationElement::DirectOnWire(OnWire::Swap(inst));
                }

                let vert_lines = (minima..=maxima)
                    .filter(|idx| !(qargs.iter().map(|q| q.0 as usize)).contains(idx));
                self.add_vertical_lines(vert_lines, false);
            }
            _ => unimplemented!("{}", format!("{:?} is not supported yet", gate))
        }

    //     let vert_lines = (minima..=maxima)
    //         .filter(|idx| !control_indices.contains(idx) && !box_indices.contains(idx));
    //     self.add_vertical_lines(inst, vert_lines);
    }

    fn add_standard_instruction(
        &mut self,
        cregbundle: bool,
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
                self.0[qargs.last().unwrap().index()] =
                    VisualizationElement::Boxed(Boxed::Single(inst));
                let maxima = {
                    if !cregbundle {
                        maxima
                    } else {
                        let shareable_clbit = circuit
                            .clbits()
                            .get(circuit.get_cargs(inst.clbits)[0])
                            .unwrap();
                        let creg = circuit
                            .cregs()
                            .iter()
                            .position(|r| r.contains(shareable_clbit))
                            .unwrap();
                        circuit.num_qubits() + creg
                    }
                };
                self.add_vertical_lines(minima + 1..=maxima, true);
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
///
/// This structure follows a column-major order, where each layer represents a column of the circuit,
#[derive(Clone)]
struct VisualizationMatrix<'a> {
    layers: Vec<VisualizationLayer<'a>>,
    circuit: &'a CircuitData,
}

impl<'a> VisualizationMatrix<'a> {
    fn from_circuit(circuit: &'a CircuitData, bundle_cregs: bool) -> PyResult<Self> {
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
                layers[i + 1].add_instruction(
                    bundle_cregs,
                    node_index_to_inst.get(node_index).unwrap(),
                    circuit,
                );
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

impl<'a> Debug for VisualizationMatrix<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for w in 0..self.num_wires() {
            for l in 0..self.num_layers() {
                let element = &self[l][w];
                let label = match &element {
                    VisualizationElement::Empty => "~",
                    VisualizationElement::VerticalLine(true) => "║",
                    VisualizationElement::VerticalLine(false) => "|",
                    VisualizationElement::Input(input) => match input {
                        ElementWireInput::Qubit(_) => "QR",
                        ElementWireInput::Clbit(_) => "CR",
                        ElementWireInput::Creg(_) => "C/",
                    },
                    VisualizationElement::DirectOnWire(on_wire) => match on_wire {
                        OnWire::Barrier => "░",
                        OnWire::Control(_) => "■",
                        OnWire::Reset => "|0>",
                        OnWire::Swap(_) => "x",
                    },
                    VisualizationElement::Boxed(_) => "[ ]",
                };
                write!(f, "{:^5}", label)?;
            }
            writeln!(f, "")?;
        }
        Ok(())
    }
}

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
        // return the max of all strings
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
    fn get_label(instruction: &PackedInstruction) -> String {
        
        if let Some(std_instruction) = instruction.op.try_standard_instruction() {
            return match std_instruction {
                StandardInstruction::Measure => "M".to_string(),
                StandardInstruction::Reset => "|0>".to_string(),
                StandardInstruction::Barrier(_) => "░".to_string(),
                StandardInstruction::Delay(delay_unit) => {
                    format!("Delay({:?}[{}])", instruction.params, delay_unit)
                }
            };
        }
        
        let custom_label = instruction.label();
        if let Some(standard_gate) = instruction.op.try_standard_gate() {
            let mut label = match standard_gate {
                StandardGate::GlobalPhase => "",
                StandardGate::H => "H",
                StandardGate::I => "I",
                StandardGate::X => "X",
                StandardGate::Y => "Y",
                StandardGate::Z => "Z",
                StandardGate::Phase => "P",
                StandardGate::R => "R",
                StandardGate::RX => "RX",
                StandardGate::RY => "RY",
                StandardGate::RZ => "RZ",
                StandardGate::S => "S",
                StandardGate::Sdg => "Sdg",
                StandardGate::SX => "√X",
                StandardGate::SXdg => "√Xdg",
                StandardGate::T => "T",
                StandardGate::Tdg => "T†",
                StandardGate::U => "U",
                StandardGate::U1 => "U1",
                StandardGate::U2 => "U2",
                StandardGate::U3 => "U3",
                StandardGate::CH => "H",
                StandardGate::CX => "X",
                StandardGate::CY => "Y",
                StandardGate::CZ => "Z",
                StandardGate::DCX => "DCX",
                StandardGate::ECR => "ECR",
                StandardGate::Swap => "",
                StandardGate::ISwap => "Iswap",
                StandardGate::CPhase => "P",
                StandardGate::CRX => "RX",
                StandardGate::CRY => "RY",
                StandardGate::CRZ => "RZ",
                StandardGate::CS => "S",
                StandardGate::CSdg => "Sdg",
                StandardGate::CSX => "√X",
                StandardGate::CU => "U",
                StandardGate::CU1 => "U1",
                StandardGate::CU3 => "U3",
                StandardGate::RXX => "RXX",
                StandardGate::RYY => "RYY",
                StandardGate::RZZ => "RZZ",
                StandardGate::RZX => "RZX",
                StandardGate::XXMinusYY => "XX-YY",
                StandardGate::XXPlusYY => "XX+YY",
                StandardGate::CCX => "X",
                StandardGate::CCZ => "Z",
                StandardGate::CSwap => "",
                StandardGate::RCCX => "RCCX",
                StandardGate::C3X => "X",
                StandardGate::C3SX => "√X",
                StandardGate::RC3X => "RC3X",
            }
            .to_string();

            if custom_label.is_some() && custom_label.unwrap() != label {
                label = custom_label.unwrap().to_string();
            }
            if standard_gate.num_params() > 0 {
                let params = instruction
                    .params_view()
                    .iter()
                    .map(|param| match param {
                        Param::Float(f) => format!("{}", f),
                        _ => format!("{:?}", param),
                    })
                    .join(",");
                label = format!("{}({})", label, params);
            }

            return format!(" {} ", label);
        }

        // Fallback for non-standard operations
        format!(" {} ", instruction.op.name().to_string())
    }

    fn from_visualization_matrix(vis_mat: &VisualizationMatrix, cregbundle: bool) -> Self {
        let mut text_drawer = TextDrawer {
            wires: vec![Vec::new(); vis_mat.num_wires()],
        };

        for (i, layer) in vis_mat.layers.iter().enumerate() {
            let layer_wires = Self::draw_layer(layer, vis_mat, cregbundle, i);
            for (j, wire) in layer_wires.iter().enumerate() {
                text_drawer.wires[j].push(wire.clone());
            }
        }
        text_drawer
    }

    fn get_layer_width(&self, ind: usize) -> usize {
        self.wires
            .iter()
            .map(|wire| wire[ind].width())
            .max()
            .unwrap_or(0)
    }

    fn draw_layer(
        layer: &VisualizationLayer,
        vis_mat: &VisualizationMatrix,
        cregbundle: bool,
        layer_ind: usize,
    ) -> Vec<ElementWire> {
        let mut wires: Vec<ElementWire> = vec![];
        for (i, element) in layer.0.iter().enumerate() {
            let wire = Self::draw_element(element.clone(), layer, vis_mat.circuit, cregbundle,i);
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
        cregbundle: bool,
        ind: usize,
    ) -> ElementWire {
        let is_start = |ind: usize, inst: &PackedInstruction| {
            let (minima, _) = get_instruction_range(
                circuit.get_qargs(inst.qubits),
                circuit.get_cargs(inst.clbits),
                circuit.num_qubits(),
            );
            ind == minima
        };

        let is_end = |ind: usize, inst: &PackedInstruction| {
            let (_, maxima) = get_instruction_range(
                circuit.get_qargs(inst.qubits),
                circuit.get_cargs(inst.clbits),
                circuit.num_qubits(),
            );
            ind == maxima
        };

        match vis_ele {
            VisualizationElement::Boxed(sub_type) => {
                // implement for cases where the box is on classical wires. The left and right connectors will change
                // from single wired to double wired.
                match sub_type {
                    Boxed::Single(inst) => {
                        let mut top_con = Q_WIRE;
                        let mut bot_con = Q_WIRE;
                        if let Some(gate) = inst.op.try_standard_gate() {
                            if gate.is_controlled_gate() {
                                let qargs = circuit.get_qargs(inst.qubits);
                                let (minima, maxima) = get_instruction_range(qargs, &[], 0);
                                if qargs.last().unwrap().index() > minima {
                                    top_con = TOP_CON;
                                }
                                if qargs.last().unwrap().index() < maxima {
                                    bot_con = BOT_CON;
                                }
                            }
                        } else if let Some(std_inst) = inst.op.try_standard_instruction() {
                            if std_inst == StandardInstruction::Measure {
                                bot_con = C_BOT_CON;
                            }
                        }
                        let label = Self::get_label(inst);
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
                        let label = Self::get_label(inst);
                        let qargs = circuit.get_qargs(inst.qubits);
                        let (minima, maxima) = get_instruction_range(qargs, &[], 0);
                        let mid = (minima + maxima) / 2;

                        let num_affected =
                            if let Some(idx) = qargs.iter().position(|&x| x.index() == ind) {
                                format!("{:^width$}", idx, width = qargs.len())
                            } else {
                                " ".to_string()
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
                        let top = if ind == minima {
                            format!(
                                "{}{}{}{}{}",
                                TOP_LEFT_BOX,
                                Q_WIRE.to_string().repeat(left_len),
                                Q_WIRE,
                                Q_WIRE.to_string().repeat(right_len),
                                TOP_RIGHT_BOX
                            )
                        } else {
                            format!(
                                "{}{}{}",
                                CONNECTING_WIRE,
                                " ".repeat(mid_section.len()),
                                CONNECTING_WIRE
                            )
                        };
                        let mid = format!("{}{}{}", Q_LEFT_CON, mid_section, Q_RIGHT_CON);
                        let bot = if ind == maxima {
                            format!(
                                "{}{}{}{}{}",
                                BOT_LEFT_BOX,
                                Q_WIRE.to_string().repeat(left_len),
                                Q_WIRE,
                                Q_WIRE.to_string().repeat(right_len),
                                BOT_RIGHT_BOX
                            )
                        } else {
                            format!(
                                "{}{}{}",
                                CONNECTING_WIRE,
                                " ".repeat(mid_section.len()),
                                CONNECTING_WIRE
                            )
                        };

                        ElementWire { top, mid, bot }
                    }
                }
            }
            VisualizationElement::DirectOnWire(on_wire) => {
                let (top, wire_symbol, bot) = match on_wire {
                    OnWire::Control(inst) => (
                        if is_start(ind, inst) {
                            "".to_string()
                        } else {
                            CONNECTING_WIRE.to_string()
                        },
                        BULLET.to_string(),
                        if is_end(ind, inst) {
                            "".to_string()
                        } else {
                            CONNECTING_WIRE.to_string()
                        },
                    ),
                    OnWire::Swap(inst) => (
                        if is_start(ind, inst) {
                            "".to_string()
                        } else {
                            CONNECTING_WIRE.to_string()
                        },
                        "X".to_string(),
                        if is_end(ind, inst) {
                            "".to_string()
                        } else {
                            CONNECTING_WIRE.to_string()
                        },
                    ),
                    OnWire::Barrier => (
                        BARRIER.to_string(),
                        BARRIER.to_string(),
                        BARRIER.to_string(),
                    ),
                    OnWire::Reset => ("   ".to_string(), "|0>".to_string(), "   ".to_string()),
                };

                ElementWire {
                    top: format!(" {} ", top),
                    mid: format!("{}{}{}", Q_WIRE, wire_symbol, Q_WIRE),
                    bot: format!(" {} ", bot),
                }
            }
            VisualizationElement::Input(wire_input) => {
                let wire_name = wire_input.get_name(circuit).unwrap();
                ElementWire {
                    top: format!("{}", " ".repeat(wire_name.len())),
                    mid: format!("{}", wire_name),
                    bot: format!("{}", " ".repeat(wire_name.len())),
                }
            }
            VisualizationElement::VerticalLine(is_double) => {
                let (connector, crossed) = if is_double {
                    (
                        CL_CONNECTING_WIRE,
                        if ind < circuit.num_qubits() {
                            CL_Q_CROSSED_WIRE
                        } else {
                            CL_CL_CROSSED_WIRE
                        },
                    )
                } else {
                    (
                        CONNECTING_WIRE,
                        if ind < circuit.num_qubits() {
                            Q_Q_CROSSED_WIRE
                        } else {
                            Q_CL_CROSSED_WIRE
                        },
                    )
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

    fn print(&self, mergewires: bool, fold: usize) -> String {
        let ranges: Vec<(usize, usize)> = {
            let mut temp_ranges = vec![];
            let mut layer_counter: usize = 1;
            while layer_counter < self.wires[0].len() {
                let mut total_width: usize = 0;
                total_width += self.get_layer_width(0);

                let start = layer_counter;
                while total_width <= fold && layer_counter < self.wires[0].len() {
                    total_width += self.get_layer_width(layer_counter);
                    layer_counter += 1;
                }
                let end = layer_counter;
                temp_ranges.push((start, end));
            }
            temp_ranges
        };

        let mut output = String::new();

        for (j, (start, end)) in ranges.iter().enumerate() {
            if !mergewires {
                for (i, element) in self.wires.iter().enumerate() {
                    let mut top_line: String = element
                        .iter()
                        .enumerate()
                        .filter(|(idx, _)| (*idx >= *start && *idx < *end) || *idx == 0) // Fix here
                        .map(|(_, wire)| wire.top.clone()) // And here
                        .collect::<Vec<String>>()
                        .join("");
                    let mut mid_line: String = element
                        .iter()
                        .enumerate()
                        .filter(|(idx, _)| (*idx >= *start && *idx < *end) || *idx == 0) // Fix here too
                        .map(|(_, wire)| wire.mid.clone()) // And here
                        .collect::<Vec<String>>()
                        .join("");
                    let mut bot_line: String = element
                        .iter()
                        .enumerate()
                        .filter(|(idx, _)| (*idx >= *start && *idx < *end) || *idx == 0) // And here
                        .map(|(_, wire)| wire.bot.clone()) // And here
                        .collect::<Vec<String>>()
                        .join("");
                    top_line.push_str(if j == ranges.len() - 1 { "" } else { "»" });
                    mid_line.push_str(if j == ranges.len() - 1 { "" } else { "»" });
                    bot_line.push_str(if j == ranges.len() - 1 { "" } else { "»" });
                    output.push_str(&format!("{}\n{}\n{}\n", top_line, mid_line, bot_line));
                }
            } else {
                let num_wires = self.wires.len();
                for i in 0..num_wires - 1 {
                    if i == 0 {
                        let mut top_line = self.wires[i]
                            .iter()
                            .enumerate()
                            .filter(|(idx, _)| (*idx >= *start && *idx < *end) || *idx == 0) // Fix here
                            .map(|(_, wire)| wire.top.clone()) // And here
                            .collect::<Vec<String>>()
                            .join("");
                        top_line.push_str(if j == ranges.len() - 1 { "" } else { "»" });
                        let mut mid_line = self.wires[i]
                            .iter()
                            .enumerate()
                            .filter(|(idx, _)| (*idx >= *start && *idx < *end) || *idx == 0) // Fix here
                            .map(|(_, wire)| wire.mid.clone()) // And here
                            .collect::<Vec<String>>()
                            .join("");
                        mid_line.push_str(if j == ranges.len() - 1 { "" } else { "»" });
                        output.push_str(&format!("{}\n{}\n", top_line, mid_line));
                    }
                    let mut bot_line = self.wires[i]
                        .iter()
                        .enumerate()
                        .filter(|(idx, _)| (*idx >= *start && *idx < *end) || *idx == 0) // Fix here
                        .map(|(_, wire)| wire.bot.clone()) // And here
                        .collect::<Vec<String>>()
                        .join("");

                    bot_line.push_str(if j == ranges.len() - 1 { "" } else { "»" });

                    let mut top_line_next = self.wires[i + 1]
                        .iter()
                        .enumerate()
                        .filter(|(idx, _)| (*idx >= *start && *idx < *end) || *idx == 0) // Fix here
                        .map(|(_, wire)| wire.top.clone()) // And here
                        .collect::<Vec<String>>()
                        .join("");

                    top_line_next.push_str(if j == ranges.len() - 1 { "" } else { "»" });
                    let mut merged_line = Self::merge_lines(&bot_line, &top_line_next, "top");
                    output.push_str(&format!("{}\n", merged_line));
                    let mut mid_line_next = self.wires[i + 1]
                        .iter()
                        .enumerate()
                        .filter(|(idx, _)| (*idx >= *start && *idx < *end) || *idx == 0) // Fix here
                        .map(|(_, wire)| wire.mid.clone()) // And here
                        .collect::<Vec<String>>()
                        .join("");
                    mid_line_next.push_str(if j == ranges.len() - 1 { "" } else { "»" });
                    output.push_str(&format!("{}\n", mid_line_next));
                }
                let last_index = num_wires - 1;
                let bot_line = self.wires[last_index].iter().map(|wire| wire.bot.clone()).collect::<Vec<String>>().join("");
                let top_line_next = self.wires[last_index + 1].iter().map(|wire| wire.top.clone()).collect::<Vec<String>>().join("");
                let merged_line = Self::merge_lines(&bot_line, &top_line_next, "top");
                output.push_str(&format!("{}\n", merged_line));
                let mid_line_next = self.wires[last_index + 1].iter().map(|wire| wire.mid.clone()).collect::<Vec<String>>().join("");
                output.push_str(&format!("{}\n", mid_line_next));
                let bot_line = self.wires[last_index].iter().map(|wire| wire.bot.clone()).collect::<Vec<String>>().join("");
                output.push_str(&format!("{}\n", bot_line))
            }
        }
        output
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
