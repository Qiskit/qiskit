// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::bit::{ClassicalRegister, ShareableClbit, ShareableQubit};
use crate::circuit_data::CircuitData;
use crate::converters::QuantumCircuitData;
use crate::dag_circuit::DAGCircuit;
use crate::dag_circuit::NodeType;
use crate::operations::{Operation, OperationRef, Param, StandardGate, StandardInstruction};
use crate::packed_instruction::PackedInstruction;
use crate::{Clbit, Qubit};
use crossterm::terminal::size;
use hashbrown::{HashMap, HashSet};
use itertools::{Itertools, MinMaxResult};
use pyo3::prelude::*;
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use std::fmt::Debug;
use std::ops::Index;

/// Draw the [CircuitData] object as string.
///
/// # Arguments:
///
/// * circuit: The CircuitData to draw.
/// * cregbundle: If true, classical bits of classical registers are bundled into one wire.
/// * mergewires: If true, adjacent wires are merged when rendered.
/// * fold: If not None, applies line wrapping using the specified amount.
///
/// # Returns:
///
/// The String representation of the circuit.
pub fn draw_circuit(
    circuit: &CircuitData,
    cregbundle: bool,
    mergewires: bool,
    fold: Option<usize>,
) -> PyResult<String> {
    let vis_mat = VisualizationMatrix::from_circuit(circuit, cregbundle)?;

    let text_drawer = TextDrawer::from_visualization_matrix(&vis_mat, cregbundle);

    let fold = match fold {
        Some(f) => f,
        None => {
            let (term_width, _) = size().unwrap_or((80, 24));
            term_width as usize
        }
    };

    Ok(text_drawer.draw(mergewires, fold))
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
            let NodeType::Operation(instruction_to_insert) = &dag.dag()[node_index] else {
                continue;
            };

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
        .chain(node_clbits.iter().max().map(|c| c.index() + num_qubits));

    match indices.minmax() {
        MinMaxResult::MinMax(min, max) => (min, max),
        MinMaxResult::OneElement(idx) => (idx, idx),
        MinMaxResult::NoElements => panic!("Encountered an instruction without qubits and clbits"),
    }
}

#[derive(Clone, PartialEq, Eq)]
enum WireInputElement<'a> {
    Qubit(&'a ShareableQubit),
    Clbit(&'a ShareableClbit),
    Creg(&'a ClassicalRegister),
}

impl WireInputElement<'_> {
    fn get_name(&self, circuit: &CircuitData) -> Option<String> {
        match self {
            Self::Qubit(qubit) => {
                let bit_info = circuit
                    .qubit_indices()
                    .get(qubit)
                    .expect("Bit should have location info");
                if !bit_info.registers().is_empty() {
                    let (register, index) = bit_info
                        .registers()
                        .first()
                        .expect("Register cannot be empty");
                    if !register.is_empty() {
                        Some(format!("{}_{}: ", register.name(), index))
                    } else {
                        Some(format!("{}: ", register.name()))
                    }
                } else {
                    None
                }
            }
            WireInputElement::Clbit(clbit) => {
                let bit_info = circuit
                    .clbit_indices()
                    .get(clbit)
                    .expect("Bit should have location info");

                if !bit_info.registers().is_empty() {
                    let (register, index) = bit_info
                        .registers()
                        .first()
                        .expect("Register cannot be empty");
                    if !register.is_empty() {
                        Some(format!("{}_{}: ", register.name(), index))
                    } else {
                        Some(format!("{}: ", register.name()))
                    }
                } else {
                    None
                }
            }
            WireInputElement::Creg(creg) => Some(format!("{}: {}/", creg.name(), creg.len())),
        }
    }
}

/// Enum for representing elements that appear directly on a wire and how they're connected.
#[derive(Clone, Debug, Copy)]
enum OnWireElement<'a> {
    Control(&'a PackedInstruction),
    Swap(&'a PackedInstruction),
    Barrier,
    Reset,
}

/// Represent elements that appear in a boxed operation.
#[derive(Clone)]
enum BoxedElement<'a> {
    Single(&'a PackedInstruction),
    Multi(&'a PackedInstruction),
}

/// Enum for representing the elements stored in the visualization matrix. The elements
/// do not directly implement visualization capabilities, but rather carry enough information
/// to enable visualization later on by the actual drawer.
#[derive(Default, Clone)]
enum VisualizationElement<'a> {
    #[default]
    /// A wire element without any associated information.
    Empty,
    /// A Vertical line element, belonging to an instruction (e.g of a controlled gate or a measure).
    VerticalLine(&'a PackedInstruction),
    /// A circuit input element (qubit, clbit, creg).
    Input(WireInputElement<'a>),
    /// An element which is drawn without a surrounding box. Used only on qubit wires.
    DirectOnWire(OnWireElement<'a>),
    // A boxed element which can span one or more wires. Used only on qubit wires.
    Boxed(BoxedElement<'a>),
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

    fn add_input(&mut self, input: WireInputElement<'a>, idx: usize) {
        self.0[idx] = VisualizationElement::Input(input);
    }

    /// Adds the required visualization elements to represent the given instruction
    fn add_instruction(
        &mut self,
        cregbundle: bool,
        inst: &'a PackedInstruction,
        circuit: &CircuitData,
        clbit_map: &[usize],
    ) {
        match inst.op.view() {
            OperationRef::StandardGate(gate) => {
                self.add_standard_gate(gate, inst, circuit);
            }
            OperationRef::StandardInstruction(std_inst) => {
                self.add_standard_instruction(cregbundle, std_inst, inst, circuit, clbit_map);
            }
            OperationRef::Unitary(_) => {
                self.add_unitary_gate(inst, circuit);
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
            self.0[*control] = VisualizationElement::DirectOnWire(OnWireElement::Control(inst));
        }
    }

    fn add_vertical_lines<I>(&mut self, vertical_lines: I, inst: &'a PackedInstruction)
    where
        I: Iterator<Item = usize>,
    {
        for vline in vertical_lines {
            self.0[vline] = VisualizationElement::VerticalLine(inst);
        }
    }

    fn add_standard_gate(
        &mut self,
        gate: StandardGate,
        inst: &'a PackedInstruction,
        circuit: &CircuitData,
    ) {
        let qargs = circuit.get_qargs(inst.qubits);
        let (minima, maxima) = get_instruction_range(qargs, &[], 0);

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
                    self.0[q] = VisualizationElement::Boxed(BoxedElement::Multi(inst));
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
                self.0[qargs[0].index()] = VisualizationElement::Boxed(BoxedElement::Single(inst));
            }
            StandardGate::CH
            | StandardGate::CX
            | StandardGate::CY
            | StandardGate::CZ
            | StandardGate::CRX
            | StandardGate::CRY
            | StandardGate::CRZ
            | StandardGate::CS
            | StandardGate::CSdg
            | StandardGate::CSX
            | StandardGate::CU
            | StandardGate::CU1
            | StandardGate::CU3
            | StandardGate::CCX
            | StandardGate::CCZ
            | StandardGate::C3X
            | StandardGate::C3SX
            | StandardGate::CPhase => {
                self.0[qargs.last().unwrap().index()] =
                    VisualizationElement::Boxed(BoxedElement::Single(inst));
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
                self.add_vertical_lines(vert_lines, inst);
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
                    self.0[qubit] = VisualizationElement::DirectOnWire(OnWireElement::Swap(inst));
                }

                let vert_lines = (minima..=maxima)
                    .filter(|idx| !(qargs.iter().map(|q| q.0 as usize)).contains(idx));
                self.add_vertical_lines(vert_lines, inst);
            }
        }
    }

    fn add_standard_instruction(
        &mut self,
        cregbundle: bool,
        std_inst: StandardInstruction,
        inst: &'a PackedInstruction,
        circuit: &CircuitData,
        clbit_map: &[usize],
    ) {
        let qargs = circuit.get_qargs(inst.qubits);
        let (minima, mut maxima) =
            get_instruction_range(qargs, circuit.get_cargs(inst.clbits), circuit.num_qubits());

        match std_inst {
            StandardInstruction::Barrier(_) => {
                for q in qargs {
                    self.0[q.index()] = VisualizationElement::DirectOnWire(OnWireElement::Barrier);
                }
            }
            StandardInstruction::Reset => {
                for q in qargs {
                    self.0[q.index()] = VisualizationElement::DirectOnWire(OnWireElement::Reset);
                }
            }
            StandardInstruction::Measure => {
                self.0[qargs.last().unwrap().index()] =
                    VisualizationElement::Boxed(BoxedElement::Single(inst));

                // Some bits may be bundled, so we need to map the Clbit index to the proper wire index
                if cregbundle {
                    maxima = clbit_map[circuit
                        .get_cargs(inst.clbits)
                        .first()
                        .expect("Measure should have a clbit arg")
                        .index()];
                }
                self.add_vertical_lines(minima + 1..=maxima, inst);
            }
            StandardInstruction::Delay(_) => {
                for q in qargs {
                    self.0[q.index()] = VisualizationElement::Boxed(BoxedElement::Single(inst));
                }
            }
        }
    }

    fn add_unitary_gate(&mut self, inst: &'a PackedInstruction, circuit: &CircuitData) {
        let qargs = circuit.get_qargs(inst.qubits);
        if qargs.len() == 1 {
            self.0[qargs.first().unwrap().index()] =
                VisualizationElement::Boxed(BoxedElement::Single(inst));
        } else {
            let (minima, maxima) = get_instruction_range(qargs, &[], 0);

            for q in minima..=maxima {
                self.0[q] = VisualizationElement::Boxed(BoxedElement::Multi(inst));
            }
        }
    }
}

/// A Plain, logical 2D representation of a circuit.
///
/// A dense representation of the circuit of size N * (M + 1), where the first
/// layer(column) represents the qubits and clbits inputs in the circuits,
/// M is the number of operation layers and N is the number of
/// qubits and clbits (or cregs if bundling is enabled).
///
/// This structure follows a column-major order, where each layer represents a column of the circuit,
#[derive(Clone)]
struct VisualizationMatrix<'a> {
    /// Layers stored in the matrix.
    layers: Vec<VisualizationLayer<'a>>,
    /// A reference to the circuit this matrix was constructed from.
    circuit: &'a CircuitData,
    /// A mapping from instruction's Clbit indices to the visualization matrix wires,
    /// to be used when mapping clbits to bit of bundled cregs
    clbit_map: Vec<usize>,
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

        let num_wires = circuit.num_qubits()
            + if !bundle_cregs {
                circuit.num_clbits()
            } else {
                // Anonymous bits are not bundled so need to be counted explicitly
                circuit.cregs_data().len() + circuit.num_clbits()
                    - circuit
                        .cregs_data()
                        .registers()
                        .iter()
                        .map(|r| r.len())
                        .sum::<usize>()
            };

        let mut layers = vec![
            VisualizationLayer(vec![VisualizationElement::default(); num_wires]);
            inst_layers.len() + 1
        ]; // Add 1 to account for the inputs layer

        let input_layer = layers.first_mut().unwrap();
        let mut input_idx = 0;
        for qubit in circuit.qubits().objects() {
            input_layer.add_input(WireInputElement::Qubit(qubit), input_idx);
            input_idx += 1;
        }

        let mut visited_cregs: HashSet<&ClassicalRegister> = HashSet::new();
        let mut clbit_map: Vec<usize> = Vec::new();
        for clbit in circuit.clbits().objects() {
            if bundle_cregs {
                let bit_location = circuit
                    .clbit_indices()
                    .get(clbit)
                    .expect("Bit should have bit info");
                if !bit_location.registers().is_empty() {
                    let creg = &bit_location
                        .registers()
                        .first()
                        .expect("Registers should not be empty")
                        .0;

                    if visited_cregs.contains(creg) {
                        clbit_map.push(input_idx - 1);
                    } else {
                        input_layer.add_input(WireInputElement::Creg(creg), input_idx);
                        visited_cregs.insert(creg);
                        clbit_map.push(input_idx);
                        input_idx += 1;
                    }
                    continue;
                }
            }

            input_layer.add_input(WireInputElement::Clbit(clbit), input_idx);
            clbit_map.push(input_idx);
            input_idx += 1;
        }

        for (i, layer) in inst_layers.iter().enumerate() {
            for node_index in layer {
                layers[i + 1].add_instruction(
                    bundle_cregs,
                    node_index_to_inst.get(node_index).unwrap(),
                    circuit,
                    &clbit_map,
                );
            }
        }

        Ok(VisualizationMatrix {
            layers,
            circuit,
            clbit_map,
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

impl Debug for VisualizationMatrix<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for w in 0..self.num_wires() {
            for l in 0..self.num_layers() {
                let element = &self[l][w];
                let label = match &element {
                    VisualizationElement::Empty => "~",
                    VisualizationElement::VerticalLine(inst) => {
                        let mut line = "|";
                        if let Some(std_inst) = inst.op.try_standard_instruction() {
                            if std_inst == StandardInstruction::Measure {
                                line = "║";
                            }
                        }
                        line
                    }
                    VisualizationElement::Input(input) => match input {
                        WireInputElement::Qubit(_) => "QR",
                        WireInputElement::Clbit(_) => "CR",
                        WireInputElement::Creg(_) => "C/",
                    },
                    VisualizationElement::DirectOnWire(on_wire) => match on_wire {
                        OnWireElement::Barrier => "░",
                        OnWireElement::Control(_) => "■",
                        OnWireElement::Reset => "|0>",
                        OnWireElement::Swap(_) => "x",
                    },
                    VisualizationElement::Boxed(_) => "[ ]",
                };
                write!(f, "{:^5}", label)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// TO DO: better name for the struct
#[derive(Clone)]
struct TextWireElement {
    top: String,
    mid: String,
    bot: String,
}

impl Debug for TextWireElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\n{}\n{}", self.top, self.mid, self.bot)
    }
}

impl TextWireElement {
    fn width(&self) -> usize {
        self.top
            .chars()
            .count()
            .max(self.mid.chars().count())
            .max(self.bot.chars().count())
    }

    fn left_pad_string(s: &mut String, pad_char: char, width: usize) {
        let current_width = s.len();
        if current_width < width {
            let pad_size = width - current_width;
            s.reserve(pad_size);
            (0..pad_size).for_each(|_| s.insert(0, pad_char));
        }
    }

    fn pad_string(s: &mut String, pad_char: char, width: usize) {
        let current_width = s.chars().count();
        if current_width < width {
            let pad_size = width - current_width;
            let left_pad = pad_size / 2;
            let right_pad = pad_size - left_pad;
            s.reserve(pad_size);
            (0..left_pad).for_each(|_| s.insert(0, pad_char));
            (0..right_pad).for_each(|_| s.push(pad_char));
        }
    }

    fn pad_wire_left(&mut self, mid_char: char, width: usize) {
        let current_width = self.width();
        if current_width < width {
            Self::left_pad_string(&mut self.top, ' ', width);
            Self::left_pad_string(&mut self.mid, mid_char, width);
            Self::left_pad_string(&mut self.bot, ' ', width);
        }
    }

    fn pad_wire(&mut self, mid_char: char, width: usize) {
        let current_width = self.width();
        if current_width < width {
            Self::pad_string(&mut self.top, ' ', width);
            Self::pad_string(&mut self.mid, mid_char, width);
            Self::pad_string(&mut self.bot, ' ', width);
        }
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

/// Textual representation of the circuit
struct TextDrawer {
    /// The array of textural wire elements corresponding to the visualization elements
    wires: Vec<Vec<TextWireElement>>,
}

impl Index<usize> for TextDrawer {
    type Output = Vec<TextWireElement>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.wires[index]
    }
}

impl TextDrawer {
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

    fn get_label(instruction: &PackedInstruction) -> String {
        match instruction.op.view() {
            OperationRef::StandardInstruction(std_instruction) => {
                match std_instruction {
                    StandardInstruction::Measure => "M".to_string(),
                    StandardInstruction::Reset => "|0>".to_string(),
                    StandardInstruction::Barrier(_) => BARRIER.to_string(),
                    StandardInstruction::Delay(delay_unit) => {
                        match instruction.params_view().first().unwrap() {
                            Param::Float(duration) => {
                                format!("Delay({}[{}])", duration, delay_unit)
                            }
                            Param::ParameterExpression(expr) => {
                                format!("Delay({}[{}])", expr, delay_unit)
                            }
                            Param::Obj(obj) => format!("Delay({:?}[{}])", obj, delay_unit), // TODO: extract the int
                        }
                    }
                }
            }
            OperationRef::StandardGate(standard_gate) => {
                static STANDARD_GATE_LABELS: [&str; crate::operations::STANDARD_GATE_SIZE] = [
                    "", "H", "I", "X", "Y", "Z", "P", "R", "Rx", "Ry", "Rz", "S", "Sdg", "√X",
                    "√Xdg", "T", "Tdg", "U", "U1", "U2", "U3", "H", "X", "Y", "Z", "Dcx", "Ecr",
                    "", "Iswap", "P", "Rx", "Ry", "Rz", "S", "Sdg", "Sx", "U", "U1", "U3", "Rxx",
                    "Ryy", "Rzz", "Rzx", "XX-YY", "XX+YY", "X", "Z", "", "Rccx", "X", "Sx",
                    "Rcccx",
                ];

                let mut label = STANDARD_GATE_LABELS[standard_gate as usize].to_string();

                if let Some(custom_label) = instruction.label.clone() {
                    if *custom_label != label {
                        label = *custom_label;
                    }
                }

                if standard_gate.num_params() > 0 {
                    let params = instruction
                        .params_view()
                        .iter()
                        .map(|param| match param {
                            Param::Float(f) => f.to_string(),
                            _ => format!("{:?}", param),
                        })
                        .join(",");
                    label = format!("{}({})", label, params);
                }
                label
            }
            OperationRef::Unitary(_) => *instruction
                .label
                .clone()
                .unwrap_or(Box::new(" Unitary ".to_string())),
            // Fallback for non-standard operations
            _ => format!(" {} ", instruction.op.name()),
        }
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
    ) -> Vec<TextWireElement> {
        let mut wires: Vec<TextWireElement> = layer
            .0
            .iter()
            .enumerate()
            .map(|(i, element)| Self::draw_element(element, vis_mat, cregbundle, i))
            .collect();

        let num_qubits = vis_mat.circuit.num_qubits();

        let layer_width = wires.iter().map(|wire| wire.width()).max().unwrap_or(0);

        for (i, wire) in wires.iter_mut().enumerate() {
            if layer_ind == 0 {
                wire.pad_wire_left(' ', layer_width);
            } else if i < num_qubits {
                wire.pad_wire(Q_WIRE, layer_width);
            } else {
                wire.pad_wire(C_WIRE, layer_width);
            }
        }
        wires
    }

    pub fn draw_element(
        vis_ele: &VisualizationElement,
        vis_mat: &VisualizationMatrix,
        cregbundle: bool,
        ind: usize,
    ) -> TextWireElement {
        let circuit = vis_mat.circuit;
        let (top, mid, bot);
        match vis_ele {
            VisualizationElement::Boxed(sub_type) => {
                // implement for cases where the box is on classical wires. The left and right connectors will change
                // from single wired to double wired.
                match sub_type {
                    BoxedElement::Single(inst) => {
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
                        let label = format!(" {} ", Self::get_label(inst));
                        let label_len = label.chars().count(); // To count unicode chars properly (e.g. in √X)
                        let left_len = (label_len - 1) / 2;
                        let right_len = label_len - left_len - 1;
                        top = format!(
                            "{}{}{}{}{}",
                            TOP_LEFT_BOX,
                            Q_WIRE.to_string().repeat(left_len),
                            top_con,
                            Q_WIRE.to_string().repeat(right_len),
                            TOP_RIGHT_BOX
                        );
                        mid = format!("{}{}{}", Q_LEFT_CON, label, Q_RIGHT_CON);
                        bot = format!(
                            "{}{}{}{}{}",
                            BOT_LEFT_BOX,
                            Q_WIRE.to_string().repeat(left_len),
                            bot_con,
                            Q_WIRE.to_string().repeat(right_len),
                            BOT_RIGHT_BOX
                        );
                    }
                    BoxedElement::Multi(inst) => {
                        let label = format!(" {} ", Self::get_label(inst));
                        let label_len = label.chars().count(); // To count unicode chars properly (e.g. in √X)
                        let qargs = circuit.get_qargs(inst.qubits);
                        let (minima, maxima) = get_instruction_range(qargs, &[], 0);
                        let mid_idx = (minima + maxima) / 2;

                        let num_affected =
                            if let Some(idx) = qargs.iter().position(|&x| x.index() == ind) {
                                format!("{:^width$}", idx, width = qargs.len())
                            } else {
                                " ".to_string()
                            };

                        let mid_section = if ind == mid_idx {
                            format!(
                                "{:^total_q$}{:^label_len$}",
                                num_affected,
                                label,
                                total_q = qargs.len(),
                                label_len = label_len
                            )
                        } else {
                            format!(
                                "{:^total_q$}{:^label_len$}",
                                num_affected,
                                " ",
                                total_q = qargs.len(),
                                label_len = label_len
                            )
                        };

                        let left_len = (mid_section.len() - 1) / 2;
                        let right_len = mid_section.len() - left_len - 1;
                        top = if ind == minima {
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
                        mid = format!("{}{}{}", Q_LEFT_CON, mid_section, Q_RIGHT_CON);
                        bot = if ind == maxima {
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
                    }
                }
            }
            VisualizationElement::DirectOnWire(on_wire) => {
                let (wire_top, wire_symbol, wire_bot) = match on_wire {
                    OnWireElement::Control(inst) => {
                        let (minima, maxima) =
                            get_instruction_range(circuit.get_qargs(inst.qubits), &[], 0);
                        (
                            if ind == minima {
                                " ".to_string()
                            } else {
                                CONNECTING_WIRE.to_string()
                            },
                            BULLET.to_string(),
                            if ind == maxima {
                                " ".to_string()
                            } else {
                                CONNECTING_WIRE.to_string()
                            },
                        )
                    }
                    OnWireElement::Swap(inst) => {
                        let (minima, maxima) =
                            get_instruction_range(circuit.get_qargs(inst.qubits), &[], 0);
                        (
                            if ind == minima {
                                " ".to_string()
                            } else {
                                CONNECTING_WIRE.to_string()
                            },
                            "X".to_string(),
                            if ind == maxima {
                                " ".to_string()
                            } else {
                                CONNECTING_WIRE.to_string()
                            },
                        )
                    }
                    OnWireElement::Barrier => (
                        BARRIER.to_string(),
                        BARRIER.to_string(),
                        BARRIER.to_string(),
                    ),
                    OnWireElement::Reset => {
                        ("   ".to_string(), "|0>".to_string(), "   ".to_string())
                    }
                };

                top = format!(" {} ", wire_top);
                mid = format!("{}{}{}", Q_WIRE, wire_symbol, Q_WIRE);
                bot = format!(" {} ", wire_bot);
            }
            VisualizationElement::Input(input) => {
                let input_name = input.get_name(circuit).unwrap_or_else(|| match input {
                    WireInputElement::Qubit(_) => format!("q_{}: ", ind),
                    WireInputElement::Clbit(_) => format!("c_{}: ", ind - circuit.num_qubits()),
                    WireInputElement::Creg(_) => unreachable!(),
                });
                top = " ".repeat(input_name.len());
                bot = " ".repeat(input_name.len());
                mid = input_name;
            }
            VisualizationElement::VerticalLine(inst) => {
                let is_measure = if let Some(std_inst) = inst.op.try_standard_instruction() {
                    std_inst == StandardInstruction::Measure
                } else {
                    false
                };

                if is_measure {
                    top = CL_CONNECTING_WIRE.to_string();

                    let clbit = circuit.get_cargs(inst.clbits).first().unwrap();
                    if ind == vis_mat.clbit_map[clbit.index()] {
                        mid = C_WIRE_CON_TOP.to_string();

                        let shareable_clbit = circuit.clbits().get(*clbit).unwrap();
                        let registers = circuit
                            .clbit_indices()
                            .get(shareable_clbit)
                            .unwrap()
                            .registers();
                        // TODO: if someone adds > 99 clbits
                        // the visualization will have an extra whitespace shift which
                        // needs to be fixed
                        bot = if cregbundle && !registers.is_empty() {
                            format!("{}", registers.first().unwrap().1)
                        } else {
                            " ".to_string()
                        }
                    } else {
                        bot = CL_CONNECTING_WIRE.to_string();
                        mid = {
                            if ind < circuit.num_qubits() {
                                CL_Q_CROSSED_WIRE
                            } else {
                                CL_CL_CROSSED_WIRE
                            }
                        }
                        .to_string();
                    }
                } else {
                    top = CONNECTING_WIRE.to_string();
                    bot = CONNECTING_WIRE.to_string();
                    mid = {
                        if ind < circuit.num_qubits() {
                            Q_Q_CROSSED_WIRE
                        } else {
                            Q_CL_CROSSED_WIRE
                        }
                    }
                    .to_string();
                }
            }
            VisualizationElement::Empty => {
                top = " ".to_string();
                bot = " ".to_string();
                mid = {
                    if ind < circuit.num_qubits() {
                        Q_WIRE
                    } else {
                        C_WIRE
                    }
                }
                .to_string();
            }
        };
        TextWireElement { top, mid, bot }
    }

    fn draw(&self, mergewires: bool, fold: usize) -> String {
        // Calculate the layer ranges for each fold of the circuit
        let num_layers = self.wires[0].len();
        // We skip the first (inputs) layer since it's printed for each fold, regardless
        // of screen width limit
        let layer_widths = (1..num_layers).map(|layer| self.get_layer_width(layer));
        let (mut start, mut current_fold) = (1usize, 0usize);
        // let mut ranges: Vec<(usize, usize)> = Vec::new();

        let ranges: Vec<(usize, usize)> = layer_widths
            .enumerate()
            .filter_map(|(i, layer_width)| {
                current_fold += layer_width;
                if current_fold > fold {
                    let range = (start, i + 1);
                    start = i + 1;
                    current_fold = layer_width;
                    Some(range)
                } else {
                    None
                }
            })
            .collect::<Vec<(usize, usize)>>()
            .into_iter()
            .chain(std::iter::once((start, num_layers)))
            .collect();

        // for (i, layer_width) in layer_widths.enumerate() {
        //     current_fold += layer_width;
        //     if current_fold > fold {
        //         ranges.push((start, i + 1));
        //         start = i + 1;
        //         current_fold = layer_width;
        //     }
        // }
        // ranges.push((start, num_layers));

        let mut output = String::new();

        let mut wire_strings: Vec<String> = Vec::new();
        for (start, end) in ranges {
            wire_strings.clear();

            for wire in &self.wires {
                let top_line: String = wire[start..end]
                    .iter()
                    .map(|elem| elem.top.clone())
                    .collect::<Vec<String>>()
                    .join("");
                let mid_line: String = wire[start..end]
                    .iter()
                    .map(|elem| elem.mid.clone())
                    .collect::<Vec<String>>()
                    .join("");
                let bot_line: String = wire[start..end]
                    .iter()
                    .map(|elem| elem.bot.clone())
                    .collect::<Vec<String>>()
                    .join("");
                wire_strings.push(format!(
                    "{}{}{}{}",
                    if start > 1 { "«" } else { "" },
                    wire[0].top,
                    top_line,
                    if end < num_layers - 1 { "»" } else { "" }
                ));
                wire_strings.push(format!(
                    "{}{}{}{}",
                    if start > 1 { "«" } else { "" },
                    wire[0].mid,
                    mid_line,
                    if end < num_layers - 1 { "»" } else { "" }
                ));
                wire_strings.push(format!(
                    "{}{}{}{}",
                    if start > 1 { "«" } else { "" },
                    wire[0].bot,
                    bot_line,
                    if end < num_layers - 1 { "»" } else { "" }
                ));
            }
            for wire_idx in 0..wire_strings.len() {
                if mergewires && wire_idx % 3 == 2 && wire_idx < wire_strings.len() - 3 {
                    // Merge the bot_line of the this wire with the top_line of the next wire
                    let merged_line =
                        Self::merge_lines(&wire_strings[wire_idx], &wire_strings[wire_idx + 1]);
                    output.push_str(&format!("{}\n", merged_line));
                } else if !mergewires || wire_idx % 3 != 0 || wire_idx == 0 {
                    // if mergewires, skip all top_line strings but the very first one
                    output.push_str(&format!("{}\n", wire_strings[wire_idx]));
                }
            }
        }

        output
    }

    pub fn merge_lines(top: &str, bot: &str) -> String {
        let mut ret = String::new();

        for (topc, botc) in top.chars().zip(bot.chars()) {
            if topc == botc {
                ret.push(topc);
            } else if "┼╪".contains(topc) && botc == ' ' {
                ret.push('│');
            } else if topc == ' ' {
                ret.push(botc);
            } else if "┬╥".contains(topc) && " ║│".contains(botc) {
                ret.push(topc);
            } else if "┬│".contains(topc) && botc == '═' {
                ret.push('╪');
            } else if "┬│".contains(topc) && botc == '─' {
                ret.push('┼');
            } else if "└┘║│░─═".contains(topc) && botc == ' ' {
                ret.push(topc);
            } else if "║╥".contains(topc) && botc == '═' {
                ret.push('╬');
            } else if "║╥".contains(topc) && botc == '─' {
                ret.push('╫');
            } else if "║╫╬".contains(topc) && botc == ' ' {
                ret.push('║');
            } else if "│┼╪".contains(topc) && botc == ' ' {
                ret.push('│');
            } else if topc == '└' && botc == '┌' {
                ret.push('├');
            } else if topc == '┘' && botc == '┐' {
                ret.push('┤');
            } else if "┐┌".contains(botc) {
                ret.push('┬');
            } else if "┘└".contains(topc) && botc == '─' {
                ret.push('┴');
            } else if botc == ' ' {
                ret.push(topc);
            } else {
                ret.push(botc);
            }
        }

        ret
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bit::{ClassicalRegister, QuantumRegister, ShareableClbit, ShareableQubit};

    fn basic_circuit() -> CircuitData {
        let qreg = QuantumRegister::new_owning("q", 2);

        let creg_1 = ClassicalRegister::new_owning("c1", 2);

        let creg_2 = ClassicalRegister::new_owning("c2", 2);

        let qubits: Vec<ShareableQubit> = (0..qreg.len())
            .map(|i| qreg.get(i).expect("index in range"))
            .collect();

        let clbits: Vec<ShareableClbit> = (0..creg_1.len())
            .map(|i| creg_1.get(i).expect("index in range"))
            .chain((0..creg_2.len()).map(|i| creg_2.get(i).expect("index in range")))
            .collect();

        let mut circuit =
            CircuitData::new(Some(qubits), Some(clbits), None, 0, Param::Float(0.0)).unwrap();

        _ = circuit.add_creg(creg_1, true);
        _ = circuit.add_creg(creg_2, true);
        _ = circuit.add_qreg(qreg, true);

        circuit
            .push_standard_gate(StandardGate::H, &[], &[Qubit::new(0)])
            .unwrap();

        circuit
            .push_standard_gate(StandardGate::CX, &[], &[Qubit::new(0), Qubit::new(1)])
            .unwrap();

        circuit
    }

    #[test]
    fn test_creg_bundle() {
        let circuit = basic_circuit();

        let result = draw_circuit(&circuit, true, false, None).unwrap();

        let expected = "      ┌───┐     \n q_0: ┤ H ├──■──\n      └───┘  │  \n           ┌─┴─┐\n q_1: ─────┤ X ├\n           └───┘\n                \nc1: 2/══════════\n                \n                \nc2: 2/══════════\n                \n";

        assert_eq!(result, expected);
    }

    #[test]
    fn test_merge_wires() {
        let circuit = basic_circuit();

        let result = draw_circuit(&circuit, false, true, None).unwrap();

        let expected = "      ┌───┐     \n q_0: ┤ H ├──■──\n      └───┘┌─┴─┐\n q_1: ─────┤ X ├\n           └───┘\nc1_0: ══════════\n                \nc1_1: ══════════\n                \nc2_0: ══════════\n                \nc2_1: ══════════\n                \n";

        assert_eq!(result, expected);
    }

    #[test]
    fn test_fold() {
        let mut circuit = basic_circuit();

        circuit
            .push_standard_gate(StandardGate::CY, &[], &[Qubit::new(0), Qubit::new(1)])
            .unwrap();

        circuit
            .push_standard_gate(StandardGate::CZ, &[], &[Qubit::new(0), Qubit::new(1)])
            .unwrap();

        let result = draw_circuit(&circuit, false, false, Some(10)).unwrap();
        let expected = "      ┌───┐     »\n q_0: ┤ H ├──■──»\n      └───┘  │  »\n           ┌─┴─┐»\n q_1: ─────┤ X ├»\n           └───┘»\n                »\nc1_0: ══════════»\n                »\n                »\nc1_1: ══════════»\n                »\n                »\nc2_0: ══════════»\n                »\n                »\nc2_1: ══════════»\n                »\n«                \n« q_0: ──■────■──\n«        │    │  \n«      ┌─┴─┐┌─┴─┐\n« q_1: ┤ Y ├┤ Z ├\n«      └───┘└───┘\n«                \n«c1_0: ══════════\n«                \n«                \n«c1_1: ══════════\n«                \n«                \n«c2_0: ══════════\n«                \n«                \n«c2_1: ══════════\n«                \n";
        assert_eq!(result, expected);
    }
}
