// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::bit::{ClassicalRegister, ShareableClbit, ShareableQubit};
use crate::circuit_data::CircuitData;
use crate::operations::{
    ControlFlowInstruction, Operation, OperationRef, Param, StandardGate, StandardInstruction,
};
use crate::packed_instruction::PackedInstruction;
use crate::{Clbit, Qubit};
use approx;
use crossterm::terminal;
use hashbrown::HashSet;
use itertools::{Itertools, MinMaxResult};
use pyo3::prelude::*;
use std::fmt::Debug;
use std::ops::{Index, IndexMut};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

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
            let (term_width, _) = terminal::size().unwrap_or((80, 24));
            term_width as usize
        }
    };
    let phase = circuit.global_phase();
    let mut output: String = match phase {
        Param::Float(f) => {
            if approx::abs_diff_eq!(*f, 0.) {
                String::new()
            } else {
                format!("global phase: {}\n", f)
            }
        }
        Param::ParameterExpression(expr) => {
            let phase = expr.to_string();
            format!("global phase: {}\n", phase)
        }
        _ => unreachable!("Global phase must be a ParameterExpression or float"),
    };
    // Strip trailing whitespace from lines.
    // On the last line ensure only a single newline ends the returned
    // string (in case we ended up with a double newline after the stripping)
    output.extend(
        text_drawer
            .draw(mergewires, fold)
            .lines()
            .flat_map(|x| [x.trim_end(), "\n"]),
    );
    let mut chars = output.chars();
    if let Some(last) = chars.next_back() {
        if last == '\n' && chars.next_back() == Some('\n') {
            output.pop();
        }
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug)]
struct MappedInstruction<'a> {
    // TODO: Document
    inst: &'a PackedInstruction,
    mapping_context: Option<usize>, // TODO: document the mapping context of the instruction bits w.r.t global matrix view
}

/// Calculate the range (inclusive) of the given instruction qubits/clbits over the wire indices.
/// The assumption is that clbits always appear after the qubits in the visualization, hence the clbit indices
/// are offset by the number of instruction qubits when calculating the range.
fn get_instruction_range(
    // TODO: make it take slices of u32?
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

#[derive(Clone, PartialEq, Eq, Debug)]
enum WireInputElement<'a> {
    Qubit(&'a ShareableQubit),
    Clbit(&'a ShareableClbit),
    Creg(&'a ClassicalRegister), // TODO: document: this is always w.r.t the top-level circuit
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
                    if register.len() > 1 {
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
                    if register.len() > 1 {
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
    Control(MappedInstruction<'a>),
    Swap(MappedInstruction<'a>),
    Barrier,
    Reset,
}

/// Represent elements that appear in a boxed operation.
#[derive(Clone, Debug)]
enum BoxedElement<'a> {
    Single(MappedInstruction<'a>),
    Multi(MappedInstruction<'a>),
}

/// Enum for representing the elements stored in the visualization matrix. The elements
/// do not directly implement visualization capabilities, but rather carry enough information
/// to enable visualization later on by the actual drawer.
#[derive(Default, Clone, Debug)]
enum VisualizationElement<'a> {
    #[default]
    /// A wire element without any associated information.
    Empty,
    /// A Vertical line element, belonging to an instruction (e.g of a controlled gate or a measure).
    VerticalLine(MappedInstruction<'a>),
    /// A circuit input element (qubit, clbit, creg).
    Input(WireInputElement<'a>),
    /// An element which is drawn without a surrounding box. Used only on qubit wires.
    DirectOnWire(OnWireElement<'a>),
    // A boxed element which can span one or more wires. Used only on qubit wires.
    Boxed(BoxedElement<'a>),
    // A frame element for control flow operations
    Frame,
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

impl<'a> IndexMut<usize> for VisualizationLayer<'a> {
    fn index_mut(&mut self, index: usize) -> &mut VisualizationElement<'a> {
        &mut self.0[index]
    }
}

impl<'a> VisualizationLayer<'a> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn add_input(&mut self, input: WireInputElement<'a>, idx: usize) {
        self.0[idx] = VisualizationElement::Input(input);
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
    /// A reference to the circuit this matrix was constructed from. TODO: this is the top level circuit
    circuit: &'a CircuitData,
    /// A mapping from instruction's Clbit indices to the visualization matrix wires,
    /// to be used when mapping clbits to bit of bundled cregs
    clbit_map: Vec<usize>,

    wire_map_contexts: Vec<(&'a CircuitData, Vec<u32>)>, // TODO: make a struct? document
}

impl<'a> VisualizationMatrix<'a> {
    fn from_circuit(circuit: &'a CircuitData, bundle_cregs: bool) -> PyResult<Self> {
        let mut vis_matrix = VisualizationMatrix {
            layers: Vec::new(),
            circuit,
            clbit_map: Vec::new(),
            wire_map_contexts: Vec::new(),
        };

        let inst_layers = vis_matrix.build_layers(circuit, None);

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

        vis_matrix.layers = vec![
            VisualizationLayer(vec![VisualizationElement::default(); num_wires]);
            inst_layers.len() + 1 // Add 1 to account for the inputs layer
        ];

        let input_layer = vis_matrix.layers.first_mut().unwrap();
        let mut input_idx = 0;
        for qubit in circuit.qubits().objects() {
            input_layer.add_input(WireInputElement::Qubit(qubit), input_idx);
            input_idx += 1;
        }

        let mut visited_cregs: HashSet<&ClassicalRegister> = HashSet::new();
        // let mut clbit_map: Vec<usize> = Vec::new();
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
                        vis_matrix.clbit_map.push(input_idx - 1);
                    } else {
                        input_layer.add_input(WireInputElement::Creg(creg), input_idx);
                        visited_cregs.insert(creg);
                        vis_matrix.clbit_map.push(input_idx);
                        input_idx += 1;
                    }
                    continue;
                }
            }

            input_layer.add_input(WireInputElement::Clbit(clbit), input_idx);
            vis_matrix.clbit_map.push(input_idx);
            input_idx += 1;
        }

        for (i, layer) in inst_layers.iter().enumerate() {
            for inst in layer {
                vis_matrix.add_instruction(i + 1, bundle_cregs, inst);
            }
        }

        Ok(vis_matrix)
    }

    /// Return a list of layers such that each layer contains a list of instructions
    /// whose qubits/clbits indices do not overlap. The instruction are packed into each layer as long as there
    /// is no qubit/clbit overlap.
    fn build_layers(
        &mut self,
        circ: &'a CircuitData,
        current_mapping: Option<usize>,
    ) -> Vec<Vec<MappedInstruction<'a>>> {
        // TODO: document why we need the pair
        let mut layers: Vec<Vec<MappedInstruction<'a>>> = Vec::new();
        let num_qubits = circ.num_qubits();
        let num_clbits = circ.num_clbits();
        let mut layers_frontier = vec![0usize; num_qubits + num_clbits]; // TODO: explain about the invariant of this vector

        for inst in circ.data() {
            if matches!(
                inst.op.view(),
                OperationRef::StandardGate(StandardGate::GlobalPhase)
            ) {
                if layers.last().is_none() {
                    layers.push(Vec::new())
                }
                layers.last_mut().unwrap().push(MappedInstruction {
                    inst,
                    mapping_context: None,
                });
                continue;
            }

            let (node_min, node_max) = get_instruction_range(
                circ.get_qargs(inst.qubits),
                circ.get_cargs(inst.clbits),
                num_qubits,
            );

            let mut layer_idx = *layers_frontier[node_min..=node_max]
                .iter()
                .max()
                .expect("Instruction range should be valid");
            if layers.len() == layer_idx {
                layers.push(Vec::new());
            }
            layers[layer_idx].push(MappedInstruction {
                inst,
                mapping_context: current_mapping,
            }); // TODO: we might don't want to do that for a control flow, since it might skew the layers for subsequence instructions
            layers_frontier[node_min..=node_max].fill(layer_idx + 1);

            // TODO: document
            if let OperationRef::ControlFlow(_) = inst.op.view() {
                let mut wire_map = circ
                    .get_qargs(inst.qubits)
                    .iter()
                    .map(|q| q.index() as u32)
                    .collect::<Vec<u32>>(); // TODO: Need to add this to wire_maps. TODO: need to use bytemuck from Ray
                if let Some(context_idx) = current_mapping {
                    let top_level_mapping = &self.wire_map_contexts[context_idx].1;
                    wire_map
                        .iter_mut()
                        .for_each(|i| *i = top_level_mapping[*i as usize]);
                }
                for block in inst.blocks_view() {
                    // TODO: for multi-block cf inst (e.g. switch), how should we mark the blocks in the matrix?
                    let block_circ = &circ.blocks()[*block];
                    self.wire_map_contexts.push((block_circ, wire_map.clone()));
                    let block_layers =
                        self.build_layers(block_circ, Some(self.wire_map_contexts.len() - 1));
                    layer_idx += block_layers.len() + 1;

                    layers.extend(block_layers);
                    if layers.len() == layer_idx {
                        layers.push(Vec::new());
                    }

                    layers[layer_idx].push(MappedInstruction {
                        inst,
                        mapping_context: current_mapping,
                    }); // TODO: test that for nesting this works fine
                    // TODO: rephrase: "close" the area by using the inst range
                    // We want the next block or instructions to start outside the frame-boundaries of this block
                    layers_frontier[node_min..=node_max].fill(layer_idx + 1); // TODO: else and switch cases don't get the extra layer (they're all part of the same instruction- do we want it?
                }
            }
        }
        layers
    }

    /// Adds the required visualization elements to represent the given instruction
    fn add_instruction(
        &mut self,
        layer_idx: usize,
        cregbundle: bool,
        inst: &MappedInstruction<'a>,
    ) {
        match inst.inst.op.view() {
            OperationRef::StandardGate(gate) => {
                self.add_standard_gate(layer_idx, gate, inst);
            }
            OperationRef::StandardInstruction(std_inst) => {
                self.add_standard_instruction(layer_idx, cregbundle, std_inst, inst);
            }
            OperationRef::Unitary(_) => {
                self.add_unitary_gate(layer_idx, inst);
            }
            OperationRef::ControlFlow(cf_inst) => {
                self.add_control_flow_op(layer_idx, cf_inst, inst);
            }
            _ => unimplemented!(
                "{}",
                format!(
                    "Visualization is not implemented for instruction of type {:?}",
                    inst.inst.op.name()
                )
            ),
        }
    }

    // TODO: document
    // TODO: is there an opportunity to spare the Vec if it's not mapped?
    fn mapped_qargs(&self, inst: &MappedInstruction) -> Vec<Qubit> {
        if let Some(mapping_idx) = inst.mapping_context {
            let mapping_context = &self.wire_map_contexts[mapping_idx];
            let qargs = mapping_context.0.get_qargs(inst.inst.qubits);
            qargs
                .iter()
                .map(|q| Qubit(mapping_context.1[q.index()]))
                .collect::<Vec<Qubit>>()
        } else {
            self.circuit.get_qargs(inst.inst.qubits).to_vec()
        }
    }

    // TODO: document
    // TODO: is there an opportunity to spare the Vec if it's not mapped?
    fn mapped_cargs(&self, inst: &MappedInstruction) -> Vec<Clbit> {
        if let Some(mapping_idx) = inst.mapping_context {
            let mapping_context = &self.wire_map_contexts[mapping_idx];
            let cargs = mapping_context.0.get_cargs(inst.inst.clbits);
            cargs
                .iter()
                .map(|c| Clbit(mapping_context.1[c.index()]))
                .collect::<Vec<Clbit>>()
        } else {
            self.circuit.get_cargs(inst.inst.clbits).to_vec()
        }
    }

    fn add_standard_gate(
        &mut self,
        layer_idx: usize,
        gate: StandardGate,
        inst: &MappedInstruction<'a>,
    ) {
        if gate == StandardGate::GlobalPhase {
            return;
        }

        let qargs = self.mapped_qargs(inst);
        let (minima, maxima) = get_instruction_range(&qargs, &[], 0);

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
                    self.layers[layer_idx][q] =
                        VisualizationElement::Boxed(BoxedElement::Multi(*inst));
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
                self.layers[layer_idx][qargs[0].index()] =
                    VisualizationElement::Boxed(BoxedElement::Single(*inst));
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
                self.layers[layer_idx][qargs.last().unwrap().index()] =
                    VisualizationElement::Boxed(BoxedElement::Single(*inst));
                if gate.num_ctrl_qubits() > 0 {
                    self.add_controls(
                        layer_idx,
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
                self.add_vertical_lines(layer_idx, vert_lines, inst);
            }
            StandardGate::GlobalPhase => {}
            StandardGate::Swap | StandardGate::CSwap => {
                // taking the last 2 elements of qargs
                if gate == StandardGate::CSwap {
                    let control = vec![qargs[0].0 as usize];
                    self.add_controls(layer_idx, inst, &control);
                }
                let swap_qubits = qargs.iter().map(|q| q.0 as usize).rev().take(2);
                for qubit in swap_qubits {
                    self.layers[layer_idx][qubit] =
                        VisualizationElement::DirectOnWire(OnWireElement::Swap(*inst));
                }

                let vert_lines = (minima..=maxima)
                    .filter(|idx| !(qargs.iter().map(|q| q.0 as usize)).contains(idx));
                self.add_vertical_lines(layer_idx, vert_lines, inst);
            }
        }
    }

    fn add_controls(
        &mut self,
        layer_idx: usize,
        inst: &MappedInstruction<'a>,
        controls: &Vec<usize>,
    ) {
        let layer = &mut self.layers[layer_idx];
        for control in controls {
            layer[*control] = VisualizationElement::DirectOnWire(OnWireElement::Control(*inst));
        }
    }

    fn add_vertical_lines<I>(
        &mut self,
        layer_idx: usize,
        vertical_lines: I,
        inst: &MappedInstruction<'a>,
    ) where
        I: Iterator<Item = usize>,
    {
        for vline in vertical_lines {
            self.layers[layer_idx][vline] = VisualizationElement::VerticalLine(*inst);
        }
    }

    fn add_standard_instruction(
        &mut self,
        layer_idx: usize,
        cregbundle: bool,
        std_inst: StandardInstruction,
        inst: &MappedInstruction<'a>,
    ) {
        let qargs = &self.mapped_qargs(inst);
        let clbits = &self.mapped_cargs(inst);

        let (minima, mut maxima) = get_instruction_range(qargs, clbits, self.circuit.num_qubits());

        match std_inst {
            StandardInstruction::Barrier(_) => {
                for q in qargs {
                    self.layers[layer_idx][q.index()] =
                        VisualizationElement::DirectOnWire(OnWireElement::Barrier);
                }
            }
            StandardInstruction::Reset => {
                for q in qargs {
                    self.layers[layer_idx][q.index()] =
                        VisualizationElement::DirectOnWire(OnWireElement::Reset);
                }
            }
            StandardInstruction::Measure => {
                self.layers[layer_idx][qargs.last().unwrap().index()] =
                    VisualizationElement::Boxed(BoxedElement::Single(*inst));

                // Some bits may be bundled, so we need to map the Clbit index to the proper wire index
                if cregbundle {
                    maxima = self.clbit_map[self
                        .circuit
                        .get_cargs(inst.inst.clbits)
                        .first()
                        .expect("Measure should have a clbit arg")
                        .index()];
                }
                self.add_vertical_lines(layer_idx, minima + 1..=maxima, inst);
            }
            StandardInstruction::Delay(_) => {
                for q in qargs {
                    self.layers[layer_idx][q.index()] =
                        VisualizationElement::Boxed(BoxedElement::Single(*inst));
                }
            }
        }
    }

    fn add_unitary_gate(&mut self, layer_idx: usize, inst: &MappedInstruction<'a>) {
        let qargs = &self.mapped_qargs(inst);

        if qargs.len() == 1 {
            self.layers[layer_idx][qargs.first().unwrap().index()] =
                VisualizationElement::Boxed(BoxedElement::Single(*inst));
        } else {
            let (minima, maxima) = get_instruction_range(qargs, &[], 0);

            for q in minima..=maxima {
                self.layers[layer_idx][q] = VisualizationElement::Boxed(BoxedElement::Multi(*inst));
            }
        }
    }

    fn add_control_flow_op(
        &mut self,
        layer_idx: usize,
        _cf_inst: &ControlFlowInstruction,
        inst: &MappedInstruction<'a>,
    ) {
        let qargs = self.mapped_qargs(inst);
        let (minima, maxima) = get_instruction_range(&qargs, &[], 0);
        (minima..=maxima).for_each(|q| self.layers[layer_idx][q] = VisualizationElement::Frame);
    }

    fn num_lanes(&self) -> usize {
        self.layers.first().map_or(0, |layer| layer.len())
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn is_lane_wire(&self, lane_idx: usize) -> bool {
        !matches!(self.layers[0][lane_idx], VisualizationElement::Frame) // TODO: use enum instead
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
        for w in 0..self.num_lanes() {
            for l in 0..self.num_layers() {
                let element = &self[l][w];
                let label = match &element {
                    VisualizationElement::Empty => "~",
                    VisualizationElement::VerticalLine(inst) => {
                        let mut line = "|";
                        if let Some(std_inst) = inst.inst.op.try_standard_instruction() {
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
                    VisualizationElement::Frame => {
                        if self.is_lane_wire(w) {
                            &FRAME_VERTICAL.to_string()
                        } else {
                            &FRAME_HORIZONTAL.to_string()
                        }
                    }
                };
                write!(f, "{:^5}", label)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
enum TextDrawerElement {
    Wire(TextWireElement),
    Ribbon(TextRibbonElement),
}

impl TextDrawerElement {
    fn width(&self) -> usize {
        match self {
            TextDrawerElement::Wire(wire_element) => wire_element.width(),
            TextDrawerElement::Ribbon(_) => 0,
        }
    }
}

#[derive(Clone, Default, Debug)]
struct TextRibbonElement {
    lane: String,
}

impl TextRibbonElement {
    fn width(&self) -> usize {
        self.lane.width()
    }

    fn stretch(&mut self, pad_char: char, width: usize) {
        let current_width = self.width();

        if current_width < width {
            let pad_size = width - current_width;

            let left_pad = pad_size / 2;
            let right_pad = pad_size - left_pad;
            self.lane.reserve(pad_size);
            (0..left_pad).for_each(|_| self.lane.insert(0, pad_char));
            (0..right_pad).for_each(|_| self.lane.push(pad_char));
        }
    }
}

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
        self.top.width().max(self.mid.width()).max(self.bot.width())
    }

    fn left_pad_string(s: &mut String, pad_char: char, width: usize) {
        let current_width = s.width();
        if current_width < width {
            let pad_size = width - current_width;
            s.reserve(pad_size);
            (0..pad_size).for_each(|_| s.insert(0, pad_char));
        }
    }

    fn pad_string(s: &mut String, pad_char: char, width: usize) {
        let current_width = s.width();
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

// impl Index<usize> for TextWireElement {
//     type Output = TextDrawerElement;

//     fn index(&self, index: usize) -> &Self::Output {
//         &self.
//     }
// }
// impl IndexMut<usize> for TextWireElement {
//     fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
//         &mut self.p[idx]
//     }
// }

const Q_WIRE: char = '─';
const C_WIRE: char = '═';
const TOP_CON: char = '┴';
const BOT_CON: char = '┬';
const C_WIRE_CON_TOP: char = '╩';
const C_BOT_CON: char = '╥';
const Q_LEFT_CON: char = '┤';
const Q_RIGHT_CON: char = '├';
const _CL_LEFT_CON: char = '╡';
const _CL_RIGHT_CON: char = '╞';
const TOP_LEFT_BOX: char = '┌';
const TOP_RIGHT_BOX: char = '┐';
const BOT_LEFT_BOX: char = '└';
const BOT_RIGHT_BOX: char = '┘';
const BARRIER: char = '░';
const BULLET: char = '■';
const CONNECTING_WIRE: char = '│';
const CL_CONNECTING_WIRE: char = '║';
const Q_Q_CROSSED_WIRE: char = '┼';
const Q_CL_CROSSED_WIRE: char = '╪';
const CL_CL_CROSSED_WIRE: char = '╬';
const CL_Q_CROSSED_WIRE: char = '╫';
const FRAME_VERTICAL: char = '┆';
const FRAME_HORIZONTAL: char = '┄';

#[derive(Debug)]
enum TextDrawerLane {
    WireLane(Vec<TextWireElement>),
    RibbonLane(Vec<TextRibbonElement>),
}

/// Textual representation of the circuit
struct TextDrawer {
    /// The array of textual element lanes corresponding to the visualization elements
    lanes: Vec<TextDrawerLane>,

    num_layers: usize,
}

// impl Index<usize> for TextDrawer {
//     type Output = Vec<TextWireElement>;

//     fn index(&self, index: usize) -> &Self::Output {
//         &self.wires[index]
//     }
// }

impl TextDrawer {
    fn from_visualization_matrix(vis_mat: &VisualizationMatrix, cregbundle: bool) -> Self {
        let mut text_drawer = TextDrawer {
            lanes: Vec::new(),
            num_layers: 0,
        };

        for lane in 0..vis_mat.num_lanes() {
            text_drawer.lanes.push(if vis_mat.is_lane_wire(lane) {
                TextDrawerLane::WireLane(Vec::new())
            } else {
                TextDrawerLane::RibbonLane(Vec::new())
            });
        }

        for (i, layer) in vis_mat.layers.iter().enumerate() {
            let layer_elements = Self::draw_layer(layer, vis_mat, cregbundle, i);

            for (j, element) in layer_elements.iter().enumerate() {
                let lane = &mut text_drawer.lanes[j];

                match (element, lane) {
                    (
                        TextDrawerElement::Wire(wire_element),
                        TextDrawerLane::WireLane(wire_lane),
                    ) => {
                        wire_lane.push(wire_element.clone());
                    }
                    (
                        TextDrawerElement::Ribbon(ribbon_element),
                        TextDrawerLane::RibbonLane(ribbon_lane),
                    ) => {
                        ribbon_lane.push(ribbon_element.clone());
                    }
                    _ => unreachable!(
                        "Layer element variant should correspond to the TextDrawerLane variant in each lane"
                    ),
                }
            }
        }
        text_drawer.num_layers = vis_mat.layers.len();

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
                            Param::ParameterExpression(expr) => expr.to_string(),
                            _ => format!("{:?}", param),
                        })
                        .join(",");
                    label = format!("{}({})", label, params);
                }
                label
            }
            OperationRef::Unitary(_) => instruction
                .label
                .as_ref()
                .map(|x| x.to_string())
                .unwrap_or(" Unitary ".to_string()),
            // Fallback for non-standard operations
            _ => {
                if let Some(ref label) = instruction.label {
                    label.to_string()
                } else {
                    format!(" {} ", instruction.op.name())
                }
            }
        }
    }

    fn get_layer_width(&self, ind: usize) -> usize {
        let layer_width = |lane: &TextDrawerLane| match lane {
            TextDrawerLane::WireLane(wire_lane) => wire_lane[ind].width(),
            TextDrawerLane::RibbonLane(ribbon_lane) => ribbon_lane[ind].width(),
        };

        self.lanes.iter().map(layer_width).max().unwrap_or(0)
    }

    fn draw_layer(
        layer: &VisualizationLayer,
        vis_mat: &VisualizationMatrix,
        cregbundle: bool,
        layer_ind: usize,
    ) -> Vec<TextDrawerElement> {
        let mut layer_elments: Vec<TextDrawerElement> = Vec::new();

        for (lane_idx, element) in layer.0.iter().enumerate() {
            if vis_mat.is_lane_wire(lane_idx) {
                layer_elments.push(TextDrawerElement::Wire(Self::draw_wire_element(
                    element, vis_mat, cregbundle, lane_idx,
                )));
            } else {
                layer_elments.push(TextDrawerElement::Ribbon(Self::draw_ribbon_element(
                    element, vis_mat, cregbundle, lane_idx,
                )));
            }
        }

        let num_qubits = vis_mat.circuit.num_qubits();

        let layer_width = layer_elments
            .iter()
            .map(|element| element.width())
            .max()
            .unwrap_or(0);

        for (i, element) in layer_elments.iter_mut().enumerate() {
            match element {
                TextDrawerElement::Wire(element) => {
                    if layer_ind == 0 {
                        element.pad_wire_left(' ', layer_width);
                    } else if i < num_qubits {
                        element.pad_wire(Q_WIRE, layer_width);
                    } else {
                        element.pad_wire(C_WIRE, layer_width);
                    }
                }
                TextDrawerElement::Ribbon(element) => {
                    element.stretch('#', layer_width);
                }
            }
        }
        layer_elments
    }

    fn draw_wire_element(
        vis_ele: &VisualizationElement,
        vis_mat: &VisualizationMatrix,
        cregbundle: bool,
        lane_idx: usize,
    ) -> TextWireElement {
        let (top, mid, bot);
        match vis_ele {
            VisualizationElement::Boxed(sub_type) => {
                // implement for cases where the box is on classical wires. The left and right connectors will change
                // from single wired to double wired.
                match sub_type {
                    BoxedElement::Single(inst) => {
                        let mut top_con = Q_WIRE;
                        let mut bot_con = Q_WIRE;
                        let mut label = format!(" {} ", Self::get_label(inst.inst));
                        if let Some(gate) = inst.inst.op.try_standard_gate() {
                            if gate.is_controlled_gate() {
                                let qargs = &vis_mat.mapped_qargs(inst);
                                let (minima, maxima) = get_instruction_range(qargs, &[], 0);
                                if qargs.last().unwrap().index() > minima {
                                    top_con = TOP_CON;
                                }
                                if qargs.last().unwrap().index() < maxima {
                                    bot_con = BOT_CON;
                                }
                                // This ensures the top_con/bot_con connectors are properly aligned with the control
                                // lines regardless of whether the text element padding size is odd or even.
                                (label.len() % 2 == 0).then(|| label.push(' '));
                            }
                        } else if let Some(std_inst) = inst.inst.op.try_standard_instruction() {
                            if std_inst == StandardInstruction::Measure {
                                bot_con = C_BOT_CON;
                            }
                        }
                        let label_len = label.width();
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
                        let label = format!(" {} ", Self::get_label(inst.inst));
                        let label_len = label.width();
                        let qargs = &vis_mat.mapped_qargs(inst);
                        let (minima, maxima) = get_instruction_range(qargs, &[], 0);
                        let mid_idx = (minima + maxima) / 2;
                        let num_affected =
                            if let Some(idx) = qargs.iter().position(|&x| x.index() == lane_idx) {
                                format!("{:^width$}", idx, width = qargs.len())
                            } else {
                                " ".to_string()
                            };
                        let mid_section = if lane_idx == mid_idx {
                            format!("{:^total_q$}{}", num_affected, label, total_q = qargs.len(),)
                        } else {
                            format!(
                                "{:^total_q$}{:^label_len$}",
                                num_affected,
                                " ",
                                total_q = qargs.len(),
                                label_len = label_len
                            )
                        };
                        let left_len = (mid_section.width() - 1) / 2;
                        let right_len = mid_section.width() - left_len - 1;
                        top = if lane_idx == minima {
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
                                " ".repeat(mid_section.width()),
                                CONNECTING_WIRE
                            )
                        };
                        mid = format!("{}{}{}", Q_LEFT_CON, mid_section, Q_RIGHT_CON);
                        bot = if lane_idx == maxima {
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
                                " ".repeat(mid_section.width()),
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
                            get_instruction_range(&vis_mat.mapped_qargs(inst), &[], 0);
                        (
                            if lane_idx == minima {
                                " ".to_string()
                            } else {
                                CONNECTING_WIRE.to_string()
                            },
                            BULLET.to_string(),
                            if lane_idx == maxima {
                                " ".to_string()
                            } else {
                                CONNECTING_WIRE.to_string()
                            },
                        )
                    }
                    OnWireElement::Swap(inst) => {
                        let (minima, maxima) =
                            get_instruction_range(&vis_mat.mapped_qargs(inst), &[], 0);
                        (
                            if lane_idx == minima {
                                " ".to_string()
                            } else {
                                CONNECTING_WIRE.to_string()
                            },
                            "X".to_string(),
                            if lane_idx == maxima {
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
                let input_name = input
                    .get_name(vis_mat.circuit)
                    .unwrap_or_else(|| match input {
                        WireInputElement::Qubit(_) => format!("q_{}: ", lane_idx),
                        WireInputElement::Clbit(_) => {
                            format!("c_{}: ", lane_idx - vis_mat.circuit.num_qubits())
                        }
                        WireInputElement::Creg(_) => unreachable!(),
                    });
                top = " ".repeat(input_name.width());
                bot = " ".repeat(input_name.width());
                mid = input_name;
            }
            VisualizationElement::VerticalLine(inst) => {
                let is_measure = if let Some(std_inst) = inst.inst.op.try_standard_instruction() {
                    std_inst == StandardInstruction::Measure
                } else {
                    false
                };

                if is_measure {
                    top = CL_CONNECTING_WIRE.to_string();

                    let clbits = &vis_mat.mapped_cargs(inst);
                    let clbit = clbits.first().unwrap();
                    if lane_idx == vis_mat.clbit_map[clbit.index()] {
                        mid = C_WIRE_CON_TOP.to_string();

                        let shareable_clbit = vis_mat.circuit.clbits().get(*clbit).unwrap();
                        let registers = vis_mat
                            .circuit
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
                            if lane_idx < vis_mat.circuit.num_qubits() {
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
                        if lane_idx < vis_mat.circuit.num_qubits() {
                            Q_Q_CROSSED_WIRE
                        } else {
                            Q_CL_CROSSED_WIRE
                        }
                    }
                    .to_string();
                }
            }
            VisualizationElement::Frame => {
                top = FRAME_VERTICAL.to_string();
                bot = FRAME_VERTICAL.to_string();
                mid = {
                    if lane_idx < vis_mat.circuit.num_qubits() {
                        Q_Q_CROSSED_WIRE
                    } else {
                        C_WIRE
                    }
                }
                .to_string();
            }
            VisualizationElement::Empty => {
                top = " ".to_string();
                bot = " ".to_string();
                mid = {
                    if lane_idx < vis_mat.circuit.num_qubits() {
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

    fn draw_ribbon_element(
        vis_ele: &VisualizationElement,
        _vis_mat: &VisualizationMatrix,
        _cregbundle: bool,
        _lane_idx: usize,
    ) -> TextRibbonElement {
        match vis_ele {
            VisualizationElement::Frame => TextRibbonElement {
                lane: FRAME_HORIZONTAL.to_string(),
            },
            _ => {
                unimplemented!()
            }
        }
    }

    fn draw(&self, mergewires: bool, fold: usize) -> String {
        // Calculate the layer ranges for each fold of the circuit
        // We skip the first (inputs) layer since it's printed for each fold, regardless
        // of screen width limit
        let layer_widths = (1..self.num_layers).map(|layer| self.get_layer_width(layer));
        let (mut start, mut current_fold) = (1usize, 0usize);

        let mut ranges: Vec<(usize, usize)> = layer_widths
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
            .collect();
        ranges.push((start, self.num_layers));

        let mut output = String::new();

        let mut lane_strings: Vec<String> = Vec::new();
        for (start, end) in ranges {
            lane_strings.clear();
            for lane in &self.lanes {
                match lane {
                    TextDrawerLane::WireLane(wire_lane) => {
                        let top_line: String = wire_lane[start..end]
                            .iter()
                            .map(|elem| elem.top.clone())
                            .collect::<Vec<String>>()
                            .join("");
                        let mid_line: String = wire_lane[start..end]
                            .iter()
                            .map(|elem| elem.mid.clone())
                            .collect::<Vec<String>>()
                            .join("");
                        let bot_line: String = wire_lane[start..end]
                            .iter()
                            .map(|elem| elem.bot.clone())
                            .collect::<Vec<String>>()
                            .join("");
                        lane_strings.push(format!(
                            "{}{}{}{}",
                            if start > 1 { "«" } else { "" },
                            wire_lane[0].top,
                            top_line,
                            if end < self.num_layers - 1 { "»" } else { "" }
                        ));
                        lane_strings.push(format!(
                            "{}{}{}{}",
                            if start > 1 { "«" } else { "" },
                            wire_lane[0].mid,
                            mid_line,
                            if end < self.num_layers - 1 { "»" } else { "" }
                        ));
                        lane_strings.push(format!(
                            "{}{}{}{}",
                            if start > 1 { "«" } else { "" },
                            wire_lane[0].bot,
                            bot_line,
                            if end < self.num_layers - 1 { "»" } else { "" }
                        ));
                    }
                    TextDrawerLane::RibbonLane(ribbon_lane) => {
                        let lane_string = ribbon_lane[start..end]
                            .iter()
                            .map(|elem| elem.lane.clone())
                            .collect::<Vec<String>>()
                            .join("");
                        lane_strings.push(format!(
                            "{}{}{}{}",
                            if start > 1 { "«" } else { "" },
                            ribbon_lane[0].lane,
                            lane_string,
                            if end < self.num_layers - 1 { "»" } else { "" }
                        ));
                    }
                }
            }

            for wire_idx in 0..lane_strings.len() {
                if mergewires && wire_idx % 3 == 2 && wire_idx < lane_strings.len() - 3 {
                    // Merge the bot_line of the this wire with the top_line of the next wire
                    let merged_line =
                        Self::merge_lines(&lane_strings[wire_idx], &lane_strings[wire_idx + 1]);
                    output.push_str(&format!("{}\n", merged_line));
                } else if !mergewires || wire_idx % 3 != 0 || wire_idx == 0 {
                    // if mergewires, skip all top_line strings but the very first one
                    output.push_str(&format!("{}\n", lane_strings[wire_idx]));
                }
            }
        }

        output
    }

    pub fn merge_lines(top: &str, bot: &str) -> String {
        let mut ret = String::new();

        for (topc, botc) in top.graphemes(true).zip(bot.graphemes(true)) {
            if topc == botc {
                ret.push_str(topc);
            } else if "┼╪".contains(topc) && botc == " " {
                ret.push('│');
            } else if topc == " " {
                ret.push_str(botc);
            } else if "┬╥".contains(topc) && " ║│".contains(botc) {
                ret.push_str(topc);
            } else if "┬│".contains(topc) && botc == "═" {
                ret.push('╪');
            } else if "┬│".contains(topc) && botc == "─" {
                ret.push('┼');
            } else if "└┘║│░─═".contains(topc) && botc == " " {
                ret.push_str(topc);
            } else if "║╥".contains(topc) && botc == "═" {
                ret.push('╬');
            } else if "║╥".contains(topc) && botc == "─" {
                ret.push('╫');
            } else if "║╫╬".contains(topc) && botc == " " {
                ret.push('║');
            } else if "│┼╪".contains(topc) && botc == " " {
                ret.push('│');
            } else if topc == "└" && botc == "┌" {
                ret.push('├');
            } else if topc == "┘" && botc == "┐" {
                ret.push('┤');
            } else if "┐┌".contains(botc) {
                ret.push('┬');
            } else if "┘└".contains(topc) && botc == "─" {
                ret.push('┴');
            } else if botc == " " {
                ret.push_str(topc);
            } else {
                ret.push_str(botc);
            }
        }

        ret
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use smallvec::smallvec;
    use std::sync::Arc;

    use super::*;
    use crate::bit::{ClassicalRegister, QuantumRegister, ShareableClbit, ShareableQubit};
    use crate::instruction::Parameters;
    use crate::operations::ControlFlow;
    use crate::operations::{
        ArrayType, ControlFlowInstruction, DelayUnit, STANDARD_GATE_SIZE, StandardInstruction,
        UnitaryGate,
    };
    use crate::packed_instruction::PackedOperation;
    use crate::parameter::parameter_expression::ParameterExpression;
    use crate::parameter::symbol_expr::Symbol;

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

        let mut circuit = CircuitData::new(Some(qubits), Some(clbits), Param::Float(0.0)).unwrap();

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

    #[cfg(not(miri))]
    #[test]
    fn test_creg_bundle() {
        let circuit = basic_circuit();

        let result = draw_circuit(&circuit, true, false, None).unwrap();

        let expected = "
      ┌───┐
 q_0: ┤ H ├──■──
      └───┘  │
           ┌─┴─┐
 q_1: ─────┤ X ├
           └───┘

c1: 2/══════════


c2: 2/══════════
";
        assert_eq!(result, expected.trim_start_matches("\n"));
    }

    #[cfg(not(miri))]
    #[test]
    fn test_merge_wires() {
        let circuit = basic_circuit();

        let result = draw_circuit(&circuit, false, true, None).unwrap();
        let expected = "
      ┌───┐
 q_0: ┤ H ├──■──
      └───┘┌─┴─┐
 q_1: ─────┤ X ├
           └───┘
c1_0: ══════════

c1_1: ══════════

c2_0: ══════════

c2_1: ══════════
";
        assert_eq!(result, expected.trim_start_matches("\n"));
    }

    #[test]
    fn test_single_bit_registers() {
        // Single-bit registers should render as "q: " and "c: " (no auto-index suffix "_0").
        let qreg = QuantumRegister::new_owning("q", 1);
        let creg = ClassicalRegister::new_owning("c", 1);

        let qubits: Vec<ShareableQubit> = (0..qreg.len())
            .map(|i| qreg.get(i).expect("index in range"))
            .collect();
        let clbits: Vec<ShareableClbit> = (0..creg.len())
            .map(|i| creg.get(i).expect("index in range"))
            .collect();

        let mut circuit = CircuitData::new(Some(qubits), Some(clbits), Param::Float(0.0)).unwrap();
        _ = circuit.add_creg(creg, true);
        _ = circuit.add_qreg(qreg, true);

        circuit
            .push_standard_gate(StandardGate::H, &[], &[Qubit::new(0)])
            .unwrap();

        let inst = PackedInstruction {
            op: StandardInstruction::Measure.into(),
            qubits: circuit.add_qargs(&[Qubit::new(0)]),
            clbits: circuit.add_cargs(&[Clbit::new(0)]),
            params: None,
            label: None,
        };
        circuit.push(inst).unwrap();

        let result = draw_circuit(&circuit, false, false, Some(100)).unwrap();
        let expected = "
   ┌───┐┌───┐
q: ┤ H ├┤ M ├
   └───┘└─╥─┘
          ║
c: ═══════╩══
";
        assert_eq!(result, expected.trim_start_matches("\n"));
    }

    #[test]
    fn test_mixed_single_and_multi_bit_registers() {
        // Single-bit registers ("q", "c") mixed with multi-bit registers ("qr", "cr").
        // Single-bit ones render without the index suffix ("q:", "c:").
        // Multi-bit ones keep the index ("qr_0:", "qr_1:", "cr_0:", "cr_1:").
        let q = QuantumRegister::new_owning("q", 1);
        let qr = QuantumRegister::new_owning("qr", 2);
        let c = ClassicalRegister::new_owning("c", 1);
        let cr = ClassicalRegister::new_owning("cr", 2);

        let qubits: Vec<ShareableQubit> = (0..q.len())
            .map(|i| q.get(i).expect("index in range"))
            .chain((0..qr.len()).map(|i| qr.get(i).expect("index in range")))
            .collect();
        let clbits: Vec<ShareableClbit> = (0..c.len())
            .map(|i| c.get(i).expect("index in range"))
            .chain((0..cr.len()).map(|i| cr.get(i).expect("index in range")))
            .collect();

        let mut circuit = CircuitData::new(Some(qubits), Some(clbits), Param::Float(0.0)).unwrap();
        _ = circuit.add_creg(c, true);
        _ = circuit.add_creg(cr, true);
        _ = circuit.add_qreg(q, true);
        _ = circuit.add_qreg(qr, true);

        circuit
            .push_standard_gate(StandardGate::H, &[], &[Qubit::new(0)])
            .unwrap();
        circuit
            .push_standard_gate(StandardGate::H, &[], &[Qubit::new(1)])
            .unwrap();

        let result = draw_circuit(&circuit, false, false, Some(100)).unwrap();
        let expected = "
      ┌───┐
   q: ┤ H ├
      └───┘
      ┌───┐
qr_0: ┤ H ├
      └───┘

qr_1: ─────


   c: ═════


cr_0: ═════


cr_1: ═════
";
        assert_eq!(result, expected.trim_start_matches("\n"));
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
        let expected = "
      ┌───┐     »
 q_0: ┤ H ├──■──»
      └───┘  │  »
           ┌─┴─┐»
 q_1: ─────┤ X ├»
           └───┘»
                »
c1_0: ══════════»
                »
                »
c1_1: ══════════»
                »
                »
c2_0: ══════════»
                »
                »
c2_1: ══════════»
                »
«
« q_0: ──■────■──
«        │    │
«      ┌─┴─┐┌─┴─┐
« q_1: ┤ Y ├┤ Z ├
«      └───┘└───┘
«
«c1_0: ══════════
«
«
«c1_1: ══════════
«
«
«c2_0: ══════════
«
«
«c2_1: ══════════
«
";
        assert_eq!(result, expected.trim_start_matches("\n"));
    }

    #[test]
    fn test_label() {
        let qubits = vec![
            ShareableQubit::new_anonymous(),
            ShareableQubit::new_anonymous(),
        ];
        let mut circuit = CircuitData::new(Some(qubits), None, Param::Float(0.0)).unwrap();
        circuit
            .push_standard_gate(StandardGate::CH, &[], &[Qubit(0), Qubit(1)])
            .unwrap();
        circuit
            .push_standard_gate(
                StandardGate::RXX,
                &[Param::Float(1.23)],
                &[Qubit(0), Qubit(1)],
            )
            .unwrap();
        let mut inst_clone = circuit.data()[1].clone();
        inst_clone.label = Some(Box::new("my_rxx".to_string()));
        circuit.push(inst_clone).unwrap();
        let mut inst_clone = circuit.data()[0].clone();
        inst_clone.label = Some(Box::new("my_ch".to_string()));
        circuit.push(inst_clone).unwrap();
        let result = draw_circuit(&circuit, false, false, Some(80)).unwrap();
        let expected = "
          ┌─────────────┐┌────────────────┐
q_0: ──■──┤0  Rxx(1.23) ├┤0  my_rxx(1.23) ├────■────
       │  │             ││                │    │
     ┌─┴─┐│             ││                │┌───┴───┐
q_1: ┤ H ├┤1            ├┤1               ├┤ my_ch ├
     └───┘└─────────────┘└────────────────┘└───────┘
";
        assert_eq!(result, expected.trim_start_matches("\n"));
    }

    #[test]
    fn test_unitary() {
        let qubits = vec![
            ShareableQubit::new_anonymous(),
            ShareableQubit::new_anonymous(),
            ShareableQubit::new_anonymous(),
            ShareableQubit::new_anonymous(),
        ];
        let one_qubit_iden = Array2::eye(2);
        let two_qubit_iden = Array2::eye(4);
        let three_qubit_iden = Array2::eye(8);
        let four_qubit_iden = Array2::eye(16);
        let mut circuit = CircuitData::new(Some(qubits), None, Param::Float(0.0)).unwrap();
        let inst = PackedInstruction {
            op: Box::new(UnitaryGate {
                array: ArrayType::NDArray(one_qubit_iden.clone()),
            })
            .into(),
            qubits: circuit.add_qargs(&[Qubit(0)]),
            clbits: circuit.cargs_interner().get_default(),
            params: None,
            label: None,
        };
        circuit.push(inst).unwrap();
        let inst = PackedInstruction {
            op: Box::new(UnitaryGate {
                array: ArrayType::NDArray(one_qubit_iden),
            })
            .into(),
            qubits: circuit.add_qargs(&[Qubit(2)]),
            clbits: circuit.cargs_interner().get_default(),
            params: None,
            label: Some(Box::new("my little identity".to_string())),
        };
        circuit.push(inst).unwrap();
        let inst = PackedInstruction {
            op: Box::new(UnitaryGate {
                array: ArrayType::NDArray(two_qubit_iden.clone()),
            })
            .into(),
            qubits: circuit.add_qargs(&[Qubit(1), Qubit(3)]),
            clbits: circuit.cargs_interner().get_default(),
            params: None,
            label: None,
        };
        circuit.push(inst).unwrap();
        let inst = PackedInstruction {
            op: Box::new(UnitaryGate {
                array: ArrayType::NDArray(two_qubit_iden),
            })
            .into(),
            qubits: circuit.add_qargs(&[Qubit(0), Qubit(2)]),
            clbits: circuit.cargs_interner().get_default(),
            params: None,
            label: Some(Box::new("my small identity".to_string())),
        };
        circuit.push(inst).unwrap();
        let inst = PackedInstruction {
            op: Box::new(UnitaryGate {
                array: ArrayType::NDArray(three_qubit_iden.clone()),
            })
            .into(),
            qubits: circuit.add_qargs(&[Qubit(1), Qubit(3), Qubit(0)]),
            clbits: circuit.cargs_interner().get_default(),
            params: None,
            label: None,
        };
        circuit.push(inst).unwrap();
        let inst = PackedInstruction {
            op: Box::new(UnitaryGate {
                array: ArrayType::NDArray(three_qubit_iden),
            })
            .into(),
            qubits: circuit.add_qargs(&[Qubit(0), Qubit(2), Qubit(3)]),
            clbits: circuit.cargs_interner().get_default(),
            params: None,
            label: Some(Box::new("my medium identity".to_string())),
        };
        circuit.push(inst).unwrap();
        let inst = PackedInstruction {
            op: Box::new(UnitaryGate {
                array: ArrayType::NDArray(four_qubit_iden.clone()),
            })
            .into(),
            qubits: circuit.add_qargs(&[Qubit(1), Qubit(3), Qubit(0), Qubit(2)]),
            clbits: circuit.cargs_interner().get_default(),
            params: None,
            label: None,
        };
        circuit.push(inst).unwrap();
        let inst = PackedInstruction {
            op: Box::new(UnitaryGate {
                array: ArrayType::NDArray(four_qubit_iden),
            })
            .into(),
            qubits: circuit.add_qargs(&[Qubit(0), Qubit(2), Qubit(3), Qubit(1)]),
            clbits: circuit.cargs_interner().get_default(),
            params: None,
            label: Some(Box::new("my bigger identity".to_string())),
        };
        circuit.push(inst).unwrap();
        let result = draw_circuit(&circuit, false, false, Some(80)).unwrap();
        let expected = "
         ┌───────────┐                    ┌─────────────────────┐┌──────────────┐»
q_0: ────┤  Unitary  ├────────────────────┤0                    ├┤ 2            ├»
         └───────────┘                    │                     ││              │»
                           ┌─────────────┐│                     ││              │»
q_1: ──────────────────────┤0            ├┤   my small identity ├┤ 0   Unitary  ├»
                           │             ││                     ││              │»
     ┌────────────────────┐│             ││                     ││              │»
q_2: ┤ my little identity ├┤    Unitary  ├┤1                    ├┤              ├»
     └────────────────────┘│             │└─────────────────────┘│              │»
                           │             │                       │              │»
q_3: ──────────────────────┤1            ├───────────────────────┤ 1            ├»
                           └─────────────┘                       └──────────────┘»
«     ┌───────────────────────┐┌───────────────┐┌────────────────────────┐
«q_0: ┤ 0                     ├┤ 2             ├┤ 0                      ├
«     │                       ││               ││                        │
«     │                       ││               ││                        │
«q_1: ┤    my medium identity ├┤ 0    Unitary  ├┤ 3   my bigger identity ├
«     │                       ││               ││                        │
«     │                       ││               ││                        │
«q_2: ┤ 1                     ├┤ 3             ├┤ 1                      ├
«     │                       ││               ││                        │
«     │                       ││               ││                        │
«q_3: ┤ 2                     ├┤ 1             ├┤ 2                      ├
«     └───────────────────────┘└───────────────┘└────────────────────────┘
";
        assert_eq!(result, expected.trim_start_matches("\n"));
    }

    #[test]
    fn test_all_standard_gates() {
        let qubits = vec![
            ShareableQubit::new_anonymous(),
            ShareableQubit::new_anonymous(),
            ShareableQubit::new_anonymous(),
            ShareableQubit::new_anonymous(),
            ShareableQubit::new_anonymous(),
        ];
        let num_wires = qubits.len() as u32;
        let mut circuit = CircuitData::new(Some(qubits), None, Param::Float(0.0)).unwrap();
        for i in 0..STANDARD_GATE_SIZE {
            let op: StandardGate = unsafe { std::mem::transmute(i as u8) };
            let num_qubits = op.num_qubits();
            let num_params = op.num_params();
            let qubits = (0..num_qubits)
                .map(|x| Qubit(x + (i as u32 % (num_wires - num_qubits))))
                .collect::<Vec<_>>();
            #[allow(clippy::approx_constant)]
            let params = (0..num_params)
                .map(|_x| 3.141.into())
                .collect::<Vec<Param>>();
            circuit.push_standard_gate(op, &params, &qubits).unwrap();
        }
        let result = draw_circuit(&circuit, false, false, Some(80)).unwrap();
        let expected = "
     ┌───┐  ┌───────────┐      ┌─────┐   ┌─────┐ ┌───────────────────────┐          »
q_0: ┤ Y ├──┤ Rx(3.141) ├──────┤ Sdg ├───┤ Tdg ├─┤ U3(3.141,3.141,3.141) ├──■───────»
     └───┘  └───────────┘      └─────┘   └─────┘ └───────────────────────┘  │       »
     ┌───┐      ┌───┐       ┌───────────┐ ┌────┐ ┌──────────────────────┐ ┌─┴─┐     »
q_1: ┤ H ├──────┤ Z ├───────┤ Ry(3.141) ├─┤ √X ├─┤ U(3.141,3.141,3.141) ├─┤ H ├──■──»
     └───┘      └───┘       └───────────┘ └────┘ └──────────────────────┘ └───┘  │  »
     ┌───┐   ┌──────────┐   ┌───────────┐┌──────┐      ┌───────────┐           ┌─┴─┐»
q_2: ┤ I ├───┤ P(3.141) ├───┤ Rz(3.141) ├┤ √Xdg ├──────┤ U1(3.141) ├───────────┤ X ├»
     └───┘   └──────────┘   └───────────┘└──────┘      └───────────┘           └───┘»
     ┌───┐┌────────────────┐    ┌───┐     ┌───┐     ┌─────────────────┐             »
q_3: ┤ X ├┤ R(3.141,3.141) ├────┤ S ├─────┤ T ├─────┤ U2(3.141,3.141) ├─────────────»
     └───┘└────────────────┘    └───┘     └───┘     └─────────────────┘             »
                                                                                    »
q_4: ───────────────────────────────────────────────────────────────────────────────»
                                                                                    »
«                                                                                     »
«q_0: ──■───────────────X─────────────────────■─────────────────────────■─────────────»
«       │               │                     │                         │             »
«     ┌─┴─┐┌───────┐    │    ┌─────────┐┌─────┴─────┐                 ┌─┴─┐           »
«q_1: ┤ Z ├┤0  Dcx ├────X────┤0  Iswap ├┤ Rx(3.141) ├──────■──────────┤ S ├───────■───»
«     └───┘│       │         │         │└───────────┘      │          └───┘       │   »
«          │       │┌───────┐│         │             ┌─────┴─────┐             ┌──┴──┐»
«q_2: ──■──┤1      ├┤0  Ecr ├┤1        ├──────■──────┤ Ry(3.141) ├──────■──────┤ Sdg ├»
«       │  └───────┘│       │└─────────┘      │      └───────────┘      │      └─────┘»
«     ┌─┴─┐         │       │           ┌─────┴─────┐             ┌─────┴─────┐       »
«q_3: ┤ Y ├─────────┤1      ├───────────┤ P(3.141)  ├─────────────┤ Rz(3.141) ├───────»
«     └───┘         └───────┘           └───────────┘             └───────────┘       »
«                                                                                     »
«q_4: ────────────────────────────────────────────────────────────────────────────────»
«                                                                                     »
«                                                     ┌──────────────┐     »
«q_0: ───────────────■────────────────────────────────┤0  Rxx(3.141) ├─────»
«                    │                                │              │     »
«     ┌──────────────┴──────────────┐                 │              │     »
«q_1: ┤ U(3.141,3.141,3.141,3.141)  ├──────■──────────┤1             ├─────»
«     └─────────────────────────────┘      │          └──────────────┘     »
«                                    ┌─────┴─────┐                         »
«q_2: ───────────────■───────────────┤ U1(3.141) ├────────────■────────────»
«                    │               └───────────┘            │            »
«                 ┌──┴──┐                         ┌───────────┴───────────┐»
«q_3: ────────────┤ Sx  ├─────────────────────────┤ U3(3.141,3.141,3.141) ├»
«                 └─────┘                         └───────────────────────┘»
«                                                                          »
«q_4: ─────────────────────────────────────────────────────────────────────»
«                                                                          »
«                     ┌──────────────┐                                                »
«q_0: ────────────────┤0  Rzx(3.141) ├────────────────────────────────────────────────»
«                     │              │                                                »
«     ┌──────────────┐│              │┌──────────────────────┐                        »
«q_1: ┤0  Ryy(3.141) ├┤1             ├┤0  XX-YY(3.141,3.141) ├────────────────────────»
«     │              │└──────────────┘│                      │                        »
«     │              │┌──────────────┐│                      │┌──────────────────────┐»
«q_2: ┤1             ├┤0  Rzz(3.141) ├┤1                     ├┤0  XX+YY(3.141,3.141) ├»
«     └──────────────┘│              │└──────────────────────┘│                      │»
«                     │              │                        │                      │»
«q_3: ────────────────┤1             ├────────────────────────┤1                     ├»
«                     └──────────────┘                        └──────────────────────┘»
«                                                                                     »
«q_4: ────────────────────────────────────────────────────────────────────────────────»
«                                                                                     »
«                  ┌─────────┐            ┌───────────┐
«q_0: ───────■─────┤ 0       ├──■─────■───┤ 0         ├
«            │     │         │  │     │   │           │
«            │     │         │  │     │   │           │
«q_1: ──■────■───■─┤ 1  Rccx ├──■─────■───┤ 1   Rcccx ├
«       │    │   │ │         │  │     │   │           │
«       │  ┌─┴─┐ │ │         │  │     │   │           │
«q_2: ──■──┤ Z ├─X─┤ 2       ├──■─────■───┤ 2         ├
«       │  └───┘ │ └─────────┘  │     │   │           │
«     ┌─┴─┐      │            ┌─┴─┐┌──┴──┐│           │
«q_3: ┤ X ├──────X────────────┤ X ├┤ Sx  ├┤ 3         ├
«     └───┘                   └───┘└─────┘└───────────┘
«
«q_4: ─────────────────────────────────────────────────
«
";
        assert_eq!(result, expected.trim_start_matches("\n"));
    }

    #[cfg(not(miri))]
    #[test]
    #[allow(clippy::approx_constant)]
    fn test_global_phase() {
        let mut circuit = basic_circuit();
        circuit.set_global_phase_param(3.14.into()).unwrap();
        let result = draw_circuit(&circuit, true, false, None).unwrap();

        let expected = "
global phase: 3.14
      ┌───┐
 q_0: ┤ H ├──■──
      └───┘  │
           ┌─┴─┐
 q_1: ─────┤ X ├
           └───┘

c1: 2/══════════


c2: 2/══════════
";
        assert_eq!(result, expected.trim_start_matches("\n"));
    }

    #[test]
    fn test_global_phase_parameterized() {
        let mut circuit = basic_circuit();
        circuit
            .set_global_phase_param(Param::ParameterExpression(Arc::new(
                ParameterExpression::from_symbol(Symbol::new("ϕ", None, None)),
            )))
            .unwrap();
        let result = draw_circuit(&circuit, true, false, Some(80)).unwrap();

        let expected = "
global phase: ϕ
      ┌───┐
 q_0: ┤ H ├──■──
      └───┘  │
           ┌─┴─┐
 q_1: ─────┤ X ├
           └───┘

c1: 2/══════════


c2: 2/══════════
";
        assert_eq!(result, expected.trim_start_matches("\n"));
    }

    #[test]
    fn test_parameterized_standard_gate() {
        let qubits = vec![
            ShareableQubit::new_anonymous(),
            ShareableQubit::new_anonymous(),
        ];
        let mut circuit = CircuitData::new(Some(qubits), None, Param::Float(0.0)).unwrap();
        let param = Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(
            Symbol::new("a", None, None),
        )));
        circuit
            .push_standard_gate(StandardGate::RXX, &[param], &[Qubit(0), Qubit(1)])
            .unwrap();
        let mut inst_clone = circuit.data()[0].clone();
        inst_clone.label = Some(Box::new("my_rxx".to_string()));
        circuit.push(inst_clone).unwrap();
        circuit
            .push_standard_gate(
                StandardGate::RZX,
                &[Param::Float(2.)],
                &[Qubit(0), Qubit(1)],
            )
            .unwrap();
        let result = draw_circuit(&circuit, false, false, Some(100)).unwrap();
        let expected = "
     ┌──────────┐┌─────────────┐┌──────────┐
q_0: ┤0  Rxx(a) ├┤0  my_rxx(a) ├┤0  Rzx(2) ├
     │          ││             ││          │
     │          ││             ││          │
q_1: ┤1         ├┤1            ├┤1         ├
     └──────────┘└─────────────┘└──────────┘
";
        assert_eq!(result, expected.trim_start_matches("\n"));
    }

    #[test]
    fn test_all_standard_instructions() {
        let qubits = vec![
            ShareableQubit::new_anonymous(),
            ShareableQubit::new_anonymous(),
            ShareableQubit::new_anonymous(),
            ShareableQubit::new_anonymous(),
        ];
        let clbits = vec![
            ShareableClbit::new_anonymous(),
            ShareableClbit::new_anonymous(),
            ShareableClbit::new_anonymous(),
            ShareableClbit::new_anonymous(),
        ];

        let mut circuit = CircuitData::new(Some(qubits), Some(clbits), Param::Float(0.0)).unwrap();
        for i in 0..circuit.num_qubits() {
            let inst = PackedInstruction {
                op: StandardInstruction::Reset.into(),
                qubits: circuit.add_qargs(&[Qubit::new(i)]),
                clbits: circuit.cargs_interner().get_default(),
                params: None,
                label: None,
            };
            circuit.push(inst).unwrap();
        }
        for i in 0..circuit.num_qubits() {
            for unit in [
                DelayUnit::NS,
                DelayUnit::PS,
                DelayUnit::US,
                DelayUnit::MS,
                DelayUnit::S,
            ] {
                let param = Param::Float(2.1);
                let inst = PackedInstruction {
                    op: StandardInstruction::Delay(unit).into(),
                    qubits: circuit.add_qargs(&[Qubit::new(i)]),
                    clbits: circuit.cargs_interner().get_default(),
                    params: Some(Box::new(Parameters::Params(smallvec![param]))),
                    label: None,
                };
                circuit.push(inst).unwrap();
            }
        }
        for i in [1, 2, 3, 4] {
            let qubits = (0..i).map(Qubit::new).collect::<Vec<_>>();
            let inst = PackedInstruction {
                op: StandardInstruction::Barrier(i as u32).into(),
                qubits: circuit.add_qargs(&qubits),
                clbits: circuit.cargs_interner().get_default(),
                params: None,
                label: None,
            };
            circuit.push(inst).unwrap();
        }
        for i in 0..circuit.num_qubits() {
            let inst = PackedInstruction {
                op: StandardInstruction::Measure.into(),
                qubits: circuit.add_qargs(&[Qubit::new(i)]),
                clbits: circuit.add_cargs(&[Clbit::new(i)]),
                params: None,
                label: None,
            };
            circuit.push(inst).unwrap();
        }

        let result = draw_circuit(&circuit, false, false, Some(100)).unwrap();
        let expected = "
          ┌────────────────┐┌────────────────┐┌────────────────┐┌────────────────┐┌───────────────┐ ░  ░ »
q_0: ─|0>─┤ Delay(2.1[ns]) ├┤ Delay(2.1[ps]) ├┤ Delay(2.1[us]) ├┤ Delay(2.1[ms]) ├┤ Delay(2.1[s]) ├─░──░─»
          └────────────────┘└────────────────┘└────────────────┘└────────────────┘└───────────────┘ ░  ░ »
          ┌────────────────┐┌────────────────┐┌────────────────┐┌────────────────┐┌───────────────┐    ░ »
q_1: ─|0>─┤ Delay(2.1[ns]) ├┤ Delay(2.1[ps]) ├┤ Delay(2.1[us]) ├┤ Delay(2.1[ms]) ├┤ Delay(2.1[s]) ├────░─»
          └────────────────┘└────────────────┘└────────────────┘└────────────────┘└───────────────┘    ░ »
          ┌────────────────┐┌────────────────┐┌────────────────┐┌────────────────┐┌───────────────┐      »
q_2: ─|0>─┤ Delay(2.1[ns]) ├┤ Delay(2.1[ps]) ├┤ Delay(2.1[us]) ├┤ Delay(2.1[ms]) ├┤ Delay(2.1[s]) ├──────»
          └────────────────┘└────────────────┘└────────────────┘└────────────────┘└───────────────┘      »
          ┌────────────────┐┌────────────────┐┌────────────────┐┌────────────────┐┌───────────────┐      »
q_3: ─|0>─┤ Delay(2.1[ns]) ├┤ Delay(2.1[ps]) ├┤ Delay(2.1[us]) ├┤ Delay(2.1[ms]) ├┤ Delay(2.1[s]) ├──────»
          └────────────────┘└────────────────┘└────────────────┘└────────────────┘└───────────────┘      »
                                                                                                         »
c_0: ════════════════════════════════════════════════════════════════════════════════════════════════════»
                                                                                                         »
                                                                                                         »
c_1: ════════════════════════════════════════════════════════════════════════════════════════════════════»
                                                                                                         »
                                                                                                         »
c_2: ════════════════════════════════════════════════════════════════════════════════════════════════════»
                                                                                                         »
                                                                                                         »
c_3: ════════════════════════════════════════════════════════════════════════════════════════════════════»
                                                                                                         »
«      ░  ░ ┌───┐
«q_0: ─░──░─┤ M ├───────────────
«      ░  ░ └─╥─┘
«      ░  ░   ║  ┌───┐
«q_1: ─░──░───╫──┤ M ├──────────
«      ░  ░   ║  └─╥─┘
«      ░  ░   ║    ║  ┌───┐
«q_2: ─░──░───╫────╫──┤ M ├─────
«      ░  ░   ║    ║  └─╥─┘
«         ░   ║    ║    ║  ┌───┐
«q_3: ────░───╫────╫────╫──┤ M ├
«         ░   ║    ║    ║  └─╥─┘
«             ║    ║    ║    ║
«c_0: ════════╩════╬════╬════╬══
«                  ║    ║    ║
«                  ║    ║    ║
«c_1: ═════════════╩════╬════╬══
«                       ║    ║
«                       ║    ║
«c_2: ══════════════════╩════╬══
«                            ║
«                            ║
«c_3: ═══════════════════════╩══
«
";
        assert_eq!(result, expected.trim_start_matches("\n"));
    }

    #[test]
    fn test_unicode() {
        let qubits = vec![
            ShareableQubit::new_anonymous(),
            ShareableQubit::new_anonymous(),
        ];
        let mut circuit = CircuitData::new(Some(qubits), None, Param::Float(0.0)).unwrap();
        let param = Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(
            Symbol::new("ϕ", None, None),
        )));
        circuit
            .push_standard_gate(StandardGate::RXX, &[param], &[Qubit(0), Qubit(1)])
            .unwrap();
        let mut inst_clone = circuit.data()[0].clone();
        inst_clone.label = Some(Box::new("μου_rxx".to_string()));
        circuit.push(inst_clone).unwrap();
        circuit
            .push_standard_gate(
                StandardGate::RZX,
                &[Param::Float(2.)],
                &[Qubit(0), Qubit(1)],
            )
            .unwrap();
        let result = draw_circuit(&circuit, false, false, Some(100)).unwrap();
        let expected = "
     ┌──────────┐┌──────────────┐┌──────────┐
q_0: ┤0  Rxx(ϕ) ├┤0  μου_rxx(ϕ) ├┤0  Rzx(2) ├
     │          ││              ││          │
     │          ││              ││          │
q_1: ┤1         ├┤1             ├┤1         ├
     └──────────┘└──────────────┘└──────────┘
";
        assert_eq!(result, expected.trim_start_matches("\n"));
    }

    #[test]
    fn test_unicode_emoji() {
        let qubits = vec![
            ShareableQubit::new_anonymous(),
            ShareableQubit::new_anonymous(),
        ];
        let mut circuit = CircuitData::new(Some(qubits), None, Param::Float(0.0)).unwrap();
        let param = Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(
            Symbol::new("🎩", None, None),
        )));
        circuit
            .push_standard_gate(StandardGate::RY, &[param.clone()], &[Qubit(1)])
            .unwrap();
        circuit
            .push_standard_gate(StandardGate::RXX, &[param], &[Qubit(0), Qubit(1)])
            .unwrap();
        let mut inst_clone = circuit.data()[0].clone();
        inst_clone.label = Some(Box::new("💶🔉".to_string()));
        circuit.push(inst_clone).unwrap();
        let mut inst_clone = circuit.data()[1].clone();
        inst_clone.label = Some(Box::new("💶🔉".to_string()));
        circuit.push(inst_clone).unwrap();
        circuit
            .push_standard_gate(
                StandardGate::RZX,
                &[Param::Float(2.)],
                &[Qubit(0), Qubit(1)],
            )
            .unwrap();
        let result = draw_circuit(&circuit, false, false, Some(100)).unwrap();
        let expected = "
               ┌───────────┐            ┌────────────┐┌──────────┐
q_0: ──────────┤0  Rxx(🎩) ├────────────┤0  💶🔉(🎩) ├┤0  Rzx(2) ├
               │           │            │            ││          │
     ┌────────┐│           │┌──────────┐│            ││          │
q_1: ┤ Ry(🎩) ├┤1          ├┤ 💶🔉(🎩) ├┤1           ├┤1         ├
     └────────┘└───────────┘└──────────┘└────────────┘└──────────┘
";
        assert_eq!(result, expected.trim_start_matches("\n"));
    }

    #[test]
    fn test_box_op() {
        let qubits = (0..3).map(|_| ShareableQubit::new_anonymous()).collect();
        let mut outer_box_circ = CircuitData::new(Some(qubits), None, Param::Float(0.0)).unwrap();
        outer_box_circ
            .push_standard_gate(StandardGate::H, &[], &[Qubit(0)])
            .unwrap();
        outer_box_circ
            .push_standard_gate(StandardGate::CX, &[], &[Qubit(0), Qubit(1)])
            .unwrap();

        let qubits = vec![ShareableQubit::new_anonymous()];
        let mut inner_box_circ = CircuitData::new(Some(qubits), None, Param::Float(0.0)).unwrap();
        inner_box_circ
            .push_standard_gate(StandardGate::S, &[], &[Qubit(0)])
            .unwrap();

        let cf_inst_inner = ControlFlowInstruction {
            control_flow: ControlFlow::Box {
                duration: None,
                annotations: Vec::new(),
            },
            num_qubits: 1,
            num_clbits: 0,
        };
        let block = outer_box_circ.add_block(inner_box_circ);
        outer_box_circ
            .push_packed_operation(
                PackedOperation::from_control_flow(cf_inst_inner.into()),
                Some(Parameters::Blocks(vec![block])),
                &[Qubit(0)],
                &[],
            )
            .unwrap();

        outer_box_circ
            .push_standard_gate(StandardGate::CCX, &[], &[Qubit(2), Qubit(0), Qubit(1)])
            .unwrap();

        let qubits = (0..5).map(|_| ShareableQubit::new_anonymous()).collect();
        let mut top_level_circuit =
            CircuitData::new(Some(qubits), None, Param::Float(0.0)).unwrap();

        let cf_inst = ControlFlowInstruction {
            control_flow: ControlFlow::Box {
                duration: None,
                annotations: Vec::new(),
            },
            num_qubits: 2,
            num_clbits: 0,
        };
        let block = top_level_circuit.add_block(outer_box_circ);
        top_level_circuit
            .push_packed_operation(
                PackedOperation::from_control_flow(cf_inst.into()),
                Some(Parameters::Blocks(vec![block])),
                &[Qubit(1), Qubit(3), Qubit(2)],
                &[],
            )
            .unwrap();
        top_level_circuit
            .push_standard_gate(StandardGate::Y, &[], &[Qubit(4)])
            .unwrap();
        top_level_circuit
            .push_standard_gate(StandardGate::Z, &[], &[Qubit(0)])
            .unwrap();

        let result = draw_circuit(&top_level_circuit, false, false, Some(100)).unwrap();

        println!("{}", result);
        let _expected = "
     ┌───┐
q_0: ┤ Z ├───────────────────────
     └───┘
       ┆  ┌───┐     ┆┌───┐┆     ┆
q_1: ──┼──┤ H ├──■──┼┤ S ├┼──■──┼
       ┆  └───┘  │  ┆└───┘┆  │  ┆
       ┆         │           │  ┆
q_2: ──┼─────────┼───────────■──┼
       ┆         │           │  ┆
       ┆       ┌─┴─┐       ┌─┴─┐┆
q_3: ──┼───────┤ X ├───────┤ X ├┼
       ┆       └───┘       └───┘┆
     ┌───┐
q_4: ┤ Y ├───────────────────────
     └───┘
";

        // TODO: test the case where the box spans classical bits

        // TODO: test the case where there are measures or PPM inside a Box op

        // TODO: test the case of box inside box where the top/bottom frames collide
        //      ┌─────── ┌───┐     ┌───┐┌─────── ┌───┐ ───────┐  ───────┐
        // q_0: ┤        ┤ H ├──■──┤ X ├┤        ┤ X ├        ├─        ├─
        //      │        └───┘  │  └─┬─┘│        └─┬─┘        │         │
        //      │             ┌─┴─┐  │  │          │          │         │
        // q_1: ┤ Box-0  ─────┤ X ├──■──┤ Box-1  ──┼──  End-1 ├─  End-0 ├─
        //      │             └───┘  │  │          │          │         │
        //      │                    │  │          │          │         │
        // q_2: ┤        ────────────■──┤        ──■──        ├─        ├─
        //      └───────                └───────       ───────┘  ───────┘
    }
}
