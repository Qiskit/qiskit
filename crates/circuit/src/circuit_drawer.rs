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
use crate::operations::{Operation, OperationRef, Param, StandardGate, StandardInstruction};
use crate::packed_instruction::PackedInstruction;
use crate::{Clbit, Qubit};
use approx;
use crossterm::terminal;
use hashbrown::HashSet;
use itertools::{Itertools, MinMaxResult};
use lexical_core::ToLexicalWithOptions;
use lexical_write_float::{self, format::STANDARD};
use pyo3::prelude::*;
use std::f64::consts::PI;
use std::fmt::Debug;
use std::ops::Index;
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
                format!(
                    "global phase: {}\n",
                    F64UiFormatter::new(5).format_with_pi(*f)
                )
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

/// Return a list of layers such that each layer contains a list of op node indices, representing instructions
/// whose qubits/clbits indices do not overlap. The instruction are packed into each layer as long as there
/// is no qubit/clbit overlap.
fn build_layers(circ: &CircuitData) -> Vec<Vec<&PackedInstruction>> {
    let mut layers: Vec<Vec<&PackedInstruction>> = Vec::new();
    let num_qubits = circ.num_qubits();
    let num_clbits = circ.num_clbits();
    let mut layers_frontier = vec![0usize; num_qubits + num_clbits];

    for inst in circ.data() {
        if matches!(
            inst.op.view(),
            OperationRef::StandardGate(StandardGate::GlobalPhase)
        ) {
            if layers.last().is_none() {
                layers.push(Vec::new())
            }
            layers.last_mut().unwrap().push(inst);
            continue;
        }
        let (node_min, node_max) = get_instruction_range(
            circ.get_qargs(inst.qubits),
            circ.get_cargs(inst.clbits),
            num_qubits,
        );

        let layer_idx = *layers_frontier[node_min..=node_max]
            .iter()
            .max()
            .expect("Instruction range should be valid");
        if layers.len() == layer_idx {
            layers.push(Vec::new());
        }
        layers[layer_idx].push(inst);
        layers_frontier[node_min..=node_max].fill(layer_idx + 1);
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
                    inst.op.name()
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
        if gate == StandardGate::GlobalPhase {
            return;
        }

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
        let inst_layers = build_layers(circuit);

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
            for inst in layer {
                layers[i + 1].add_instruction(bundle_cregs, inst, circuit, &clbit_map);
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

/// A formatter for UI rendering of floating-point numbers
///
/// Supports formatting similar to Python's `g` or C printf's `%g` format specifiers
/// as well as formatting of multiples and fractions of pi.
///
/// Example outputs:
/// ```text
/// F64UiFormatter::new(4).format(1.23456)        → 1.235
/// F64UiFormatter::new(4).format(123.456)        → 123.5
/// F64UiFormatter::new(5).format(12345678.0)     → 1.2346e7
/// F64UiFormatter::new(5).format(-0.00001234)    → -1.234e-5
/// F64UiFormatter::new(5).format_with_pi(5π/6)   → 5π/6
/// ```
struct F64UiFormatter {
    buffer: Vec<u8>,
    options: lexical_write_float::Options,
}

impl F64UiFormatter {
    fn new(num_significant_digits: usize) -> Self {
        let options = lexical_write_float::Options::builder()
            .max_significant_digits(core::num::NonZeroUsize::new(num_significant_digits))
            .positive_exponent_break(core::num::NonZeroI32::new(num_significant_digits as i32))
            .negative_exponent_break(core::num::NonZeroI32::new(
                -(num_significant_digits as i32) + 1,
            ))
            .trim_floats(true)
            .build_strict();

        F64UiFormatter {
            buffer: vec![0u8; options.buffer_size_const::<f64, STANDARD>()],
            options,
        }
    }

    /// Formats the input number based on the formatting options.
    /// This Can be called multiple times, but the internal buffer is overwritten on each call.
    fn format(&mut self, num: f64) -> &str {
        let buf = num.to_lexical_with_options::<STANDARD>(&mut self.buffer, &self.options);
        std::str::from_utf8_mut(buf).expect("Byte representation should be valid")
    }

    /// Tries to format the string as a multiple or simple fraction of pi if possible,
    /// otherwise falls back to the simpler [F64UiFormatter::format] logic
    fn format_with_pi(&mut self, num: f64) -> String {
        format_float_pi(num).unwrap_or_else(|| self.format(num).to_owned())
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
                                format!(
                                    "Delay({}[{}])",
                                    F64UiFormatter::new(5).format(*duration),
                                    delay_unit
                                )
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
                            Param::Float(f) => {
                                F64UiFormatter::new(5).format_with_pi(*f).to_string()
                            }
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
                        let mut label = format!(" {} ", Self::get_label(inst));
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
                                // This ensures the top_con/bot_con connectors are properly aligned with the control
                                // lines regardless of whether the text element padding size is odd or even.
                                (label.len() % 2 == 0).then(|| label.push(' '));
                            }
                        } else if let Some(std_inst) = inst.op.try_standard_instruction() {
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
                        let label = format!(" {} ", Self::get_label(inst));
                        let label_len = label.width();
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
                                " ".repeat(mid_section.width()),
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
                top = " ".repeat(input_name.width());
                bot = " ".repeat(input_name.width());
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
        ranges.push((start, num_layers));

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

/// Computes if a number is close to an integer
/// fraction or multiple of PI and returns the
/// corresponding string.
///
/// Args:
///     f : Number to check.
///
/// Returns:
///     The string representation of output. None if no Pi formatting is found.  
pub fn format_float_pi(f: f64) -> Option<String> {
    const DENOMINATOR: i64 = 16;
    // epsilon value defines the threshold to detect pi.
    const EPS: f64 = 1e-9;

    // pi_str is needed to match the output expected according to the format needed
    let pi_str = "π";

    // f_abs and sign help us working through each steps
    let f_abs = f.abs();
    let sign = if f < 0.0 { "-" } else { "" };

    // Detecting 0 before moving on
    if f_abs < EPS {
        return Some("0".to_string());
    }

    // First check is for whole multiples of pi
    let val = f_abs / PI;
    let round = val.round();
    if val >= 1.0 - EPS && (val - round).abs() < EPS {
        let round = round as usize;
        return Some(if round == 1 {
            format!("{}{}", sign, pi_str)
        } else {
            format!("{}{}{}", sign, round, pi_str)
        });
    }

    // Second is a check for powers of pi
    if f_abs > PI {
        if let Some(k) = (2..=4).find(|k| (f_abs - PI.powi(*k)).abs() < EPS) {
            return Some(format!("{}{}^{}", sign, pi_str, k));
        }
    }

    // Third is a check for a number larger than DENOMINATOR * pi, not a
    // multiple or power of pi, since no fractions will exceed DENOMINATOR * pi
    if f_abs > (DENOMINATOR as f64 * PI) {
        return None;
    }

    // Fourth check is for fractions for 1*pi in the numer and any
    // number in the denom.
    let val = PI / f_abs;
    let round = val.round();
    if round >= 1.0 && (val - round).abs() < EPS {
        let d = round as usize;
        let str_out = format!("{}{}/{}", sign, pi_str, d);
        return Some(str_out);
    }

    // Fifth check is for fractions of the form (numer/denom) * pi or (numer/denom) / pi
    // where 1 <= numer,denom <= DENOMINATOR, which are not covered in the previous checks.
    // Ex. 15pi/16, 2pi/5, 15pi/2, 16pi/9 or 15/16pi, 2/5pi, 15/2pi, 16/9pi
    for denom in 1..=DENOMINATOR {
        for numer in 1..=DENOMINATOR {
            let up = numer as f64 / denom as f64;
            let val = up * PI;
            if (f_abs - val).abs() < EPS {
                let str_out = format!("{}{}{}/{}", sign, numer, pi_str, denom);
                return Some(str_out);
            }
            let val = up / PI;
            if (f_abs - val).abs() < EPS {
                let str_out = match denom {
                    1 => format!("{}{}/{}", sign, numer, pi_str),
                    d => format!("{}{}/{}{}", sign, numer, d, pi_str),
                };
                return Some(str_out);
            }
        }
    }

    // fall back when no conversion is possible
    None
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use smallvec::smallvec;
    use std::sync::Arc;

    use super::*;
    use crate::bit::{ClassicalRegister, QuantumRegister, ShareableClbit, ShareableQubit};
    use crate::instruction::Parameters;
    use crate::operations::{
        ArrayType, DelayUnit, STANDARD_GATE_SIZE, StandardInstruction, UnitaryGate,
    };
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

    #[cfg(not(miri))]
    #[test]
    fn test_f64_formatting() {
        let qubits = vec![
            ShareableQubit::new_anonymous(),
            ShareableQubit::new_anonymous(),
        ];
        let mut circuit = CircuitData::new(Some(qubits), None, Param::Float(0.8 * PI)).unwrap();

        circuit
            .push_standard_gate(StandardGate::RX, &[Param::Float(1.234567)], &[Qubit(0)])
            .unwrap();
        circuit
            .push_standard_gate(StandardGate::RX, &[Param::Float(123.4567)], &[Qubit(0)])
            .unwrap();

        let expr = ParameterExpression::from_symbol(Symbol::new("ϕ", None, None))
            .mul(&ParameterExpression::from_f64(1.23456))
            .unwrap();
        let param = Param::ParameterExpression(Arc::new(expr));
        circuit
            .push_standard_gate(StandardGate::RY, &[param], &[Qubit(0)])
            .unwrap();
        circuit
            .push_standard_gate(StandardGate::RZ, &[Param::Float(123456789f64)], &[Qubit(1)])
            .unwrap();

        circuit
            .push_standard_gate(StandardGate::RX, &[Param::Float(0.1234567)], &[Qubit(1)])
            .unwrap();
        circuit
            .push_standard_gate(StandardGate::RX, &[Param::Float(0.0000123456)], &[Qubit(1)])
            .unwrap();
        circuit
            .push_standard_gate(
                StandardGate::RX,
                &[Param::Float(2.0 / 3.0 * PI)],
                &[Qubit(1)],
            )
            .unwrap();

        let result = draw_circuit(&circuit, true, true, None).unwrap();
        let expected = "
global phase: 4π/5
      ┌────────────┐ ┌────────────┐ ┌───────────────┐
q_0: ─┤ Rx(1.2346) ├─┤ Rx(123.46) ├─┤ Ry(1.23456*ϕ) ├────────────
     ┌┴────────────┴┐├────────────┴┐├───────────────┤┌──────────┐
q_1: ┤ Rz(1.2346e8) ├┤ Rx(0.12346) ├┤ Rx(1.2346e-5) ├┤ Rx(2π/3) ├
     └──────────────┘└─────────────┘└───────────────┘└──────────┘
";

        assert_eq!(result, expected.trim_start_matches("\n"));
    }

    #[test]
    fn test_format_float_pi() {
        let test_points = [
            (0.0, Some("0")),
            (-0.0, Some("0")),
            (1e-12, Some("0")),
            (PI, Some("π")),
            (-PI, Some("-π")),
            (2.0 * PI, Some("2π")),
            (3.0 * PI, Some("3π")),
            (10.0 * PI, Some("10π")),
            (16.0 * PI, Some("16π")),
            (-2.0 * PI, Some("-2π")),
            (-5.0 * PI, Some("-5π")),
            (PI.powi(2), Some("π^2")),
            (-PI.powi(2), Some("-π^2")),
            (PI.powi(3), Some("π^3")),
            (PI.powi(4), Some("π^4")),
            (PI / 2.0, Some("π/2")),
            (PI / 3.0, Some("π/3")),
            (PI / 4.0, Some("π/4")),
            (PI / 6.0, Some("π/6")),
            (-PI / 2.0, Some("-π/2")),
            (2.0 * PI / 3.0, Some("2π/3")),
            (3.0 * PI / 4.0, Some("3π/4")),
            (5.0 * PI / 6.0, Some("5π/6")),
            (7.0 * PI / 4.0, Some("7π/4")),
            (15.0 * PI / 16.0, Some("15π/16")),
            (-2.0 * PI / 3.0, Some("-2π/3")),
            (1.0 / PI, Some("1/π")),
            (2.0 / PI, Some("2/π")),
            (1.0 / (2.0 * PI), Some("1/2π")),
            (3.0 / (4.0 * PI), Some("3/4π")),
            (-1.0 / PI, Some("-1/π")),
            (-1.0 / (2.0 * PI), Some("-1/2π")),
            (-18.0 / 16.0 * PI, Some("-9π/8")),
            (60.0 / 44.0 / PI, Some("15/11π")),
            (17.0 * PI + 1.0, None),
            (100.0, None),
            (1.0, None),
            (2.0, None),
            (1.5, None),
            (-7.3, None),
            (PI + 1e-6, None),
            (PI - 1e-6, None),
            (PI / 2.0 + 1e-6, None),
            (17.0 * PI / 2.0, None),
            (9.0 / (17.0 * PI), None),
        ];

        for test in test_points {
            assert_eq!(format_float_pi(test.0), test.1.map(|s| s.to_string()));
        }
    }

    #[test]
    fn test_f64_ui_formatter() {
        let test_data_5_sig_digits = [
            (-1.23, "-1.23"),
            (1.23456, "1.2346"),
            (-12.34567, "-12.346"),
            (123456.78, "123460"),
            (-0.0001, "-0.0001"),
            (12.34 * 1_000_000.0, "1.234e7"),
            (-0.00001, "-1e-5"),
            (12345678.000001, "1.2346e7"),
            (15.0 * PI / 16.0, "15π/16"),
            (-2.0 * PI / 3.0, "-2π/3"),
        ];

        let mut formatter = F64UiFormatter::new(5);
        for test in test_data_5_sig_digits {
            assert_eq!(test.1.to_owned(), formatter.format_with_pi(test.0));
        }
    }
}
