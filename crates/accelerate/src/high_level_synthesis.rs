// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use hashbrown::HashMap;
use hashbrown::HashSet;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::types::PyTuple;
use pyo3::Bound;
use pyo3::IntoPyObjectExt;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::circuit_instruction::ExtraInstructionAttributes;
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::converters::dag_to_circuit;
use qiskit_circuit::converters::QuantumCircuitData;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::imports::HLS_SYNTHESIZE_OP_USING_PLUGINS;
use qiskit_circuit::imports::{QUANTUM_CIRCUIT, QUBIT};
use qiskit_circuit::operations::Operation;
use qiskit_circuit::operations::OperationRef;
use qiskit_circuit::operations::{radd_param, Param};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::Clbit;
use qiskit_circuit::Qubit;

use crate::equivalence::EquivalenceLibrary;
use crate::nlayout::PhysicalQubit;
use crate::target_transpiler::exceptions::TranspilerError;
use crate::target_transpiler::Target;

#[cfg(feature = "cache_pygates")]
use std::sync::OnceLock;

/// Track global qubits by their state.
/// The global qubits are numbered by consecutive integers starting at `0`,
/// and the states are distinguished into clean (:math:`|0\rangle`)
/// and dirty (unknown).
#[pyclass]
#[derive(Clone, Debug)]
pub struct QubitTracker {
    /// The total number of global qubits
    num_qubits: usize,
    /// Stores the state for each qubit: `true` means clean, `false` means dirty
    state: Vec<bool>,
    /// Stores whether qubits are allowed be used
    enabled: Vec<bool>,
    /// Used internally for keeping the computations in `O(n)`
    ignored: Vec<bool>,
}

#[pymethods]
impl QubitTracker {
    #[new]
    pub fn new(num_qubits: usize, qubits_initially_zero: bool) -> Self {
        QubitTracker {
            num_qubits,
            state: vec![qubits_initially_zero; num_qubits],
            enabled: vec![true; num_qubits],
            ignored: vec![false; num_qubits],
        }
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Sets state of the given qubits to dirty
    fn set_dirty(&mut self, qubits: Vec<usize>) {
        for q in qubits {
            self.state[q] = false;
        }
    }

    /// Sets state of the given qubits to clean
    fn set_clean(&mut self, qubits: Vec<usize>) {
        for q in qubits {
            self.state[q] = true;
        }
    }

    /// Disables using the given qubits
    fn disable(&mut self, qubits: Vec<usize>) {
        for q in qubits {
            self.enabled[q] = false;
        }
    }

    /// Enable using the given qubits
    fn enable(&mut self, qubits: Vec<usize>) {
        for q in qubits {
            self.enabled[q] = true;
        }
    }

    /// Returns the number of enabled clean qubits, ignoring the given qubits
    /// ToDo: check if it's faster to avoid using ignored
    fn num_clean(&mut self, ignored_qubits: Vec<usize>) -> usize {
        for q in &ignored_qubits {
            self.ignored[*q] = true;
        }

        let count = (0..self.num_qubits)
            .filter(|q| !self.ignored[*q] && self.enabled[*q] && self.state[*q])
            .count();

        for q in &ignored_qubits {
            self.ignored[*q] = false;
        }

        count
    }

    /// Returns the number of enabled dirty qubits, ignoring the given qubits
    /// ToDo: check if it's faster to avoid using ignored
    fn num_dirty(&mut self, ignored_qubits: Vec<usize>) -> usize {
        for q in &ignored_qubits {
            self.ignored[*q] = true;
        }

        let count = (0..self.num_qubits)
            .filter(|q| !self.ignored[*q] && self.enabled[*q] && !self.state[*q])
            .count();

        for q in &ignored_qubits {
            self.ignored[*q] = false;
        }

        count
    }

    /// Get `num_qubits` enabled qubits, excluding `ignored_qubits`, and returning the
    /// clean qubits first.
    /// ToDo: check if it's faster to avoid using ignored
    fn borrow(&mut self, num_qubits: usize, ignored_qubits: Vec<usize>) -> Vec<usize> {
        for q in &ignored_qubits {
            self.ignored[*q] = true;
        }

        let clean_ancillas = (0..self.num_qubits)
            .filter(|q| !self.ignored[*q] && self.enabled[*q] && self.state[*q]);
        let dirty_ancillas = (0..self.num_qubits)
            .filter(|q| !self.ignored[*q] && self.enabled[*q] && !self.state[*q]);
        let out: Vec<usize> = clean_ancillas
            .chain(dirty_ancillas)
            .take(num_qubits)
            .collect();

        for q in &ignored_qubits {
            self.ignored[*q] = false;
        }
        out
    }

    /// Copies the contents
    fn copy(&self) -> Self {
        QubitTracker {
            num_qubits: self.num_qubits,
            state: self.state.clone(),
            enabled: self.enabled.clone(),
            ignored: self.ignored.clone(),
        }
    }

    /// Replaces the state of the given qubits by their state in the `other` tracker
    fn replace_state(&mut self, other: QubitTracker, qubits: Vec<usize>) {
        for q in qubits {
            self.state[q] = other.state[q]
        }
    }

    /// Pretty-prints
    pub fn __str__(&self) -> String {
        let mut out = String::from("QubitTracker(");
        for q in 0..self.num_qubits {
            out.push_str(&q.to_string());
            out.push(':');
            out.push(' ');
            if !self.enabled[q] {
                out.push('_');
            } else if self.state[q] {
                out.push('0');
            } else {
                out.push('*');
            }
            if q != self.num_qubits - 1 {
                out.push(';');
                out.push(' ');
            } else {
                out.push(')');
            }
        }
        out
    }
}

/// Internal class that encapsulates immutable data required by the HighLevelSynthesis transpiler pass.
#[pyclass]
#[derive(Clone, Debug)]
pub struct HighLevelSynthesisData {
    // The high-level-synthesis config that specifies the synthesis methods
    // to use for high-level-objects in the circuit.
    // This is only accessedfrom the Python space.
    hls_config: Py<PyAny>,

    // The high-level-synthesis plugin manager that specifies the synthesis methods
    // available for various high-level-objects.
    // This is only accessedfrom the Python space.
    hls_plugin_manager: Py<PyAny>,

    // The names of high-level objects with available synthesis methods.
    // This is an optimization to avoid calling Python code.
    hls_op_names: Vec<String>,

    // Optional, directed graph represented as a coupling map.
    // This is only accessedfrom the Python space (when passing the coupling map to
    // high-level synthesis plugins).
    coupling_map: Py<PyAny>,

    // Optional, the backend target to use for this pass. If it is specified,
    // it will be used instead of the coupling map.
    // ToDo: currently this is a problematic member of the class.
    target: Option<Target>,

    // The equivalence library used (instructions in this library will not
    // be unrolled by this pass)
    equivalence_library: Option<EquivalenceLibrary>,

    // Supported instructions in case that target is not specified.
    device_insts: HashSet<String>,

    // A flag indicating whether the qubit indices of high-level-objects in the
    // circuit correspond to qubit indices on the target backend
    use_qubit_indices: bool,

    // The minimum number of qubits for operations in the input dag to translate.
    min_qubits: usize,

    // Indicates whether to use custom definitions.
    unroll_definitions: bool,
}

#[pymethods]
impl HighLevelSynthesisData {
    #[new]
    #[pyo3(signature=(/, hls_config, hls_plugin_manager, hls_op_names, coupling_map, target, equivalence_library, device_insts, use_qubit_indices, min_qubits, unroll_definitions))]
    #[allow(clippy::too_many_arguments)]
    fn __new__(
        hls_config: Py<PyAny>,
        hls_plugin_manager: Py<PyAny>,
        hls_op_names: Vec<String>,
        coupling_map: Py<PyAny>,
        target: Option<Target>,
        equivalence_library: Option<EquivalenceLibrary>,
        device_insts: HashSet<String>,
        use_qubit_indices: bool,
        min_qubits: usize,
        unroll_definitions: bool,
    ) -> Self {
        Self {
            hls_config,
            hls_plugin_manager,
            hls_op_names,
            coupling_map,
            target,
            equivalence_library,
            device_insts,
            use_qubit_indices,
            min_qubits,
            unroll_definitions,
        }
    }

    pub fn get_hls_config(&self) -> &Py<PyAny> {
        &self.hls_config
    }

    pub fn get_hls_plugin_manager(&self) -> &Py<PyAny> {
        &self.hls_plugin_manager
    }

    pub fn get_coupling_map(&self) -> &Py<PyAny> {
        &self.coupling_map
    }

    // ToDo: this is a problematic line that involves both cloning the target and
    // triggers a compiler warning about the visibility of target.
    pub fn get_target(&self) -> Option<Target> {
        self.target.clone()
    }

    pub fn get_use_qubit_indices(&self) -> bool {
        self.use_qubit_indices
    }

    pub fn __str__(&self) -> String {
        let mut out = String::from("HighLevelSynthesisData: ");
        out.push_str(&self.min_qubits.to_string());
        out
    }
}

/// Check whether an operation is natively supported.
pub fn instruction_supported(data: &HighLevelSynthesisData, name: &str, qubits: &[Qubit]) -> bool {
    match &data.target {
        Some(target) => {
            if target.num_qubits.is_some() {
                if data.use_qubit_indices {
                    let physical_qubits = qubits
                        .iter()
                        .map(|q| PhysicalQubit(q.index() as u32))
                        .collect();
                    target.instruction_supported(name, Some(&physical_qubits))
                } else {
                    target.instruction_supported(name, None)
                }
            } else {
                data.device_insts.contains(name)
            }
        }
        None => data.device_insts.contains(name),
    }
}

/// Check whether an operation does not need to be synthesized.
pub fn definitely_skip_op(
    data: &HighLevelSynthesisData,
    op: &PackedOperation,
    qubits: &[Qubit],
) -> bool {
    if qubits.len() < data.min_qubits {
        return true;
    }

    if op.directive() {
        return true;
    }

    if op.control_flow() {
        return false;
    }

    // If the operation is natively supported, we can skip it.
    if instruction_supported(data, op.name(), qubits) {
        return true;
    }

    // If there are avilable plugins for this operation, we should try them
    // before checking the equivalence library.
    if data.hls_op_names.iter().any(|s| s == op.name()) {
        return false;
    }

    if let Some(equiv_lib) = &data.equivalence_library {
        if equiv_lib.has_entry(op) {
            return true;
        }
    }

    false
}

/// Recursively synthesizes a circuit. This circuit is either the original circuit,
/// the definition circuit for one of the gates, or a circuit returned by a plugin.
///
/// The input to this function is the circuit to be synthesized and the global
/// qubits over which it is defined.
///
/// The output is the synthesized circuit and the global qubits over which it is
/// defined. Note that by using auxiliary qubits, the output circuit may be defined
/// over more qubits than the input circuit.
///
/// The function also updates in-place the qubit tracker, which keeps track of the
/// state of each global qubits (whether it's clean, dirty, or cannot be used).
fn run_on_circuitdata(
    py: Python,
    input_circuit: &CircuitData,
    input_qubits: &[usize],
    data: &HighLevelSynthesisData,
    tracker: &mut QubitTracker,
) -> PyResult<(CircuitData, Vec<usize>)> {
    if input_circuit.num_qubits() != input_qubits.len() {
        return Err(TranspilerError::new_err(
            "HighLevelSynthesis: the input to 'run_on_circuitdata' is incorrect.",
        ));
    }

    // We iteratively process circuit instructions in the order they appear in the input circuit,
    // and add the synthesized instructions to the output circuit. Note that in the process the
    // output circuit may need to be extended with additional qubits. In addition, we keep track
    // of the state of the global qubits using the qubits tracker.
    //
    // Note: This is a first version of a potentially more elaborate approach to find
    // good operation/ancilla allocations. The current approach is greedy and just gives
    // all available ancilla qubits to the current operation ("the-first-takes-all" approach).
    // It does not distribute ancilla qubits between different operations present in the circuit.

    let mut output_circuit: CircuitData = CircuitData::clone_empty_like(py, input_circuit, None)?;
    let mut output_qubits = input_qubits.to_vec();

    // The "inverse" map from the global qubits to the output circuit's qubits.
    // This map may be extended if additional auxiliary qubits get used.
    let mut global_to_local: HashMap<usize, usize> =
        HashMap::from_iter(output_qubits.iter().enumerate().map(|(i, j)| (*j, i)));

    for inst in input_circuit.iter() {
        // op's qubits as viewed globally
        let op_qubits = input_circuit
            .get_qargs(inst.qubits)
            .iter()
            .map(|q| input_qubits[q.index()])
            .collect::<Vec<usize>>();

        // Start by handling special operations.
        // In the future, we can also consider other possible optimizations, e.g.:
        //   - improved qubit tracking after a SWAP gate
        //   - automatically simplify control gates with control at 0.
        if ["id", "delay", "barrier"].contains(&inst.op.name()) {
            output_circuit.push(py, inst.clone())?;
            // tracker is not updated, these are no-ops
            continue;
        }

        if inst.op.name() == "reset" {
            output_circuit.push(py, inst.clone())?;
            tracker.set_clean(op_qubits);
            continue;
        }

        // Check if synthesis for this operation can be skipped
        let op_qargs: Vec<Qubit> = op_qubits.iter().map(|q| Qubit(*q as u32)).collect();
        if definitely_skip_op(data, &inst.op, &op_qargs) {
            output_circuit.push(py, inst.clone())?;
            tracker.set_dirty(op_qubits);
            continue;
        }

        // Recursively handle control-flow.
        // Currently we do not allow subcircuits within the control flow to use auxiliary qubits
        // and mark all the usable qubits as dirty. This is done in order to avoid complications
        // that different subcircuits may choose to use different auxiliary global qubits, and to
        // avoid complications related to tracking qubit status for while- loops.
        // In the future, this handling can potentially be improved.
        if inst.op.control_flow() {
            let quantum_circuit_cls = QUANTUM_CIRCUIT.get_bound(py);
            if let OperationRef::Instruction(py_inst) = inst.op.view() {
                let old_blocks_as_bound_obj = py_inst.instruction.bind(py);
                let old_blocks = old_blocks_as_bound_obj.getattr("blocks")?;
                let old_blocks = old_blocks.downcast::<PyTuple>()?;

                let mut new_blocks: Vec<Bound<PyAny>> = Vec::with_capacity(old_blocks.len());
                let mut block_tracker = tracker.clone();
                let to_disable: Vec<usize> = (0..tracker.num_qubits())
                    .filter(|q| !op_qubits.contains(q))
                    .collect();
                block_tracker.disable(to_disable);
                block_tracker.set_dirty(op_qubits.clone());
                for block in old_blocks {
                    let old_block: QuantumCircuitData = block.extract()?;
                    let (new_block, _) = run_on_circuitdata(
                        py,
                        &old_block.data,
                        &op_qubits,
                        data,
                        &mut block_tracker,
                    )?;
                    let new_block = new_block.into_bound_py_any(py)?;
                    // ToDo: check global phase!
                    let new_block_circuit =
                        quantum_circuit_cls.call_method1("copy_empty_like", (block,))?;

                    new_block_circuit.setattr("_data", new_block.as_ref())?;
                    new_blocks.push(new_block_circuit);
                }
                let replaced_blocks =
                    old_blocks_as_bound_obj.call_method1("replace_blocks", (new_blocks,))?;
                let synthesized_op: OperationFromPython = replaced_blocks.extract()?;

                let packed_instruction = PackedInstruction {
                    op: synthesized_op.operation,
                    qubits: inst.qubits,
                    clbits: inst.clbits,
                    params: inst.params.clone(),
                    extra_attrs: inst.extra_attrs.clone(),
                    #[cfg(feature = "cache_pygates")]
                    py_op: std::sync::OnceLock::new(),
                };
                output_circuit.push(py, packed_instruction)?;
                tracker.set_dirty(op_qubits);
                continue;
            }
        }

        // Now we synthesize the operation.
        // The function synthesize_operation returns either None if the operation does not need to be
        // synthesized, or returns a quantum circuit together with the global qubits on which this
        // circuit is defined. Note that the synthesized circuit may involve auxiliary
        // global qubits not used by the input circuit.
        let synthesize_operation_result = synthesize_operation(
            py,
            data,
            tracker,
            &op_qubits,
            &inst.op,
            inst.params_view(),
            &inst.extra_attrs,
        )?;

        match synthesize_operation_result {
            None => {
                // If the synthesis did not change anything, we add the operation to the output circuit
                // and update the qubit tracker.
                output_circuit.push(py, inst.clone())?;
                tracker.set_dirty(op_qubits);
            }
            Some((synthesized_circuit, synthesized_circuit_qubits)) => {
                // This pedantic check can possibly be removed.
                if synthesized_circuit.num_qubits() != synthesized_circuit_qubits.len() {
                    return Err(TranspilerError::new_err(
                        "HighLevelSynthesis: the output from 'synthesize_operation' is incorrect.",
                    ));
                }

                // If the synthesized circuit uses (auxiliary) global qubits that are not in the output circuit,
                // we add these qubits to the output circuit.
                if synthesized_circuit_qubits.len() > op_qubits.len() {
                    let qubit_cls = QUBIT.get_bound(py);

                    for q in &synthesized_circuit_qubits {
                        if !global_to_local.contains_key(q) {
                            let new_qubit_index = output_qubits.len();
                            let bit = qubit_cls.call0()?;
                            global_to_local.insert(*q, new_qubit_index);
                            output_qubits.push(*q);
                            output_circuit.add_qubit(py, &bit, false)?;
                        }
                    }
                }

                // Add the operations from the circuit synthesized for the current operation to the output circuit.
                // The correspondence between qubits is:
                // qubit index in the synthesized circuit -> corresponding global qubit -> corresponding qubit in the output circuit
                let qubit_map: HashMap<usize, usize> = HashMap::from_iter(
                    synthesized_circuit_qubits
                        .iter()
                        .enumerate()
                        .map(|(i, q)| (i, global_to_local[q])),
                );

                for inst_inner in synthesized_circuit.iter() {
                    let inst_inner_qubits = synthesized_circuit.get_qargs(inst_inner.qubits);
                    let inst_inner_clbits = synthesized_circuit.get_cargs(inst_inner.clbits);

                    let inst_outer_qubits: Vec<Qubit> = inst_inner_qubits
                        .iter()
                        .map(|q| Qubit(qubit_map[&q.index()] as u32))
                        .collect();
                    let inst_outer_clbits: Vec<Clbit> = inst_inner_clbits
                        .iter()
                        .map(|c| Clbit(c.index() as u32))
                        .collect();

                    output_circuit.push_packed_operation(
                        inst_inner.op.clone(),
                        inst_inner.params_view(),
                        &inst_outer_qubits,
                        &inst_outer_clbits,
                    )?;
                }

                let updated_global_phase = radd_param(
                    output_circuit.global_phase().clone(),
                    synthesized_circuit.global_phase().clone(),
                    py,
                );
                output_circuit.set_global_phase(py, updated_global_phase)?;
            }
        }
    }

    // Another pedantic check that can possibly be removed.
    if output_circuit.num_qubits() != output_qubits.len() {
        return Err(TranspilerError::new_err(
            "HighLevelSynthesis: the output from 'run_on_circuitdata' is incorrect.",
        ));
    }

    Ok((output_circuit, output_qubits))
}

/// Recursively synthesizes a single operation.
///
/// The input to this function is the operation to be synthesized (consisting of a
/// packed operation, params and extra attributes) and a list of global qubits over
/// which this operation is defined.
///
/// The function returns the synthesized circuit and the global qubits over which this
/// synthesized circuit is defined. Note that by using auxiliary qubits, the output circuit
/// may be defined over more qubits than the input operation. In addition, the output
/// circuit may be ``None``, which means that the operation should remain as it is.
///
/// The function also updates in-place the qubit tracker which keeps track of the state of
/// each global qubit (whether it's clean, dirty, or cannot be used).
fn synthesize_operation(
    py: Python,
    data: &HighLevelSynthesisData,
    tracker: &mut QubitTracker,
    input_qubits: &[usize],
    op: &PackedOperation,
    params: &[Param],
    extra_attrs: &ExtraInstructionAttributes,
) -> PyResult<Option<(CircuitData, Vec<usize>)>> {
    if op.num_qubits() != input_qubits.len() as u32 {
        return Err(TranspilerError::new_err(
            "HighLevelSynthesis: the input to 'synthesize_operation' is incorrect.",
        ));
    }

    let mut output_circuit_and_qubits: Option<(CircuitData, Vec<usize>)> = None;

    // If this function is called, the operation is not supported by the target, however may have
    // high-level synthesis plugins, and/or be in the equivalence library, and/or have a definition
    // circuit. The priorities are as follows:
    // - First, we try running the battery of high-level synthesis plugins.
    // - Second, we check if the operation is present in the equivalence library.
    // - Third, we unroll custom definitions.
    //
    // If we obtain a new quantum circuit, it needs to be recursively synthesized, so
    // that the final result only consists of supported operations. If there is no
    // change, we return None.

    // Try to synthesize using plugins.
    if data.hls_op_names.iter().any(|s| s == op.name()) {
        output_circuit_and_qubits = synthesize_op_using_plugins(
            py,
            data,
            tracker,
            input_qubits,
            &op.view(),
            params,
            extra_attrs,
        )?;
    }

    // Check if present in the equivalent library.
    if output_circuit_and_qubits.is_none() {
        if let Some(equiv_lib) = &data.equivalence_library {
            if equiv_lib.has_entry(op) {
                return Ok(None);
            }
        }
    }

    // Extract definition.
    if output_circuit_and_qubits.is_none() && data.unroll_definitions {
        let definition_circuit = op.definition(params);
        match definition_circuit {
            Some(definition_circuit) => {
                output_circuit_and_qubits = Some((definition_circuit, input_qubits.to_vec()));
            }
            None => {
                return Err(TranspilerError::new_err(format!(
                    "HighLevelSynthesis is unable to synthesize {:?}",
                    op.name()
                )));
            }
        }
    }

    // Output circuit is a quantum circuit which we want to process recursively.
    // Currently, neither 'synthesize_op_using_plugins' nor 'get_custom_definition'
    // update the tracker (we might want to change this in the future), which makes
    // sense because we have not synthesized the output circuit yet.
    // So we pass the tracker to 'run_on_circuitdata' but make sure to restore the status of
    // clean ancilla qubits after the circuit is synthesized. In order to do that,
    // we save the current state of the tracker.
    if let Some((current_circuit, current_qubits)) = output_circuit_and_qubits {
        let saved_tracker = tracker.copy();
        let (synthesized_circuit, synthesized_qubits) =
            run_on_circuitdata(py, &current_circuit, &current_qubits, data, tracker)?;

        if synthesized_qubits.len() > input_qubits.len() {
            let qubits_to_replace: Vec<usize> =
                (input_qubits.len()..synthesized_qubits.len()).collect();
            tracker.replace_state(saved_tracker, qubits_to_replace);
        }

        output_circuit_and_qubits = Some((synthesized_circuit, synthesized_qubits));
    }

    Ok(output_circuit_and_qubits)
}

/// Attempts to synthesize an operation using available plugins.
///
/// The input to this function is the operation to be synthesized and a list of global
/// qubits over which this operation is defined.
///
/// The function returns either the synthesized quantum circuit and the global qubits over
/// which it's defined, or ``None`` in the case that no synthesis methods are available or
/// applicable (for instance, when there is an insufficient number of auxiliary qubits).
///
/// Internally, this function calls the Python function.
///
/// Currently, this function does not update the qubit tracker, which is handled upstream.
fn synthesize_op_using_plugins(
    py: Python,
    data: &HighLevelSynthesisData,
    tracker: &QubitTracker,
    input_qubits: &[usize],
    op: &OperationRef,
    params: &[Param],
    extra_attrs: &ExtraInstructionAttributes,
) -> PyResult<Option<(CircuitData, Vec<usize>)>> {
    let mut output_circuit_and_qubits: Option<(CircuitData, Vec<usize>)> = None;

    let op_py = match op {
        OperationRef::StandardGate(standard) => standard
            .create_py_op(py, Some(params), extra_attrs)?
            .into_any(),
        OperationRef::StandardInstruction(instruction) => instruction
            .create_py_op(py, Some(params), extra_attrs)?
            .into_any(),
        OperationRef::Gate(gate) => gate.gate.clone_ref(py),
        OperationRef::Instruction(instruction) => instruction.instruction.clone_ref(py),
        OperationRef::Operation(operation) => operation.operation.clone_ref(py),
        OperationRef::Unitary(unitary) => unitary.create_py_op(py, extra_attrs)?.into_any(),
    };

    // ToDo: how can we avoid cloning data and tracker?
    let res = HLS_SYNTHESIZE_OP_USING_PLUGINS
        .get_bound(py)
        .call1((op_py, input_qubits, data.clone(), tracker.clone()))?
        .extract::<Option<(QuantumCircuitData, Vec<usize>)>>()?;

    if let Some((quantum_circuit_data, qubits)) = res {
        output_circuit_and_qubits = Some((quantum_circuit_data.data, qubits));
    }

    Ok(output_circuit_and_qubits)
}

/// Synthesizes an operation.
///
/// This function is called by the default plugin for annotated operations to
/// synthesize the base operation.
#[pyfunction]
#[pyo3(signature = (py_op, input_qubits, data, tracker))]
pub fn py_synthesize_operation(
    py: Python,
    py_op: Bound<PyAny>,
    input_qubits: Vec<usize>,
    data: &HighLevelSynthesisData,
    tracker: &mut QubitTracker,
) -> PyResult<Option<(CircuitData, Vec<usize>)>> {
    let op: OperationFromPython = py_op.extract()?;
    synthesize_operation(
        py,
        data,
        tracker,
        &input_qubits,
        &op.operation,
        &op.params,
        &op.extra_attrs,
    )
}

/// Runs HighLevelSynthesis transpiler pass.
///
/// This is the main function called from the Python space. If the pass does not need
/// to do anything, it returns None, meaning that the DAG should remain unchanged.
/// Otherwise, the new DAG is returned.
#[pyfunction]
#[pyo3(signature = (dag, data, qubits_initially_zero))]
pub fn py_run_on_dag(
    py: Python,
    dag: &DAGCircuit,
    data: &HighLevelSynthesisData,
    qubits_initially_zero: bool,
) -> PyResult<Option<DAGCircuit>> {
    // Fast-path: check if HighLevelSynthesis can be skipped altogether. This is only
    // done at the top-level since this does not track the qubit states.
    let mut fast_path: bool = true;

    for (node_index, inst) in dag.op_nodes(false) {
        let qubits = dag.get_qargs(inst.qubits);
        if !dag.has_calibration_for_index(py, node_index)?
            && !definitely_skip_op(data, &inst.op, qubits)
        {
            fast_path = false;
            break;
        }
    }

    if fast_path {
        Ok(None)
    } else {
        // Regular-path: we synthesize the circuit recursively. Except for
        // this conversion from DAGCircuit to CircuitData and back, all
        // the recursive functions work with CircuitData objects only.
        let circuit = dag_to_circuit(py, dag, false)?;

        let num_qubits = circuit.num_qubits();
        let input_qubits: Vec<usize> = (0..num_qubits).collect();
        let mut tracker = QubitTracker::new(num_qubits, qubits_initially_zero);

        let (output_circuit, _) =
            run_on_circuitdata(py, &circuit, &input_qubits, data, &mut tracker)?;

        let new_dag = convert_circuit_to_dag_with_data(py, dag, &output_circuit)?;

        Ok(Some(new_dag))
    }
}

/// Converts circuit to DAGCircuit, while taking the missing DAGCircuit data from dag.
fn convert_circuit_to_dag_with_data(
    py: Python,
    dag: &DAGCircuit,
    circuit: &CircuitData,
) -> PyResult<DAGCircuit> {
    println!("==> IN B");

    let mut new_dag = dag.copy_empty_like(py, "alike")?;
    new_dag.set_global_phase(circuit.global_phase().clone())?;
    let qarg_map = new_dag.merge_qargs(circuit.qargs_interner(), |bit| Some(*bit));
    let carg_map = new_dag.merge_cargs(circuit.cargs_interner(), |bit: &Clbit| Some(*bit));

    new_dag.try_extend(
        py,
        circuit.iter().map(|instr| -> PyResult<PackedInstruction> {
            Ok(PackedInstruction {
                // op: instr.op.clone(),
                op: instr.op.py_deepcopy(py, None)?,
                qubits: qarg_map[instr.qubits],
                clbits: carg_map[instr.clbits],
                params: instr.params.clone(),
                extra_attrs: instr.extra_attrs.clone(),
                #[cfg(feature = "cache_pygates")]
                py_op: OnceLock::new(),
            })
        }),
    )?;

    println!("==> IN X");

    Ok(new_dag)
}

pub fn high_level_synthesis_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_on_dag))?;
    m.add_wrapped(wrap_pyfunction!(py_synthesize_operation))?;

    m.add_class::<QubitTracker>()?;
    m.add_class::<HighLevelSynthesisData>()?;
    Ok(())
}
