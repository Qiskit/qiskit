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

use std::sync::OnceLock;

use indexmap::IndexSet;
use qiskit_circuit::{
    circuit_data::CircuitData,
    converters::dag_to_circuit,
    dag_circuit::DAGCircuit,
    operations::{Operation, StandardGate},
};
use qiskit_synthesis::two_qubit_decompose::{
    RXXEquivalent, TwoQubitBasisDecomposer, TwoQubitControlledUDecomposer,
};
use qiskit_transpiler::{
    passes::{run_consolidate_blocks, DecomposerType},
    target::Target,
};
use smallvec::SmallVec;

use crate::pointers::const_ptr_as_ref;

/// The collection of compatible KAK decomposition gates in a set in order
/// of priority.
static KAK_GATES: OnceLock<IndexSet<StandardGate>> = OnceLock::new();

/// The collection of compatible KAK decomposition parametric gates in a set
/// in order of priority.
static KAK_GATES_PARAM: OnceLock<IndexSet<StandardGate>> = OnceLock::new();

/// Helper function that extracts the decomposer and basis gate directly from the [Target].
#[inline]
fn get_decomposer_and_basis_gate(
    target: Option<&Target>,
    approximation_degree: f64,
) -> (DecomposerType, StandardGate) {
    if let Some(target) = target {
        // Targets from C should only support
        let target_basis_gates: IndexSet<StandardGate> = target
            .operations()
            .filter_map(|op| op.operation.try_standard_gate())
            .collect();
        let target_basis_param_supported = KAK_GATES_PARAM
            .get_or_init(|| {
                IndexSet::from_iter([
                    StandardGate::RXX,
                    StandardGate::RZZ,
                    StandardGate::RYY,
                    StandardGate::RZX,
                    StandardGate::CPhase,
                    StandardGate::CRX,
                    StandardGate::CRY,
                    StandardGate::CRZ,
                ])
            })
            .intersection(&target_basis_gates)
            .next()
            .copied();
        if let Some(gate) = target_basis_param_supported {
            return (
                DecomposerType::TwoQubitControlledU(
                    TwoQubitControlledUDecomposer::new(RXXEquivalent::Standard(gate), "ZXZ")
                        .unwrap_or_else(|_| {
                            panic!(
                                "Error while creating Controlled U decomposer using a {} gate.",
                                gate.name()
                            )
                        }),
                ),
                gate,
            );
        }
        let target_basis_supported = KAK_GATES
            .get_or_init(|| {
                IndexSet::from_iter([
                    StandardGate::CX,
                    StandardGate::CZ,
                    StandardGate::ECR,
                    StandardGate::ISwap,
                ])
            })
            .intersection(&target_basis_gates)
            .next()
            .copied();
        if let Some(gate) = target_basis_supported {
            return (DecomposerType::TwoQubitBasis(
                TwoQubitBasisDecomposer::new_inner(
                    gate.into(),
                    SmallVec::default(),
                    gate.matrix(&[]).unwrap_or_else(|| panic!("Error while obtaining the matrix form of gate '{}' without params.", gate.name())).view(),
                    approximation_degree,
                    "U",
                    None,
                )
                .unwrap_or_else(|_| panic!("Error while creating Basis Decomposer using a {} gate.",
                    gate.name()))),
                gate
            );
        }
    }
    let gate = StandardGate::CX;
    (
        DecomposerType::TwoQubitBasis(
            TwoQubitBasisDecomposer::new_inner(
                gate.into(),
                SmallVec::default(),
                gate.matrix(&[])
                    .expect("Error while obtaining the matrix form of gate 'cx' without params.")
                    .view(),
                1.0,
                "U",
                None,
            )
            .expect("Error while creating Basis Decomposer using a 'cx' gate."),
        ),
        gate,
    )
}

/// @ingroup QkTranspilerPasses
/// Run the ConsolidateBlocks pass on a circuit.
///
/// ConsolidateBlocks is a transpiler pass that consolidates consecutive blocks of
/// gates operating on the same qubits into a Unitary gate, to later on be
/// resynthesized, which leads to a more optimal subcircuit.
///
/// @param circuit A pointer to the circuit to run ConsolidateBlocks on.
/// @param target A pointer to the target to run ConsolidateBlocks on.
/// @param approximation_degree A float between `[0.0, 1.0]`. Lower approximates more.
/// @param force_consolidate: Force block consolidation.
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit`` and
/// if ``target`` is not a valid pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_consolidate_blocks(
    circuit: *const CircuitData,
    target: *const Target,
    approximation_degree: f64,
    force_consolidate: bool,
) -> *mut CircuitData {
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    let target = unsafe {
        if target.is_null() {
            None
        } else {
            Some(const_ptr_as_ref(target))
        }
    };
    let approximation_degree = if approximation_degree.is_nan() {
        1.0
    } else {
        approximation_degree
    };
    let mut circ_as_dag = DAGCircuit::from_circuit_data(circuit, true, None, None, None, None)
        .expect("Error while converting from CircuitData to DAGCircuit.");
    let (decomposer, basis_gate) = get_decomposer_and_basis_gate(target, approximation_degree);

    // Call the pass
    run_consolidate_blocks(
        &mut circ_as_dag,
        decomposer,
        basis_gate.name(),
        force_consolidate,
        target,
        None,
    )
    .expect("Error running the consolidate blocks pass.");

    let result_circuit = dag_to_circuit(&circ_as_dag, true)
        .expect("Error while converting from DAGCircuit to CircuitData.");
    Box::into_raw(Box::new(result_circuit))
}
