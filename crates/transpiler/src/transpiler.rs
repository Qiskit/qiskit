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

use hashbrown::{HashMap, HashSet};

use crate::commutation_checker;
use crate::equivalence::EquivalenceLibrary;
use crate::passes::*;
use crate::target::Target;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::converters::dag_to_circuit;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::nlayout::NLayout;
use qiskit_circuit::operations::StandardGate;
use qiskit_circuit::PhysicalQubit;

fn generate_default_commutation_checker() -> commutation_checker::CommutationChecker {
    todo!();
    //    let library = commutation_checker::CommutationLibrary {
    //        library: Some(library),
    //    };
    //    let gates = [
    //        "p", "z", "u1", "rz", "t", "s", "x", "rx", "cx", "cy", "cz", "h", "y"
    //    ].into_iter().map(|x| x.to_string().collect();
    //    commutation_checker::CommutationChecker {
    //        library,
    //        cache: HashMap::new(),
    //        cache_max_entries: 1_000_000,
    //        current_cache_entries: 0,
    //        gates,
    //    }
}

fn generate_default_equivalence_library() -> EquivalenceLibrary {
    todo!();
}

fn compose_permutations(old_permutation: &[usize], new_permutation: &[usize]) {
    todo!();
}

static SELF_INVERSE_GATES_FOR_CANCELLATION: [StandardGate; 9] = [
    StandardGate::CX,
    StandardGate::ECR,
    StandardGate::CY,
    StandardGate::CZ,
    StandardGate::X,
    StandardGate::Y,
    StandardGate::Z,
    StandardGate::H,
    StandardGate::Swap,
];

static INVERSE_PAIRS_FOR_CANCELLATION: [[StandardGate; 2]; 3] = [
    [StandardGate::T, StandardGate::Tdg],
    [StandardGate::S, StandardGate::Sdg],
    [StandardGate::SX, StandardGate::SXdg],
];

pub struct TranspileLayout {
    initial_layout: NLayout,
    final_permutation: Vec<PhysicalQubit>,
}

/// A transpilation function for Rust native circuits for use in the C API. This will not cover
/// things that only exist in the Python API such as custom gates or control flow. When those
/// concepts exist in the rust data model this function must be expanded before adding them to the
/// C API.
pub fn transpile(
    qc: &CircuitData,
    target: &Target,
    optimization_level: u8,
    seed: Option<u64>,
) -> PyResult<CircuitData> {
    if ![0u8, 1, 2, 3].contains(optimization_level) {
        panic!("Invalid optimization level specified {optimization_level}");
    }
    let dag = DAGCircuit::from_circuit_data(dag, false)?;
    let approximation_degree = Some(1.0);
    let commutation_checker = generate_default_commutation_checker();
    let equivalence_library = generate_default_equivalence_library();

    let unroll_3q_or_more = |dag: &mut DAGCircuit| -> PyResult<DAGCircuit> {
        // This will panic if there is a 3q unitary until qsd is ported
        let mut out_dag = run_unitary_synthesis(
            dag,
            (0..dag.num_qubits()).collect(),
            3,
            Some(target),
            HashSet::new(),
            ["unitary".to_string(), "swap".to_string()]
                .into_iter()
                .collect(),
            HashSet::new(),
            approximation_degree,
            None,
            None,
            false,
        )?;
        run_unroll_3q_or_more(&mut out_dag, target)?;
        out_dag = run_basis_translator(
            &mut out_dag,
            equivalence_library,
            HashMap::new(),
            3,
            None,
            Some(target),
            None,
        )?;
        Ok(out_dag)
    };

    // Init stage
    unroll_3q_or_more(&mut dag, target);
    let mut initial_permutation = None;
    if optimization_level == 1 {
        run_inverse_cancellation(
            &mut dag,
            &SELF_INVERSE_GATES_FOR_CANCELLATION,
            &INVERSE_PAIRS_FOR_CANCELLATION,
        );
    } else if optimization_level == 2 || optimization_level == 3 {
        let (new_dag, permutation) = run_elide_permutations(&dag)?;
        dag = new_dag;
        initial_permutation = Some(permutation);
        run_remove_diagonal_before_measure(&mut dag);
        run_remove_identity_equiv(&mut dag, approximation_degree, Some(target));
        run_inverse_cancellation(
            &mut dag,
            &SELF_INVERSE_GATES_FOR_CANCELLATION,
            &INVERSE_PAIRS_FOR_CANCELLATION,
        )?;
        cancel_commutations(&mut dag, &mut commutation_checker, None, 1.0)?;
        run_consolidate_blocks(&mut dag, target, approximation_degree, false)?;
        let result = run_split_2q_unitaries(&mut dag, true)?;
        if let Some(result) = result {
            dag = result.0;
            let split_2q_permutation = permutation.1;
            initial_permutation = if let Some(initial_permutation) = initial_permutation {
                Some(compose_permutations(
                    initial_permutation,
                    split_2q_permutation,
                ));
            } else {
                Some(split_2q_permutation)
            }
        }
    }
    let mut layout: Option<TranspileLayout> = None;
    // layout stage
    if optimization_level == 0 {
        let initial_layout = trivial_layout(&mut dag);
        layout = apply_layout(&mut dag, initial_layout, None, None, &mut layout, true);
    } else if optimization_level == 1 {
        let initial_layout = trivial_layout(&mut dag);
        if let Some(initial_layout) = initial_layout {
            layout = apply_layout(&mut dag, initial_layout, None, None, &mut layout, target);
        } else {
            if let Some(vf2_result) =
                vf2_layout_pass(&dag, target, false, Some(5_000_000), None, Some(2500))?
            {
                layout = apply_layout(&mut dag, vf2_result, None, None, &mut layout, target);
            } else {
                let (result, initial_layout, final_layout) = sabre_layout_and_routing(
                    &mut dag,
                    target,
                    heuristic,
                    2,
                    20,
                    20,
                    seed,
                    Vec::new(),
                    false,
                )?;
                dag = result;
                layout = Some(TranspileLayout {
                    initial_layout,
                    final_permutation: final_layout.virt_to_phys,
                });
            }
        }
    } else if optimization_level == 2 {
        if let Some(vf2_result) =
            vf2_layout_pass(&dag, target, false, Some(5_000_000), None, Some(2500))?
        {
            layout = apply_layout(&mut dag, vf2_result, None, None, &mut layout, target);
        } else {
            let (result, initial_layout, final_layout) = sabre_layout_and_routing(
                &mut dag,
                target,
                heuristic,
                2,
                20,
                20,
                seed,
                Vec::new(),
                false,
            )?;
            dag = result;
            layout = Some(TranspileLayout {
                initial_layout,
                final_layout: final_layout.virt_to_phys,
            });
        }
    } else {
        if let Some(vf2_result) =
            vf2_layout_pass(&dag, target, false, Some(30_000_000), None, Some(250_000))?
        {
            layout = apply_layout(&mut dag, vf2_result, eli, None, None, &mut layout, target);
        } else {
            let (result, initial_layout, final_layout) = sabre_layout_and_routing(
                &mut dag,
                target,
                heuristic,
                4,
                20,
                20,
                seed,
                Vec::new(),
                false,
            );
            dag = result;
            layout = Some(TranspileLayout {
                initial_layout,
                final_layout: final_layout.virt_to_phys,
            });
        }
    }
    // Routing stage
    let vf2_post_result = if optimization_level == 0 {
        if !run_check_map(&dag, &target) {
            let (out_dag, final_layout) = sabre_routing(
                &dag,
                target,
                heuristic,
                layout.initial_layout,
                5,
                seed,
                Some(true),
            )?;
            dag = out_dag;
            layout.final_permutation = final_layout.virt_to_phys;
        }
        None
    } else if optimization_level == 1 {
        vf2_post_layout_pass(&dag, target, false, Some(50_000), None, Some(2_500)).unwrap()
    } else if optimization_level == 2 {
        vf2_post_layout_pass(&dag, target, false, Some(50_000), None, Some(2_500)).unwrap()
    } else {
        vf2_post_layout_pass(&dag, target, false, Some(30_000_000), None, Some(250_000)).unwrap()
    };
    if let Some(post_layout) = vf2_post_result {
        apply_layout(
            &mut dag,
            post_layout,
            layout.final_layout,
            &mut layout,
            target,
        );
    }
    let translation = |dag: &mut dag| {
        dag = run_unitary_synthesis(
            &mut dag,
            (0..dag.num_qubits()).collect(),
            0,
            Some(target),
            HashSet::new(),
            ["unitary"].into_iter().collect(),
            HashSet::new(),
            approximation_degree,
            None,
            None,
            false,
        )
        .unwrap();
        dag = run_basis_translator(
            dag,
            equivalence_library,
            HashMap::new(),
            0,
            None,
            Some(target),
            None,
        )
        .unwrap();
        if !check_direction_target(&dag, target).unwrap() {
            dag = fix_direction_target(&mut dag, target).unwrap();
            if gates_missing_from_target(&dag, target)? {
                dag = run_basis_translator(
                    dag,
                    equivalence_library,
                    HashMap::new(),
                    0,
                    None,
                    Some(target),
                    None,
                )
                .unwrap();
            }
        }
    };
    translation(&mut dag);
    // optimization stage
    let mut depth: Option<usize> = None;
    let mut size: Option<usize> = None;
    let mut new_depth = None;
    let mut new_size = None;
    if optimization_level == 1 {
        new_depth = Some(dag.depth(false)?);
        new_size = Some(dag.size(false)?);
        while new_depth != depth || new_size != size {
            depth = new_depth;
            size = new_size;
            run_optimize_1q_gates_decomposition(&mut dag, Some(target), None, None)?;
            run_inverse_cancellation(
                &mut dag,
                &SELF_INVERSE_GATES_FOR_CANCELLATION,
                &INVERSE_PAIRS_FOR_CANCELLATION,
            )?;
            if gates_missing_from_target(&dag, target)? {
                translation(&mut dag);
            }
            new_depth = Some(dag.depth(false)?);
            new_size = Some(dag.size(false)?);
        }
    } else if optimization_level == 2 {
        run_consolidate_blocks(&mut dag, target, approximation_degree, false);
        dag = run_unitary_synthesis(
            &mut dag,
            (0..dag.num_qubits()).collect(),
            0,
            Some(target),
            HashSet::new(),
            ["unitary"].into_iter().map(|x| x.to_string()).collect(),
            HashSet::new(),
            approximation_degree,
            None,
            None,
            false,
        )?;
        new_depth = Some(dag.depth(false)?);
        new_size = Some(dag.size(false)?);
        while new_depth != depth || new_size != size {
            depth = new_depth;
            size = new_size;
            run_remove_identity_equiv(&mut dag, approximation_degree, Some(target))?;
            run_optimize_1q_gates_decomposition(&mut dag, Some(target), None, None)?;
            cancel_commutations(&mut dag, &mut commutation_checker, None, 1.0)?;
            if gates_missing_from_target(&dag, target)? {
                translation(&mut dag);
            }
            new_depth = Some(dag.depth(false)?);
            new_size = Some(dag.size(false)?);
        }
    } else if optimization_level == 3 {
        let mut best_dag = dag.clone();
        let mut best_depth = None;
        let mut best_size = None;
        let mut count = 0;
        let min_point_check = || {
            if best_depth.is_none() || best_size.is_none() {
                best_depth = new_depth;
                best_size = new_size;
                best_dag = dag.clone();
                true
            } else if (new_depth, new_size) > (best_depth, best_size) {
                count += 1;
                true
            } else if (new_depth, new_size) < (best_depth, best_size) {
                count = 1;
                best_depth = new_depth;
                best_size = new_size;
                true
            } else if (new_depth, new_size) == (best_depth, best_size) {
                false
            } else {
                true
            }
        };
        while min_point_check() {
            depth = new_depth;
            size = new_size;
            run_consolidate_blocks(&mut dag, target, approximation_degree, false);
            dag = run_unitary_synthesis(
                &mut dag,
                (0..dag.num_qubits()).collect(),
                0,
                Some(target),
                HashSet::new(),
                ["unitary"].into_iter().map(|x| x.to_string()).collect(),
                HashSet::new(),
                approximation_degree,
                None,
                None,
                false,
            )?;
            run_remove_identity_equiv(&mut dag, approximation_degree, Some(target));
            run_optimize_1q_gates_decomposition(&mut dag, Some(target), None, None)?;
            cancel_commutations(&mut dag, &mut commutation_checker, None, 1.0)?;
            if gates_missing_from_target(&dag, target)? {
                translation(&mut dag);
            }
            best_dag = dag.clone();
            new_depth = Some(dag.depth(false)?);
            new_size = Some(dag.size(false)?);
        }
    }
    dag_to_circuit(&dag, false)
}
