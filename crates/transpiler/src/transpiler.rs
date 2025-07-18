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

use pyo3::prelude::PyResult;

use crate::commutation_checker::get_standard_commutation_checker;
use crate::passes::*;
use crate::target::Target;
use crate::transpile_layout::TranspileLayout;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::converters::dag_to_circuit;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::{PhysicalQubit, Qubit, VirtualQubit};

/// A transpilation function for Rust native circuits for use in the C API. This will not cover
/// things that only exist in the Python API such as custom gates or control flow. When those
/// concepts exist in the rust data model this function must be expanded before adding them to the
/// C API.
pub fn transpile(
    circuit: &CircuitData,
    target: &Target,
    optimization_level: u8,
    approximation_degree: Option<f64>,
    seed: Option<u64>,
) -> PyResult<(CircuitData, TranspileLayout)> {
    if !(0..=3u8).contains(&optimization_level) {
        panic!("Invalid optimization level specified {optimization_level}");
    }
    let dag = DAGCircuit::from_circuit_data(circuit, false, None, None, None, None)?;
    let mut commutation_checker = get_standard_commutation_checker();
    let mut equivalence_library = get_standard_equivalence_library();
    let mut transpile_layout: TranspileLayout = TranspileLayout::new(None, None, dag.qubits().objects().to_owned(), dag.num_qubits() as u32);

    let unroll_3q_or_more = |dag: &mut DAGCircuit| -> PyResult<()> {
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
        run_unroll_3q_or_more(&mut out_dag, Some(target))?;
        out_dag = run_basis_translator(
            &mut out_dag,
            equivalence_library,
            HashMap::new(),
            3,
            None,
            Some(target),
            None,
        )?;
        *dag = out_dag;
        Ok(())
    };

    // Init stage
    unroll_3q_or_more(&mut dag);
    let mut transpile_layout = TranspileLayout::new(
        None,
        None,
        dag.qubits().objects().clone(),
        dag.num_qubits() as u32,
    );
    if optimization_level == 1 {
        run_inverse_cancellation_standard_gates(&mut dag);
    } else if optimization_level == 2 || optimization_level == 3 {
        if let Some((new_dag, permutation)) = run_elide_permutations(&dag)? {
            dag = new_dag;
            let permutation: Vec<Qubit> = permutation.into_iter().map(Qubit::new).collect();
            transpile_layout.compose_output_permutation(&permutation, false);
        };
        run_remove_diagonal_before_measure(&mut dag);
        run_remove_identity_equiv(&mut dag, approximation_degree, Some(target));
        run_inverse_cancellation_standard_gates(&mut dag);
        cancel_commutations(&mut dag, &mut commutation_checker, None, 1.0)?;
        run_consolidate_blocks(&mut dag, 1.0, false, None)?;
        let result = run_split_2q_unitaries(
            &mut dag,
            approximation_degree
                .unwrap_or(1.0 - f64::EPSILON)
                .min(1.0 - f64::EPSILON),
            true,
        )?;
        if let Some(result) = result {
            dag = result.0;
            let split_2q_permutation = result.1.into_iter().map(Qubit::new).collect::<Vec<_>>();
            transpile_layout.compose_output_permutation(&split_2q_permutation, true);
        }
    }
    // layout stage

    let sabre_heuristic =
        sabre::Heuristic::new(None, None, None, 10 * target.num_qubits().unwrap())
            .with_basic(1.0, sabre::SetScaling::Constant)
            .with_lookahead(0.5, 20, sabre::SetScaling::Size)
            .with_decay(0.001, 5);

    if optimization_level == 0 {
        // Apply a trivial layout
        apply_layout(&mut dag, &mut transpile_layout, target.num_qubits(), |x| {
            PhysicalQubit(x.0)
        });
    } else if optimization_level == 1 {
        if run_check_map(dag, target).is_none() {
            apply_layout(&mut dag, &mut transpile_layout, target.num_qubits(), |x| {
                PhysicalQubit(x.0)
            });
        } else if let Some(vf2_result) =
            vf2_layout_pass(&dag, target, false, Some(5_000_000), None, Some(2500), None)?
        {
            apply_layout(&mut dag, &mut transpile_layout, target.num_qubits(), |x| {
                vf2_result[x]
            });
        } else {
            let (result, initial_layout, final_layout) = sabre::sabre_layout_and_routing(
                &mut dag,
                target,
                sabre_heuristic,
                2,
                20,
                20,
                seed,
                Vec::new(),
                false,
            )?;
            dag = result;
            transpile_layout.initial_layout = Some(initial_layout);
            let permutation: Vec<Qubit> = final_layout
                .virt_to_phys
                .into_iter()
                .map(|x| Qubit(x.0))
                .collect();
            transpile_layout.compose_output_permutation(&permutation, true);
        }
    } else if optimization_level == 2 {
        if let Some(vf2_result) =
            vf2_layout_pass(&dag, target, false, Some(5_000_000), None, Some(2500), None)?
        {
            apply_layout(&mut dag, &mut transpile_layout, target.num_qubits(), |x| {
                vf2_result[x]
            });
        } else {
            let (result, initial_layout, final_layout) = sabre::sabre_layout_and_routing(
                &mut dag,
                target,
                sabre_heuristic,
                2,
                20,
                20,
                seed,
                Vec::new(),
                false,
            )?;
            dag = result;
            transpile_layout.initial_layout = Some(initial_layout);
            let permutation: Vec<Qubit> = final_layout
                .virt_to_phys
                .into_iter()
                .map(|x| Qubit(x.0))
                .collect();
            transpile_layout.compose_output_permutation(&permutation, true);
        }
    } else {
        if let Some(vf2_result) = vf2_layout_pass(
            &dag,
            target,
            false,
            Some(30_000_000),
            None,
            Some(250_000),
            None,
        )? {
            apply_layout(&mut dag, &mut transpile_layout, target.num_qubits(), |x| {
                vf2_result[x]
            });
        } else {
            let (result, initial_layout, final_layout) = sabre::sabre_layout_and_routing(
                &mut dag,
                target,
                sabre_heuristic,
                4,
                20,
                20,
                seed,
                Vec::new(),
                false,
            );
            dag = result;
            transpile_layout.initial_layout = Some(initial_layout);
            let permutation: Vec<Qubit> = final_layout
                .virt_to_phys
                .into_iter()
                .map(|x| Qubit(x.0))
                .collect();
            transpile_layout.compose_output_permutation(&permutation, true);
        }
    }
    // Routing stage
    let vf2_post_result = if optimization_level == 0 {
        if run_check_map(&dag, &target).is_none() {
            let (out_dag, final_layout) = sabre::sabre_routing(
                &dag,
                target,
                sabre_heuristic,
                transpile_layout.initial_layout,
                5,
                seed,
                Some(true),
            )?;
            dag = out_dag;
            transpile_layout.compose_output_permutation(final_layout.virt_to_phys, true);
        }
        None
    } else if optimization_level == 1 {
        vf2_post_layout_pass(&dag, target, false, Some(50_000), None, Some(2_500), None).unwrap()
    } else if optimization_level == 2 {
        vf2_post_layout_pass(&dag, target, false, Some(50_000), None, Some(2_500), None).unwrap()
    } else {
        vf2_post_layout_pass(
            &dag,
            target,
            false,
            Some(30_000_000),
            None,
            Some(250_000),
            None,
        )
        .unwrap()
    };
    if let Some(post_layout) = vf2_post_result {
        update_layout(&mut dag, &mut transpile_layout, |qubit| {
            Qubit(post_layout[VirtualQubit(qubit.0)].0)
        });
    }
    // Translation Stage
    let translation = |dag: &mut DAGCircuit| {
        dag = run_unitary_synthesis(
            &mut dag,
            (0..dag.num_qubits()).collect(),
            0,
            Some(target),
            HashSet::new(),
            ["unitary".to_string()].into_iter().collect(),
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
        Ok(())
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
            run_inverse_cancellation_standard_gates(&mut dag);
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
            ["unitary".to_string()].into_iter().collect(),
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
            run_remove_identity_equiv(&mut dag, approximation_degree, Some(target));
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
    Ok((dag_to_circuit(&dag, false)?, transpile_layout))
}
