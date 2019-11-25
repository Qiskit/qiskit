# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=no-member,invalid-name,missing-docstring,no-name-in-module
# pylint: disable=attribute-defined-outside-init,unsubscriptable-object
# pylint: disable=unused-wildcard-import,wildcard-import


from qiskit.transpiler.passes import *
from qiskit.converters import circuit_to_dag

from .utils import random_circuit


class PassBenchmarks:

    params = ([1, 2, 5, 8, 14, 20],
              [8, 128, 1024])

    param_names = ['n_qubits', 'depth']
    timeout = 300

    def setup(self, n_qubits, depth):
        seed = 42
        self.circuit = random_circuit(n_qubits, depth, measure=True,
                                      conditional=True, reset=True, seed=seed)
        self.dag = circuit_to_dag(self.circuit)
        self.basis_gates = ['u1', 'u2', 'u3', 'cx', 'id']
        self.unrolled_dag = Unroller(self.basis_gates).run(self.dag)
        commutative_analysis = CommutationAnalysis()
        commutative_analysis.run(
            self.dag)
        self.commutation_set = commutative_analysis.property_set[
            'commutation_set']
        collect_blocks = Collect2qBlocks()
        collect_blocks.run(self.dag)
        self.block_list = collect_blocks.property_set['block_list']

    def time_unroller(self, _, __):
        Unroller(self.basis_gates).run(self.dag)

    def peakmem_unroller(self, _, __):
        Unroller(self.basis_gates).run(self.dag)

    def track_unroller_depth(self, _, __):
        return Unroller(self.basis_gates).run(self.dag).depth()

    def time_depth_pass(self, _, __):
        Depth().run(self.dag)

    def peakmem_depth_pass(self, _, __):
        Depth().run(self.dag)

    def time_size_pass(self, _, __):
        Size().run(self.dag)

    def peakmem_size_pass(self, _, __):
        Size().run(self.dag)

    def time_width_pass(self, _, __):
        Width().run(self.dag)

    def peakmem_width_pass(self, _, __):
        Width().run(self.dag)

    def time_count_ops_pass(self, _, __):
        CountOps().run(self.dag)

    def peakemem_count_ops_pass(self, _, __):
        CountOps().run(self.dag)

    def time_count_ops_longest_path(self, _, __):
        CountOpsLongestPath().run(self.dag)

    def peakmem_count_ops_longest_path(self, _, __):
        CountOpsLongestPath().run(self.dag)

    def time_num_tensor_factors(self, _, __):
        NumTensorFactors().run(self.dag)

    def peakmem_num_tensor_factors(self, _, __):
        NumTensorFactors().run(self.dag)

    def time_resource_optimization(self, _, __):
        ResourceEstimation().run(self.dag)

    def peakmem_resoure_optimization(self, _, __):
        ResourceEstimation().run(self.dag)

    def time_cx_cancellation(self, _, __):
        CXCancellation().run(self.dag)

    def peakmem_cx_cancellation(self, _, __):
        CXCancellation().run(self.dag)

    def time_dag_longest_path(self, _, __):
        DAGLongestPath().run(self.dag)

    def peakmem_dag_longest_path(self, _, __):
        DAGLongestPath().run(self.dag)

    def time_merge_adjacent_barriers(self, _, __):
        MergeAdjacentBarriers().run(self.dag)

    def peakmem_merge_adjacent_barriers(self, _, __):
        MergeAdjacentBarriers().run(self.dag)

    def time_optimize_1q(self, _, __):
        Optimize1qGates().run(self.unrolled_dag)

    def peakmem_optimize_1q(self, _, __):
        Optimize1qGates().run(self.unrolled_dag)

    def track_optimize_1q_depth(self, _, __):
        return Optimize1qGates().run(self.unrolled_dag).depth()

    def time_decompose_pass(self, _, __):
        Decompose().run(self.dag)

    def peakmem_decompose_pass(self, _, __):
        Decompose().run(self.dag)

    def track_decompose_depth(self, _, __):
        return Decompose().run(self.dag).depth()

    def time_unroll_3q_or_more(self, _, __):
        Unroll3qOrMore().run(self.dag)

    def peakmem_unroll_3q_or_more(self, _, __):
        Unroll3qOrMore().run(self.dag)

    def track_unroll_3q_or_more_depth(self, _, __):
        return Unroll3qOrMore().run(self.dag).depth()

    def time_commutation_analysis(self, _, __):
        CommutationAnalysis().run(self.dag)

    def peakmem_commutation_analysis(self, _, __):
        CommutationAnalysis().run(self.dag)

    def time_remove_reset_in_zero_state(self, _, __):
        RemoveResetInZeroState().run(self.dag)

    def peakemem_remove_reset_in_zero_state(self, _, __):
        RemoveResetInZeroState().run(self.dag)

    def track_remove_reset_in_zero_state(self, _, __):
        return RemoveResetInZeroState().run(self.dag).depth()

    def time_collect_2q_blocks(self, _, __):
        Collect2qBlocks().run(self.dag)

    def peakmem_collect_2q_blocks(self, _, __):
        Collect2qBlocks().run(self.dag)

    def time_commutative_cancellation(self, _, __):
        _pass = CommutativeCancellation()
        _pass.property_set['commutation_set'] = self.commutation_set
        _pass.run(self.dag)

    def peakmem_commutative_cancellation(self, _, __):
        _pass = CommutativeCancellation()
        _pass.property_set['commutation_set'] = self.commutation_set
        _pass.run(self.dag)

    def track_commutative_cancellation_depth(self, _, __):
        _pass = CommutativeCancellation()
        _pass.property_set['commutation_set'] = self.commutation_set
        return _pass.run(self.dag).depth()

    def time_optimize_swap_before_measure(self, _, __):
        OptimizeSwapBeforeMeasure().run(self.dag)

    def peakmem_optimize_swap_before_measure(self, _, __):
        OptimizeSwapBeforeMeasure().run(self.dag)

    def track_optimize_swap_before_measure_depth(self, _, __):
        return OptimizeSwapBeforeMeasure().run(self.dag).depth()

    def time_consolidate_blocks(self, _, __):
        _pass = ConsolidateBlocks()
        _pass.property_set['block_list'] = self.block_list
        _pass.run(self.dag)

    def peakmem_consolidate_blocks(self, _, __):
        _pass = ConsolidateBlocks()
        _pass.property_set['block_list'] = self.block_list
        _pass.run(self.dag)

    def track_consolidate_blocks_depth(self, _, __):
        _pass = ConsolidateBlocks()
        _pass.property_set['block_list'] = self.block_list
        return _pass.run(self.dag).depth()

    def time_barrier_before_final_measurements(self, _, __):
        BarrierBeforeFinalMeasurements().run(self.dag)

    def peakmem_barrier_before_final_measurement(self, _, __):
        BarrierBeforeFinalMeasurements().run(self.dag)

    def track_barrier_before_final_measurement(self, _, __):
        BarrierBeforeFinalMeasurements().run(self.dag).depth()

    def time_remove_diagonal_gates_before_measurement(self, _, __):
        RemoveDiagonalGatesBeforeMeasure().run(self.dag)

    def peakmem_remove_diagonal_gates_before_measurement(self, _, __):
        RemoveDiagonalGatesBeforeMeasure().run(self.dag)

    def track_remove_diagonal_gates_before_measurement(self, _, __):
        return RemoveDiagonalGatesBeforeMeasure().run(self.dag).run()

    def time_remove_final_measurements(self, _, __):
        RemoveFinalMeasurements().run(self.dag)

    def peakmem_remove_final_measurements(self, _, __):
        RemoveFinalMeasurements().run(self.dag)

    def track_remove_final_measurements_depth(self, _, __):
        return RemoveFinalMeasurements().run(self.dag).depth()
