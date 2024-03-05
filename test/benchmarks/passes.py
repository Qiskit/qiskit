# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
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
# pylint: disable=unused-wildcard-import,wildcard-import,undefined-variable


from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as SEL
from qiskit.transpiler.passes import *
from qiskit.converters import circuit_to_dag
from qiskit.circuit.library import CXGate

from .utils import random_circuit


class Collect2QPassBenchmarks:
    params = ([5, 14, 20], [1024])

    param_names = ["n_qubits", "depth"]
    timeout = 300

    def setup(self, n_qubits, depth):
        seed = 42
        self.circuit = random_circuit(
            n_qubits, depth, measure=True, conditional=True, reset=True, seed=seed
        )
        self.dag = circuit_to_dag(self.circuit)
        collect_blocks = Collect2qBlocks()
        collect_blocks.run(self.dag)
        self.block_list = collect_blocks.property_set["block_list"]

    def time_consolidate_blocks(self, _, __):
        _pass = ConsolidateBlocks()
        _pass.property_set["block_list"] = self.block_list
        _pass.run(self.dag)


class CommutativeAnalysisPassBenchmarks:
    params = ([5, 14, 20], [1024])

    param_names = ["n_qubits", "depth"]
    timeout = 300

    def setup(self, n_qubits, depth):
        seed = 42
        self.circuit = random_circuit(
            n_qubits, depth, measure=True, conditional=True, reset=True, seed=seed
        )
        self.dag = circuit_to_dag(self.circuit)
        commutative_analysis = CommutationAnalysis()
        commutative_analysis.run(self.dag)
        self.commutation_set = commutative_analysis.property_set["commutation_set"]

    def time_commutative_cancellation(self, _, __):
        _pass = CommutativeCancellation()
        _pass.property_set["commutation_set"] = self.commutation_set
        _pass.run(self.dag)


class MultipleBasisPassBenchmarks:
    params = (
        [5, 14, 20],
        [1024],
        [["u", "cx", "id"], ["rx", "ry", "rz", "r", "rxx", "id"], ["rz", "x", "sx", "cx", "id"]],
    )

    param_names = ["n_qubits", "depth", "basis_gates"]
    timeout = 300

    def setup(self, n_qubits, depth, basis_gates):
        seed = 42
        self.circuit = random_circuit(n_qubits, depth, measure=True, seed=seed)
        self.dag = circuit_to_dag(self.circuit)
        self.basis_gates = basis_gates

    def time_optimize_1q_decompose(self, _, __, ___):
        Optimize1qGatesDecomposition(self.basis_gates).run(self.dag)

    def time_optimize_1q_commutation(self, _, __, ___):
        Optimize1qGatesSimpleCommutation(self.basis_gates).run(self.dag)

    def time_basis_translator(self, _, __, ___):
        BasisTranslator(SEL, self.basis_gates).run(self.dag)


class PassBenchmarks:
    params = ([5, 14, 20], [1024])

    param_names = ["n_qubits", "depth"]
    timeout = 300

    def setup(self, n_qubits, depth):
        seed = 42
        self.circuit = random_circuit(
            n_qubits, depth, measure=True, conditional=True, reset=True, seed=seed
        )
        self.dag = circuit_to_dag(self.circuit)
        self.basis_gates = ["u1", "u2", "u3", "cx", "id"]

    def time_depth_pass(self, _, __):
        Depth().run(self.dag)

    def time_size_pass(self, _, __):
        Size().run(self.dag)

    def time_width_pass(self, _, __):
        Width().run(self.dag)

    def time_count_ops_pass(self, _, __):
        CountOps().run(self.dag)

    def time_count_ops_longest_path(self, _, __):
        CountOpsLongestPath().run(self.dag)

    def time_num_tensor_factors(self, _, __):
        NumTensorFactors().run(self.dag)

    def time_resource_optimization(self, _, __):
        ResourceEstimation().run(self.dag)

    def time_cx_cancellation(self, _, __):
        InverseCancellation([CXGate()]).run(self.dag)

    def time_dag_longest_path(self, _, __):
        DAGLongestPath().run(self.dag)

    def time_merge_adjacent_barriers(self, _, __):
        MergeAdjacentBarriers().run(self.dag)

    def time_decompose_pass(self, _, __):
        Decompose().run(self.dag)

    def time_unroll_3q_or_more(self, _, __):
        Unroll3qOrMore().run(self.dag)

    def time_commutation_analysis(self, _, __):
        CommutationAnalysis().run(self.dag)

    def time_remove_reset_in_zero_state(self, _, __):
        RemoveResetInZeroState().run(self.dag)

    def time_collect_2q_blocks(self, _, __):
        Collect2qBlocks().run(self.dag)

    def time_optimize_swap_before_measure(self, _, __):
        OptimizeSwapBeforeMeasure().run(self.dag)

    def time_barrier_before_final_measurements(self, _, __):
        BarrierBeforeFinalMeasurements().run(self.dag)

    def time_remove_diagonal_gates_before_measurement(self, _, __):
        RemoveDiagonalGatesBeforeMeasure().run(self.dag)

    def time_remove_final_measurements(self, _, __):
        RemoveFinalMeasurements().run(self.dag)

    def time_contains_instruction(self, _, __):
        ContainsInstruction("cx").run(self.dag)

    def time_gates_in_basis(self, _, __):
        GatesInBasis(self.basis_gates).run(self.dag)

    def time_remove_barriers(self, _, __):
        RemoveBarriers().run(self.dag)


class MultiQBlockPassBenchmarks:
    params = ([5, 14, 20], [1024], [1, 2, 3, 4, 5])

    param_names = ["n_qubits", "depth", "max_block_size"]
    timeout = 300

    def setup(self, n_qubits, depth, _):
        seed = 42
        self.circuit = random_circuit(
            n_qubits, depth, measure=True, conditional=True, reset=True, seed=seed
        )
        self.dag = circuit_to_dag(self.circuit)

    def time_collect_multiq_block(self, _, __, max_block_size):
        CollectMultiQBlocks(max_block_size).run(self.dag)
