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
# pylint: disable=unused-wildcard-import,wildcard-import,undefined-variable

from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import *
from qiskit.converters import circuit_to_dag

from .backends import fake_singapore
from .utils import random_circuit


class PassBenchmarks:

    params = ([5, 14, 20],
              [1024])

    param_names = ['n_qubits', 'depth']
    timeout = 300

    def setup(self, n_qubits, depth):
        seed = 42
        self.circuit = random_circuit(n_qubits, depth, measure=True,
                                      conditional=True, reset=True, seed=seed,
                                      max_operands=2)
        self.fresh_dag = circuit_to_dag(self.circuit)
        self.basis_gates = ['u1', 'u2', 'u3', 'cx', 'iid']
        self.cmap = [[0, 1], [1, 0], [1, 2], [1, 6], [2, 1], [2, 3], [3, 2],
                     [3, 4], [3, 8], [4, 3], [5, 6], [5, 10], [6, 1], [6, 5],
                     [6, 7], [7, 6], [7, 8], [7, 12], [8, 3], [8, 7], [8, 9],
                     [9, 8], [9, 14], [10, 5], [10, 11], [11, 10], [11, 12],
                     [11, 16], [12, 7], [12, 11], [12, 13], [13, 12], [13, 14],
                     [13, 18], [14, 9], [14, 13], [15, 16], [16, 11], [16, 15],
                     [16, 17], [17, 16], [17, 18], [18, 13], [18, 17],
                     [18, 19], [19, 18]]
        self.coupling_map = CouplingMap(self.cmap)

        layout_pass = DenseLayout(self.coupling_map)
        layout_pass.run(self.fresh_dag)
        self.layout = layout_pass.property_set['layout']
        full_ancilla_pass = FullAncillaAllocation(self.coupling_map)
        full_ancilla_pass.property_set['layout'] = self.layout
        self.full_ancilla_dag = full_ancilla_pass.run(self.fresh_dag)
        enlarge_pass = EnlargeWithAncilla()
        enlarge_pass.property_set['layout'] = self.layout
        self.enlarge_dag = enlarge_pass.run(self.full_ancilla_dag)
        apply_pass = ApplyLayout()
        apply_pass.property_set['layout'] = self.layout
        self.dag = apply_pass.run(self.enlarge_dag)
        self.backend_props = fake_singapore.FakeSingapore().properties()

    def time_stochastic_swap(self, _, __):
        swap = StochasticSwap(self.coupling_map, seed=42)
        swap.property_set['layout'] = self.layout
        swap.run(self.dag)

    def peakmem_stochastic_swap(self, _, __):
        swap = StochasticSwap(self.coupling_map, seed=42)
        swap.property_set['layout'] = self.layout
        swap.run(self.dag)

    def track_stochastic_swap_depth(self, _, __):
        swap = StochasticSwap(self.coupling_map, seed=42)
        swap.property_set['layout'] = self.layout
        return swap.run(self.dag).depth()

    def track_stochastic_swap_swap_count(self, _, __):
        swap = StochasticSwap(self.coupling_map, seed=42)
        swap.property_set['layout'] = self.layout
        return swap.run(self.dag).count_ops().get('swap')

    # Disable lookahead swap benchmarks due to timeout.
    # def time_lookahead_swap(self, _, __):
    #     swap = LookaheadSwap(self.coupling_map)
    #     swap.property_set['layout'] = self.layout
    #     swap.run(self.dag)

    # def peakmem_lookahead_swap(self, _, __):
    #     swap = LookaheadSwap(self.coupling_map)
    #     swap.property_set['layout'] = self.layout
    #     swap.run(self.dag)

    # def track_lookahead_swap_depth(self, _, __):
    #     swap = LookaheadSwap(self.coupling_map)
    #     swap.property_set['layout'] = self.layout
    #     return swap.run(self.dag).depth()

    # def track_lookahead_swap_swap_count(self, _, __):
    #     swap = LookaheadSwap(self.coupling_map)
    #     swap.property_set['layout'] = self.layout
    #     return swap.run(self.dag).depth().count_ops().get('swap')

    def time_basic_swap(self, _, __):
        swap = BasicSwap(self.coupling_map)
        swap.property_set['layout'] = self.layout
        swap.run(self.dag)

    def peakmem_basic_swap(self, _, __):
        swap = BasicSwap(self.coupling_map)
        swap.property_set['layout'] = self.layout
        swap.run(self.dag)

    def track_basic_swap_depth(self, _, __):
        swap = BasicSwap(self.coupling_map)
        swap.property_set['layout'] = self.layout
        return swap.run(self.dag).depth()

    def track_basic_swap_swap_count(self, _, __):
        swap = BasicSwap(self.coupling_map)
        swap.property_set['layout'] = self.layout
        return swap.run(self.dag).depth().count_ops().get('swap')

    def time_csp_layout(self, _, __):
        CSPLayout(self.coupling_map, seed=42).run(self.fresh_dag)

    def peakmem_csp_layout(self, _, __):
        CSPLayout(self.coupling_map, seed=42).run(self.fresh_dag)

    def time_dense_layout(self, _, __):
        DenseLayout(self.coupling_map).run(self.fresh_dag)

    def peakmem_dense_layout(self, _, __):
        DenseLayout(self.coupling_map).run(self.fresh_dag)

    def time_layout_2q_distance(self, _, __):
        layout = Layout2qDistance(self.coupling_map)
        layout.property_set['layout'] = self.layout
        layout.run(self.dag)

    def peakmem_layout_2q_distance(self, _, __):
        layout = Layout2qDistance(self.coupling_map)
        layout.property_set['layout'] = self.layout
        layout.run(self.dag)

    def time_cxdirection(self, _, __):
        CXDirection(self.coupling_map).run(self.dag)

    def peakmem_cxdirection(self, _, __):
        CXDirection(self.coupling_map).run(self.dag)

    def track_cxdirection_depth(self, _, __):
        return CXDirection(self.coupling_map).run(self.dag).depth()

    def track_cxdirection_cnot_count(self, _, __):
        return CXDirection(
            self.coupling_map).run(self.dag).count_ops().get('cx')

    def time_apply_layout(self, _, __):
        layout = ApplyLayout()
        layout.property_set['layout'] = self.layout
        layout.run(self.dag)

    def peakmem_apply_layout(self, _, __):
        layout = ApplyLayout()
        layout.property_set['layout'] = self.layout
        layout.run(self.dag)

    def time_full_ancilla_allocation(self, _, __):
        ancilla = FullAncillaAllocation(self.coupling_map)
        ancilla.property_set['layout'] = self.layout
        ancilla.run(self.fresh_dag)

    def peakmem_full_ancilla_allocation(self, _, __):
        ancilla = FullAncillaAllocation(self.coupling_map)
        ancilla.property_set['layout'] = self.layout
        ancilla.run(self.fresh_dag)

    def time_enlarge_with_ancilla(self, _, __):
        ancilla = EnlargeWithAncilla()
        ancilla.property_set['layout'] = self.layout
        ancilla.run(self.full_ancilla_dag)

    def peakmem_enlarge_with_ancilla(self, _, __):
        ancilla = EnlargeWithAncilla()
        ancilla.property_set['layout'] = self.layout
        ancilla.run(self.full_ancilla_dag)

    def time_check_map(self, _, __):
        CheckMap(self.coupling_map).run(self.dag)

    def peakmem_check_map(self, _, __):
        CheckMap(self.coupling_map).run(self.dag)

    def time_check_cx_direction(self, _, __):
        CheckCXDirection(self.coupling_map).run(self.dag)

    def peakmem_check_cx_direction(self, _, __):
        CheckCXDirection(self.coupling_map).run(self.dag)

    def time_trivial_layout(self, _, __):
        TrivialLayout(self.coupling_map).run(self.fresh_dag)

    def peakmem_trivial_layout(self, _, __):
        TrivialLayout(self.coupling_map).run(self.fresh_dag)

    def time_set_layout(self, _, __):
        SetLayout(self.layout).run(self.fresh_dag)

    def peakmem_set_layout(self, _, __):
        SetLayout(self.layout).run(self.fresh_dag)

    def time_noise_adaptive_layout(self, _, __):
        NoiseAdaptiveLayout(self.backend_props).run(self.fresh_dag)

    def peakmem_noise_adaptive_layout(self, _, __):
        NoiseAdaptiveLayout(self.backend_props).run(self.fresh_dag)
