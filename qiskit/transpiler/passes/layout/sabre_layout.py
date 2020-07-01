# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Layout selection using the SABRE bidirectional search approach from Li et al.
"""

import logging
import numpy as np

from qiskit.converters import dag_to_circuit
from qiskit.transpiler.passes.layout.set_layout import SetLayout
from qiskit.transpiler.passes.layout.full_ancilla_allocation import FullAncillaAllocation
from qiskit.transpiler.passes.layout.enlarge_with_ancilla import EnlargeWithAncilla
from qiskit.transpiler.passes.layout.apply_layout import ApplyLayout
from qiskit.transpiler.passes.routing import SabreSwap
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError

logger = logging.getLogger(__name__)


class SabreLayout(AnalysisPass):
    """Choose a Layout via iterative bidirectional routing of the input circuit.

    Starting with a random initial `Layout`, the algorithm does a full routing
    of the circuit (via the `routing_pass` method) to end up with a
    `final_layout`. This final_layout is then used as the initial_layout for
    routing the reverse circuit. The algorithm iterates a number of times until
    it finds an initial_layout that reduces full routing cost.

    This method exploits the reversibility of quantum circuits, and tries to
    include global circuit information in the choice of initial_layout.

    **References:**

    [1] Li, Gushu, Yufei Ding, and Yuan Xie. "Tackling the qubit mapping problem
    for NISQ-era quantum devices." ASPLOS 2019.
    `arXiv:1809.02573 <https://arxiv.org/pdf/1809.02573.pdf>`_
    """

    def __init__(self, coupling_map, routing_pass=None, seed=None,
                 max_iterations=3):
        """SabreLayout initializer.

        Args:
            coupling_map (Coupling): directed graph representing a coupling map.
            routing_pass (BasePass): the routing pass to use while iterating.
            seed (int): seed for setting a random first trial layout.
            max_iterations (int): number of forward-backward iterations.
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.routing_pass = routing_pass
        self.seed = seed
        self.max_iterations = max_iterations

    def run(self, dag):
        """Run the SabreLayout pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to find layout for.

        Raises:
            TranspilerError: if dag wider than self.coupling_map
        """
        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError('More virtual qubits exist than physical.')

        # Choose a random initial_layout.
        if self.seed is None:
            self.seed = np.random.randint(0, np.iinfo(np.int32).max)
        rng = np.random.default_rng(self.seed)

        physical_qubits = rng.choice(self.coupling_map.size(),
                                     len(dag.qubits), replace=False)
        physical_qubits = rng.permutation(physical_qubits)
        initial_layout = Layout({q: dag.qubits[i]
                                 for i, q in enumerate(physical_qubits)})

        if self.routing_pass is None:
            self.routing_pass = SabreSwap(self.coupling_map, 'decay', seed=self.seed)

        # Do forward-backward iterations.
        circ = dag_to_circuit(dag)
        for i in range(self.max_iterations):
            for _ in ('forward', 'backward'):
                pm = self._layout_and_route_passmanager(initial_layout)
                new_circ = pm.run(circ)

                # Update initial layout and reverse the unmapped circuit.
                pass_final_layout = pm.property_set['final_layout']
                final_layout = self._compose_layouts(initial_layout,
                                                     pass_final_layout,
                                                     circ.qregs)
                initial_layout = final_layout
                circ = circ.reverse_ops()

            # Diagnostics
            logger.info('After round %d, num_swaps: %d',
                        i+1, new_circ.count_ops().get('swap', 0))
            logger.info('new initial layout')
            logger.info(initial_layout)

        self.property_set['layout'] = initial_layout

    def _layout_and_route_passmanager(self, initial_layout):
        """Return a passmanager for a full layout and routing.

        We use a factory to remove potential statefulness of passes.
        """
        layout_and_route = [SetLayout(initial_layout),
                            FullAncillaAllocation(self.coupling_map),
                            EnlargeWithAncilla(),
                            ApplyLayout(),
                            self.routing_pass]
        pm = PassManager(layout_and_route)
        return pm

    def _compose_layouts(self, initial_layout, pass_final_layout, qregs):
        """Return the real final_layout resulting from the composition
        of an initial_layout with the final_layout reported by a pass.

        The routing passes internally start with a trivial layout, as the
        layout gets applied to the circuit prior to running them. So the
        "final_layout" they report must be amended to account for the actual
        initial_layout that was selected.
        """
        trivial_layout = Layout.generate_trivial_layout(*qregs)
        pass_final_layout = Layout({trivial_layout[v.index]: p
                                    for v, p in pass_final_layout.get_virtual_bits().items()})
        qubit_map = Layout.combine_into_edge_map(initial_layout, trivial_layout)
        final_layout = {v: pass_final_layout[qubit_map[v]]
                        for v, _ in initial_layout.get_virtual_bits().items()}
        return Layout(final_layout)
