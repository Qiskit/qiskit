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

"""A pass for choosing a Layout of a circuit onto a Coupling graph, as a
Constraint Satisfaction Problem. It tries to find a solution that fully
satisfy the circuit, i.e. no further swap is needed. If no solution is
found, no ``property_set['layout']`` is set.
"""
import random

from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError


class CSPLayout(AnalysisPass):
    """
    If possible, chooses a Layout as a CSP, using backtracking.
    """

    def __init__(self, coupling_map, strict_direction=False, seed=None, call_limit=None):
        """
        If possible, chooses a Layout as a CSP, using backtracking. If not possible,
        does not set the layout property.

        Args:
            coupling_map (Coupling): Directed graph representing a coupling map.
            strict_direction (bool): If True, considers the direction of the coupling map.
                                     Default is False.
            seed (int): Sets the seed of the PRNG.
            call_limit (int): Amount of times that
                ``constraint.RecursiveBacktrackingSolver.recursiveBacktracking`` will be called.
                The default is None, which means, no call limit.
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.strict_direction = strict_direction
        self.call_limit = call_limit
        self.seed = seed

    def run(self, dag):
        try:
            from constraint import Problem, RecursiveBacktrackingSolver, AllDifferentConstraint
        except ImportError:
            raise TranspilerError('CSPLayout requires python-constraint to run. '
                                  'Run pip install python-constraint')
        qubits = dag.qubits()
        cxs = set()

        for gate in dag.twoQ_gates():
            cxs.add((qubits.index(gate.qargs[0]),
                     qubits.index(gate.qargs[1])))
        edges = self.coupling_map.get_edges()

        class CustomSolver(RecursiveBacktrackingSolver):
            """A wrap to RecursiveBacktrackingSolver to support ``call_limit``"""

            def __init__(self, call_limit=None):
                self.call_limit = call_limit
                super().__init__()

            def recursiveBacktracking(self,  # pylint: disable=invalid-name
                                      solutions, domains, vconstraints, assignments, single):
                """Like ``constraint.RecursiveBacktrackingSolver.recursiveBacktracking`` but
                limited in the amount of calls by ``self.call_limit`` """
                self.call_limit -= 1
                if self.call_limit < 0:
                    return None
                return super().recursiveBacktracking(solutions, domains, vconstraints, assignments,
                                                     single)

        if self.call_limit is None:
            problem = Problem(RecursiveBacktrackingSolver())
        else:
            problem = Problem(CustomSolver(call_limit=self.call_limit))

        problem.addVariables(list(range(len(qubits))), self.coupling_map.physical_qubits)

        problem.addConstraint(AllDifferentConstraint())  # each wire is map to a single qbit

        if self.strict_direction:
            def constraint(control, target):
                return (control, target) in edges
        else:
            def constraint(control, target):
                return (control, target) in edges or (target, control) in edges

        for pair in cxs:
            problem.addConstraint(constraint, [pair[0], pair[1]])

        random.seed(self.seed)
        solution = problem.getSolution()

        if solution is None:
            return

        self.property_set['layout'] = Layout({v: qubits[k] for k, v in solution.items()})
