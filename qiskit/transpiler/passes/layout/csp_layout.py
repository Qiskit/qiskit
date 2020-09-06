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
from time import time
from constraint import Problem, RecursiveBacktrackingSolver, AllDifferentConstraint

from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass


class CustomSolver(RecursiveBacktrackingSolver):
    """A wrap to RecursiveBacktrackingSolver to support ``call_limit``"""

    def __init__(self, call_limit=None, time_limit=None):
        self.call_limit = call_limit
        self.time_limit = time_limit
        self.call_current = None
        self.time_start = None
        self.time_current = None
        super().__init__()

    def limit_reached(self):
        """Checks if a limit is reached."""
        if self.call_current is not None:
            self.call_current += 1
            if self.call_current > self.call_limit:
                return True
        if self.time_start is not None:
            self.time_current = time() - self.time_start
            if self.time_current > self.time_limit:
                return True
        return False

    def getSolution(self,  # pylint: disable=invalid-name
                    domains, constraints, vconstraints):
        """Wrap RecursiveBacktrackingSolver.getSolution to add the limits."""
        if self.call_limit is not None:
            self.call_current = 0
        if self.time_limit is not None:
            self.time_start = time()
        return super().getSolution(domains, constraints, vconstraints)

    def recursiveBacktracking(self,  # pylint: disable=invalid-name
                              solutions, domains, vconstraints, assignments, single):
        """Like ``constraint.RecursiveBacktrackingSolver.recursiveBacktracking`` but
        limited in the amount of calls by ``self.call_limit`` """
        if self.limit_reached():
            return None
        return super().recursiveBacktracking(solutions, domains, vconstraints, assignments,
                                             single)


class CSPLayout(AnalysisPass):
    """If possible, chooses a Layout as a CSP, using backtracking."""

    def __init__(self, coupling_map, strict_direction=False, seed=None, call_limit=1000,
                 time_limit=10):
        """If possible, chooses a Layout as a CSP, using backtracking.

        If not possible, does not set the layout property. In all the cases,
        the property `CSPLayout_stop_reason` will be added with one of the
        following values:

        * solution found: If a perfect layout was found.
        * nonexistent solution: If no perfect layout was found and every combination was checked.
        * call limit reached: If no perfect layout was found and the call limit was reached.
        * time limit reached: If no perfect layout was found and the time limit was reached.

        Args:
            coupling_map (Coupling): Directed graph representing a coupling map.
            strict_direction (bool): If True, considers the direction of the coupling map.
                                     Default is False.
            seed (int): Sets the seed of the PRNG.
            call_limit (int): Amount of times that
                ``constraint.RecursiveBacktrackingSolver.recursiveBacktracking`` will be called.
                None means no call limit. Default: 1000.
            time_limit (int): Amount of seconds that the pass will try to find a solution.
                None means no time limit. Default: 10 seconds.
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.strict_direction = strict_direction
        self.call_limit = call_limit
        self.time_limit = time_limit
        self.seed = seed

    def run(self, dag):
        qubits = dag.qubits
        cxs = set()

        for gate in dag.two_qubit_ops():
            cxs.add((qubits.index(gate.qargs[0]),
                     qubits.index(gate.qargs[1])))
        edges = self.coupling_map.get_edges()

        if self.time_limit is None and self.call_limit is None:
            solver = RecursiveBacktrackingSolver()
        else:
            solver = CustomSolver(call_limit=self.call_limit, time_limit=self.time_limit)

        problem = Problem(solver)
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
            stop_reason = 'nonexistent solution'
            if isinstance(solver, CustomSolver):
                if solver.time_current is not None and solver.time_current >= self.time_limit:
                    stop_reason = 'time limit reached'
                elif solver.call_current is not None and solver.call_current >= self.call_limit:
                    stop_reason = 'call limit reached'
        else:
            stop_reason = 'solution found'
            self.property_set['layout'] = Layout({v: qubits[k] for k, v in solution.items()})

        self.property_set['CSPLayout_stop_reason'] = stop_reason
