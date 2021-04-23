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
import warnings
from time import time
import numpy as np
from constraint import Problem, RecursiveBacktrackingSolver, AllDifferentConstraint, Constraint, Unassigned

from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass

from .layout_scorer import LayoutScorer

class CustomSolver(RecursiveBacktrackingSolver):
    """ A wrap to RecursiveBacktrackingSolver to support 
        ``call_limit``, ``time_limit`` and ``solution_limit``
    """

    def __init__(self, call_limit=None, time_limit=None, solution_limit=None):

        self.call_limit = call_limit
        self.time_limit = time_limit
        self.solution_limit = solution_limit

        self.call_current = None
        self.time_start = None
        self.time_current = None
        self.solution_current = None

        super().__init__()

    def callLimitReached(self):
        """ Check if the call_limit is reached """
        if self.call_current is not None:
            self.call_current += 1
            if self.call_current > self.call_limit:
                return True
 
        return False

    def timeLimitReached(self):
        """ Check if the time_limit is reached """
        if self.time_start is not None:
            self.time_current = time() - self.time_start
            if self.time_current > self.time_limit:
                return True

        return False

    def solutionLimitReached(self, solutions):
        """ Check if the solution_limit is reached """
        if self.solution_current is not None:
            self.solution_current = len(solutions)
            if self.solution_current >= self.solution_limit:
                return True

        return False

    def getSolution(self,
                    domains, constraints, vconstraints):
        """Wrap RecursiveBacktrackingSolver.getSolution to add the limits."""
        if self.call_limit is not None:
            self.call_current = 0
        if self.time_limit is not None:
            self.time_start = time()
        if self.solution_limit is not None:
            self.solution_current = 0
        return super().getSolution(domains, constraints, vconstraints)

    def getSolutions(self,
                     domains, constraints, vconstraints):
        """ Wrap RecursiveBacktrackingSolver.getSolutions to add the limits."""
        if self.call_limit is not None:
            self.call_current = 0
        if self.time_limit is not None:
            self.time_start = time()
        if self.solution_limit is not None:
            self.solution_current = 0
        return super().getSolutions(domains, constraints, vconstraints)

    def recursiveBacktracking(self,  # pylint: disable=invalid-name
                              solutions, domains, vconstraints, assignments, single):
        """Like ``constraint.RecursiveBacktrackingSolver.recursiveBacktracking`` but
        limited to the introduced limits """
        if (
                self.callLimitReached() or
                self.timeLimitReached() or
                self.solutionLimitReached(solutions)
            ):
            return solutions
        return super().recursiveBacktracking(solutions, domains, vconstraints, assignments, single)


class CustomConstraint(Constraint):
    """Based on constraint.FunctionConstraint, reduces all the generality to check if a
    2-sized-tuple (control, target) is in a set (edges).
    """

    def __init__(self, edges, assigned=True):
        self._edges = edges
        self._assigned = assigned

    def __call__(self, variables, domains, assignments, forwardcheck=False):
        parms = (assignments.get(variables[0], Unassigned),
                 assignments.get(variables[1], Unassigned))
        if Unassigned in parms:
            return (self._assigned or parms in self._edges) and (
                    not forwardcheck or
                    parms == (Unassigned, Unassigned) or
                    self.forwardCheck(variables, domains, assignments)
            )
        return parms in self._edges


class CSPLayout(AnalysisPass):
    """If possible, chooses a Layout as a CSP, using backtracking."""

    def __init__(self, coupling_map, strict_direction=False, seed=None, call_limit=1000,
                 time_limit=10, solution_limit=1, backend_properties=None):
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
            solution_limit (int): Limit the number of solutions, which should be obtained 
                by the solver. Default: 1.
            backend_properties (BackendProperties): The properties of the backend, needed if 
                solution_limit is greater one and a solution needs to be picked from the bunch. 
                Default: None.
        Raises:
            Warning: "Can only check multiple solutions when backend properties are given. \
                      Defaulting to limiting solutions!"
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.strict_direction = strict_direction
        self.call_limit = call_limit
        self.time_limit = time_limit
        self.solution_limit = solution_limit
        self.backend_properties = backend_properties
        self.seed = seed

        if self.solution_limit < 1:
            self.solution_limit = 1
            warnings.warn("Can only check positive number of solutions. "
                          "Defaulting to one solution!", RuntimeWarning)
        elif self.solution_limit > 1 and not self.backend_properties:
            self.solution_limit = 1
            warnings.warn("Can only check multiple solutions when backend properties are given. "
                          "Defaulting to one solution!", RuntimeWarning)

    def run(self, dag):
        """ run the layout method """

        if (
                self.time_limit is None and
                self.call_limit is None and
                self.solution_limit == 1
            ):
            csp_solver = RecursiveBacktrackingSolver()
        else:
            csp_solver = CustomSolver(call_limit=self.call_limit, 
                                      time_limit=self.time_limit,
                                      solution_limit=self.solution_limit)

        problem = self._get_csp_problem(dag, csp_solver)

        if self.solution_limit == 1:
            solution_list = [problem.getSolution()]
        else:
            solution_list = problem.getSolutions()

        if any(solution_list):
            # solution_list has entries
            stop_reason = 'solution found'
            if self.solution_limit == 1:
                solution = solution_list[0]
            else:
                layout_scorer = LayoutScorer(self.coupling_map, self.backend_properties)
                sol_layouts = [Layout({v: dag.qubits[k] for k, v in solution.items()})
                               for solution in solution_list]
                layout_fidelities = [layout_scorer.evaluate(dag, layout) for layout in sol_layouts]
                max_fid_idx = np.argsort(layout_fidelities)[-1]
                solution = solution_list[max_fid_idx]

            self.property_set['layout'] = Layout(
                {v: dag.qubits[k] for k, v in solution.items()})
            self.property_set['CSPLayout_stop_reason'] = stop_reason

        else:
            # solution_list is empty
            stop_reason = 'nonexistent solution'
            if (
                    csp_solver.time_current is not None and
                    csp_solver.time_current >= self.time_limit
                ):
                stop_reason = 'time limit reached'
            elif (
                    csp_solver.call_current is not None and
                    csp_solver.call_current >= self.call_limit
                ):
                stop_reason = 'call limit reached'
                
        self.property_set['CSPLayout_stop_reason'] = stop_reason

    def _get_csp_problem(self, dag, solver):
        """ Create a CSP Problem """
        logical_edges = set()
        physical_edges = set()

        for gate in dag.two_qubit_ops():
            logical_edges.add((dag.qubits.index(gate.qargs[0]),
                               dag.qubits.index(gate.qargs[1])))
        physical_edges = set(self.coupling_map.get_edges())

        if not self.strict_direction:
            logical_edges = {tuple(sorted(edge)) for edge in logical_edges}
            physical_edges = {tuple(sorted(edge)) for edge in physical_edges}

        variables = list(range(len(dag.qubits)))
        variable_domains = list(self.coupling_map.physical_qubits)
        random.Random(self.seed).shuffle(variable_domains)

        problem = Problem(solver)
        problem.addVariables(variables, variable_domains)
        problem.addConstraint(AllDifferentConstraint())  # each wire is map to a single qbit

        for edge in logical_edges:
            problem.addConstraint(CustomConstraint(physical_edges), (edge[0], edge[1]))

        return problem