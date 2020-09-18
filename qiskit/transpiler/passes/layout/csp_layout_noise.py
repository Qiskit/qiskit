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
import numpy as np
from constraint import RecursiveBacktrackingSolver

from qiskit.circuit import Gate, Measure
from qiskit.transpiler.layout import Layout

from .csp_layout import CSPLayout
from ._crb_solver import CRBSolver


class CSPLayoutNoise(CSPLayout):
    """If possible, chooses a Layout as a CSP, using backtracking."""

    def __init__(self,
                 coupling_map,
                 backend_prop=None,
                 strict_direction=True,
                 seed=None,
                 call_limit=1000,
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
            backend_prop (BackendProperties): Properties of the backend like gate errors.
                                              Default is None.
            strict_direction (bool): If True, considers the direction of the coupling map.
                                     Default is True.
            seed (int): Sets the seed of the PRNG.
            call_limit (int): Amount of times that
                ``constraint.RecursiveBacktrackingSolver.recursiveBacktracking`` will be called.
                None means no call limit. Default: 1000.
            time_limit (int): Amount of seconds that the pass will try to find a solution.
                None means no time limit. Default: 10 seconds.
        """
        self.backend_prop = backend_prop
        super().__init__(coupling_map,
                         strict_direction=strict_direction,
                         seed=seed,
                         call_limit=call_limit,
                         time_limit=time_limit)

    def run(self, dag):
        qubits = dag.qubits

        if self.time_limit is None and self.call_limit is None:
            solver = RecursiveBacktrackingSolver()
        else:
            solver = CRBSolver(call_limit=self.call_limit, time_limit=self.time_limit)

        solutions = self._get_csp_solutions(solver, dag)

        if not solutions:
            stop_reason = "nonexistent solution"
            if isinstance(solver, CRBSolver):
                if solver.time_current is not None and solver.time_current >= self.time_limit:
                    stop_reason = "time limit reached"
                elif solver.call_current is not None and solver.call_current >= self.call_limit:
                    stop_reason = "call limit reached"
        else:
            if self.backend_prop:
                gate_errors = [self._calc_sol_gate_error(dag, sol) for sol in solutions]
            else:
                gate_errors = [0]
            min_err_idx = np.argsort(gate_errors)[0]
            sol = solutions[min_err_idx]

            stop_reason = "solution found"
            self.property_set["layout"] = Layout({v: qubits[k] for k, v in sol.items()})

        self.property_set["CSPLayout_stop_reason"] = stop_reason

    def _calc_sol_gate_error(self, dag, solution):
        """Calculate the gate error of the solution"""
        gate_error = 0
        for node in dag.op_nodes(include_directives=False):
            physical_qubits = [solution[qubit.index] for qubit in node.qargs]
            if isinstance(node.op, Gate):
                if len(node.qargs) == 1:
                    # exact gate not known, use average
                    gate_error += 1/2 * (self.backend_prop.gate_error("u2", physical_qubits) +
                                         self.backend_prop.gate_error("u3", physical_qubits))
                elif len(node.qargs) == 2:
                    gate_error += self.backend_prop.gate_error("cx", physical_qubits)
            elif isinstance(node.op, Measure):
                gate_error += self.backend_prop.readout_error(*physical_qubits)
        return gate_error
