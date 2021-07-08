# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Integer programming model for quantum circuit compilation."""

import logging

try:
    from docplex.mp.model import Model

    HAS_DOCPLEX = True
except ImportError:
    HAS_DOCPLEX = False

try:
    import cplex  # pylint: disable=unused-import

    HAS_CPLEX = True
except ImportError:
    HAS_CPLEX = False

from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout

logger = logging.getLogger(__name__)


class BIPMappingModel:
    """Internal model to create and solve a BIP problem for mapping.

    Attributes:
        problem (Model):
            A CPLEX problem model object, which is set by calling
            :method:`create_cpx_problem`. After calling :method:`solve_cpx_problem`,
            the solution will be stored in :attr:`solution`). None if it's not yet set.
    """

    def __init__(self, dag, coupling_map, dummy_timesteps=None):
        """
        Args:
            dummy_timesteps (int):
                Number of dummy time steps, after each real layer of gates, to
                allow arbitrary swaps between neighbors.

        Raises:
            MissingOptionalLibraryError: If docplex is not installed
            TranspilerError: If size of virtual qubits and physical qubits differ, or
                if coupling_map is not symmetric (bidirectional).
        """
        if not HAS_DOCPLEX:
            raise MissingOptionalLibraryError(
                libname="DOcplex",
                name="Decision Optimization CPLEX Modeling for Python",
                pip_install="pip install docplex",
            )

        if len(dag.qubits) != coupling_map.size():
            raise TranspilerError(
                "BIPMappingModel assumes the same size of virtual and physical qubits."
            )
        if not coupling_map.is_symmetric:
            raise TranspilerError(
                "BIPMappingModel assumes the coupling_map is symmetric (bidirectional)."
            )

        self._dag = dag
        self._coupling = coupling_map

        self.problem = None
        self.solution = None
        self.num_vqubits = len(self._dag.qubits)
        self.num_pqubits = self._coupling.size()
        self._arcs = self._coupling.get_edges()

        # pylint: disable=unnecessary-comprehension
        initial_layout = Layout.generate_trivial_layout(*dag.qregs.values())
        self._virtual_to_index = {v: i for i, v in enumerate(initial_layout.get_virtual_bits())}
        self._index_to_virtual = {i: v for i, v in enumerate(initial_layout.get_virtual_bits())}

        # Construct internal circuit model
        # Extract layers with 2-qubit gates
        self._to_su4layer = []
        self.su4layers = []
        for lay in dag.layers():
            laygates = []
            subdag = lay["graph"]
            for node in subdag.two_qubit_ops():
                i1 = self._virtual_to_index[node.qargs[0]]
                i2 = self._virtual_to_index[node.qargs[1]]
                laygates.append((i1, i2))
            if laygates:
                self._to_su4layer.append(len(self.su4layers))
                self.su4layers.append(laygates)
            else:
                self._to_su4layer.append(-1)
        # Add dummy time steps inbetween su4layers. Dummy time steps can only contain SWAPs.
        self.gates = []  # layered 2q-gates with dummy steps
        for k, lay in enumerate(self.su4layers):
            self.gates.append(lay)
            if k == len(self.su4layers) - 1:  # do not add dummy steps after the last layer
                break
            self.gates.extend([[]] * dummy_timesteps)

        logger.info("Num virtual qubits: %d", self.num_vqubits)
        logger.info("Num physical qubits: %d", self.num_pqubits)
        logger.info("Model depth: %d", self.depth)
        logger.info("Dummy steps: %d", dummy_timesteps)

    @property
    def depth(self):
        """Number of time-steps (including dummy steps)."""
        return len(self.gates)

    def is_su4layer(self, depth: int) -> bool:
        """Check if the depth-th layer is su4layer (layer containing 2q-gates) or not.

        Args:
            depth: Depth of the ordinary layer

        Returns:
            True if the depth-th layer is su4layer, otherwise False
        """
        return self._to_su4layer[depth] >= 0

    def to_su4layer_depth(self, depth: int) -> int:
        """Return the depth as a su4layer. If the depth-th layer is not a su4layer, return -1.

        Args:
            depth: Depth of the ordinary layer

        Returns:
            su4layer depth if the depth-th layer is a su4layer, otherwise -1
        """
        return self._to_su4layer[depth]

    # pylint: disable=invalid-name
    def _is_dummy_step(self, t: int):
        """Check if the time-step t is a dummy step or not."""
        return len(self.gates[t]) == 0

    def create_cpx_problem(self, objective: str, line_symm: bool = False):
        """Create integer programming model to compile a circuit.

        Args:
            objective:
                Type of objective function:

                * ``'error_rate'``: [Not implemented] predicted error rate of the circuit
                * ``'depth'``: depth (number of timesteps) of the circuit
                * ``'balanced'``: [Not implemented] weighted sum of error_rate and depth

            line_symm:
                Use symmetry breaking constrainst for line topology. Should
                only be True if the hardware graph is a chain/line/path.

        Raises:
            TranspilerError: if unknow objective type is specified.
        """
        mdl = Model()

        # *** Define main variables ***
        # Add w variables
        w = {}
        for t in range(self.depth):
            for q in range(self.num_vqubits):
                for j in range(self.num_pqubits):
                    w[t, q, j] = mdl.binary_var(name=f"w_{t}_{q}_{j}")
        # Add y variables
        y = {}
        for t in range(self.depth):
            for (p, q) in self.gates[t]:
                for (i, j) in self._arcs:
                    y[t, p, q, i, j] = mdl.binary_var(name=f"y_{t}_{p}_{q}_{i}_{j}")
        # Add x variables
        x = {}
        for t in range(self.depth - 1):
            for q in range(self.num_vqubits):
                for i in range(self.num_pqubits):
                    x[t, q, i, i] = mdl.binary_var(name=f"x_{t}_{q}_{i}_{i}")
                    for j in self._coupling.neighbors(i):
                        x[t, q, i, j] = mdl.binary_var(name=f"x_{t}_{q}_{i}_{j}")

        # *** Define main constraints ***
        # Assignment constraints for w variables
        for t in range(self.depth):
            for q in range(self.num_vqubits):
                mdl.add_constraint(
                    sum(w[t, q, j] for j in range(self.num_pqubits)) == 1,
                    ctname=f"assignment_vqubits_{q}_at_{t}",
                )
        for t in range(self.depth):
            for j in range(self.num_pqubits):
                mdl.add_constraint(
                    sum(w[t, q, j] for q in range(self.num_vqubits)) == 1,
                    ctname=f"assignment_pqubits_{j}_at_{t}",
                )
        # Each gate must be implemented
        for t in range(self.depth):
            for (p, q) in self.gates[t]:
                mdl.add_constraint(
                    sum(y[t, p, q, i, j] for (i, j) in self._arcs) == 1,
                    ctname=f"implement_gate_{p}_{q}_at_{t}",
                )
        # Gate can be implemented iff both of its qubits are located at the associated nodes
        for t in range(self.depth):
            for (p, q) in self.gates[t]:
                for (i, j) in self._arcs:
                    # Apply McCormick to y[t, p, q, i, j] == w[t, p, i] * w[t, q, j]
                    mdl.add_constraint(
                        y[t, p, q, i, j] >= w[t, p, i] + w[t, q, j] - 1,
                        ctname=f"McCormickLB_{p}_{q}_{i}_{j}_at_{t}",
                    )
                    mdl.add_constraint(
                        y[t, p, q, i, j] <= w[t, p, i],
                        ctname=f"McCormickUB1_{p}_{q}_{i}_{j}_at_{t}",
                    )
                    mdl.add_constraint(
                        y[t, p, q, i, j] <= w[t, q, j],
                        ctname=f"McCormickUB2_{p}_{q}_{i}_{j}_at_{t}",
                    )
        # Logical qubit flow-out constraints
        for t in range(self.depth - 1):  # Flow out; skip last time step
            for q in range(self.num_vqubits):
                for i in range(self.num_pqubits):
                    mdl.add_constraint(
                        w[t, q, i]
                        == x[t, q, i, i] + sum(x[t, q, i, j] for j in self._coupling.neighbors(i)),
                        ctname=f"flow_out_{q}_{i}_at_{t}",
                    )
        # Logical qubit flow-in constraints
        for t in range(1, self.depth):  # Flow in; skip first time step
            for q in range(self.num_vqubits):
                for i in range(self.num_pqubits):
                    mdl.add_constraint(
                        w[t, q, i]
                        == x[t - 1, q, i, i]
                        + sum(x[t - 1, q, j, i] for j in self._coupling.neighbors(i)),
                        ctname=f"flow_in_{q}_{i}_at_{t}",
                    )
        # If a gate is implemented, involved qubits cannot swap with other positions
        for t in range(self.depth - 1):
            for (p, q) in self.gates[t]:
                for (i, j) in self._arcs:
                    mdl.add_constraint(
                        x[t, p, i, j] == x[t, q, j, i], ctname=f"swap_{p}_{q}_{i}_{j}_at_{t}"
                    )
        # Qubit not in gates can flip with their neighbors
        for t in range(self.depth - 1):
            q_no_gate = list(range(self.num_vqubits))
            for (p, q) in self.gates[t]:
                q_no_gate.remove(p)
                q_no_gate.remove(q)
            for (i, j) in self._arcs:
                mdl.add_constraint(
                    sum(x[t, q, i, j] for q in q_no_gate) == sum(x[t, p, j, i] for p in q_no_gate),
                    ctname=f"swap_no_gate_{i}_{j}_at_{t}",
                )

        # *** Define supplemental variables ***
        # Add z variables to count dummy steps (supplemental variables for symmetry breaking)
        z = {}
        for t in range(self.depth):
            if self._is_dummy_step(t):
                z[t] = mdl.binary_var(name=f"z_{t}")

        # *** Define supplemental constraints ***
        # See if a dummy time step is needed
        for t in range(self.depth):
            if self._is_dummy_step(t):
                for q in range(self.num_vqubits):
                    mdl.add_constraint(
                        sum(x[t, q, i, j] for (i, j) in self._arcs) <= z[t],
                        ctname=f"dummy_ts_needed_for_vqubit_{q}_at_{t}",
                    )
        # Symmetry breaking between dummy time steps
        for t in range(self.depth - 1):
            # This is a dummy time step and the next one is dummy too
            if self._is_dummy_step(t) and self._is_dummy_step(t + 1):
                # We cannot use the next time step unless this one is used too
                mdl.add_constraint(z[t] >= z[t + 1], ctname=f"dummy_precedence_{t}")
        # Symmetry breaking on the line -- only works on line topology!
        if line_symm:
            for h in range(1, self.num_vqubits):
                mdl.add_constraint(
                    sum(w[0, p, 0] for p in range(h))
                    + sum([w[0, q, self.num_pqubits - 1] for q in range(h, self.num_vqubits)])
                    >= 1,
                    ctname=f"sym_break_line_{h}",
                )

        # *** Define objevtive function ***
        if objective == "depth":
            objexr = sum(z[t] for t in range(self.depth) if self._is_dummy_step(t))
            for t in range(self.depth - 1):
                for q in range(self.num_vqubits):
                    for (i, j) in self._arcs:
                        objexr += 0.01 * x[t, q, i, j]
            mdl.minimize(objexr)
        elif objective == "error_rate":
            self._set_error_rate_obj(mdl)
        elif objective == "balanced":
            self._set_balanced_obj(mdl)
        else:
            raise TranspilerError(f"Unknown objective type: {objective}")

        self.problem = mdl
        logger.info("BIP problem stats: %s", self.problem.statistics)

    @staticmethod
    def _set_error_rate_obj(model):
        """Set the minimum error rate objective function."""
        raise NotImplementedError("objective: 'error_rate' is not implemented")

    @staticmethod
    def _set_balanced_obj(model):
        """Set the minimum balanced (weighted sum of error_rate and depth) objective function."""
        raise NotImplementedError("objective: 'balanced' is not implemented")

    def solve_cpx_problem(self, time_limit: float = 60, threads: int = None) -> str:
        """Solve the BIP problem using CPLEX.

        Args:
            time_limit:
                Time limit (seconds) given to CPLEX.

            threads:
                Number of threads to be allowed for CPLEX to use.

        Returns:
            Status string that CPLEX returned after solving the BIP problem.

        Raises:
            MissingOptionalLibraryError: If CPLEX is not installed
        """
        if not HAS_CPLEX:
            raise MissingOptionalLibraryError(
                libname="CPLEX",
                name="CplexOptimizer",
                pip_install="pip install cplex",
            )
        self.problem.set_time_limit(time_limit)
        if threads is not None:
            self.problem.context.cplex_parameters.threads = threads
        self.problem.context.cplex_parameters.randomseed = 777

        self.solution = self.problem.solve()

        status = self.problem.solve_details.status
        logger.info("BIP solution status: %s", status)
        return status

    def get_layout(self, t: int) -> Layout:
        """Get layout at time-step t.

        Args:
            t: Time-step

        Returns:
            Layout
        """
        dic = {}
        for q in range(self.num_vqubits):
            for i in range(self.num_pqubits):
                if self.solution.get_value(f"w_{t}_{q}_{i}") > 0.5:
                    dic[self._index_to_virtual[q]] = i
        layout = Layout(dic)
        for reg in self._dag.qregs.values():
            layout.add_register(reg)
        return layout

    def get_swaps(self, t: int) -> list:
        """Get swaps (pairs of physical qubits) inserted at time-step ``t``.

        Args:
            t: Time-step (<= depth - 1)

        Returns:
            List of swaps (pairs of physical qubits (integers))
        """
        swaps = []
        for (i, j) in self._arcs:
            if i >= j:
                continue
            for q in range(self.num_vqubits):
                if self.solution.get_value(f"x_{t}_{q}_{i}_{j}") > 0.5:
                    swaps.append((i, j))
        return swaps
