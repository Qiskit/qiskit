# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Optimize consecutive gates of identical gate type by combining them into a single gate."""

from typing import Optional

from qiskit.circuit.library.standard_gates.p import PhaseGate
from qiskit.circuit.library.standard_gates.rx import RXGate
from qiskit.circuit.library.standard_gates.rxx import RXXGate
from qiskit.circuit.library.standard_gates.ry import RYGate
from qiskit.circuit.library.standard_gates.ryy import RYYGate
from qiskit.circuit.library.standard_gates.rz import RZGate
from qiskit.circuit.library.standard_gates.rzx import RZXGate
from qiskit.circuit.library.standard_gates.rzz import RZZGate
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

_SUPPORTED_GATES = {
    "p": PhaseGate,
    "rx": RXGate,
    "ry": RYGate,
    "rz": RZGate,
    "rxx": RXXGate,
    "ryy": RYYGate,
    "rzz": RZZGate,
    "rzx": RZXGate,
}


class OptimizeConsecutive(TransformationPass):
    """Optimize chains of rotation gates of the same type by combining them into a single gate."""

    def __init__(self, basis: Optional[list[str]] = None, eps: float = 1e-15):
        """Optimize1qGates initializer.

        Args:
            basis (list[str]): Basis gates to consider, e.g. `['rz', 'rx']`. For the effects
                of this pass, the basis is the set intersection between the `basis` parameter and
                the set `{'rx','rz','rzz'}`.
            eps (float): EPS to check against

        Raises:
            ValueError: if `basis` contain any basis gate that is not supported.
        """
        super().__init__()
        if basis:
            for gate_type in basis:
                if gate_type not in _SUPPORTED_GATES:
                    raise ValueError(f"Gate type ({gate_type}) is not supported")
            self.basis = set(basis)
        else:
            self.basis = set(_SUPPORTED_GATES)
        self.eps = eps

    def run(self, dag):
        """Run the OptimizeConsecutive pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """

        for gate_type in self.basis:
            runs = dag.collect_runs([gate_type])
            for run in runs:
                if len(run) == 1:
                    continue
                name = run[0].op.name
                if len(run[0].params) != 1:
                    raise TranspilerError("internal error")

                parameter = run[0].op.params[0]
                for current_node in run[1:]:
                    if len(current_node.params) != 1:
                        raise TranspilerError("internal error")
                    parameter += current_node.op.params[0]
                if isinstance(parameter, ParameterExpression) and parameter.is_real():
                    parameter = float(parameter)

                if not isinstance(parameter, ParameterExpression) and abs(parameter) <= self.eps:
                    for current_node in run:
                        dag.remove_op_node(current_node)
                else:
                    new_op = _SUPPORTED_GATES[name](parameter)
                    dag.substitute_node(run[0], new_op, inplace=True)
                    for current_node in run[1:]:
                        dag.remove_op_node(current_node)
        return dag
