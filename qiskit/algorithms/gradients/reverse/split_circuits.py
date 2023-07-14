# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Split a circuit into subcircuits, each containing a single parameterized gate."""

from __future__ import annotations

from collections.abc import Iterable
from qiskit.circuit import QuantumCircuit, ParameterExpression, Parameter


def split(
    circuit: QuantumCircuit,
    parameters: Iterable[Parameter] | None = None,
) -> tuple[list[QuantumCircuit], list[list[Parameter]]]:
    """Split the circuit at ParameterExpressions.

    Args:
        circuit: The circuit to split.
        parameters: The parameters at which to split. If None, split at each parameter.

    Returns:
        A list of the split circuits along with a list of which parameters are in the subcircuits.
    """
    circuits = []
    corresponding_parameters = []

    sub = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    for inst in circuit.data:
        # check if new split must be created
        if parameters is None:
            params = [
                param
                for param in inst.operation.params
                if isinstance(param, ParameterExpression) and len(param.parameters) > 0
            ]
        else:
            if inst.operation.definition is not None:
                free_inst_params = inst.operation.definition.parameters
            else:
                free_inst_params = {}

            params = [p for p in parameters if p in free_inst_params]

        new_split = bool(len(params) > 0)

        if new_split:
            sub.append(inst)
            circuits.append(sub)
            corresponding_parameters.append(params)
            sub = QuantumCircuit(*circuit.qregs, *circuit.cregs)
        else:
            sub.append(inst)

    # handle leftover gates
    if len(sub.data) > 0:
        circuits[-1].compose(sub, inplace=True)

    return circuits, corresponding_parameters
