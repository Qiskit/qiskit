# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Bind values to a parametrized circuit, accepting binds for non-existing parameters in the circuit."""

from __future__ import annotations
from collections.abc import Iterable

from qiskit.circuit import QuantumCircuit, Parameter

# pylint: disable=inconsistent-return-statements
def bind(
    circuits: QuantumCircuit | Iterable[QuantumCircuit],
    parameter_binds: dict[Parameter, float],
    inplace: bool = False,
) -> QuantumCircuit | Iterable[QuantumCircuit] | None:
    """Bind parameters in a circuit (or list of circuits).

    This method also allows passing parameter binds to parameters that are not in the circuit,
    and thereby differs to :meth:`.QuantumCircuit.bind_parameters`.

    Args:
        circuits: Input circuit(s).
        parameter_binds: A dictionary with ``{Parameter: float}`` pairs determining the values to
            which the free parameters in the circuit(s) are bound.
        inplace: If ``True``, bind the values in place, otherwise return circuit copies.

    Returns:
        The bound circuits, if ``inplace=False``, otherwise None.

    """
    if not isinstance(circuits, Iterable):
        circuits = [circuits]
        return_list = False
    else:
        return_list = True

    bound = []
    for circuit in circuits:
        existing_parameter_binds = {p: parameter_binds[p] for p in circuit.parameters}
        bound.append(circuit.assign_parameters(existing_parameter_binds, inplace=inplace))

    if not inplace:
        return bound if return_list else bound[0]
