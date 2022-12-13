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

"""Bind a parameters to a circuit, accepting parameters not existing in the circuit."""

# pylint: disable=inconsistent-return-statements
def bind(circuits, parameter_binds, inplace=False):
    if not isinstance(circuits, list):
        existing_parameter_binds = {p: parameter_binds[p] for p in circuits.parameters}
        return circuits.assign_parameters(existing_parameter_binds, inplace=inplace)

    bound = []
    for circuit in circuits:
        existing_parameter_binds = {p: parameter_binds[p] for p in circuit.parameters}
        bound.append(circuit.assign_parameters(existing_parameter_binds, inplace=inplace))

    if not inplace:
        return bound
