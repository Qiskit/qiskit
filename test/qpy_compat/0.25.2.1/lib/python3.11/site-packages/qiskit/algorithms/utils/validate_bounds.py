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

"""Validate parameter bounds."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit


def validate_bounds(circuit: QuantumCircuit) -> list[tuple[float | None, float | None]]:
    """
    Validate the bounds provided by a quantum circuit against its number of parameters.
    If no bounds are obtained, return ``None`` for all lower and upper bounds.

    Args:
        circuit: A parameterized quantum circuit.

    Returns:
        A list of tuples (lower_bound, upper_bound)).

    Raises:
        ValueError: If the number of bounds does not the match the number of circuit parameters.
    """
    if hasattr(circuit, "parameter_bounds") and circuit.parameter_bounds is not None:
        bounds = circuit.parameter_bounds
        if len(bounds) != circuit.num_parameters:
            raise ValueError(
                f"The number of bounds ({len(bounds)}) does not match the number of "
                f"parameters in the circuit ({circuit.num_parameters})."
            )
    else:
        bounds = [(None, None)] * circuit.num_parameters

    return bounds
