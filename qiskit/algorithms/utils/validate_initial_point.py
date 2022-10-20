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

"""Validate an initial point."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.utils import algorithm_globals


def validate_initial_point(
    point: Sequence[float] | None, circuit: QuantumCircuit
) -> Sequence[float]:
    r"""
    Validate a choice of initial point against a choice of circuit. If no point is provided, a
    random point will be generated within certain parameter bounds. It will first look to the
    circuit for these bounds. If the circuit does not specify bounds, bounds of :math:`-2\pi`,
    :math:`2\pi` will be used.

    Args:
        point: An initial point.
        circuit: A parameterized quantum circuit.

    Returns:
        A validated initial point.

    Raises:
        ValueError: If the dimension of the initial point does not match the number of circuit
        parameters.
    """
    expected_size = circuit.num_parameters

    if point is None:
        # get bounds if circuit has them set, otherwise use [-2pi, 2pi] for each parameter
        bounds = getattr(circuit, "parameter_bounds", None)
        if bounds is None:
            bounds = [(-2 * np.pi, 2 * np.pi)] * expected_size

        # replace all Nones by [-2pi, 2pi]
        lower_bounds = []
        upper_bounds = []
        for lower, upper in bounds:
            lower_bounds.append(lower if lower is not None else -2 * np.pi)
            upper_bounds.append(upper if upper is not None else 2 * np.pi)

        # sample from within bounds
        point = algorithm_globals.random.uniform(lower_bounds, upper_bounds)

    elif len(point) != expected_size:
        raise ValueError(
            f"The dimension of the initial point ({len(point)}) does not match the "
            f"number of parameters in the circuit ({expected_size})."
        )

    return point
