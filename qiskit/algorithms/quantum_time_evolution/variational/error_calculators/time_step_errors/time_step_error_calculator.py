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
from typing import Iterable

import math
import numpy as np


def _get_error_term(
    d_t, eps_t, grad_err, energy: float, h_squared: float, h_norm: float, stddev: float
):
    """
    Compute the error term for a given time step and a point in the simulation time
    Args:
        d_t: time step
        j: jth step in VarQITE
    Returns: eps_j(delta_t)
    """
    if eps_t < 0:
        eps_t = 0
        print("Warn eps_t neg. clipped to 0")
    if eps_t == 0 and grad_err == 0:
        eps_t_next = 0
    else:
        energy_factor = _get_energy_factor(eps_t, energy, stddev, h_norm)
        y = _get_max_bures(eps_t, energy, energy_factor, h_squared, d_t)
        eps_t_next = y + d_t * grad_err + d_t * energy_factor

    # TODO Write terms to csv file
    return eps_t_next


# TODO extract grid search to reduce code duplication
def _get_max_bures(
    eps: float, e: float, e_factor: float, h_squared: float, delta_t: float
) -> float:
    """
    Compute  max_alpha B(I + delta_t(E_t-H)|psi_t>, I + delta_t(E_t-H)|psi*_t>(alpha))
    Args:
        eps: Error from the previous time step
        e: Energy <psi_t|H|psi_t>
        e_factor: Upper bound to |E - E*|
        h_squared: <psi_t|H^2|psi_t>
        delta_t: time step
    Returns: max_alpha B(I + delta_t(E_t-H)|psi_t>, I + delta_t(E_t-H)|psi*_t>(alpha))
    """

    c_alpha = lambda a: np.sqrt(
        (1 - np.abs(a)) ** 2 + 2 * a * (1 - np.abs(a)) * e + a ** 2 * h_squared
    )

    def overlap(alpha: Iterable[float]) -> float:
        """
        Compute generalized Bures metric B(I + delta_t(E_t-H)|psi_t>, I + delta_t(
        E_t-H)|psi*_t>(alpha))
        Args:
            alpha: optimization parameter alpha
        Returns: B(I + delta_t(E_t-H)|psi_t>, I + delta_t(E_t-H)|psi*_t>(alpha))
        """

        alpha = alpha[0]

        x = np.abs(
            (
                (1 + 2 * delta_t * e) * (1 - np.abs(alpha) + alpha * e)
                - 2 * delta_t * ((1 - np.abs(alpha)) * e + alpha * h_squared)
            )
            / c_alpha(alpha)
        )

        return x

    def constraint(alpha: Iterable[float]) -> float:
        """
        This constraint ensures that the optimization chooses a |psi*_t> which is in
        accordance with the prior state error
        Args:
            alpha: optimization value
        Returns: |<|psi_t|psi*_t>| - (1 + eps^2/2)
        """
        alpha = alpha[0]
        return np.abs((1 - np.abs(alpha) + alpha * e) / c_alpha(alpha)) - 1 + eps ** 2 / 2

    x = None
    # TODO Use again finer grid of 10**6
    # Grid search over alphas for the optimization
    a_grid = np.append(np.linspace(-1, 1, 10 ** 6), 0)
    for a in a_grid:
        returned_bures = overlap([a])
        if (
            not math.isnan(returned_bures)
            and constraint([a]) >= 0
            and (x is None or returned_bures < x)
        ):
            x = returned_bures

    max_bures = np.sqrt(2 + 2 * delta_t * e_factor - 2 * x)
    return max_bures


def _get_energy_factor(eps_t: float, energy: float, stddev: float, h_norm: float):
    """
    Get upper bound to |E-E*|
    Args:
        eps_t:
        energy:
        stddev:
    Returns: upper bound to |E-E*|
    """

    def optimize_energy_factor(alpha):
        return np.abs(alpha[0] * energy - np.sqrt(1 - (1 - alpha[0]) ** 2) * stddev)

    x = None
    # Grid search over alphas for the optimization
    if eps_t ** 2 / 2 > 1:
        grid = np.linspace(0, 1, int(1e7))
    else:
        grid = np.linspace(0, eps_t ** 2 / 2, int(1e7))
    for a in grid:
        returned_x = optimize_energy_factor([a])
        if math.isnan(returned_x):
            print("eps_t", eps_t)
            raise Warning("optimization fun is nan")
        else:
            # Check if the current bures metric is bigger than the max.
            if x is None or returned_x > x:
                x = returned_x

    return eps_t ** 2 * h_norm + 2 * x
