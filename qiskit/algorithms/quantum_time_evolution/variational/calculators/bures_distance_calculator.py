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
from typing import List, Union

import numpy as np
from scipy.optimize import minimize


def _calculate_bures_distance(
    state1: Union[List, np.ndarray],
    state2: Union[List, np.ndarray],
    optimizer: str = "COBYLA",
    tolerance: float = 1e-6,
) -> float:
    """
    Find the Bures metric between two normalized pure states
    Args:
        state1: Target state
        state2: Trained state with potential phase mismatch
    Returns:
        global phase agnostic l2 norm value
    """

    def bures_dist(phi: float):
        return np.linalg.norm(np.subtract(state1, np.exp(1j * phi) * state2), ord=2)

    bures_distance = minimize(fun=bures_dist, x0=np.array([0]), method=optimizer, tol=tolerance)
    return bures_distance.fun
