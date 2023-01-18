# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helper class for assigning values to parameters."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from qiskit.circuit import ParameterExpression
from qiskit.circuit.parametertable import ParameterView


def _get_parameters(array: np.ndarray) -> ParameterView:
    """Retrieves parameters from a numpy array as a ``ParameterView``."""
    ret = set()
    for a in array:
        if isinstance(a, ParameterExpression):
            ret |= a.parameters
    return ParameterView(ret)


def _assign_parameters(array: np.ndarray, parameter_values: Sequence[float]) -> np.ndarray:
    """Binds ``ParameterExpression``s in a numpy array to provided values."""
    parameter_dict = dict(zip(_get_parameters(array), parameter_values))
    for i, a in enumerate(array):
        if isinstance(a, ParameterExpression):
            for key in a.parameters & parameter_dict.keys():
                a = a.assign(key, parameter_dict[key])
            if not a.parameters:
                a = complex(a)
            array[i] = a
    return array
