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

"""Helper class for assigning values to parameters."""
import numpy as np

from qiskit.circuit import ParameterExpression, Parameter
from qiskit.circuit.parametertable import ParameterView
from qiskit.quantum_info import SparsePauliOp


def get_parameters(array: np.array) -> ParameterView:
    """Retrieves parameters from an np.array as a ``ParameterView``."""
    ret = set()
    for a in array:
        if isinstance(a, ParameterExpression):
            ret |= a.parameters
    return ParameterView(ret)


def assign_parameters(array: np.array, parameter_values: np.array) -> np.array:
    """Binds ``ParameterExpression``s in an np.array to provided values."""
    if isinstance(parameter_values, list):
        parameter_values = dict(zip(get_parameters(array), parameter_values))
    for i, a in enumerate(array):
        if isinstance(a, ParameterExpression):
            for key in a.parameters & parameter_values.keys():
                a = a.assign(key, parameter_values[key])
            if not a.parameters:
                a = complex(a)
            array[i] = a
    return array
