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
from typing import Union

from qiskit.opflow import StateFn, Gradient, CircuitGradient


def build(observable, ansatz, parameters, grad_method: Union[str, CircuitGradient] = "lin_comb"):
    operator = ~StateFn(observable) @ StateFn(ansatz)
    return Gradient(grad_method).convert(operator, parameters)
