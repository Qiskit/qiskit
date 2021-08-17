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

from qiskit.opflow import StateFn, QFI, CircuitQFI


def build(observable, ansatz, parameters, qfi_method: Union[str, CircuitQFI] = "lin_comb_full"):
    operator = ~StateFn(observable) @ StateFn(ansatz)
    # TODO why?
    operator = operator.oplist[-1]
    return QFI(qfi_method).convert(operator, parameters)
