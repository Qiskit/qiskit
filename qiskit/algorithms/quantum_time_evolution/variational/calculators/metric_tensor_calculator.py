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
from typing import Union, Optional, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression
from qiskit.opflow import QFI, CircuitQFI, CircuitStateFn, OperatorBase, Z

# TODO basis to be passed, real measure Z, imaginary -iY observable
from qiskit.opflow.gradients.circuit_qfis import LinCombFull


def calculate(
    ansatz: QuantumCircuit,
    parameters: Optional[Union[ParameterVector, ParameterExpression, List[ParameterExpression]]],
    qfi_method: Union[str, CircuitQFI] = "lin_comb_full",
    basis: OperatorBase = Z,
    phase_fix: bool = True
):
    operator = CircuitStateFn(ansatz)

    if qfi_method == "lin_comb_full":
        return LinCombFull().convert(operator, parameters, aux_meas_op=basis, phase_fix=phase_fix)

    return QFI(qfi_method).convert(operator, parameters)
