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
from abc import abstractmethod
from typing import Union

from qiskit.algorithms.quantum_time_evolution.variational.principles.variational_principle import (
    VariationalPrinciple,
)
from qiskit.opflow import CircuitQFI, CircuitGradient


class RealVariationalPrinciple(VariationalPrinciple):
    def __init__(self, observable, ansatz, parameters,
                 qfi_method: Union[str, CircuitQFI] = 'lin_comb_full',
                 grad_method: Union[str, CircuitGradient] = 'lin_comb',
                 is_error_supported: bool = False):
        super().__init__(observable, ansatz, parameters, qfi_method, grad_method,
                         is_error_supported)

    @staticmethod
    @abstractmethod
    def _calc_metric_tensor(raw_metric_tensor):
        pass

    @staticmethod
    @abstractmethod
    def _calc_evolution_grad(raw_evolution_grad):
        pass
