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
from abc import ABC, abstractmethod


class VariationalPrinciple(ABC):
    def __init__(self, raw_metric_tensor, raw_evolution_grad, is_error_supported: bool):
        self._is_error_supported = is_error_supported
        self._metric_tensor = self._calc_metric_tensor(raw_metric_tensor)
        self._evolution_grad = self._calc_evolution_grad(raw_evolution_grad)

    @staticmethod
    @abstractmethod
    def _calc_metric_tensor(raw_metric_tensor):
        pass

    @staticmethod
    @abstractmethod
    def _calc_evolution_grad(raw_evolution_grad):
        pass

    @property
    def metric_tensor(self):
        return self._metric_tensor

    @property
    def evolution_grad(self):
        return self._evolution_grad
