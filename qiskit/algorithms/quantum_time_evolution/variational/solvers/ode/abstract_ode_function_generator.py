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
from abc import abstractmethod, ABC
from typing import Iterable, Union, Dict, Optional

from qiskit.algorithms.quantum_time_evolution.variational.principles.variational_principle import (
    VariationalPrinciple,
)
from qiskit.algorithms.quantum_time_evolution.variational.solvers.var_qte_linear_solver import (
    VarQteLinearSolver,
)
from qiskit.circuit import Parameter
from qiskit.opflow import CircuitSampler
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance


class AbstractOdeFunctionGenerator(ABC):
    def __init__(
        self,
        param_dict: Dict[Parameter, Union[float, complex]],
        variational_principle: VariationalPrinciple,
        grad_circ_sampler: CircuitSampler,
        metric_circ_sampler: CircuitSampler,
        nat_grad_circ_sampler: CircuitSampler,
        regularization: Optional[str] = None,
        backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
        t_param: Parameter = None,
    ):
        self._param_dict = param_dict
        self._variational_principle = variational_principle
        self._grad_circ_sampler = grad_circ_sampler
        self._metric_circ_sampler = metric_circ_sampler
        self._nat_grad_circ_sampler = nat_grad_circ_sampler
        self._regularization = regularization
        self._backend = backend
        self._linear_solver = VarQteLinearSolver(
            self._grad_circ_sampler,
            self._metric_circ_sampler,
            self._nat_grad_circ_sampler,
            self._regularization,
            self._backend,
        )
        self._t_param = t_param

    @abstractmethod
    def var_qte_ode_function(self, t: float, parameters_values: Iterable) -> Iterable:
        current_param_dict = dict(zip(self._param_dict.keys(), parameters_values))
        nat_grad_res = self._linear_solver._solve_sle(
            self._variational_principle, current_param_dict, self._t_param, t
        )

        return nat_grad_res
