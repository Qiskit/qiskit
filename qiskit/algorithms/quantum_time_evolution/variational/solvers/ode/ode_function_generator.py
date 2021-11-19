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
from typing import Union, Dict, Optional, Iterable

from qiskit.algorithms.quantum_time_evolution.variational.principles.variational_principle import (
    VariationalPrinciple,
)
from qiskit.algorithms.quantum_time_evolution.variational.solvers.ode.abstract_ode_function_generator import (
    AbstractOdeFunctionGenerator,
)
from qiskit.circuit import Parameter
from qiskit.opflow import CircuitSampler
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance


class OdeFunctionGenerator(AbstractOdeFunctionGenerator):
    def __init__(
        self,
        param_dict: Dict[Parameter, Union[float, complex]],
        variational_principle: VariationalPrinciple,
        grad_circ_sampler: CircuitSampler,
        metric_circ_sampler: CircuitSampler,
        energy_sampler: CircuitSampler,
        regularization: Optional[str] = None,
        backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
        t_param: Parameter = None,
    ):
        super().__init__(
            param_dict,
            variational_principle,
            grad_circ_sampler,
            metric_circ_sampler,
            energy_sampler,
            regularization,
            backend,
            t_param,
        )

    def var_qte_ode_function(self, t: float, parameters_values: Iterable) -> Iterable:
        return super().var_qte_ode_function(t, parameters_values)
