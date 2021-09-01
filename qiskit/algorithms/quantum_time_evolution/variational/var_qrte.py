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
from typing import Optional, Union, List

import numpy as np
from scipy.integrate import ode, OdeSolver

from qiskit.algorithms.quantum_time_evolution.evolution_base import EvolutionBase
from qiskit.algorithms.quantum_time_evolution.results.evolution_gradient_result import (
    EvolutionGradientResult,
)
from qiskit.algorithms.quantum_time_evolution.results.evolution_result import EvolutionResult
from qiskit.algorithms.quantum_time_evolution.variational.principles.real\
    .real_variational_principle import (
    RealVariationalPrinciple,
)
from qiskit.algorithms.quantum_time_evolution.variational.var_qte import VarQte
from qiskit.opflow import OperatorBase, Gradient, StateFn, ComposedOp, CircuitStateFn, \
    PauliExpectation
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance


class VarQrte(VarQte, EvolutionBase):
    def __init__(
            self,
            variational_principle: RealVariationalPrinciple,
            regularization: Optional[str] = None,
            num_time_steps: int = 10,
            init_parameter_values: Optional[Union[List, np.ndarray]] = None,
            ode_solver: Optional[Union[OdeSolver, ode]] = None,
            backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
            snapshot_dir: Optional[str] = None,
            error_based_ode: bool = False,
    ):
        super().__init__(
            variational_principle,
            regularization,
            num_time_steps,
            init_parameter_values,
            ode_solver,
            backend,
            snapshot_dir,
            error_based_ode,
        )

    def evolve(
            self,
            hamiltonian: OperatorBase,
            time: float,
            initial_state: StateFn = None,
            observable: OperatorBase = None,
            t_param=None,
            hamiltonian_value_dict=None,
    ) -> EvolutionResult:

        """
        Apply Variational Quantum Time Evolution (VarQTE) w.r.t. the given operator
        Args:
            operator:
                Operator used vor Variational Quantum Real Time Evolution (VarQRTE)
                The coefficient of the operator determines the evolution time.
                If the coefficient is real this method implements VarQRTE.
                The operator may for now ONLY be given as a composed op consisting of a
                Hermitian observable and a CircuitStateFn.
        Returns:
            StateFn (parameters are bound) which represents an approximation to the respective
            time evolution.
        """
        self._parameters = list(hamiltonian_value_dict.keys())
        self._variational_principle._lazy_init(hamiltonian, initial_state, self._parameters)
        self._state = initial_state
        self._operator = self._variational_principle._operator
        if not isinstance(self._operator, ComposedOp) or len(self._operator.oplist) != 2:
            raise TypeError('Please provide the operator as a ComposedOp consisting of the '
                            'observable and the state (as CircuitStateFn).')
        if not isinstance(self._operator[-1], CircuitStateFn):
            raise TypeError('Please provide the state as a CircuitStateFn.')

        # For VarQRTE we need to add a -i factor to the operator coefficient.
        # TODO should this also happen in VarPrinciple?
        self._operator = 1j * self._operator / self._operator.coeff
        self._operator_eval = PauliExpectation().convert(self._operator / self._operator.coeff)

        self._init_grad_objects()
        # Step size
        # dt = np.abs(self._operator.coeff)
        # Run ODE Solver
        parameter_values = self._ode_solver._run(time, self._init_parameter_values)
        # return evolved
        return self._state.assign_parameters(dict(zip(self._parameters,
                                                      parameter_values)))

    @abstractmethod
    def gradient(
            self,
            hamiltonian: OperatorBase,
            time: float,
            initial_state: StateFn,
            gradient_object: Gradient,
            observable: OperatorBase = None,
            t_param=None,
            hamiltonian_value_dict=None,
            gradient_params=None,
    ) -> EvolutionGradientResult:
        raise NotImplementedError()
