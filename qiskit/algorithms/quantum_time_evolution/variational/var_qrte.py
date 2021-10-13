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
from typing import Optional, Union

from qiskit.algorithms.quantum_time_evolution.evolution_base import EvolutionBase
from qiskit.algorithms.quantum_time_evolution.results.evolution_gradient_result import (
    EvolutionGradientResult,
)
from qiskit.algorithms.quantum_time_evolution.variational.error_calculators.gradient_errors.real_error_calculator import (
    RealErrorCalculator,
)
from qiskit.algorithms.quantum_time_evolution.variational.principles.real.real_variational_principle import (
    RealVariationalPrinciple,
)
from qiskit.algorithms.quantum_time_evolution.variational.solvers.ode.ode_function_generator import (
    OdeFunctionGenerator,
)
from qiskit.algorithms.quantum_time_evolution.variational.solvers.ode.var_qte_ode_solver import (
    VarQteOdeSolver,
)
from qiskit.algorithms.quantum_time_evolution.variational.var_qte import VarQte
from qiskit.opflow import (
    OperatorBase,
    Gradient,
    StateFn,
    ComposedOp,
    CircuitStateFn,
    PauliExpectation,
)
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance


class VarQrte(VarQte, EvolutionBase):
    def __init__(
        self,
        variational_principle: RealVariationalPrinciple,
        regularization: Optional[str] = None,
        backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
        error_based_ode: Optional[bool] = False,
        epsilon: Optional[float] = 10e-6,
    ):
        super().__init__(
            variational_principle,
            regularization,
            backend,
            error_based_ode,
            epsilon,
        )

    def evolve(
        self,
        hamiltonian: OperatorBase,
        time: float,
        initial_state: Optional[StateFn] = None,
        observable: Optional[OperatorBase] = None,
        t_param=None,
        hamiltonian_value_dict=None,
    ) -> StateFn:

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
        init_state_parameters = list(initial_state.parameters)
        init_state_param_dict, init_state_parameter_values = self._create_init_state_param_dict(
            hamiltonian_value_dict, init_state_parameters
        )

        self._variational_principle._lazy_init(
            hamiltonian, initial_state, init_state_param_dict, self._regularization
        )
        self.bind_initial_state(
            StateFn(initial_state), init_state_param_dict
        )  # in this case this is ansatz
        self._operator = self._variational_principle._operator
        if not isinstance(self._operator, ComposedOp) or len(self._operator.oplist) != 2:
            raise TypeError(
                "Please provide the operator as a ComposedOp consisting of the "
                "observable and the state (as CircuitStateFn)."
            )
        if not isinstance(self._operator[-1], CircuitStateFn):
            raise TypeError("Please provide the state as a CircuitStateFn.")

        # For VarQRTE we need to add a -i factor to the operator coefficient.
        # TODO should this also happen in VarPrinciple?
        self._operator = 1j * self._operator / self._operator.coeff
        self._operator_eval = PauliExpectation().convert(self._operator / self._operator.coeff)

        self._init_grad_objects()
        error_calculator = RealErrorCalculator(
            self._h_squared,
            self._operator,
            self._h_squared_circ_sampler,
            self._operator_circ_sampler,
            init_state_param_dict,
        )

        ode_function_generator = OdeFunctionGenerator(
            error_calculator,
            init_state_param_dict,
            self._variational_principle,
            self._grad_circ_sampler,
            self._metric_circ_sampler,
            self._nat_grad_circ_sampler,
            self._regularization,
            self._backend,
            self._error_based_ode,
            t_param,
        )

        ode_solver = VarQteOdeSolver(init_state_parameter_values, ode_function_generator)
        # Run ODE Solver
        parameter_values = ode_solver._run(time)

        # initial state here is not with self because we need a parametrized state (input to this
        # method)
        param_dict_from_ode = dict(zip(init_state_parameters, parameter_values))
        return initial_state.assign_parameters(param_dict_from_ode)

    def gradient(
        self,
        hamiltonian: OperatorBase,
        time: float,
        initial_state: StateFn,
        gradient_object: Gradient,
        observable: Optional[OperatorBase] = None,
        t_param=None,
        hamiltonian_value_dict=None,
        gradient_params=None,
    ) -> EvolutionGradientResult:
        raise NotImplementedError()
