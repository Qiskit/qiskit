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
from qiskit.algorithms.quantum_time_evolution.variational.error_calculators.gradient_errors \
    .imaginary_error_calculator import (
    ImaginaryErrorCalculator,
)
from qiskit.algorithms.quantum_time_evolution.variational.principles.imaginary \
    .imaginary_variational_principle import (
    ImaginaryVariationalPrinciple,
)
from qiskit.algorithms.quantum_time_evolution.variational.solvers.ode.var_qte_ode_solver import (
    VarQteOdeSolver,
)
from qiskit.opflow import (
    OperatorBase,
    Gradient,
    StateFn,
    CircuitStateFn,
    ComposedOp,
    PauliExpectation,
)
from qiskit.algorithms.quantum_time_evolution.variational.var_qte import VarQte
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance


class VarQite(VarQte, EvolutionBase):
    def __init__(
            self,
            variational_principle: ImaginaryVariationalPrinciple,
            regularization: Optional[str] = None,
            backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
            error_based_ode: Optional[bool] = False,
            optimizer: str = "COBYLA",
            epsilon: Optional[float] = 10e-6,
    ):
        r"""
        Args:
            variational_principle: Variational Principle to be used.
            regularization: Use the following regularization with a least square method to solve the
                underlying system of linear equations
                Can be either None or ``'ridge'`` or ``'lasso'`` or ``'perturb_diag'``
                ``'ridge'`` and ``'lasso'`` use an automatic optimal parameter search
                If regularization is None but the metric is ill-conditioned or singular then
                a least square solver is used without regularization
            backend: Backend used to evaluate the quantum circuit outputs
            error_based_ode: If False use the provided variational principle to get the parameter
                                updates.
                             If True use the argument that minimizes the error error_bounds.
            optimizer: Optimizer used in case error_based_ode is true.
            epsilon: # TODO, not sure where this will be used.
        """
        super().__init__(
            variational_principle,
            regularization,
            backend,
            error_based_ode,
            optimizer,
            epsilon,
        )

    def evolve(
            self,
            hamiltonian: OperatorBase,
            time: float,
            initial_state: OperatorBase = None,
            observable: OperatorBase = None,
            t_param=None,
            hamiltonian_value_dict=None,
    ) -> StateFn:
        """
        Apply Variational Quantum Imaginary Time Evolution (VarQITE) w.r.t. the given
        operator
        Args:
            hamiltonian:
                ⟨ψ(ω)|H|ψ(ω)〉
                Operator used vor Variational Quantum Imaginary Time Evolution (VarQITE)
                The coefficient of the operator (operator.coeff) determines the evolution
                time.
                The operator may be given either as a composed op consisting of a Hermitian
                observable and a CircuitStateFn or a ListOp of a CircuitStateFn with a
                ComboFn.
                The latter case enables the evaluation of a Quantum Natural Gradient.
            time: Total time of evolution.
            initial_state: Quantum state to be evolved.
            observable: Observable to be evolved. Not supported by VarQite.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
            hamiltonian_value_dict: Dictionary that maps all parameters in a Hamiltonian to
                                    certain values, including the t_param.
        Returns:
            StateFn (parameters are bound) which represents an approximation to the
            respective
            time evolution.
        """
        if observable is not None:
            raise TypeError(
                "Observable argument provided. Observable evolution not supported by VarQite."
            )

        init_state_parameters = list(initial_state.parameters)
        init_state_param_dict, init_state_parameter_values = self._create_init_state_param_dict(
            hamiltonian_value_dict, init_state_parameters
        )

        # TODO bind Hamiltonian?

        self._variational_principle._lazy_init(
            hamiltonian, initial_state, init_state_param_dict, self._regularization
        )
        self.bind_initial_state(
            StateFn(initial_state), init_state_param_dict
        )  # in this case this is ansatz
        self._operator = self._variational_principle._operator

        if not isinstance(self._operator[-1], CircuitStateFn):
            raise TypeError("Please provide the respective Ansatz as a CircuitStateFn.")
        elif not isinstance(self._operator, ComposedOp) and not all(
                isinstance(op, CircuitStateFn) for op in self._operator.oplist
        ):
            raise TypeError(
                "Please provide the operator either as ComposedOp or as ListOp of a "
                "CircuitStateFn potentially with a combo function."
            )

        # Convert the operator that holds the Hamiltonian and ansatz into a NaturalGradient operator
        self._operator_eval = PauliExpectation().convert(self._operator)

        self._init_grad_objects()
        error_calculator = ImaginaryErrorCalculator(
            self._h_squared,
            self._operator,
            self._h_squared_circ_sampler,
            self._operator_circ_sampler,
            init_state_param_dict,
            self._backend,
        )

        ode_function_generator = self._create_ode_function_generator(error_calculator,
                                                                     init_state_param_dict, t_param)

        ode_solver = VarQteOdeSolver(init_state_parameter_values, ode_function_generator)
        # Run ODE Solver
        parameter_values = ode_solver._run(time)
        # return evolved
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
            observable: OperatorBase = None,
            t_param=None,
            hamiltonian_value_dict=None,
            gradient_params=None,
    ) -> EvolutionGradientResult:
        raise NotImplementedError()
