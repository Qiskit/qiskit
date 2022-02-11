# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Variational Quantum Real Time Evolution algorithm."""

from typing import Optional, Union, Dict, List

from scipy.integrate import OdeSolver, RK45

from qiskit.algorithms.optimizers import Optimizer, COBYLA
from qiskit.algorithms.time_evolution.evolution_result import EvolutionResult
from qiskit.algorithms.time_evolution.real.qrte import Qrte
from qiskit.algorithms.time_evolution.variational.error_calculators.gradient_errors\
    .real_error_calculator import (
    RealErrorCalculator,
)
from qiskit.algorithms.time_evolution.variational.variational_principles.real\
    .real_variational_principle import (
    RealVariationalPrinciple,
)
from qiskit.algorithms.time_evolution.variational.solvers.ode.abstract_ode_function_generator \
    import (
    AbstractOdeFunctionGenerator,
)
from qiskit.algorithms.time_evolution.variational.algorithms.var_qte import VarQte
from qiskit.circuit import Parameter
from qiskit.opflow import (
    OperatorBase,
    Gradient,
    StateFn,
)
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance


class VarQrte(VarQte, Qrte):
    """Variational Quantum Real Time Evolution algorithm."""

    def __init__(
        self,
        variational_principle: RealVariationalPrinciple,
        regularization: Optional[str] = None,
        # TODO: Should we keep this more general? And pass here a natural gradient object?
        backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
        error_based_ode: Optional[bool] = False,
        ode_solver_callable: OdeSolver = RK45,
        optimizer: Optimizer = COBYLA,
        optimizer_tolerance: float = 1e-6,
        allowed_imaginary_part: float = 1e-7,
        allowed_num_instability_error: float = 1e-7,
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
                             Deprecated if error is not being computed.
            ode_solver_callable: ODE solver callable that follows a SciPy OdeSolver interface.
            optimizer: Optimizer used in case error_based_ode is true.
            optimizer_tolerance: Numerical tolerance of an optimizer used for convergence to a minimum.
            allowed_imaginary_part: Allowed value of an imaginary part that can be neglected if no
                                    imaginary part is expected.
            allowed_num_instability_error: The amount of negative value that is allowed to be
                                           rounded up to 0 for quantities that are expected to be
                                           non-negative.
        """
        super().__init__(
            variational_principle,
            regularization,
            backend,
            error_based_ode,
            ode_solver_callable,
            optimizer,
            optimizer_tolerance,
            allowed_imaginary_part,
            allowed_num_instability_error,
        )

    def evolve(
        self,
        hamiltonian: OperatorBase,
        time: float,
        initial_state: Optional[StateFn] = None,
        observable: Optional[OperatorBase] = None,
        t_param: Optional[Parameter] = None,
        hamiltonian_value_dict: Optional[Dict[Parameter, Union[float, complex]]] = None,
    ) -> EvolutionResult:

        """
        Apply Variational Quantum Real Time Evolution (VarQRTE) w.r.t. the given operator.
        Args:
            hamiltonian:
                ⟨ψ(ω)|H(t, theta)|ψ(ω)〉
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
                                    certain values, including the t_param. If no state parameters
                                    are provided, they are generated randomly.
        Returns:
            StateFn (parameters are bound) which represents an approximation to the respective
            time evolution.
        """
        init_state_param_dict = self._create_init_state_param_dict(
            hamiltonian_value_dict, list(initial_state.parameters)
        )

        evolved_object = super()._evolve_helper(
            self._create_real_ode_function_generator,
            init_state_param_dict,
            hamiltonian,
            time,
            initial_state,
            observable,
        )

        return EvolutionResult(evolved_object)

    def _create_real_ode_function_generator(
        self,
        init_state_param_dict: Dict[Parameter, Union[float, complex]],
        t_param: Optional[Parameter] = None,
    ) -> AbstractOdeFunctionGenerator:
        """
        Creates an ODE function generator for the real time evolution, i.e. with an
        RealErrorCalculator in case of an error-based evolution.
        Args:
            init_state_param_dict: Dictionary mapping parameters to their initial values for a
                                quantum state.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
        Returns:
            Instantiated real ODE function generator.
        """
        if self._error_based_ode:
            error_calculator = RealErrorCalculator(
                self._hamiltonian_squared,
                self._operator,
                self._h_squared_circ_sampler,
                self._operator_circ_sampler,
                self._backend,
                self._allowed_imaginary_part,
                self._allowed_num_instability_error,
            )
            return super()._create_ode_function_generator(
                error_calculator, init_state_param_dict, t_param
            )
        else:
            return super()._create_ode_function_generator(None, init_state_param_dict, t_param)

    def gradient(
        self,
        hamiltonian: OperatorBase,
        time: float,
        initial_state: StateFn,
        gradient_object: Gradient,
        observable: Optional[OperatorBase] = None,
        t_param: Optional[Parameter] = None,
        hamiltonian_value_dict: Optional[Dict[Parameter, Union[float, complex]]] = None,
        gradient_params: Optional[List[Parameter]] = None,
    ):
        """Performs Variational Quantum Real Time Evolution of gradient expressions."""
        pass
