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
from collections import defaultdict
from typing import Union, Optional

from qiskit.algorithms.quantum_time_evolution.builders.implementations.trotterizations.suzuki \
    import (
    Suzuki,
)
from qiskit.algorithms.quantum_time_evolution.builders.implementations.trotterizations \
    .trotter_mode_enum import (
    TrotterModeEnum,
)
from qiskit.algorithms.quantum_time_evolution.builders.implementations.trotterizations \
    .trotterization_factory import (
    TrotterizationFactory,
)
from qiskit.algorithms.quantum_time_evolution.real.qrte import Qrte
from qiskit.algorithms.quantum_time_evolution.results.evolution_gradient_result import (
    EvolutionGradientResult,
)
from qiskit.algorithms.quantum_time_evolution.results.evolution_result import EvolutionResult
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.opflow import OperatorBase, StateFn, Gradient, commutator, SummedOp, PauliSumOp, PauliOp


class TrotterQrte(Qrte):
    def __init__(self, mode: TrotterModeEnum, reps: int = 1):
        self._mode = mode
        self._reps = reps

    def evolve(
            self,
            hamiltonian: OperatorBase,
            time: float,
            initial_state: StateFn = None,
            observable: OperatorBase = None,
            t_param: Parameter = None,
            hamiltonian_value_dict=None,
    ) -> EvolutionResult:

        hamiltonian = self._try_binding_params(hamiltonian, hamiltonian_value_dict)
        self._validate_input(initial_state, observable)

        trotter = TrotterizationFactory.build(self._mode, self._reps)
        trotterized_evolution_op = trotter.build(time * hamiltonian)

        if initial_state is not None:
            return (trotterized_evolution_op @ initial_state).eval()
        if observable is not None:
            return trotterized_evolution_op.adjoint() @ observable @ trotterized_evolution_op

    def _try_binding_params(self, hamiltonian, hamiltonian_value_dict):
        # PauliSumOp does not allow parametrized coefficients
        if isinstance(hamiltonian, SummedOp):
            op_list = []
            for op in hamiltonian.oplist:
                if hamiltonian_value_dict is not None:
                    op_bound = op.bind_parameters(hamiltonian_value_dict)
                else:
                    op_bound = op
                if len(op_bound.parameters) > 0:
                    raise ValueError(
                        f"Did not manage to bind all parameters in the Hamiltonian, "
                        f"these parameters encountered: {op_bound.parameters}.")
                op_list.append(op_bound)
            return SummedOp(op_list)
        #for an observable, we might have an OperatorBase... TODO
        elif isinstance(hamiltonian, PauliOp):
            return hamiltonian.bind_parameters(hamiltonian_value_dict)
        else:
            return hamiltonian

    def _validate_input(self, initial_state, observable):
        if initial_state is None and observable is None:
            raise ValueError(
                "TrotterQrte requires an initial state or an observable to be evolved; None "
                "provided."
            )
        elif initial_state is not None and observable is not None:
            raise ValueError(
                "TrotterQrte requires an initial state or an observable to be evolved; both "
                "provided."
            )

    def gradient(
            self,
            hamiltonian: Union[PauliSumOp, SummedOp],
            time: float,
            initial_state: StateFn,
            gradient_object: Optional[Gradient],
            observable: OperatorBase = None,
            t_param=None,
            hamiltonian_value_dict=None,
            gradient_params=None,
    ) -> EvolutionGradientResult:
        if observable is None:
            raise NotImplementedError(
                "Observable not provided. Probability gradients are not yet supported by "
                "TrotterQrte. "
            )
        if gradient_object is not None:
            raise Warning(
                "TrotterQrte does not support custom Gradient method. Provided Gradient object is "
                "ignored.")
        self._validate_hamiltonian_form(hamiltonian)

        if t_param in gradient_params:
            epsilon = 0.01
            hamiltonian = self._try_binding_params(hamiltonian, hamiltonian_value_dict)
            evolved_state1 = self.evolve(hamiltonian, time + epsilon, initial_state, t_param=t_param)
            evolved_state2 = self.evolve(hamiltonian, time - epsilon, initial_state, t_param=t_param)
            expected_val_1 = ~StateFn(observable) @ evolved_state1
            expected_val_2 = ~StateFn(observable) @ evolved_state2
            finite_difference = (expected_val_1 - expected_val_2) / (2 * epsilon)
            return finite_difference.eval()
        elif set(gradient_params) == set(hamiltonian.parameters):
            gradients = defaultdict(float)
            if isinstance(hamiltonian, SummedOp):
                for gradient_param in gradient_params:
                    # the whole SummedOp might be multiplied by a parameter of interest
                    if gradient_param == hamiltonian.coeff:
                        gradient = self._calc_term_gradient(hamiltonian, hamiltonian,
                                                            initial_state, observable, t_param,
                                                            time, hamiltonian_value_dict)
                        gradients[gradient_param] += gradient
                    for hamiltonian_term in hamiltonian.oplist:
                        if gradient_param in hamiltonian_term.parameters:
                            gradient = self._calc_term_gradient(hamiltonian, hamiltonian_term,
                                                                initial_state, observable, t_param,
                                                                time, hamiltonian_value_dict)
                            gradients[gradient_param] += gradient
            # PauliSumOp has a coefficient which is complex or a parameter. If complex, it will be
            # skipped for a particular gradient as expected.
            elif isinstance(hamiltonian, PauliSumOp):
                for gradient_param in gradient_params:
                    # the whole PauliSumOp might be multiplied by a parameter of interest
                    if gradient_param == hamiltonian.coeff:
                        gradient = self._calc_term_gradient(hamiltonian, hamiltonian,
                                                            initial_state, observable, t_param,
                                                            time, hamiltonian_value_dict)
                        gradients[gradient_param] += gradient
                    for hamiltonian_term in hamiltonian:
                        if gradient_param in hamiltonian_term.parameters:
                            gradient = self._calc_term_gradient(hamiltonian,
                                                                hamiltonian_term.primitive,
                                                                initial_state, observable, t_param,
                                                                time, hamiltonian_value_dict)
                            gradients[gradient_param] += gradient

            return gradients
        elif any(gradient_params) not in (hamiltonian.parameters or [t_param]):
            raise ValueError("gradient_params provided that are not found in hamiltonian.params "
                             "and not a t_param.")

    def _calc_term_gradient(self, hamiltonian,
                            hamiltonian_term: Union[OperatorBase, PauliSumOp, SummedOp],
                            initial_state, observable, t_param,
                            time, hamiltonian_value_dict):
        hamiltonian_term = self._try_binding_params(hamiltonian_term, hamiltonian_value_dict)
        custom_observable = commutator(1j * time * hamiltonian_term, observable)
        hamiltonian = self._try_binding_params(hamiltonian, hamiltonian_value_dict)

        evolved_state = self.evolve(hamiltonian, time, initial_state, t_param,
                                    hamiltonian_value_dict=hamiltonian_value_dict)

        gradient = ~StateFn(custom_observable) @ evolved_state
        gradient = gradient.eval()
        return gradient

    def _validate_hamiltonian_form(self, hamiltonian: Union[OperatorBase, PauliSumOp, SummedOp]):
        if isinstance(hamiltonian, SummedOp):
            for op in hamiltonian.oplist:
                if not isinstance(op.coeff, ParameterExpression):
                    raise ValueError(
                        "Term of a Hamiltonian has a coefficient that is not a "
                        "ParameterExpression. It is not allowed.")
                if len(op.coeff.parameters) > 1:
                    raise ValueError(
                        "Term of a Hamiltonian has a coefficient that depends on several "
                        "parameters. Only dependence on a single parameter is allowed.")
                # TODO check if param linear

        elif isinstance(hamiltonian, PauliSumOp):
            for op in hamiltonian:
                if len(op.coeffs) > 1:
                    raise ValueError(
                        "Term of a Hamiltonian has multiple coefficients. It is not allowed.")
                if not isinstance(op.coeffs[0], ParameterExpression):
                    raise ValueError(
                        "Term of a Hamiltonian has a coefficient that is not a "
                        "ParameterExpression. It is not allowed.")
                if len(op.coeffs[0].parameters) > 1:
                    raise ValueError(
                        "Term of a Hamiltonian has a coefficient that depends on several "
                        "parameters. Only dependence on a single parameter is allowed.")
                # TODO check if param linear
        else:
            raise ValueError("Hamiltonian not a SummedOp or PauliSumOp")
