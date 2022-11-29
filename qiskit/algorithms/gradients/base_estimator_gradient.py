# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Abstract base class of gradient for ``Estimator``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from copy import copy

import numpy as np

from qiskit import transpile
from qiskit.circuit import Parameter, QuantumCircuit, ParameterExpression
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.primitives.utils import _circuit_key
from qiskit.providers import Options
from qiskit.algorithms import AlgorithmJob
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .estimator_gradient_result import EstimatorGradientResult
from .utils import GradientCircuit


class BaseEstimatorGradient(ABC):
    """Base class for an ``EstimatorGradient`` to compute the gradients of the expectation value."""

    def __init__(
        self,
        estimator: BaseEstimator,
        options: Options | None = None,
    ):
        """
        Args:
            estimator: The estimator used to compute the gradients.
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting
        """
        self._estimator: BaseEstimator = estimator
        self._default_options = Options()
        if options is not None:
            self._default_options.update_options(**options)
        self._gradient_circuit_cache: dict[QuantumCircuit, GradientCircuit] = {}

    def run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None] | None = None,
        **options,
    ) -> AlgorithmJob:
        """Run the job of the estimator gradient on the given circuits.

        Args:
            circuits: The list of quantum circuits to compute the gradients.
            observables: The list of observables.
            parameter_values: The list of parameter values to be bound to the circuit.
            parameters: The sequence of parameters to calculate only the gradients of
                the specified parameters. Each sequence of parameters corresponds to a circuit in
                ``circuits``. Defaults to None, which means that the gradients of all parameters in
                each circuit are calculated.
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting

        Returns:
            The job object of the gradients of the expectation values. The i-th result corresponds to
            ``circuits[i]`` evaluated with parameters bound as ``parameter_values[i]``. The j-th
            element of the i-th result corresponds to the gradient of the i-th circuit with respect
            to the j-th parameter.

        Raises:
            ValueError: Invalid arguments are given.
        """
        # if ``parameters`` is none, all parameters in each circuit are differentiated.
        if parameters is None:
            parameters = [None for _ in range(len(circuits))]
        # Validate the arguments.
        self._validate_arguments(circuits, observables, parameter_values, parameters)
        # The priority of run option is as follows:
        # options in ``run`` method > gradient's default options > primitive's default setting.
        opts = copy(self._default_options)
        opts.update_options(**options)
        job = AlgorithmJob(
            self._run, circuits, observables, parameter_values, parameters, **opts.__dict__
        )
        job.submit()
        return job

    @abstractmethod
    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None],
        **options,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        raise NotImplementedError()

    def _validate_arguments(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None] | None = None,
    ) -> None:
        """Validate the arguments of the ``run`` method.

        Args:
            circuits: The list of quantum circuits to compute the gradients.
            observables: The list of observables.
            parameter_values: The list of parameter values to be bound to the circuit.
            parameters: The Sequence of Sequence of Parameters to calculate only the gradients of
                the specified parameters. Each Sequence of Parameters corresponds to a circuit in
                ``circuits``. Defaults to None, which means that the gradients of all parameters in
                each circuit are calculated.

        Raises:
            ValueError: Invalid arguments are given.
        """
        if len(circuits) != len(parameter_values):
            raise ValueError(
                f"The number of circuits ({len(circuits)}) does not match "
                f"the number of parameter value sets ({len(parameter_values)})."
            )

        if len(circuits) != len(observables):
            raise ValueError(
                f"The number of circuits ({len(circuits)}) does not match "
                f"the number of observables ({len(observables)})."
            )

        if parameters is not None:
            if len(circuits) != len(parameters):
                raise ValueError(
                    f"The number of circuits ({len(circuits)}) does not match "
                    f"the number of the specified parameter sets ({len(parameters)})."
                )

        for i, (circuit, parameter_value) in enumerate(zip(circuits, parameter_values)):
            if not circuit.num_parameters:
                raise ValueError(f"The {i}-th circuit is not parameterised.")
            if len(parameter_value) != circuit.num_parameters:
                raise ValueError(
                    f"The number of values ({len(parameter_value)}) does not match "
                    f"the number of parameters ({circuit.num_parameters}) for the {i}-th circuit."
                )

        for i, (circuit, observable) in enumerate(zip(circuits, observables)):
            if circuit.num_qubits != observable.num_qubits:
                raise ValueError(
                    f"The number of qubits of the {i}-th circuit ({circuit.num_qubits}) does "
                    f"not match the number of qubits of the {i}-th observable "
                    f"({observable.num_qubits})."
                )
        for i, (circuit, parameter) in enumerate(zip(circuits, parameters)):
            if parameter is not None:
                if not set(parameter).issubset(circuit.parameters):
                    raise ValueError(
                        f"The {i}-th parameter set contains parameters not present in the "
                        f"{i}-th circuit."
                    )

    def _preprocess(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None],
        supported_gates: Sequence[str],
    ):
        """Preprocess the gradient."""
        g_circuits, g_parameter_values, g_parameters = [], [], []
        for circuit, parameter_value_, parameters_ in zip(circuits, parameter_values, parameters):
            if not _circuit_key(circuit) in self._gradient_circuit_cache:
                transpiled_circuit = transpile(
                    circuit, basis_gates=supported_gates, optimization_level=0
                )
                self._gradient_circuit_cache[
                    _circuit_key(circuit)
                ] = self._assign_unique_parameters(transpiled_circuit)
            gradient_circuit = self._gradient_circuit_cache[_circuit_key(circuit)]
            g_circuits.append(gradient_circuit.gradient_circuit)
            g_parameter_values.append(
                self._make_gradient_parameter_values(circuit, gradient_circuit, parameter_value_)
            )
            g_parameters.append(
                self._make_gradient_parameters(circuit, gradient_circuit, parameters_)
            )
        return g_circuits, g_parameter_values, g_parameters

    def _assign_unique_parameters(
        self,
        circuit: QuantumCircuit,
    ) -> GradientCircuit:
        """Assign unique parameters to the circuit.

        Args:
            circuit: The circuit to assign unique parameters.

        Returns:
            The circuit with unique parameters.
        """
        gradient_circuit = circuit.copy_empty_like(f"{circuit.name}_gradient")
        parameter_map = defaultdict(list)
        gradient_parameter_map = {}
        num_gradient_parameters = 0
        for instruction, qargs, cargs in circuit.data:
            if instruction.is_parameterized():
                new_inst_params = []
                for angle in instruction.params:
                    new_parameter = Parameter(f"gθ{num_gradient_parameters}")
                    new_inst_params.append(new_parameter)
                    num_gradient_parameters += 1
                    for parameter in angle.parameters:
                        parameter_map[parameter].append((new_parameter, angle.gradient(parameter)))
                    gradient_parameter_map[new_parameter] = angle
                instruction.params = new_inst_params
            gradient_circuit.append(instruction, qargs, cargs)
        # For the global phase
        gradient_circuit.global_phase = circuit.global_phase
        if isinstance(gradient_circuit.global_phase, ParameterExpression):
            substitution_map = {}
            for parameter in gradient_circuit.global_phase.parameters:
                if parameter in parameter_map:
                    substitution_map[parameter] = parameter_map[parameter][0][0]
                else:
                    new_parameter = Parameter(f"gθ{num_gradient_parameters}")
                    substitution_map[parameter] = new_parameter
                    parameter_map[parameter].append(new_parameter, 1)
                    num_gradient_parameters += 1
            gradient_circuit.global_phase = gradient_circuit.global_phase.subs(substitution_map)

        return GradientCircuit(gradient_circuit, parameter_map, gradient_parameter_map)

    def _make_gradient_parameter_values(
        self,
        circuit: QuantumCircuit,
        gradient_circuit: GradientCircuit,
        parameter_values: np.ndarray,
    ) -> np.ndarray:
        """Makes parameter values for the gradient circuit.

        Args:
            circuit: The original quantum circuit
            gradient_circuit: The gradient circuit
            parameter_values: The parameter values for the original circuit
            parameter_set: The parameter set to calculate gradients

        Returns:
            The parameter values for the gradient circuit.
        """
        g_circuit = gradient_circuit.gradient_circuit
        g_parameter_values = np.zeros(len(g_circuit.parameters))
        for i, g_parameter in enumerate(g_circuit.parameters):
            expr = gradient_circuit.gradient_parameter_map[g_parameter]
            bound_expr = expr.bind(
                {p: parameter_values[circuit.parameters.data.index(p)] for p in expr.parameters}
            )

            g_parameter_values[i] = float(bound_expr)
        return g_parameter_values

    def _make_gradient_parameters(
        self,
        circuit: QuantumCircuit,
        gradient_circuit: GradientCircuit,
        parameters: Sequence[Parameter] | None,
    ) -> Sequence[Parameter] | None:
        """Makes parameters for the gradient circuit.

        Args:
            circuit: The original quantum circuit
            gradient_circuit: The gradient circuit
            parameters: The parameters for the original circuit

        Returns:
            The parameters for the gradient circuit.
        """
        if parameters is None:
            return None

        g_parameters = []
        for parameter in circuit.parameters:
            if parameter in parameters:
                g_parameters.extend(
                    g_parameter for g_parameter, _ in gradient_circuit.parameter_map[parameter]
                )
        return list(set(g_parameters))

    def _make_parameter_set(self, circuit: QuantumCircuit, parameters: Sequence[Parameter]):
        """Make a set of parameters from ``parameters`` that are in ``circuit``.

        Args:
            circuit: The circuit to make the parameter set.
            parameters: The parameters to make the parameter set.

        Returns:
            The set of parameters. If ``parameters`` is None, then the set of all parameters in ``circuit``
            is returned.
        """
        return set(circuit.parameters) if parameters is None else set(parameters)

    def _postprocess(
        self,
        results: Sequence[EstimatorGradientResult],
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None],
    ) -> EstimatorGradientResult:
        """Postprocess the gradient."""
        original_gradients, original_metadata = [], []
        for circuit, parameter_values_, parameters_, gradient in zip(
            circuits, parameter_values, parameters, results.gradients
        ):
            parameter_set = self._make_parameter_set(circuit, parameters_)
            original_gradient = np.zeros(len(parameter_set))
            gradient_circuit = self._gradient_circuit_cache[_circuit_key(circuit)]
            g_parameters_ = self._make_gradient_parameters(circuit, gradient_circuit, parameters_)
            g_parameter_set = self._make_parameter_set(
                gradient_circuit.gradient_circuit, g_parameters_
            )
            result_indices_ = [param for param in circuit.parameters if param in parameter_set]
            g_result_indices_ = [
                param
                for param in gradient_circuit.gradient_circuit.parameters
                if param in g_parameter_set
            ]
            g_result_indices = {param: i for i, param in enumerate(g_result_indices_)}

            for i, parameter in enumerate(result_indices_):
                for g_parameter, coeff in gradient_circuit.parameter_map[parameter]:
                    if isinstance(coeff, ParameterExpression):
                        local_map = {
                            p: parameter_values_[circuit.parameters.data.index(p)]
                            for p in coeff.parameters
                        }
                        bound_coeff = coeff.bind(local_map)
                    else:
                        bound_coeff = coeff
                    original_gradient[i] += bound_coeff * gradient[g_result_indices[g_parameter]]
            original_gradients.append(original_gradient)
            original_metadata.append(
                [{"parameters": [p for p in circuit.parameters if p in parameter_set]}]
            )
        return EstimatorGradientResult(
            gradients=original_gradients, metadata=results.metadata, options=results.options
        )

    @property
    def options(self) -> Options:
        """Return the union of estimator options setting and gradient default options,
        where, if the same field is set in both, the gradient's default options override
        the primitive's default setting.

        Returns:
            The gradient default + estimator options.
        """
        return self._get_local_options(self._default_options.__dict__)

    def update_default_options(self, **options):
        """Update the gradient's default options setting.

        Args:
            **options: The fields to update the default options.
        """

        self._default_options.update_options(**options)

    def _get_local_options(self, options: Options) -> Options:
        """Return the union of the primitive's default setting,
        the gradient default options, and the options in the ``run`` method.
        The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.

        Args:
            options: The fields to update the options

        Returns:
            The gradient default + estimator + run options.
        """
        opts = copy(self._estimator.options)
        opts.update_options(**options)
        return opts
