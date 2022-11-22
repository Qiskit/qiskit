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
from copy import copy, deepcopy

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
from .utils import GradientCircuit, GradientCircuit2


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

    def _assign_unique_parameters(
        self,
        circuit: QuantumCircuit,
        supported_gates: Sequence[str] = None,
        add_virtual_var: bool = False,
    ) -> QuantumCircuit:
        """Assign unique parameters to the circuit.

        Args:
            circuit: The circuit to assign unique parameters.

        Returns:
            The circuit with unique parameters.
        """
        circuit2 = transpile(circuit, basis_gates=supported_gates, optimization_level=0)
        gradient_circuit = circuit2.copy_empty_like(f"g_{circuit2.name}")
        used_parameter_dict = defaultdict(int)
        parameter_map = defaultdict(list)
        virtual_parameter_map = {}
        coeff_map = {}
        num_virtual_parameters = 0
        for instruction, qargs, cargs in circuit2.data:
            # If the instruction is a parameterized gate, assign unique parameters to it.
            if instruction.is_parameterized():
                new_inst_params = []
                # For a gate with multiple angles, e.g. a U gate with 3 angles, (theta, phi, lambda).
                for angle in instruction.params:
                    subs_map = {}
                    # For an angle with multiple parameters, e.g. theta = a + b.
                    for param in angle.parameters:
                        if param in used_parameter_dict:
                            used_parameter_dict[param] += 1
                        new_param = Parameter(f"g{param.name}_{used_parameter_dict[param]}")
                        parameter_map[param].append(new_param)
                        subs_map[param] = new_param
                        # Coefficient used in the chain rule,
                        # i.e. da/d(theta) in df/d(theta) = df/da * da/d(theta).
                        coeff_map[new_param] = angle.gradient(param)
                    # Substitute the new parameters with the existing parameters.
                    new_angle = angle.subs(subs_map)
                    if add_virtual_var:
                        if not isinstance(new_angle, Parameter):
                            virtual_parameter = Parameter(f"vθ_{num_virtual_parameters}")
                            num_virtual_parameters += 1
                            for param in new_angle.parameters:
                                virtual_parameter_map[param] = virtual_parameter
                            new_angle = new_angle + virtual_parameter
                    new_inst_params.append(new_angle)
                instruction.params = new_inst_params
            gradient_circuit.append(instruction, qargs, cargs)
        # For the global phase.
        if isinstance(gradient_circuit.global_phase, ParameterExpression):
            subs_map = {}
            for param in gradient_circuit.global_phase.parameters:
                if param in used_parameter_dict:
                    new_param = parameter_map[param][0]
                else:
                    new_param = Parameter(f"g{param.name}_{used_parameter_dict[param]}")
                subs_map[param] = new_param
            gradient_circuit.global_phase = gradient_circuit.global_phase.subs(subs_map)
        print(gradient_circuit.draw())
        print(gradient_circuit.parameters)
        return GradientCircuit2(
            gradient_circuit=gradient_circuit,
            gradient_parameter_map=parameter_map,
            gradient_virtual_parameter_map=virtual_parameter_map,
            coeff_map=coeff_map,
        )

    def _recombine_results(
        self, circuits: Sequence[QuantumCircuit], result: EstimatorGradientResult
    ):
        """Recombine the results from the gradient circuits into the results of original circuits."""
        original_gradients = []
        for circuit, gradient, metadata in zip(circuits, result.gradients, result.metadata):
            parameters = metadata["parameters"]
            original_gradient = np.zeros(len(parameters))
            gradient_circuit = self._gradient_circuit_cache[_circuit_key(circuit)]
            idx = 0
            for i, parameter in enumerate(parameters):
                print(i, parameter)
                num_results = len(gradient_circuit.gradient_parameter_map[parameter])
                # print(f"num_results: {num_results}")
                original_gradient[i] = np.sum(gradient[idx : idx + num_results])
                idx += num_results
            original_gradients.append(original_gradient)
        return EstimatorGradientResult(
            gradients=original_gradients, metadata=result.metadata, options=result.options
        )

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
