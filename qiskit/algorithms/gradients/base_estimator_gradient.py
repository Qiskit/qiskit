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
from .utils import GradientCircuit


class BaseEstimatorGradient(ABC):
    """Base class for an ``EstimatorGradient`` to compute the gradients of the expectation value."""

    def __init__(
        self,
        estimator: BaseEstimator,
        options: Options | None = None,
        supported_gates: Sequence[str] | None = None,
        add_virtual_var: bool = False,
        skip_pre_postprocess: bool = False,
    ):
        """
        Args:
            estimator: The estimator used to compute the gradients.
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting
            supported_gates: The list of supported gates to decompose circuits. Defaults to None.
            add_virtual_var: Whether to add virtual variables to circuits. This is used for the
                parameter shift gradient. Defaults to False.
            skip_pre_postprocess: Whether to skip the preprocess and postprocess. Defaults to False.
        """
        self._estimator: BaseEstimator = estimator
        self._default_options = Options()
        if options is not None:
            self._default_options.update_options(**options)
        self._gradient_circuit_cache: dict[QuantumCircuit, GradientCircuit] = {}
        self._supported_gates: Sequence[str] | None = supported_gates
        self._add_virtual_var: bool = add_virtual_var
        self._skip_pre_postprocess: bool = skip_pre_postprocess

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

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None],
        **options,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        # Preprocess the gradient.
        if not self._skip_pre_postprocess:
            self._preprocess(circuits)

        # Calculate the gradient.
        result = self._run_unique(circuits, observables, parameter_values, parameters, **options)

        # Postprocess the gradient.
        if not self._skip_pre_postprocess:
            result = self._postprocess(circuits, result)

        return result

    # @abstractmethod
    def _run_unique(
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

    def _preprocess(self, circuits: Sequence[QuantumCircuit]):
        """Preprocess the gradient."""
        for circuit in circuits:
            gradient_circuit = self._gradient_circuit_cache.get(_circuit_key(circuit))
            if gradient_circuit is None:
                self._gradient_circuit_cache[
                    _circuit_key(circuit)
                ] = self._assign_unique_parameters(circuit)

    def _assign_unique_parameters(self, circuit: QuantumCircuit):
        """Assign unique parameters to the parameters in ``circuit`` and return the new circuit with
        the parameter map and the coefficient map.

        Args:
            circuit: The circuit to assign unique parameters.

        Returns:
            The new circuit with the parameter map and the coefficient map.
        """

        circuit2 = transpile(circuit, basis_gates=self._supported_gates, optimization_level=0)
        g_circuit = circuit2.copy_empty_like(f"g_{circuit2.name}")
        param_inst_dict = defaultdict(list)
        g_parameter_map = defaultdict(list)
        g_virtual_parameter_map = {}
        num_virtual_parameter_variables = 0
        coeff_map = {}

        for inst in circuit2.data:
            new_inst = deepcopy(inst)
            qubit_indices = [circuit2.qubits.index(qubit) for qubit in inst[1]]
            new_inst.qubits = tuple(g_circuit.qubits[qubit_index] for qubit_index in qubit_indices)

            # Assign new unique parameters when the instruction is parameterized.
            if inst.operation.is_parameterized():
                parameters = inst.operation.params
                new_inst_parameters = []
                # For a gate with multiple parameters e.g. a U gate
                for parameter in parameters:
                    subs_map = {}
                    # For a gate parameter with multiple parameter variables.
                    # e.g. ry(θ) with θ = (2x + y)
                    for parameter_variable in parameter.parameters:
                        if parameter_variable in param_inst_dict:
                            new_parameter_variable = Parameter(
                                f"g{parameter_variable.name}_{len(param_inst_dict[parameter_variable])+1}"
                            )
                        else:
                            new_parameter_variable = Parameter(f"g{parameter_variable.name}_1")
                        subs_map[parameter_variable] = new_parameter_variable
                        param_inst_dict[parameter_variable].append(inst)
                        g_parameter_map[parameter_variable].append(new_parameter_variable)
                        # Coefficient to calculate derivative i.e. dw/dt in df/dw * dw/dt
                        coeff_map[new_parameter_variable] = parameter.gradient(parameter_variable)
                    # Substitute the parameter variables with the corresponding new parameter
                    # variables in ``subs_map``.
                    new_parameter = parameter.subs(subs_map)
                    if self._add_virtual_var:
                        # If new_parameter is not a single parameter variable, then add a new virtual
                        # parameter variable. e.g. ry(θ) with θ = (2x + y) becomes ry(θ + virtual_variable)
                        if not isinstance(new_parameter, Parameter):
                            virtual_parameter_variable = Parameter(
                                f"vθ_{num_virtual_parameter_variables+1}"
                            )
                            num_virtual_parameter_variables += 1
                            for new_parameter_variable in new_parameter.parameters:
                                g_virtual_parameter_map[
                                    new_parameter_variable
                                ] = virtual_parameter_variable
                            new_parameter = new_parameter + virtual_parameter_variable
                    new_inst_parameters.append(new_parameter)
                new_inst.operation.params = new_inst_parameters
            g_circuit.append(new_inst)

        # for global phase
        subs_map = {}
        if isinstance(g_circuit.global_phase, ParameterExpression):
            for parameter_variable in g_circuit.global_phase.parameters:
                if parameter_variable in param_inst_dict:
                    new_parameter_variable = g_parameter_map[parameter_variable][0]
                else:
                    new_parameter_variable = Parameter(f"g{parameter_variable.name}_1")
                subs_map[parameter_variable] = new_parameter_variable
            g_circuit.global_phase = g_circuit.global_phase.subs(subs_map)

        return GradientCircuit(
            circuit=circuit2,
            gradient_circuit=g_circuit,
            gradient_virtual_parameter_map=g_virtual_parameter_map,
            gradient_parameter_map=g_parameter_map,
            coeff_map=coeff_map,
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

    def _postprocess(self, circuits: Sequence[QuantumCircuit], result: EstimatorGradientResult):
        """Postprocess the gradient."""
        original_gradients = []
        for circuit, gradient, metadata in zip(circuits, result.gradients, result.metadata):
            parameters = metadata["parameters"]
            original_gradient = np.zeros(len(parameters))
            gradient_circuit = self._gradient_circuit_cache[_circuit_key(circuit)]
            idx = 0
            for i, parameter in enumerate(parameters):
                num_results = len(gradient_circuit.gradient_parameter_map[parameter])
                # print(f"num_results: {num_results}")
                original_gradient[i] = np.sum(gradient[idx : idx + num_results])
                idx += num_results
            original_gradients.append(original_gradient)
        return EstimatorGradientResult(
            gradients=original_gradients, metadata=result.metadata, options=result.options
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
