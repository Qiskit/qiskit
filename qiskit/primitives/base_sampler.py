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
r"""
===================
Overview of Sampler
===================

Sampler class calculates probabilities or quasi-probabilities of bitstrings from quantum circuits.

A sampler is initialized with the following elements.

* quantum circuits (:math:`\psi_i(\theta)`): list of (parameterized) quantum circuits.
  (a list of :class:`~qiskit.circuit.QuantumCircuit`))

* parameters: a list of parameters of the quantum circuits.
  (:class:`~qiskit.circuit.parametertable.ParameterView` or
  a list of :class:`~qiskit.circuit.Parameter`).

The sampler is run with the following inputs.

* circuit indexes: a list of indices of the circuits to evaluate.

* parameter values (:math:`\theta_k`): list of sets of parameter values
  to be bound to the parameters of the quantum circuits.
  (list of list of float)

The output is a :class:`~qiskit.primitives.SamplerResult` which contains probabilities
or quasi-probabilities of bitstrings,
plus optional metadata like error bars in the samples.

The sampler object is expected to be closed after use or
accessed within "with" context
and the objects are called with parameter values and run options
(e.g., ``shots`` or number of shots).

Here is an example of how sampler is used.

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RealAmplitudes

    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure_all()

    # executes a Bell circuit
    with Sampler(circuits=[bell], parameters=[[]]) as sampler:
        result = sampler(parameters=[[]], circuits=[0])
        print([q.binary_probabilities() for q in result.quasi_dists])

    # executes three Bell circuits
    with Sampler([bell]*3, [[]] * 3) as sampler:
        result = sampler([0, 1, 2], [[]]*3)
        print([q.binary_probabilities() for q in result.quasi_dists])

    # executes three Bell circuits with objects.
    # Objects can be passed instead of indices.
    # Note that passing objects has an overhead
    # since the corresponding indices need to be searched.
    with Sampler([bell]) as sampler:
        result = sampler([bell, bell, bell])
        print([q.binary_probabilities() for q in result.quasi_dists])

    # parameterized circuit
    pqc = RealAmplitudes(num_qubits=2, reps=2)
    pqc.measure_all()
    pqc2 = RealAmplitudes(num_qubits=2, reps=3)
    pqc2.measure_all()

    theta1 = [0, 1, 1, 2, 3, 5]
    theta2 = [1, 2, 3, 4, 5, 6]
    theta3 = [0, 1, 2, 3, 4, 5, 6, 7]

    with Sampler(circuits=[pqc, pqc2], parameters=[pqc.parameters, pqc2.parameters]) as sampler:
        result = sampler([0, 0, 1], [theta1, theta2, theta3])

        # result of pqc(theta1)
        print(result.quasi_dists[0].binary_probabilities())

        # result of pqc(theta2)
        print(result.quasi_dists[1].binary_probabilities())

        # result of pqc2(theta3)
        print(result.quasi_dists[2].binary_probabilities())

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from copy import copy

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.parametertable import ParameterView
from qiskit.exceptions import QiskitError
from qiskit.utils.deprecation import deprecate_arguments

from .sampler_result import SamplerResult
from .utils import _finditer


class BaseSampler(ABC):
    """Sampler base class

    Base class of Sampler that calculates quasi-probabilities of bitstrings from quantum circuits.
    """

    def __init__(
        self,
        circuits: Iterable[QuantumCircuit] | QuantumCircuit,
        parameters: Iterable[Iterable[Parameter]] | None = None,
    ):
        """
        Args:
            circuits: Quantum circuits to be executed.
            parameters: Parameters of each of the quantum circuits.
                Defaults to ``[circ.parameters for circ in circuits]``.

        Raises:
            QiskitError: For mismatch of circuits and parameters list.
        """
        if isinstance(circuits, QuantumCircuit):
            circuits = (circuits,)
        self._circuits = tuple(circuits)

        # To guarantee that they exist as instance variable.
        # With only dynamic set, the python will not know if the attribute exists or not.
        self._circuit_ids = self._circuit_ids

        if parameters is None:
            self._parameters = tuple(circ.parameters for circ in self._circuits)
        else:
            self._parameters = tuple(ParameterView(par) for par in parameters)
            if len(self._parameters) != len(self._circuits):
                raise QiskitError(
                    f"Different number of parameters ({len(self._parameters)}) "
                    f"and circuits ({len(self._circuits)})"
                )

    def __new__(
        cls,
        circuits: Iterable[QuantumCircuit] | QuantumCircuit,
        *args,  # pylint: disable=unused-argument
        parameters: Iterable[Iterable[Parameter]] | None = None,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ):

        self = super().__new__(cls)
        if isinstance(circuits, Iterable):
            circuits = copy(circuits)
            self._circuit_ids = [id(circuit) for circuit in circuits]
        else:
            self._circuit_ids = [id(circuits)]
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    @abstractmethod
    def close(self):
        """Close the session and free resources"""
        ...

    @property
    def circuits(self) -> tuple[QuantumCircuit, ...]:
        """Quantum circuits to be sampled.

        Returns:
            The quantum circuits to be sampled.
        """
        return self._circuits

    @property
    def parameters(self) -> tuple[ParameterView, ...]:
        """Parameters of quantum circuits.

        Returns:
            List of the parameters in each quantum circuit.
        """
        return self._parameters

    @deprecate_arguments({"circuit_indices": "circuits"})
    def __call__(
        self,
        circuits: Sequence[int | QuantumCircuit],
        parameter_values: Sequence[Sequence[float]] | None = None,
        **run_options,
    ) -> SamplerResult:
        """Run the sampling of bitstrings.

        Args:
            circuits: the list of circuit indices or circuit objects.
            parameter_values: Parameters to be bound to the circuit.
            run_options: Backend runtime options used for circuit execution.

        Returns:
            The result of the sampler. The i-th result corresponds to
            ``self.circuits[circuits[i]]`` evaluated with parameters bound as
            ``parameter_values[i]``.

        Raises:
            QiskitError: For mismatch of object id.
            QiskitError: For mismatch of length of Sequence.
        """
        # Support ndarray
        if isinstance(parameter_values, np.ndarray):
            parameter_values = parameter_values.tolist()

        # Allow objects
        try:
            circuits = [
                next(_finditer(id(circuit), self._circuit_ids))
                if not isinstance(circuit, (int, np.integer))
                else circuit
                for circuit in circuits
            ]
        except StopIteration as err:
            raise QiskitError(
                "The circuits passed when calling sampler is not one of the circuits used to "
                "initialize the session."
            ) from err

        # Allow optional
        if parameter_values is None:
            for i in circuits:
                if len(self._circuits[i].parameters) != 0:
                    raise QiskitError(
                        f"The {i}-th circuit ({len(circuits)}) is parameterised,"
                        "but parameter values are not given."
                    )
            parameter_values = [[]] * len(circuits)

        # Validation
        if len(circuits) != len(parameter_values):
            raise QiskitError(
                f"The number of circuits ({len(circuits)}) does not match "
                f"the number of parameter value sets ({len(parameter_values)})."
            )

        for i, value in zip(circuits, parameter_values):
            if len(value) != len(self._parameters[i]):
                raise QiskitError(
                    f"The number of values ({len(value)}) does not match "
                    f"the number of parameters ({len(self._parameters[i])}) for the {i}-th circuit."
                )

        if max(circuits) >= len(self.circuits):
            raise QiskitError(
                f"The number of circuits is {len(self.circuits)}, "
                f"but the index {max(circuits)} is given."
            )

        return self._call(
            circuits=circuits,
            parameter_values=parameter_values,
            **run_options,
        )

    @abstractmethod
    def _call(
        self,
        circuits: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> SamplerResult:
        ...
