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
from functools import wraps
from typing import cast

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.parametertable import ParameterView
from qiskit.exceptions import QiskitError

from .sampler_result import SamplerResult
from .utils import _finditer


class BaseSampler(ABC):
    """Sampler base class

    Base class of Sampler that calculates quasi-probabilities of bitstrings from quantum circuits.
    """

    def __init__(
        self,
        circuits: Iterable[QuantumCircuit],
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
        self._circuits = tuple(circuits)
        self._circuit_ids = (
            self._circuit_ids if self._circuit_ids else [id(i) for i in self._circuits]
        )
        if parameters is None:
            self._parameters = tuple(circ.parameters for circ in self._circuits)
        else:
            self._parameters = tuple(ParameterView(par) for par in parameters)
            if len(self._parameters) != len(self._circuits):
                raise QiskitError(
                    f"Different number of parameters ({len(self._parameters)} "
                    f"and circuits ({len(self._circuits)}"
                )

    def __init_subclass__(cls):
        original_init_method = cls.__init__

        @wraps(cls.__init__)
        def init_wrapper(
            self: BaseSampler,
            circuits: Iterable[QuantumCircuit],
            *args,
            parameters: Iterable[Iterable[Parameter]] | None = None,
            **kwargs,
        ):
            if isinstance(circuits, QuantumCircuit):
                circuits = [circuits]
            else:
                circuits = list(circuits)
            self._circuit_ids = [id(i) for i in circuits]

            if parameters is None:
                original_init_method(self, circuits, *args, **kwargs)
            else:
                original_init_method(self, circuits, *args, parameters, **kwargs)

        setattr(cls, "__init__", init_wrapper)

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

    def __call__(
        self,
        circuit_indices: Sequence[int | QuantumCircuit] | None = None,
        parameter_values: Sequence[Sequence[float]] | Sequence[float] | None = None,
        **run_options,
    ) -> SamplerResult:
        """Run the sampling of bitstrings.

        Args:
            circuit_indices: Indices of the circuits to evaluate.
            parameter_values: Parameters to be bound to the circuit.
            run_options: Backend runtime options used for circuit execution.

        Returns:
            The result of the sampler. The i-th result corresponds to
            ``self.circuits[circuit_indices[i]]`` evaluated with parameters bound as
            ``parameter_values[i]``.

        Raises:
            QiskitError: For mismatch of object id.
            QiskitError: For mismatch of length of Sequence.
        """
        # Support ndarray
        if isinstance(parameter_values, np.ndarray):
            parameter_values = parameter_values.tolist()

        # Allow lift
        if circuit_indices is not None and not isinstance(circuit_indices, (np.ndarray, Sequence)):
            circuit_indices = [circuit_indices]
        if parameter_values is not None and not isinstance(
            parameter_values[0], (np.ndarray, Sequence)
        ):
            parameter_values = cast("Sequence[float]", parameter_values)
            parameter_values = [parameter_values]

        # Allow broadcasting
        if circuit_indices is None and len(self._circuits) == 1 and parameter_values is not None:
            circuit_indices = [0] * len(parameter_values)

        # Allow optional
        if circuit_indices is None:
            circuit_indices = list(range(len(self._circuits)))
        if parameter_values is None:
            parameter_values = [[]] * len(circuit_indices)

        # Allow objects
        try:
            circuit_indices = [
                next(_finditer(i, self._circuit_ids)) if not isinstance(i, (int, np.integer)) else i
                for i in circuit_indices
            ]
        except StopIteration as err:
            raise QiskitError("The object id does not match.") from err

        # Validation
        if len(circuit_indices) != len(parameter_values):
            raise QiskitError(
                f"The number of circuit indices ({len(circuit_indices)}) does not match "
                f"the number of parameter value sets ({len(parameter_values)})."
            )

        return self._call(
            circuit_indices=circuit_indices,
            parameter_values=parameter_values,
            **run_options,
        )

    @abstractmethod
    def _call(
        self,
        circuit_indices: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> SamplerResult:
        ...
