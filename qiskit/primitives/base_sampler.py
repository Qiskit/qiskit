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
=======
Sampler
=======

Sampler class calculates probabilities or quasi-probabilities of bitstrings from quantum circuits.

A sampler is initialized with the following elements.

* quantum circuits (:math:`\psi_i(\theta)`): list of (parameterized) quantum circuits.
  (a list of :class:`~qiskit.circuit.QuantumCircuit`))

* parameters: a list of parameters of the quantum circuits.
  (:class:`~qiskit.circuit.parametertable.ParameterView` or
  a list of :class:`~qiskit.circuit.Parameter`).

The estimator is run with the following inputs.

* circuit indexes: a list of indices of the circuits to evaluate.

* parameter values (:math:`\theta_k`): list of sets of parameter values
  to be bound to the parameters of the quantum circuits.
  (list of list of float)

The output is a SamplerResult which contains probabilities or quasi-probabilities of bitstrings,
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
from typing import Iterable, Sequence

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.parametertable import ParameterView
from qiskit.exceptions import QiskitError

from .sampler_result import SamplerResult


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
            circuits: quantum circuits to be executed
            parameters: parameters of quantum circuits
                Defaults to ``[circ.parameters for circ in circuits]``

        Raises:
            QiskitError: for mismatch of circuits and parameters list.
        """
        self._circuits = tuple(circuits)
        if parameters is None:
            self._parameters = tuple(circ.parameters for circ in self._circuits)
        else:
            self._parameters = tuple(ParameterView(par) for par in parameters)
            if len(self._parameters) != len(self._circuits):
                raise QiskitError(
                    f"Different number of parameters ({len(self._parameters)} "
                    f"and circuits ({len(self._circuits)}"
                )

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
        """Quantum circuits

        Returns:
            quantum circuits
        """
        return self._circuits

    @property
    def parameters(self) -> tuple[ParameterView, ...]:
        """Parameters of quantum circuits

        Returns:
            Parameter list of the quantum circuits
        """
        return self._parameters

    @abstractmethod
    def __call__(
        self,
        circuits: Sequence[int],
        parameters: Sequence[Sequence[float]],
        **run_options,
    ) -> SamplerResult:
        """Run the sampling of bitstrings.

        Args:
            circuits: indexes of the circuits to evaluate.
            parameters: parameters to be bound.
            run_options: backend runtime options used for circuit execution.

        Returns:
            the result of Sampler. The i-th result corresponds to
            ``self.circuits[circuits[i]]`` evaluated with parameters bound as ``parameters[i]``.
        """
        ...
