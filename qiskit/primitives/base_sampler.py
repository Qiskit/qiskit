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

Sampler class estimates quasi-probabilities of bitstrings from quantum circuits.

The input consists of following elements.

* quantum circuits (:math:`\psi_i(\theta)`): list of (parametrized) quantum circuits.
  (a list of :class:`~qiskit.circuit.QuantumCircuit`))

* parameters: a list of parameters of the quantum circuits.
  (:class:`~qiskit.circuit.parametertable.ParameterView` or
  a list of :class:`~qiskit.circuit.Parameter`).

* circuit indexes: a list of indices of the circuits to evaluate.

* parameter values (:math:`\theta_k`): list of sets of parameter values
  to be bound to the parameters of the quantum circuits.
  (list of list of float)

The output is the quasi-probabilities of bitstrings.

The sampler object is expected to be closed after use or
accessed within "with" context
and the objects are called with parameter values and run options
(e.g., ``shots`` or number of shots).

Here is an example of how sampler is used.

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RealAmplitudes

    bell = QuantumCircuit(2, 2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure(0, 0)
    bell.measure(1, 1)

    # executes a Bell circuit
    with Sampler(circuits=[bell], parameters=[[]]) as sampler:
        result = sampler([[]])
        print([q.binary_probabilities() for q in result.quasi_dists])

    # executes three Bell circuits
    with Sampler(circuits=[bell]*3) as sampler:
        result = sampler([[]]*3)
        print([q.binary_probabilities() for q in result.quasi_dists])

    # parametrized circuit
    pqc = QuantumCircuit(2, 2)
    pqc.compose(RealAmplitudes(num_qubits=2, reps=2), inplace=True)
    pqc.measure(0, 0)
    pqc.measure(1, 1)

    theta1 = [0, 1, 1, 2, 3, 5]
    theta2 = [1, 2, 3, 4, 5, 6]

    with Sampler(circuits=[pqc], parameters=[pcq.parameters]) as sampler:
        result1 = sampler([theta1, theta2], [0, 0])

        # result of pqc(theta1)
        print([q.binary_probabilities() for q in result[0].quasi_dists])

        # result of pqc(theta2)
        print([q.binary_probabilities() for q in result[1].quasi_dists])

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence

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
        circuits: Sequence[QuantumCircuit],
        parameters: Sequence[Sequence[Parameter]] | None = None,
    ):
        """
        Args:
            circuits (list[QuantumCircuit]): quantum circuits to be executed
            parameters (list[list[Parameter]]): parameters of quantum circuits
                Defaults to `[circ.parameters for circ in circuits]`

        Raises:
            QiskitError: for mismatch of circuits and parameters list.
        """
        self._circuits = tuple(circuits)
        if parameters is None:
            self._parameters = tuple(circ.parameters for circ in circuits)
        else:
            if len(parameters) != len(circuits):
                raise QiskitError(
                    f"Different number of parameters ({len(parameters)} and circuits ({len(circuits)}"
                )
            self._parameters = tuple(ParameterView(par) for par in parameters)

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
            tuple[QuantumCircuit]: quantum circuits
        """
        return self._circuits

    @property
    def parameters(self) -> tuple[ParameterView, ...]:
        """Parameters of quantum circuits

        Returns:
            tuple[ParameterView]: Parameter list of the quantum circuits
        """
        return self._parameters

    @abstractmethod
    def run(
        self,
        circuits: Sequence[int],
        parameters: Sequence[Sequence[float]],
        **run_options,
    ) -> SamplerResult:
        """Run the sampling of bitstrings.

        Args:
            parameters (list[list[float]]): parameters to be bound.
            circuits (list[int]): indexes of the circuits to evaluate.
            run_options: backend runtime options used for circuit execution.

        Returns:
            SamplerResult: the result of Sampler. The i-th result corresponds to
                self.circuits[circuits[i]] evaluated with parameters bound as parameters[i]
        """
        ...

    def __call__(
        self,
        circuits: Sequence[int],
        parameters: Sequence[Sequence[float]],
        **run_options,
    ) -> SamplerResult:
        """Run the sampling of bitstrings.

        Args:
            parameters (list[list[float]]): parameters to be bound.
            circuits (list[int]): indexes of the circuits to evaluate.
            run_options: backend runtime options used for circuit execution.

        Returns:
            SamplerResult: the result of Sampler. The i-th result corresponds to
                self.circuits[circuits[i]]
            evaluated with parameters bound as parameters[i]
        """
        return self.run(circuits, parameters, **run_options)


SamplerFactory = Callable[..., BaseSampler]
