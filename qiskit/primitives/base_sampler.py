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

* parameter values (:math:`\theta_k`): 1-dimensional or 2-dimensional array of values
  to be bound to the parameters of the quantum circuits.
  (list of float or list of list of float)

The output is the quasi-probabilities of bitstrings.

The sampler object is expected to be initialized with "with" statement
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
    with Sampler(circuits=bell) as sampler:
        result = sampler(shots=1000)
        print([q.binary_probabilities() for q in result.quasi_dists])

    # executes three Bell circuits
    with Sampler(circuits=[bell, bell, bell], backend=aer_simulator) as sampler:
        result = sampler(shots=1000)
        print([q.binary_probabilities() for q in result.quasi_dists])

    # parametrized circuit
    pqc = QuantumCircuit(2, 2)
    pqc.compose(RealAmplitudes(num_qubits=2, reps=2), inplace=True)
    pqc.measure(0, 0)
    pqc.measure(1, 1)

    theta1 = [0, 1, 1, 2, 3, 5]
    theta2 = [1, 2, 3, 4, 5, 6]

    with Sampler(circuits=pqc, backend=aer_simulator) as sampler:
        result1 = sampler([theta1, theta2], shots=3000)

        # result of pqc(theta1)
        print([q.binary_probabilities() for q in result[0].quasi_dists])

        # result of pqc(theta2)
        print([q.binary_probabilities() for q in result[1].quasi_dists])

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.parametertable import ParameterView

from .sampler_result import SamplerResult


class BaseSampler(ABC):
    """ Sampler base class

    Base class of Sampler that calculates quasi-probabilities of bitstrings from quantum circuits.
    """

    def __init__(
        self,
        circuits: list[QuantumCircuit],
        parameters: Union[ParameterView, list[Parameter]],
    ):
        """
        Args:
            circuits (list[QuantumCircuit]): quantum circuits to be executed
            parameters (Union[ParameterView, list[Parameter]]): parameters of quantum circuits
        """
        self._circuits = circuits
        self._parameters = parameters

    @abstractmethod
    def __enter__(self):
        ...

    @abstractmethod
    def __exit__(self, ex_type, ex_value, trace):
        ...

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.__call__ = cls.run

    @property
    def circuits(self) -> list[QuantumCircuit]:
        """Quantum circuits

        Returns:
            list[QuantumCircuit]: quantum circuits
        """
        return self._circuits

    @property
    def parameters(self) -> Union[ParameterView, list[Parameter]]:
        """Parameters of quantum circuits

        Returns:
            Union[ParameterView, list[Parameter]]: Parameter list of the quantum circuits
        """
        return self._parameters

    @abstractmethod
    def run(
        self,
        parameters: Optional[Union[list[float], list[list[float]]]] = None,
        **run_options,
    ) -> SamplerResult:
        """Run the sampling of bitstrings.

        Args:
            parameters (Optional[Union[list[float], list[list[float]]]]): parameters to be bound.
            run_options: backend runtime options used for circuit execution.

        Returns:
            SamplerResult: the result of Sampler.
        """
        ...
