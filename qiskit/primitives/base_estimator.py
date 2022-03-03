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
=========
Estimator
=========

Estimator class estimates expectation values of quantum circuits and observables.

An estimator object is initialized with multiple quantum circuits and observables
and users can specify pairs of quantum circuits and observables
to estimate the expectation values.

The input consists of following elements.

* quantum circuits (:math:`\psi_i(\theta)`): list of (parametrized) quantum circuits
  or state vectors to be converted into quantum circuits.
  (a list of :class:`~qiskit.circuit.QuantumCircuit`))

* observables (:math:`H_j`): a list of :class:`~qiskit.quantum_info.SparsePauliOp`.

* grouping: a list of pairs :class:`~qiskit.primitive.Group` of an
  index of the quantum circuits and an index of the observable.
  A tuple of two integers can be used as alternative of :class:`~qiskit.primitive.Group`.
  For example, ``Group(i, j)`` or ``(i, j)`` corresponds to a pair :math:`\psi_i` and :math:`H_j`.

* parameters: a list of parameters of the quantum circuits.
  (:class:`~qiskit.circuit.parametertable.ParameterView` or
  a list of :class:`~qiskit.circuit.Parameter`).

* parameter values (:math:`\theta_k`): 1-dimensional or 2-dimensional array of values
  to be bound to the parameters of the quantum circuits.
  (list of float or list of list of float)

The output is the expectation value or a list of expectation values.

.. math::

    \Braket{\psi_i(\theta_k)|H_j|\psi_i(\theta_k)}


The estimator object is expected to be initialized with "with" statement
and the objects are called with parameter values and run options
(e.g., ``shots`` or number of shots).

Here is an example of how estimator is used.

.. code-block:: python

    from qiskit.circuit.library import RealAmplitudes
    from qiskit.quantum_info import SparsePauliOp

    psi1 = RealAmplitudes(num_qubits=2, reps=2)
    psi2 = RealAmplitudes(num_qubits=2, reps=3)

    params1 = psi1.parameters
    params2 = psi2.parameters

    H1 = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
    H2 = SparsePauliOp.from_list([("IZ", 1)])
    H3 = SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)])

    with Estimator([psi1, psi2], [H1, H2, H3], list(params1) + list(params2)) as e:
        # first 6 values correspond to the parameters of psi1
        # last 8 values correspond to the parameters of psi2
        theta1 = [0, 1, 1, 2, 3, 5] + [0] * 8
        theta2 = [0] * 6 + [0, 1, 1, 2, 3, 5, 8, 13]
        theta3 = [1, 2, 3, 4, 5, 6] + [0] * 8

        # calculate [ <psi1(theta1)|H1|psi1(theta1)> ]
        psi1_H1_result = e(theta1, shots=1024, grouping=[(0, 0)])
        print(psi1_H1_result)

        # calculate [ <psi1(theta1)|H2|psi1(theta1)>, <psi1(theta1)|H3|psi1(theta1)> ]
        psi1_H23_result = e(theta1, shots=1024, grouping=[(0, 1), (0, 2)])
        print(psi1_H23_result)

        # calculate [ <psi2(theta2)|H2|psi2(theta2)> ]
        psi2_H2_result = e(theta2, shots=1024, grouping=[(1, 1)])
        print(psi2_H2_result)

        # calculate [ <psi1(theta1)|H1|psi1(theta1)>, <psi1(theta3)|H1|psi1(theta3)> ]
        psi1_H1_result2 = e([theta1, theta3], shots=1024, grouping=[(0, 0)])
        print(psi1_H1_result2)

        # calculate [ <psi1(theta1)|H1|psi1(theta1)>, <psi2(theta2)|H2|psi2(theta2)>, <psi1(theta3)|H3|psi1(theta3)> ]
        psi12_H23_result = e([theta1, theta2, theta3], shots=1024, grouping=[(0, 0), (1, 1), (0, 2)])
        print(psi12_H23_result)

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.parametertable import ParameterView
from qiskit.quantum_info import SparsePauliOp

from .estimator_result import EstimatorResult


@dataclass(frozen=True)
class Group:
    """The dataclass represents indices of circuit and observable."""

    circuit_index: int
    observable_index: int


class BaseEstimator(ABC):
    """Estimator base class.

    Base class for Estimator that estimates expectation values of quantum circuits and observables.
    """

    def __init__(
        self,
        circuits: list[QuantumCircuit],
        observables: list[SparsePauliOp],
        parameters: Union[ParameterView, list[Parameter]],
    ):
        """
        Args:
            circuits (list[QuantumCircuit]): quantum circuits that represents quantum states
            observables (list[SparsePauliOp]): observables
            parameters (Union[ParameterView, list[Parameter]]): parameters of quantum circuits
        """
        self._circuits = circuits
        self._observables = observables
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
        """Quantum circuits that represents quantum states.

        Returns:
            list[QuantumCircuit]: quantum circuits
        """
        return self._circuits

    @property
    def observables(self) -> list[SparsePauliOp]:
        """Observables

        Returns:
            list[SparsePauliOp]: observables
        """
        return self._observables

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
        grouping: Optional[list[Union[Group, tuple[int, int]]]] = None,
        **run_options,
    ) -> EstimatorResult:
        """Run the estimation of expectation value(s).

        Args:
            parameters (Optional[Union[list[float], list[list[float]]]]): parameters to be bound.
            grouping (Optional[list[Union[Group, tuple[int, int]]]]): the list of Group or tuple of circuit index and observable index.
            run_options: backend runtime options used for circuit execution.

        Returns:
            EstimatorResult: the result of Estimator.
        """
        ...
