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

An estimator is initialized with the following elements.

* quantum circuits (:math:`\psi_i(\theta)`): list of (parameterized) quantum circuits
  (a list of :class:`~qiskit.circuit.QuantumCircuit`))

* observables (:math:`H_j`): a list of :class:`~qiskit.quantum_info.SparsePauliOp`.

The estimator is called with the following inputs.

* circuit indexes: a list of indexes of the quantum circuits.

* observable indexes: a list of indexes of the observables.

* parameters: a list of parameters of the quantum circuits.
  (:class:`~qiskit.circuit.parametertable.ParameterView` or
  a list of :class:`~qiskit.circuit.Parameter`).

* parameter values (:math:`\theta_k`): list of sets of values
  to be bound to the parameters of the quantum circuits.
  (list of list of float)

The output is an EstimatorResult which contains a list of expectation values plus
optional metadata like confidence intervals for the estimation.

.. math::

    \langle\psi_i(\theta_k)|H_j|\psi_i(\theta_k)\rangle


The estimator object is expected to be `close()` d after use or
accessed inside "with" context
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

    with Estimator([psi1, psi2], [H1, H2, H3], [params1, params2]) as e:
        theta1 = [0, 1, 1, 2, 3, 5]
        theta2 = [0, 1, 1, 2, 3, 5, 8, 13]
        theta3 = [1, 2, 3, 4, 5, 6]

        # calculate [ <psi1(theta1)|H1|psi1(theta1)> ]
        psi1_H1_result = e([0], [0], [theta1])
        print(psi1_H1_result)

        # calculate [ <psi1(theta1)|H2|psi1(theta1)>, <psi1(theta1)|H3|psi1(theta1)> ]
        psi1_H23_result = e([0, 0], [1, 2], [theta1]*2)
        print(psi1_H23_result)

        # calculate [ <psi2(theta2)|H2|psi2(theta2)> ]
        psi2_H2_result = e([1], [1], [theta2])
        print(psi2_H2_result)

        # calculate [ <psi1(theta1)|H1|psi1(theta1)>, <psi1(theta3)|H1|psi1(theta3)> ]
        psi1_H1_result2 = e([0, 0], [0, 0], [theta1, theta3])
        print(psi1_H1_result2)

        # calculate [ <psi1(theta1)|H1|psi1(theta1)>,
        #             <psi2(theta2)|H2|psi2(theta2)>,
        #             <psi1(theta3)|H3|psi1(theta3)> ]
        psi12_H123_result = e([0, 0, 0], [0, 1, 2], [theta1, theta2, theta3])
        print(psi12_H23_result)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.parametertable import ParameterView
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import SparsePauliOp

from .estimator_result import EstimatorResult


class BaseEstimator(ABC):
    """Estimator base class.

    Base class for Estimator that estimates expectation values of quantum circuits and observables.
    """

    def __init__(
        self,
        circuits: Iterable[QuantumCircuit],
        observables: Iterable[SparsePauliOp],
        parameters: Optional[Iterable[Iterable[Parameter]]] = None,
    ):
        """
        Creating an instance of an Estimator, or using one in a ``with`` context opens a session that
        holds resources until the instance is ``close()`` ed or the context is exited.

        Args:
            circuits: quantum circuits that represent quantum states
            observables: observables
            parameters: parameters of quantum circuits, specifying the order in which values
            will be bound.
                Defaults to ``[circ.parameters for circ in circuits]``
                The indexing is such that ``parameters[i, j]`` is the j-th formal parameter of
                ``circuits[i]``.

        Raises:
            QiskitError: for mismatch of circuits and parameters list.
        """
        self._circuits = tuple(circuits)
        self._observables = tuple(observables)
        if parameters is None:
            self._parameters = tuple(circ.parameters for circ in self._circuits)
        else:
            self._parameters = tuple(ParameterView(par) for par in parameters)
            if len(self._parameters) != len(self._circuits):
                raise QiskitError(
                    f"Different number of parameters ({len(self._parameters)} and "
                    f"circuits ({len(self._circuits)}"
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
        """Quantum circuits that represents quantum states.

        Returns:
            quantum circuits
        """
        return self._circuits

    @property
    def observables(self) -> tuple[SparsePauliOp, ...]:
        """Observables to be estimated

        Returns:
            observables
        """
        return self._observables

    @property
    def parameters(self) -> tuple[ParameterView, ...]:
        """Parameters of quantum circuits

        Returns:
            parameters, where ``parameters[i][j]`` is the j-th parameter of the i-th circuit.
        """
        return self._parameters

    @abstractmethod
    def __call__(
        self,
        circuits: Sequence[int],
        observables: Sequence[int],
        parameters: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:
        """Run the estimation of expectation value(s).

        ``circuits``, ``observables``, and ``parameters`` should have the same length.
        The i-th element of the result is the expectation of observable

        .. code-block:: python

            obs = self.observables[observables[i]]

        for the state prepared by

        .. code-block:: python

            circ = self.circuits[circuits[i]]

        with bound parameters

        .. code-block:: python

            values = parameters[i].

        Args:
            circuits: the list of circuit indices.
            observables: the list of observable indices.
            parameters: concrete parameters to be bound.
            run_options: runtime options used for circuit execution.

        Returns:
            EstimatorResult: the result of Estimator.
        """
        ...
