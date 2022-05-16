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

.. estimator-desc:

=====================
Overview of Estimator
=====================

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

The output is an :class:`~qiskit.primitives.EstimatorResult` which contains a list of
expectation values plus optional metadata like confidence intervals for the estimation.

.. math::

    \langle\psi_i(\theta_k)|H_j|\psi_i(\theta_k)\rangle


The estimator object is expected to be ``close()`` d after use or
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
        result = e([0], [0], [theta1])
        print(result)

        # calculate [ <psi1(theta1)|H2|psi1(theta1)>, <psi1(theta1)|H3|psi1(theta1)> ]
        result2 = e([0, 0], [1, 2], [theta1]*2)
        print(result2)

        # calculate [ <psi2(theta2)|H2|psi2(theta2)> ]
        result3 = e([1], [1], [theta2])
        print(result3)

        # calculate [ <psi1(theta1)|H1|psi1(theta1)>, <psi1(theta3)|H1|psi1(theta3)> ]
        result4 = e([0, 0], [0, 0], [theta1, theta3])
        print(result4)

        # calculate [ <psi1(theta1)|H1|psi1(theta1)>,
        #             <psi2(theta2)|H2|psi2(theta2)>,
        #             <psi1(theta3)|H3|psi1(theta3)> ]
        result5 = e([0, 1, 0], [0, 1, 2], [theta1, theta2, theta3])
        print(result5)
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
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .estimator_result import EstimatorResult
from .utils import _finditer, _findname


class BaseEstimator(ABC):
    """Estimator base class.

    Base class for Estimator that estimates expectation values of quantum circuits and observables.
    """

    def __init__(
        self,
        circuits: Iterable[QuantumCircuit],
        observables: Iterable[SparsePauliOp],
        parameters: Iterable[Iterable[Parameter]] | None = None,
    ):
        """
        Creating an instance of an Estimator, or using one in a ``with`` context opens a session that
        holds resources until the instance is ``close()`` ed or the context is exited.

        Args:
            circuits: Quantum circuits that represent quantum states.
            observables: Observables.
            parameters: Parameters of quantum circuits, specifying the order in which values
                will be bound. Defaults to ``[circ.parameters for circ in circuits]``
                The indexing is such that ``parameters[i, j]`` is the j-th formal parameter of
                ``circuits[i]``.

        Raises:
            QiskitError: For mismatch of circuits and parameters list.
        """
        self._circuits = tuple(circuits)
        self._observables = tuple(observables)
        self._circuit_ids = (
            self._circuit_ids if self._circuit_ids else [circuit.name for circuit in self._circuits]
        )
        self._observable_ids = (
            self._observable_ids
            if self._observable_ids
            else [id(observable) for observable in self._observables]
        )
        if parameters is None:
            self._parameters = tuple(circ.parameters for circ in self._circuits)
        else:
            self._parameters = tuple(ParameterView(par) for par in parameters)
            if len(self._parameters) != len(self._circuits):
                raise QiskitError(
                    f"Different number of parameters ({len(self._parameters)} and "
                    f"circuits ({len(self._circuits)}"
                )
            for i, (circ, params) in enumerate(zip(self._circuits, self._parameters)):
                if circ.num_parameters != len(params):
                    raise QiskitError(
                        f"Different numbers of parameters of {i}-th circuit: "
                        f"expected {circ.num_parameters}, actual {len(params)}."
                    )

    def __init_subclass__(cls):
        original_init_method = cls.__init__

        @wraps(cls.__init__)
        def init_wrapper(
            self: BaseEstimator,
            circuits: Iterable[QuantumCircuit],
            observables: Iterable[SparsePauliOp],
            *args,
            parameters: Iterable[Iterable[Parameter]] | None = None,
            **kwargs,
        ):
            if isinstance(circuits, QuantumCircuit):
                circuits = [circuits]
            else:
                circuits = list(circuits)
            if isinstance(observables, (PauliSumOp, BaseOperator)):
                observables = [observables]
            else:
                observables = list(observables)
            self._circuit_ids = [circuit.name for circuit in circuits]
            self._observable_ids = [id(observable) for observable in observables]

            if parameters is None:
                original_init_method(self, circuits, observables, *args, **kwargs)
            else:
                original_init_method(self, circuits, observables, *args, parameters, **kwargs)

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
        """Quantum circuits that represents quantum states.

        Returns:
            The quantum circuits.
        """
        return self._circuits

    @property
    def observables(self) -> tuple[SparsePauliOp, ...]:
        """Observables to be estimated.

        Returns:
            The observables.
        """
        return self._observables

    @property
    def parameters(self) -> tuple[ParameterView, ...]:
        """Parameters of the quantum circuits.

        Returns:
            Parameters, where ``parameters[i][j]`` is the j-th parameter of the i-th circuit.
        """
        return self._parameters

    def __call__(
        self,
        circuit_indices: Sequence[int | QuantumCircuit] | None = None,
        observable_indices: Sequence[int | SparsePauliOp] | None = None,
        parameter_values: Sequence[Sequence[float]] | Sequence[float] | None = None,
        **run_options,
    ) -> EstimatorResult:
        """Run the estimation of expectation value(s).

        ``circuit_indices``, ``observable_indices``, and ``parameter_values`` should have the same
        length. The i-th element of the result is the expectation of observable

        .. code-block:: python

            obs = self.observables[observable_indices[i]]

        for the state prepared by

        .. code-block:: python

            circ = self.circuits[circuit_indices[i]]

        with bound parameters

        .. code-block:: python

            values = parameter_values[i].

        Args:
            circuit_indices: the list of circuit indices.
            observable_indices: the list of observable indices.
            parameter_values: concrete parameters to be bound.
            run_options: runtime options used for circuit execution.

        Returns:
            EstimatorResult: The result of the estimator.

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
        if observable_indices is not None and not isinstance(
            observable_indices, (np.ndarray, Sequence)
        ):
            observable_indices = [observable_indices]
        if parameter_values is not None and not isinstance(
            parameter_values[0], (np.ndarray, Sequence)
        ):
            parameter_values = cast("Sequence[float]", parameter_values)
            parameter_values = [parameter_values]

        # Allow broadcasting
        if (
            circuit_indices is None
            and len(self._circuits) == 1
            and observable_indices is None
            and len(self._observables) == 1
            and parameter_values is not None
        ):
            circuit_indices = [0] * len(parameter_values)
            observable_indices = [0] * len(parameter_values)

        # Allow optional
        if circuit_indices is None:
            circuit_indices = list(range(len(self._circuits)))
        if observable_indices is None:
            observable_indices = list(range(len(self._observables)))
        if parameter_values is None:
            parameter_values = [[]] * len(circuit_indices)

        # Allow objects
        try:
            circuit_indices = [
                next(_findname(circuit, self._circuit_ids))
                if not isinstance(circuit, (int, np.integer))
                else circuit
                for circuit in circuit_indices
            ]
            observable_indices = [
                next(_finditer(observable, self._observable_ids))
                if not isinstance(observable, (int, np.integer))
                else observable
                for observable in observable_indices
            ]
        except StopIteration as err:
            raise QiskitError("The object id does not match.") from err

        # Validation
        if len(circuit_indices) != len(observable_indices):
            raise QiskitError(
                f"The number of circuit indices ({len(circuit_indices)}) does not match "
                f"the number of observable indices ({len(observable_indices)})."
            )
        if len(circuit_indices) != len(parameter_values):
            raise QiskitError(
                f"The number of circuit indices ({len(circuit_indices)}) does not match "
                f"the number of parameter value sets ({len(parameter_values)})."
            )

        return self._call(
            circuit_indices=circuit_indices,
            observable_indices=observable_indices,
            parameter_values=parameter_values,
            **run_options,
        )

    @abstractmethod
    def _call(
        self,
        circuit_indices: Sequence[int],
        observable_indices: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:
        ...
