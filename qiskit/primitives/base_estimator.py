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

        # Objects can be passed instead of indices.
        # calculate [ <psi2(theta2)|H2|psi2(theta2)> ]
        # Note that passing objects has an overhead
        # since the corresponding indices need to be searched.
        result6 = e([psi2], [H2], [theta2])
        print(result6)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from copy import copy

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.parametertable import ParameterView
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.utils.deprecation import deprecate_arguments

from .estimator_result import EstimatorResult
from .utils import _finditer


class BaseEstimator(ABC):
    """Estimator base class.

    Base class for Estimator that estimates expectation values of quantum circuits and observables.
    """

    def __init__(
        self,
        circuits: Iterable[QuantumCircuit] | QuantumCircuit,
        observables: Iterable[SparsePauliOp] | SparsePauliOp,
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
        if isinstance(circuits, QuantumCircuit):
            circuits = (circuits,)
        self._circuits = tuple(circuits)

        if isinstance(observables, SparsePauliOp):
            observables = (observables,)
        self._observables = tuple(observables)

        # To guarantee that they exist as instance variable.
        # With only dynamic set, the python will not know if the attribute exists or not.
        self._circuit_ids = self._circuit_ids
        self._observable_ids = self._observable_ids

        if parameters is None:
            self._parameters = tuple(circ.parameters for circ in self._circuits)
        else:
            self._parameters = tuple(ParameterView(par) for par in parameters)
            if len(self._parameters) != len(self._circuits):
                raise QiskitError(
                    f"Different number of parameters ({len(self._parameters)}) and "
                    f"circuits ({len(self._circuits)})"
                )
            for i, (circ, params) in enumerate(zip(self._circuits, self._parameters)):
                if circ.num_parameters != len(params):
                    raise QiskitError(
                        f"Different numbers of parameters of {i}-th circuit: "
                        f"expected {circ.num_parameters}, actual {len(params)}."
                    )

    def __new__(
        cls,
        circuits: Iterable[QuantumCircuit] | QuantumCircuit,
        observables: Iterable[SparsePauliOp] | SparsePauliOp,
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
        if isinstance(observables, Iterable):
            observables = copy(observables)
            self._observable_ids = [id(observable) for observable in observables]
        else:
            self._observable_ids = [id(observables)]
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

    @deprecate_arguments({"circuit_indices": "circuits", "observable_indices": "observables"})
    def __call__(
        self,
        circuits: Sequence[int | QuantumCircuit],
        observables: Sequence[int | SparsePauliOp],
        parameter_values: Sequence[Sequence[float]] | None = None,
        **run_options,
    ) -> EstimatorResult:
        """Run the estimation of expectation value(s).

        ``circuits``, ``observables``, and ``parameter_values`` should have the same
        length. The i-th element of the result is the expectation of observable

        .. code-block:: python

            obs = self.observables[observables[i]]

        for the state prepared by

        .. code-block:: python

            circ = self.circuits[circuits[i]]

        with bound parameters

        .. code-block:: python

            values = parameter_values[i].

        Args:
            circuits: the list of circuit indices or circuit objects.
            observables: the list of observable indices or observable objects.
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
                "The circuits passed when calling estimator is not one of the circuits used to "
                "initialize the session."
            ) from err
        try:
            observables = [
                next(_finditer(id(observable), self._observable_ids))
                if not isinstance(observable, (int, np.integer))
                else observable
                for observable in observables
            ]
        except StopIteration as err:
            raise QiskitError(
                "The observables passed when calling estimator is not one of the observables used to "
                "initialize the session."
            ) from err

        # Allow optional
        if parameter_values is None:
            for i in circuits:
                if len(self._circuits[i].parameters) != 0:
                    raise QiskitError(
                        f"The {i}-th circuit is parameterised,"
                        "but parameter values are not given."
                    )
            parameter_values = [[]] * len(circuits)

        # Validation
        if len(circuits) != len(observables):
            raise QiskitError(
                f"The number of circuits ({len(circuits)}) does not match "
                f"the number of observables ({len(observables)})."
            )
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

        for circ_i, obs_i in zip(circuits, observables):
            circuit_num_qubits = self.circuits[circ_i].num_qubits
            observable_num_qubits = self.observables[obs_i].num_qubits
            if circuit_num_qubits != observable_num_qubits:
                raise QiskitError(
                    f"The number of qubits of the {circ_i}-th circuit ({circuit_num_qubits}) does "
                    f"not match the number of qubits of the {obs_i}-th observable "
                    f"({observable_num_qubits})."
                )

        if max(circuits) >= len(self.circuits):
            raise QiskitError(
                f"The number of circuits is {len(self.circuits)}, "
                f"but the index {max(circuits)} is given."
            )
        if max(observables) >= len(self.observables):
            raise QiskitError(
                f"The number of circuits is {len(self.observables)}, "
                f"but the index {max(observables)} is given."
            )

        return self._call(
            circuits=circuits,
            observables=observables,
            parameter_values=parameter_values,
            **run_options,
        )

    @abstractmethod
    def _call(
        self,
        circuits: Sequence[int],
        observables: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:
        ...
