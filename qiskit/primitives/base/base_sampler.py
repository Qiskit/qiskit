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

A sampler is initialized with an empty parameter set. The sampler is used to
create a :class:`~qiskit.providers.JobV1`, via the :meth:`qiskit.primitives.Sampler.run()`
method. This method is called with the following parameters

* quantum circuits (:math:`\psi_i(\theta)`): list of (parameterized) quantum circuits.
  (a list of :class:`~qiskit.circuit.QuantumCircuit` objects)

* parameter values (:math:`\theta_k`): list of sets of parameter values
  to be bound to the parameters of the quantum circuits.
  (list of list of float)

The method returns a :class:`~qiskit.providers.JobV1` object, calling
:meth:`qiskit.providers.JobV1.result()` yields a :class:`~qiskit.primitives.SamplerResult`
object, which contains probabilities or quasi-probabilities of bitstrings,
plus optional metadata like error bars in the samples.

Here is an example of how sampler is used.

.. code-block:: python

    from qiskit.primitives import Sampler
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RealAmplitudes

    # a Bell circuit
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure_all()

    # two parameterized circuits
    pqc = RealAmplitudes(num_qubits=2, reps=2)
    pqc.measure_all()
    pqc2 = RealAmplitudes(num_qubits=2, reps=3)
    pqc2.measure_all()

    theta1 = [0, 1, 1, 2, 3, 5]
    theta2 = [0, 1, 2, 3, 4, 5, 6, 7]

    # initialization of the sampler
    sampler = Sampler()

    # Sampler runs a job on the Bell circuit
    job = sampler.run(circuits=[bell], parameter_values=[[]], parameters=[[]])
    job_result = job.result()
    print([q.binary_probabilities() for q in job_result.quasi_dists])

    # Sampler runs a job on the parameterized circuits
    job2 = sampler.run(
        circuits=[pqc, pqc2],
        parameter_values=[theta1, theta2],
        parameters=[pqc.parameters, pqc2.parameters])
    job_result = job2.result()
    print([q.binary_probabilities() for q in job_result.quasi_dists])
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from copy import copy
from typing import Generic, TypeVar

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.parametertable import ParameterView
from qiskit.providers import JobV1 as Job

from .base_primitive import BasePrimitive

T = TypeVar("T", bound=Job)


class BaseSampler(BasePrimitive, Generic[T]):
    """Sampler base class

    Base class of Sampler that calculates quasi-probabilities of bitstrings from quantum circuits.
    """

    __hash__ = None

    def __init__(
        self,
        *,
        options: dict | None = None,
    ):
        """
        Args:
            options: Default options.
        """
        self._circuits = []
        self._parameters = []
        super().__init__(options)

    def run(
        self,
        circuits: QuantumCircuit | Sequence[QuantumCircuit],
        parameter_values: Sequence[float] | Sequence[Sequence[float]] | None = None,
        **run_options,
    ) -> T:
        """Run the job of the sampling of bitstrings.

        Args:
            circuits: One of more circuit objects.
            parameter_values: Parameters to be bound to the circuit.
            run_options: Backend runtime options used for circuit execution.

        Returns:
            The job object of the result of the sampler. The i-th result corresponds to
            ``circuits[i]`` evaluated with parameters bound as ``parameter_values[i]``.

        Raises:
            ValueError: Invalid arguments are given.
        """
        # Singular validation
        circuits = self._validate_circuits(circuits)
        parameter_values = self._validate_parameter_values(
            parameter_values,
            default=[()] * len(circuits),
        )

        # Cross-validation
        self._cross_validate_circuits_parameter_values(circuits, parameter_values)

        # Options
        run_opts = copy(self.options)
        run_opts.update_options(**run_options)

        return self._run(
            circuits,
            parameter_values,
            **run_opts.__dict__,
        )

    @abstractmethod
    def _run(
        self,
        circuits: tuple[QuantumCircuit, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ) -> T:
        raise NotImplementedError("The subclass of BaseSampler must implment `_run` method.")

    # TODO: validate measurement gates are present
    @classmethod
    def _validate_circuits(
        cls,
        circuits: Sequence[QuantumCircuit] | QuantumCircuit,
    ) -> tuple[QuantumCircuit, ...]:
        circuits = super()._validate_circuits(circuits)
        for i, circuit in enumerate(circuits):
            if circuit.num_clbits == 0:
                raise ValueError(
                    f"The {i}-th circuit does not have any classical bit. "
                    "Sampler requires classical bits, plus measurements "
                    "on the desired qubits."
                )
        return circuits

    @property
    def circuits(self) -> tuple[QuantumCircuit, ...]:
        """Quantum circuits to be sampled.

        Returns:
            The quantum circuits to be sampled.
        """
        return tuple(self._circuits)

    @property
    def parameters(self) -> tuple[ParameterView, ...]:
        """Parameters of quantum circuits.

        Returns:
            List of the parameters in each quantum circuit.
        """
        return tuple(self._parameters)
