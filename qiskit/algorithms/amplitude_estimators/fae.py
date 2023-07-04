# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Faster Amplitude Estimation."""

from __future__ import annotations
import warnings
import numpy as np

from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.providers import Backend
from qiskit.primitives import BaseSampler
from qiskit.utils import QuantumInstance
from qiskit.utils.deprecation import deprecate_arg, deprecate_func
from qiskit.algorithms.exceptions import AlgorithmError

from .amplitude_estimator import AmplitudeEstimator, AmplitudeEstimatorResult
from .estimation_problem import EstimationProblem


class FasterAmplitudeEstimation(AmplitudeEstimator):
    """The Faster Amplitude Estimation algorithm.

    The Faster Amplitude Estimation (FAE) [1] algorithm is a variant of Quantum Amplitude
    Estimation (QAE), where the Quantum Phase Estimation (QPE) by an iterative Grover search,
    similar to [2].

    Due to the iterative version of the QPE, this algorithm does not require any additional
    qubits, as the originally proposed QAE [3] and thus the resulting circuits are less complex.

    References:

        [1]: K. Nakaji. Faster Amplitude Estimation, 2020;
            `arXiv:2002.02417 <https://arxiv.org/pdf/2003.02417.pdf>`_
        [2]: D. Grinko et al. Iterative Amplitude Estimation, 2019;
            `arXiv:1912.05559 <http://arxiv.org/abs/1912.05559>`_
        [3]: G. Brassard et al. Quantum Amplitude Amplification and Estimation, 2000;
            `arXiv:quant-ph/0005055 <http://arxiv.org/abs/quant-ph/0005055>`_

    """

    @deprecate_arg(
        "quantum_instance",
        additional_msg=(
            "Instead, use the ``sampler`` argument. See https://qisk.it/algo_migration for a "
            "migration guide."
        ),
        since="0.24.0",
    )
    def __init__(
        self,
        delta: float,
        maxiter: int,
        rescale: bool = True,
        quantum_instance: QuantumInstance | Backend | None = None,
        sampler: BaseSampler | None = None,
    ) -> None:
        r"""
        Args:
            delta: The probability that the true value is outside of the final confidence interval.
            maxiter: The number of iterations, the maximal power of Q is `2 ** (maxiter - 1)`.
            rescale: Whether to rescale the problem passed to `estimate`.
            quantum_instance: Deprecated: The quantum instance or backend
                to run the circuits.
            sampler: A sampler primitive to evaluate the circuits.

        .. note::

            This algorithm overwrites the number of shots set in the ``quantum_instance``
            argument, but will reset them to the initial number after running.

        """
        super().__init__()
        # set quantum instance
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.quantum_instance = quantum_instance
        self._shots = (int(1944 * np.log(2 / delta)), int(972 * np.log(2 / delta)))
        self._rescale = rescale
        self._delta = delta
        self._maxiter = maxiter
        self._num_oracle_calls = 0
        self._sampler = sampler

    @property
    def sampler(self) -> BaseSampler | None:
        """Get the sampler primitive.

        Returns:
            The sampler primitive to evaluate the circuits.
        """
        return self._sampler

    @sampler.setter
    def sampler(self, sampler: BaseSampler) -> None:
        """Set sampler primitive.

        Args:
            sampler: A sampler primitive to evaluate the circuits.
        """
        self._sampler = sampler

    @property
    @deprecate_func(
        since="0.24.0",
        is_property=True,
        additional_msg="See https://qisk.it/algo_migration for a migration guide.",
    )
    def quantum_instance(self) -> QuantumInstance | None:
        """Deprecated. Get the quantum instance.

        Returns:
            The quantum instance used to run this algorithm.
        """
        return self._quantum_instance

    @quantum_instance.setter
    @deprecate_func(
        since="0.24.0",
        is_property=True,
        additional_msg="See https://qisk.it/algo_migration for a migration guide.",
    )
    def quantum_instance(self, quantum_instance: QuantumInstance | Backend) -> None:
        """Deprecated. Set quantum instance.

        Args:
            quantum_instance: The quantum instance used to run this algorithm.
        """
        if isinstance(quantum_instance, Backend):
            quantum_instance = QuantumInstance(quantum_instance)
        self._quantum_instance = quantum_instance

    def _cos_estimate(self, estimation_problem, k, shots):
        if self._quantum_instance is None and self._sampler is None:
            raise ValueError("A quantum instance or sampler must be provided.")

        if self._sampler is not None:
            circuit = self.construct_circuit(estimation_problem, k, measurement=True)
            try:
                job = self._sampler.run([circuit], shots=shots)
                result = job.result()
            except Exception as exc:
                raise AlgorithmError("The job was not completed successfully. ") from exc

            if shots is None:
                shots = 1
            self._num_oracle_calls += (2 * k + 1) * shots

            # sum over all probabilities where the objective qubits are 1
            prob = 0
            for bit, probabilities in result.quasi_dists[0].binary_probabilities().items():
                # check if it is a good state
                if estimation_problem.is_good_state(bit):
                    prob += probabilities

            cos_estimate = 1 - 2 * prob
        elif self._quantum_instance.is_statevector:
            circuit = self.construct_circuit(estimation_problem, k, measurement=False)
            statevector = self._quantum_instance.execute(circuit).get_statevector()

            # sum over all amplitudes where the objective qubits are 1
            prob = 0
            for i, amplitude in enumerate(statevector):
                # get bitstring of objective qubits
                full_state = bin(i)[2:].zfill(circuit.num_qubits)[::-1]
                state = "".join([full_state[i] for i in estimation_problem.objective_qubits])

                # check if it is a good state
                if estimation_problem.is_good_state(state[::-1]):
                    prob = prob + np.abs(amplitude) ** 2

            cos_estimate = 1 - 2 * prob
        else:
            circuit = self.construct_circuit(estimation_problem, k, measurement=True)

            self._quantum_instance.run_config.shots = shots
            counts = self._quantum_instance.execute(circuit).get_counts()
            self._num_oracle_calls += (2 * k + 1) * shots

            good_counts = 0
            for state, count in counts.items():
                if estimation_problem.is_good_state(state):
                    good_counts += count

            cos_estimate = 1 - 2 * good_counts / shots

        return cos_estimate

    def _chernoff(self, cos, shots) -> list[float]:
        width = np.sqrt(np.log(2 / self._delta) * 12 / shots)
        confint = [np.maximum(-1, cos - width), np.minimum(1, cos + width)]
        return confint

    def construct_circuit(
        self, estimation_problem: EstimationProblem, k: int, measurement: bool = False
    ) -> QuantumCircuit | tuple[QuantumCircuit, list[int]]:
        r"""Construct the circuit :math:`Q^k X |0\rangle>`.

        The A operator is the unitary specifying the QAE problem and Q the associated Grover
        operator.

        Args:
            estimation_problem: The estimation problem for which to construct the circuit.
            k: The power of the Q operator.
            measurement: Boolean flag to indicate if measurements should be included in the
                circuits.

        Returns:
            The circuit :math:`Q^k X |0\rangle`.
        """
        num_qubits = max(
            estimation_problem.state_preparation.num_qubits,
            estimation_problem.grover_operator.num_qubits,
        )
        circuit = QuantumCircuit(num_qubits, name="circuit")

        # add classical register if needed
        if measurement:
            c = ClassicalRegister(len(estimation_problem.objective_qubits))
            circuit.add_register(c)

        # add A operator
        circuit.compose(estimation_problem.state_preparation, inplace=True)

        # add Q^k
        if k != 0:
            circuit.compose(estimation_problem.grover_operator.power(k), inplace=True)

            # add optional measurement
        if measurement:
            # real hardware can currently not handle operations after measurements, which might
            # happen if the circuit gets transpiled, hence we're adding a safeguard-barrier
            circuit.barrier()
            circuit.measure(estimation_problem.objective_qubits, c[:])

        return circuit

    def estimate(self, estimation_problem: EstimationProblem) -> "FasterAmplitudeEstimationResult":
        """Run the amplitude estimation algorithm on provided estimation problem.

        Args:
            estimation_problem: The estimation problem.

        Returns:
            An amplitude estimation results object.

        Raises:
            ValueError: A quantum instance or Sampler must be provided.
            AlgorithmError: Sampler run error.
        """
        if self._quantum_instance is None and self._sampler is None:
            raise ValueError("A quantum instance or sampler must be provided.")

        self._num_oracle_calls = 0
        user_defined_shots = (
            self._quantum_instance._run_config.shots if self._quantum_instance is not None else None
        )

        if self._rescale:
            problem = estimation_problem.rescale(0.25)
        else:
            problem = estimation_problem

        if self._quantum_instance is not None and self._quantum_instance.is_statevector:
            cos = self._cos_estimate(problem, k=0, shots=1)
            theta = np.arccos(cos) / 2
            theta_ci = [theta, theta]
            theta_cis = [theta_ci]
            num_steps = num_first_stage_steps = 1
        else:
            theta_ci = [0, np.arcsin(0.25)]
            first_stage = True
            j_0 = self._maxiter

            theta_cis = [theta_ci]
            num_first_stage_steps = 0
            num_steps = 0

            def cos_estimate(power, shots):
                return self._cos_estimate(problem, power, shots)

            for j in range(1, self._maxiter + 1):
                num_steps += 1
                if first_stage:
                    num_first_stage_steps += 1
                    c = cos_estimate(2 ** (j - 1), self._shots[0])
                    chernoff_ci = self._chernoff(c, self._shots[0])
                    theta_ci = [np.arccos(x) / (2 ** (j + 1) + 2) for x in chernoff_ci[::-1]]

                    if 2 ** (j + 1) * theta_ci[1] >= 3 * np.pi / 8 and j < self._maxiter:
                        j_0 = j
                        v = 2**j * np.sum(theta_ci)
                        first_stage = False
                else:
                    cos = cos_estimate(2 ** (j - 1), self._shots[1])
                    cos_2 = cos_estimate(2 ** (j - 1) + 2 ** (j_0 - 1), self._shots[1])
                    sin = (cos * np.cos(v) - cos_2) / np.sin(v)
                    rho = np.arctan2(sin, cos)
                    n = int(((2 ** (j + 1) + 2) * theta_ci[1] - rho + np.pi / 3) / (2 * np.pi))

                    theta_ci = [
                        (2 * np.pi * n + rho + sign * np.pi / 3) / (2 ** (j + 1) + 2)
                        for sign in [-1, 1]
                    ]
                theta_cis.append(theta_ci)

        theta = np.mean(theta_ci)
        rescaling = 4 if self._rescale else 1
        value = (rescaling * np.sin(theta)) ** 2
        value_ci = ((rescaling * np.sin(theta_ci[0])) ** 2, (rescaling * np.sin(theta_ci[1])) ** 2)

        result = FasterAmplitudeEstimationResult()
        result.num_oracle_queries = self._num_oracle_calls
        result.num_steps = num_steps
        result.num_first_state_steps = num_first_stage_steps
        if self._quantum_instance is not None and self._quantum_instance.is_statevector:
            result.success_probability = 1.0
        else:
            result.success_probability = 1 - (2 * self._maxiter - j_0) * self._delta

        result.estimation = value
        result.estimation_processed = problem.post_processing(value)
        result.confidence_interval = value_ci
        result.confidence_interval_processed = tuple(problem.post_processing(x) for x in value_ci)
        result.theta_intervals = theta_cis

        # reset shots to what the user had defined
        if self._quantum_instance is not None:
            self._quantum_instance._run_config.shots = user_defined_shots

        return result


class FasterAmplitudeEstimationResult(AmplitudeEstimatorResult):
    """The result object for the Faster Amplitude Estimation algorithm."""

    def __init__(self) -> None:
        super().__init__()
        self._success_probability: float | None = None
        self._num_steps: int | None = None
        self._num_first_state_steps: int | None = None
        self._theta_intervals: list[list[float]] | None = None

    @property
    def success_probability(self) -> float:
        """Return the success probability of the algorithm."""
        return self._success_probability

    @success_probability.setter
    def success_probability(self, probability: float) -> None:
        """Set the success probability of the algorithm."""
        self._success_probability = probability

    @property
    def num_steps(self) -> int:
        """Return the total number of steps taken in the algorithm."""
        return self._num_steps

    @num_steps.setter
    def num_steps(self, num_steps: int) -> None:
        """Set the total number of steps taken in the algorithm."""
        self._num_steps = num_steps

    @property
    def num_first_state_steps(self) -> int:
        """Return the number of steps taken in the first step of algorithm."""
        return self._num_first_state_steps

    @num_first_state_steps.setter
    def num_first_state_steps(self, num_steps: int) -> None:
        """Set the number of steps taken in the first step of algorithm."""
        self._num_first_state_steps = num_steps

    @property
    def theta_intervals(self) -> list[list[float]]:
        """Return the confidence intervals for the angles in each iteration."""
        return self._theta_intervals

    @theta_intervals.setter
    def theta_intervals(self, value: list[list[float]]) -> None:
        """Set the confidence intervals for the angles in each iteration."""
        self._theta_intervals = value
