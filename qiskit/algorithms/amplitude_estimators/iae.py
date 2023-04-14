# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Iterative Quantum Amplitude Estimation Algorithm."""

from __future__ import annotations
from typing import cast
import warnings
import numpy as np
from scipy.stats import beta

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.providers import Backend
from qiskit.primitives import BaseSampler
from qiskit.utils import QuantumInstance
from qiskit.utils.deprecation import deprecate_arg, deprecate_func

from .amplitude_estimator import AmplitudeEstimator, AmplitudeEstimatorResult
from .estimation_problem import EstimationProblem
from ..exceptions import AlgorithmError


class IterativeAmplitudeEstimation(AmplitudeEstimator):
    r"""The Iterative Amplitude Estimation algorithm.

    This class implements the Iterative Quantum Amplitude Estimation (IQAE) algorithm, proposed
    in [1]. The output of the algorithm is an estimate that,
    with at least probability :math:`1 - \alpha`, differs by epsilon to the target value, where
    both alpha and epsilon can be specified.

    It differs from the original QAE algorithm proposed by Brassard [2] in that it does not rely on
    Quantum Phase Estimation, but is only based on Grover's algorithm. IQAE iteratively
    applies carefully selected Grover iterations to find an estimate for the target amplitude.

    References:
        [1]: Grinko, D., Gacon, J., Zoufal, C., & Woerner, S. (2019).
             Iterative Quantum Amplitude Estimation.
             `arXiv:1912.05559 <https://arxiv.org/abs/1912.05559>`_.
        [2]: Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000).
             Quantum Amplitude Amplification and Estimation.
             `arXiv:quant-ph/0005055 <http://arxiv.org/abs/quant-ph/0005055>`_.
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
        epsilon_target: float,
        alpha: float,
        confint_method: str = "beta",
        min_ratio: float = 2,
        quantum_instance: QuantumInstance | Backend | None = None,
        sampler: BaseSampler | None = None,
    ) -> None:
        r"""
        The output of the algorithm is an estimate for the amplitude `a`, that with at least
        probability 1 - alpha has an error of epsilon. The number of A operator calls scales
        linearly in 1/epsilon (up to a logarithmic factor).

        Args:
            epsilon_target: Target precision for estimation target `a`, has values between 0 and 0.5
            alpha: Confidence level, the target probability is 1 - alpha, has values between 0 and 1
            confint_method: Statistical method used to estimate the confidence intervals in
                each iteration, can be 'chernoff' for the Chernoff intervals or 'beta' for the
                Clopper-Pearson intervals (default)
            min_ratio: Minimal q-ratio (:math:`K_{i+1} / K_i`) for FindNextK
            quantum_instance: Deprecated: Quantum Instance or Backend
            sampler: A sampler primitive to evaluate the circuits.

        Raises:
            AlgorithmError: if the method to compute the confidence intervals is not supported
            ValueError: If the target epsilon is not in (0, 0.5]
            ValueError: If alpha is not in (0, 1)
            ValueError: If confint_method is not supported
        """
        # validate ranges of input arguments
        if not 0 < epsilon_target <= 0.5:
            raise ValueError(f"The target epsilon must be in (0, 0.5], but is {epsilon_target}.")

        if not 0 < alpha < 1:
            raise ValueError(f"The confidence level alpha must be in (0, 1), but is {alpha}")

        if confint_method not in {"chernoff", "beta"}:
            raise ValueError(
                f"The confidence interval method must be chernoff or beta, but is {confint_method}."
            )

        super().__init__()

        # set quantum instance
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.quantum_instance = quantum_instance

        # store parameters
        self._epsilon = epsilon_target
        self._alpha = alpha
        self._min_ratio = min_ratio
        self._confint_method = confint_method
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

    @property
    def epsilon_target(self) -> float:
        """Returns the target precision ``epsilon_target`` of the algorithm.

        Returns:
            The target precision (which is half the width of the confidence interval).
        """
        return self._epsilon

    @epsilon_target.setter
    def epsilon_target(self, epsilon: float) -> None:
        """Set the target precision of the algorithm.

        Args:
            epsilon: Target precision for estimation target `a`.
        """
        self._epsilon = epsilon

    def _find_next_k(
        self,
        k: int,
        upper_half_circle: bool,
        theta_interval: tuple[float, float],
        min_ratio: float = 2.0,
    ) -> tuple[int, bool]:
        """Find the largest integer k_next, such that the interval (4 * k_next + 2)*theta_interval
        lies completely in [0, pi] or [pi, 2pi], for theta_interval = (theta_lower, theta_upper).

        Args:
            k: The current power of the Q operator.
            upper_half_circle: Boolean flag of whether theta_interval lies in the
                upper half-circle [0, pi] or in the lower one [pi, 2pi].
            theta_interval: The current confidence interval for the angle theta,
                i.e. (theta_lower, theta_upper).
            min_ratio: Minimal ratio K/K_next allowed in the algorithm.

        Returns:
            The next power k, and boolean flag for the extrapolated interval.

        Raises:
            AlgorithmError: if min_ratio is smaller or equal to 1
        """
        if min_ratio <= 1:
            raise AlgorithmError("min_ratio must be larger than 1 to ensure convergence")

        # initialize variables
        theta_l, theta_u = theta_interval
        old_scaling = 4 * k + 2  # current scaling factor, called K := (4k + 2)

        # the largest feasible scaling factor K cannot be larger than K_max,
        # which is bounded by the length of the current confidence interval
        max_scaling = int(1 / (2 * (theta_u - theta_l)))
        scaling = max_scaling - (max_scaling - 2) % 4  # bring into the form 4 * k_max + 2

        # find the largest feasible scaling factor K_next, and thus k_next
        while scaling >= min_ratio * old_scaling:
            theta_min = scaling * theta_l - int(scaling * theta_l)
            theta_max = scaling * theta_u - int(scaling * theta_u)

            if theta_min <= theta_max <= 0.5 and theta_min <= 0.5:
                # the extrapolated theta interval is in the upper half-circle
                upper_half_circle = True
                return int((scaling - 2) / 4), upper_half_circle

            elif theta_max >= 0.5 and theta_max >= theta_min >= 0.5:
                # the extrapolated theta interval is in the upper half-circle
                upper_half_circle = False
                return int((scaling - 2) / 4), upper_half_circle

            scaling -= 4

        # if we do not find a feasible k, return the old one
        return int(k), upper_half_circle

    def construct_circuit(
        self, estimation_problem: EstimationProblem, k: int = 0, measurement: bool = False
    ) -> QuantumCircuit:
        r"""Construct the circuit :math:`\mathcal{Q}^k \mathcal{A} |0\rangle`.

        The A operator is the unitary specifying the QAE problem and Q the associated Grover
        operator.

        Args:
            estimation_problem: The estimation problem for which to construct the QAE circuit.
            k: The power of the Q operator.
            measurement: Boolean flag to indicate if measurements should be included in the
                circuits.

        Returns:
            The circuit implementing :math:`\mathcal{Q}^k \mathcal{A} |0\rangle`.
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

    def _good_state_probability(
        self,
        problem: EstimationProblem,
        counts_or_statevector: dict[str, int] | np.ndarray,
        num_state_qubits: int,
    ) -> tuple[int, float] | float:
        """Get the probability to measure '1' in the last qubit.

        Args:
            problem: The estimation problem, used to obtain the number of objective qubits and
                the ``is_good_state`` function.
            counts_or_statevector: Either a counts-dictionary (with one measured qubit only!) or
                the statevector returned from the statevector_simulator.
            num_state_qubits: The number of state qubits.

        Returns:
            If a dict is given, return (#one-counts, #one-counts/#all-counts),
            otherwise Pr(measure '1' in the last qubit).
        """
        if isinstance(counts_or_statevector, dict):
            one_counts = 0
            for state, counts in counts_or_statevector.items():
                if problem.is_good_state(state):
                    one_counts += counts

            return int(one_counts), one_counts / sum(counts_or_statevector.values())
        else:
            statevector = counts_or_statevector
            num_qubits = int(np.log2(len(statevector)))  # the total number of qubits

            # sum over all amplitudes where the objective qubit is 1
            prob = 0
            for i, amplitude in enumerate(statevector):
                # consider only state qubits and revert bit order
                bitstr = bin(i)[2:].zfill(num_qubits)[-num_state_qubits:][::-1]
                objectives = [bitstr[index] for index in problem.objective_qubits]
                if problem.is_good_state(objectives):
                    prob = prob + np.abs(amplitude) ** 2

            return prob

    def estimate(
        self, estimation_problem: EstimationProblem
    ) -> "IterativeAmplitudeEstimationResult":
        """Run the amplitude estimation algorithm on provided estimation problem.

        Args:
            estimation_problem: The estimation problem.

        Returns:
            An amplitude estimation results object.

        Raises:
            ValueError: A quantum instance or Sampler must be provided.
            AlgorithmError: Sampler job run error.
        """
        if self._quantum_instance is None and self._sampler is None:
            raise ValueError("A quantum instance or sampler must be provided.")

        # initialize memory variables
        powers = [0]  # list of powers k: Q^k, (called 'k' in paper)
        ratios = []  # list of multiplication factors (called 'q' in paper)
        theta_intervals = [[0, 1 / 4]]  # a priori knowledge of theta / 2 / pi
        a_intervals = [[0.0, 1.0]]  # a priori knowledge of the confidence interval of the estimate
        num_oracle_queries = 0
        num_one_shots = []

        # maximum number of rounds
        max_rounds = (
            int(np.log(self._min_ratio * np.pi / 8 / self._epsilon) / np.log(self._min_ratio)) + 1
        )
        upper_half_circle = True  # initially theta is in the upper half-circle

        if self._quantum_instance is not None and self._quantum_instance.is_statevector:
            # for statevector we can directly return the probability to measure 1
            # note, that no iterations here are necessary
            # simulate circuit
            circuit = self.construct_circuit(estimation_problem, k=0, measurement=False)
            ret = self._quantum_instance.execute(circuit)

            # get statevector
            statevector = ret.get_statevector(circuit)

            # calculate the probability of measuring '1'
            num_qubits = circuit.num_qubits - circuit.num_ancillas
            prob = self._good_state_probability(estimation_problem, statevector, num_qubits)
            prob = cast(float, prob)  # tell MyPy it's a float and not Tuple[int, float ]

            a_confidence_interval = [prob, prob]  # type: list[float]
            a_intervals.append(a_confidence_interval)

            theta_i_interval = [
                np.arccos(1 - 2 * a_i) / 2 / np.pi for a_i in a_confidence_interval  # type: ignore
            ]
            theta_intervals.append(theta_i_interval)
            num_oracle_queries = 0  # no Q-oracle call, only a single one to A

        else:
            num_iterations = 0  # keep track of the number of iterations
            # number of shots per iteration
            shots = 0
            # do while loop, keep in mind that we scaled theta mod 2pi such that it lies in [0,1]
            while theta_intervals[-1][1] - theta_intervals[-1][0] > self._epsilon / np.pi:
                num_iterations += 1

                # get the next k
                k, upper_half_circle = self._find_next_k(
                    powers[-1],
                    upper_half_circle,
                    theta_intervals[-1],  # type: ignore
                    min_ratio=self._min_ratio,
                )

                # store the variables
                powers.append(k)
                ratios.append((2 * powers[-1] + 1) / (2 * powers[-2] + 1))

                # run measurements for Q^k A|0> circuit
                circuit = self.construct_circuit(estimation_problem, k, measurement=True)
                counts = {}
                if self._quantum_instance is not None:
                    ret = self._quantum_instance.execute(circuit)
                    # get the counts and store them
                    counts = ret.get_counts(circuit)
                    shots = self._quantum_instance._run_config.shots
                else:
                    try:
                        job = self._sampler.run([circuit])
                        ret = job.result()
                    except Exception as exc:
                        raise AlgorithmError("The job was not completed successfully. ") from exc

                    shots = ret.metadata[0].get("shots")
                    if shots is None:
                        circuit = self.construct_circuit(estimation_problem, k=0, measurement=True)
                        try:
                            job = self._sampler.run([circuit])
                            ret = job.result()
                        except Exception as exc:
                            raise AlgorithmError(
                                "The job was not completed successfully. "
                            ) from exc

                        # calculate the probability of measuring '1'
                        prob = 0.0
                        for bit, probabilities in ret.quasi_dists[0].binary_probabilities().items():
                            # check if it is a good state
                            if estimation_problem.is_good_state(bit):
                                prob += probabilities

                        a_confidence_interval = [prob, prob]
                        a_intervals.append(a_confidence_interval)

                        theta_i_interval = [
                            np.arccos(1 - 2 * a_i) / 2 / np.pi for a_i in a_confidence_interval
                        ]
                        theta_intervals.append(theta_i_interval)
                        num_oracle_queries = 0  # no Q-oracle call, only a single one to A
                        break

                    counts = {
                        k: round(v * shots)
                        for k, v in ret.quasi_dists[0].binary_probabilities().items()
                    }

                # calculate the probability of measuring '1', 'prob' is a_i in the paper
                num_qubits = circuit.num_qubits - circuit.num_ancillas
                # type: ignore
                one_counts, prob = self._good_state_probability(
                    estimation_problem, counts, num_qubits
                )

                num_one_shots.append(one_counts)

                # track number of Q-oracle calls
                num_oracle_queries += shots * k

                # if on the previous iterations we have K_{i-1} == K_i, we sum these samples up
                j = 1  # number of times we stayed fixed at the same K
                round_shots = shots
                round_one_counts = one_counts
                if num_iterations > 1:
                    while (
                        powers[num_iterations - j] == powers[num_iterations]
                        and num_iterations >= j + 1
                    ):
                        j = j + 1
                        round_shots += shots
                        round_one_counts += num_one_shots[-j]

                # compute a_min_i, a_max_i
                if self._confint_method == "chernoff":
                    a_i_min, a_i_max = _chernoff_confint(prob, round_shots, max_rounds, self._alpha)
                else:  # 'beta'
                    a_i_min, a_i_max = _clopper_pearson_confint(
                        round_one_counts, round_shots, self._alpha / max_rounds
                    )

                # compute theta_min_i, theta_max_i
                if upper_half_circle:
                    theta_min_i = np.arccos(1 - 2 * a_i_min) / 2 / np.pi
                    theta_max_i = np.arccos(1 - 2 * a_i_max) / 2 / np.pi
                else:
                    theta_min_i = 1 - np.arccos(1 - 2 * a_i_max) / 2 / np.pi
                    theta_max_i = 1 - np.arccos(1 - 2 * a_i_min) / 2 / np.pi

                # compute theta_u, theta_l of this iteration
                scaling = 4 * k + 2  # current K_i factor
                theta_u = (int(scaling * theta_intervals[-1][1]) + theta_max_i) / scaling
                theta_l = (int(scaling * theta_intervals[-1][0]) + theta_min_i) / scaling
                theta_intervals.append([theta_l, theta_u])

                # compute a_u_i, a_l_i
                a_u = np.sin(2 * np.pi * theta_u) ** 2
                a_l = np.sin(2 * np.pi * theta_l) ** 2
                a_u = cast(float, a_u)
                a_l = cast(float, a_l)
                a_intervals.append([a_l, a_u])

        # get the latest confidence interval for the estimate of a
        confidence_interval = tuple(a_intervals[-1])

        # the final estimate is the mean of the confidence interval
        estimation = np.mean(confidence_interval)

        result = IterativeAmplitudeEstimationResult()
        result.alpha = self._alpha
        result.post_processing = estimation_problem.post_processing
        result.num_oracle_queries = num_oracle_queries

        result.estimation = estimation
        result.epsilon_estimated = (confidence_interval[1] - confidence_interval[0]) / 2
        result.confidence_interval = confidence_interval

        result.estimation_processed = estimation_problem.post_processing(estimation)
        confidence_interval = tuple(
            estimation_problem.post_processing(x) for x in confidence_interval
        )
        result.confidence_interval_processed = confidence_interval
        result.epsilon_estimated_processed = (confidence_interval[1] - confidence_interval[0]) / 2
        result.estimate_intervals = a_intervals
        result.theta_intervals = theta_intervals
        result.powers = powers
        result.ratios = ratios

        return result


class IterativeAmplitudeEstimationResult(AmplitudeEstimatorResult):
    """The ``IterativeAmplitudeEstimation`` result object."""

    def __init__(self) -> None:
        super().__init__()
        self._alpha: float | None = None
        self._epsilon_target: float | None = None
        self._epsilon_estimated: float | None = None
        self._epsilon_estimated_processed: float | None = None
        self._estimate_intervals: list[list[float]] | None = None
        self._theta_intervals: list[list[float]] | None = None
        self._powers: list[int] | None = None
        self._ratios: list[float] | None = None
        self._confidence_interval_processed: tuple[float, float] | None = None

    @property
    def alpha(self) -> float:
        r"""Return the confidence level :math:`\alpha`."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        r"""Set the confidence level :math:`\alpha`."""
        self._alpha = value

    @property
    def epsilon_target(self) -> float:
        """Return the target half-width of the confidence interval."""
        return self._epsilon_target

    @epsilon_target.setter
    def epsilon_target(self, value: float) -> None:
        """Set the target half-width of the confidence interval."""
        self._epsilon_target = value

    @property
    def epsilon_estimated(self) -> float:
        """Return the estimated half-width of the confidence interval."""
        return self._epsilon_estimated

    @epsilon_estimated.setter
    def epsilon_estimated(self, value: float) -> None:
        """Set the estimated half-width of the confidence interval."""
        self._epsilon_estimated = value

    @property
    def epsilon_estimated_processed(self) -> float:
        """Return the post-processed estimated half-width of the confidence interval."""
        return self._epsilon_estimated_processed

    @epsilon_estimated_processed.setter
    def epsilon_estimated_processed(self, value: float) -> None:
        """Set the post-processed estimated half-width of the confidence interval."""
        self._epsilon_estimated_processed = value

    @property
    def estimate_intervals(self) -> list[list[float]]:
        """Return the confidence intervals for the estimate in each iteration."""
        return self._estimate_intervals

    @estimate_intervals.setter
    def estimate_intervals(self, value: list[list[float]]) -> None:
        """Set the confidence intervals for the estimate in each iteration."""
        self._estimate_intervals = value

    @property
    def theta_intervals(self) -> list[list[float]]:
        """Return the confidence intervals for the angles in each iteration."""
        return self._theta_intervals

    @theta_intervals.setter
    def theta_intervals(self, value: list[list[float]]) -> None:
        """Set the confidence intervals for the angles in each iteration."""
        self._theta_intervals = value

    @property
    def powers(self) -> list[int]:
        """Return the powers of the Grover operator in each iteration."""
        return self._powers

    @powers.setter
    def powers(self, value: list[int]) -> None:
        """Set the powers of the Grover operator in each iteration."""
        self._powers = value

    @property
    def ratios(self) -> list[float]:
        r"""Return the ratios :math:`K_{i+1}/K_{i}` for each iteration :math:`i`."""
        return self._ratios

    @ratios.setter
    def ratios(self, value: list[float]) -> None:
        r"""Set the ratios :math:`K_{i+1}/K_{i}` for each iteration :math:`i`."""
        self._ratios = value

    @property
    def confidence_interval_processed(self) -> tuple[float, float]:
        """Return the post-processed confidence interval."""
        return self._confidence_interval_processed

    @confidence_interval_processed.setter
    def confidence_interval_processed(self, value: tuple[float, float]) -> None:
        """Set the post-processed confidence interval."""
        self._confidence_interval_processed = value


def _chernoff_confint(
    value: float, shots: int, max_rounds: int, alpha: float
) -> tuple[float, float]:
    """Compute the Chernoff confidence interval for `shots` i.i.d. Bernoulli trials.

    The confidence interval is

        [value - eps, value + eps], where eps = sqrt(3 * log(2 * max_rounds/ alpha) / shots)

    but at most [0, 1].

    Args:
        value: The current estimate.
        shots: The number of shots.
        max_rounds: The maximum number of rounds, used to compute epsilon_a.
        alpha: The confidence level, used to compute epsilon_a.

    Returns:
        The Chernoff confidence interval.
    """
    eps = np.sqrt(3 * np.log(2 * max_rounds / alpha) / shots)
    lower = np.maximum(0, value - eps)
    upper = np.minimum(1, value + eps)
    return lower, upper


def _clopper_pearson_confint(counts: int, shots: int, alpha: float) -> tuple[float, float]:
    """Compute the Clopper-Pearson confidence interval for `shots` i.i.d. Bernoulli trials.

    Args:
        counts: The number of positive counts.
        shots: The number of shots.
        alpha: The confidence level for the confidence interval.

    Returns:
        The Clopper-Pearson confidence interval.
    """
    lower, upper = 0, 1

    # if counts == 0, the beta quantile returns nan
    if counts != 0:
        lower = beta.ppf(alpha / 2, counts, shots - counts + 1)

    # if counts == shots, the beta quantile returns nan
    if counts != shots:
        upper = beta.ppf(1 - alpha / 2, counts + 1, shots - counts)

    return lower, upper
