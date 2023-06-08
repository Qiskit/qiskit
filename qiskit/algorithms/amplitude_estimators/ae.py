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

"""The Quantum Phase Estimation-based Amplitude Estimation algorithm."""

from __future__ import annotations
from collections import OrderedDict
import warnings
import numpy as np
from scipy.stats import chi2, norm
from scipy.optimize import bisect

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.providers import Backend
from qiskit.primitives import BaseSampler
from qiskit.utils import QuantumInstance
from qiskit.utils.deprecation import deprecate_arg, deprecate_func
from .amplitude_estimator import AmplitudeEstimator, AmplitudeEstimatorResult
from .ae_utils import pdf_a, derivative_log_pdf_a, bisect_max
from .estimation_problem import EstimationProblem
from ..exceptions import AlgorithmError


class AmplitudeEstimation(AmplitudeEstimator):
    r"""The Quantum Phase Estimation-based Amplitude Estimation algorithm.

    This class implements the original Quantum Amplitude Estimation (QAE) algorithm, introduced by
    [1]. This canonical version uses quantum phase estimation along with a set of :math:`m`
    additional evaluation qubits to find an estimate :math:`\tilde{a}`, that is restricted to the
    grid

    .. math::

        \tilde{a} \in \{\sin^2(\pi  y / 2^m) : y = 0, ..., 2^{m-1}\}

    More evaluation qubits produce a finer sampling grid, therefore the accuracy of the algorithm
    increases with :math:`m`.

    Using a maximum likelihood post processing, this grid constraint can be circumvented.
    This improved estimator is implemented as well, see [2] Appendix A for more detail.

    .. note::

        This class does not support the :attr:`.EstimationProblem.is_good_state` property,
        as for phase estimation-based QAE, the oracle that identifes the good states
        must be encoded in the Grover operator. To set custom oracles, the
        :attr:`.EstimationProblem.grover_operator` attribute can be set directly.

    References:
        [1]: Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000).
             Quantum Amplitude Amplification and Estimation.
             `arXiv:quant-ph/0005055 <http://arxiv.org/abs/quant-ph/0005055>`_.
        [2]: Grinko, D., Gacon, J., Zoufal, C., & Woerner, S. (2019).
             Iterative Quantum Amplitude Estimation.
             `arXiv:1912.05559 <https://arxiv.org/abs/1912.05559>`_.
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
        num_eval_qubits: int,
        phase_estimation_circuit: QuantumCircuit | None = None,
        iqft: QuantumCircuit | None = None,
        quantum_instance: QuantumInstance | Backend | None = None,
        sampler: BaseSampler | None = None,
    ) -> None:
        r"""
        Args:
            num_eval_qubits: The number of evaluation qubits.
            phase_estimation_circuit: The phase estimation circuit used to run the algorithm.
                Defaults to the standard phase estimation circuit from the circuit library,
                `qiskit.circuit.library.PhaseEstimation` when None.
            iqft: The inverse quantum Fourier transform component, defaults to using a standard
                implementation from `qiskit.circuit.library.QFT` when None.
            quantum_instance: Deprecated: The backend (or `QuantumInstance`) to execute
                the circuits on.
            sampler: A sampler primitive to evaluate the circuits.

        Raises:
            ValueError: If the number of evaluation qubits is smaller than 1.
        """
        if num_eval_qubits < 1:
            raise ValueError("The number of evaluation qubits must at least be 1.")

        super().__init__()

        # set quantum instance
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.quantum_instance = quantum_instance

        # get parameters
        self._m = num_eval_qubits
        self._M = 2**num_eval_qubits  # pylint: disable=invalid-name

        self._iqft = iqft
        self._pec = phase_estimation_circuit
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
        additional_msg="See https://qisk.it/algo_migration for a migration guide.",
        since="0.24.0",
        is_property=True,
    )
    def quantum_instance(self) -> QuantumInstance | None:
        """Deprecated: Get the quantum instance.

        Returns:
            The quantum instance used to run this algorithm.
        """
        return self._quantum_instance

    @quantum_instance.setter
    @deprecate_func(
        additional_msg="See https://qisk.it/algo_migration for a migration guide.",
        since="0.24.0",
        is_property=True,
    )
    def quantum_instance(self, quantum_instance: QuantumInstance | Backend) -> None:
        """Deprecated: Set quantum instance.

        Args:
            quantum_instance: The quantum instance used to run this algorithm.
        """
        if isinstance(quantum_instance, Backend):
            quantum_instance = QuantumInstance(quantum_instance)
        self._quantum_instance = quantum_instance

    def construct_circuit(
        self, estimation_problem: EstimationProblem, measurement: bool = False
    ) -> QuantumCircuit:
        """Construct the Amplitude Estimation quantum circuit.

        Args:
            estimation_problem: The estimation problem for which to construct the QAE circuit.
            measurement: Boolean flag to indicate if measurements should be included in the circuit.

        Returns:
            The QuantumCircuit object for the constructed circuit.
        """
        # use custom Phase Estimation circuit if provided
        if self._pec is not None:
            pec = self._pec

        # otherwise use the circuit library -- note that this does not include the A operator
        else:
            from qiskit.circuit.library import PhaseEstimation

            pec = PhaseEstimation(self._m, estimation_problem.grover_operator, iqft=self._iqft)

        # combine the Phase Estimation circuit with the A operator
        circuit = QuantumCircuit(*pec.qregs)
        circuit.compose(
            estimation_problem.state_preparation,
            list(range(self._m, circuit.num_qubits)),
            inplace=True,
        )
        circuit.compose(pec, inplace=True)

        # add measurements if necessary
        if measurement:
            cr = ClassicalRegister(self._m)
            circuit.add_register(cr)
            circuit.measure(list(range(self._m)), list(range(self._m)))

        return circuit

    def evaluate_measurements(
        self,
        circuit_results: dict[str, int] | np.ndarray,
        threshold: float = 1e-6,
    ) -> tuple[dict[float, float], dict[int, float]]:
        """Evaluate the results from the circuit simulation.

        Given the probabilities from statevector simulation of the QAE circuit, compute the
        probabilities that the measurements y/gridpoints a are the best estimate.

        Args:
            circuit_results: The circuit result from the QAE circuit. Can be either a counts dict
                or a statevector or a quasi-probabilities dict.
            threshold: Measurements with probabilities below the threshold are discarded.

        Returns:
            Dictionaries containing the a gridpoints with respective probabilities and
                y measurements with respective probabilities, in this order.
        """
        # compute grid sample and measurement dicts
        if isinstance(circuit_results, dict):
            if set(map(type, circuit_results.values())) == {int}:
                samples, measurements = self._evaluate_count_results(circuit_results)
            else:
                samples, measurements = self._evaluate_quasi_probabilities_results(circuit_results)
        else:
            samples, measurements = self._evaluate_statevector_results(circuit_results)

        # cutoff probabilities below the threshold
        samples = {a: p for a, p in samples.items() if p > threshold}
        measurements = {y: p for y, p in measurements.items() if p > threshold}

        return samples, measurements

    def _evaluate_statevector_results(self, statevector):
        # map measured results to estimates
        measurements = OrderedDict()  # type: OrderedDict
        num_qubits = int(np.log2(len(statevector)))
        for i, amplitude in enumerate(statevector):
            b = bin(i)[2:].zfill(num_qubits)[::-1]
            y = int(b[: self._m], 2)  # chop off all except the evaluation qubits
            measurements[y] = measurements.get(y, 0) + np.abs(amplitude) ** 2

        samples = OrderedDict()  # type: OrderedDict
        for y, probability in measurements.items():
            if y >= int(self._M / 2):
                y = self._M - y
            # due to the finite accuracy of the sine, we round the result to 7 decimals
            a = np.round(np.power(np.sin(y * np.pi / 2**self._m), 2), decimals=7)
            samples[a] = samples.get(a, 0) + probability

        return samples, measurements

    def _evaluate_quasi_probabilities_results(self, circuit_results):
        # construct probabilities
        measurements = OrderedDict()
        samples = OrderedDict()
        for state, probability in circuit_results.items():
            # reverts the last _m items
            y = int(state[: -self._m - 1 : -1], 2)
            measurements[y] = probability
            a = np.round(np.power(np.sin(y * np.pi / 2**self._m), 2), decimals=7)
            samples[a] = samples.get(a, 0.0) + probability

        return samples, measurements

    def _evaluate_count_results(self, counts) -> tuple[dict[float, float], dict[int, float]]:
        # construct probabilities
        measurements: dict[int, float] = OrderedDict()
        samples: dict[float, float] = OrderedDict()
        shots = sum(counts.values())
        for state, count in counts.items():
            y = int(state.replace(" ", "")[: self._m][::-1], 2)
            probability = count / shots
            measurements[y] = probability
            a = np.round(np.power(np.sin(y * np.pi / 2**self._m), 2), decimals=7)
            samples[a] = samples.get(a, 0.0) + probability

        return samples, measurements

    @staticmethod
    def compute_mle(
        result: "AmplitudeEstimationResult", apply_post_processing: bool = False
    ) -> float:
        """Compute the Maximum Likelihood Estimator (MLE).

        Args:
            result: An amplitude estimation result object.
            apply_post_processing: If True, apply the post processing to the MLE before returning
                it.

        Returns:
            The MLE for the provided result object.
        """
        m = result.num_evaluation_qubits
        M = 2**m  # pylint: disable=invalid-name
        qae = result.estimation

        # likelihood function
        a_i = np.asarray(list(result.samples.keys()))
        p_i = np.asarray(list(result.samples.values()))

        def loglikelihood(a):
            return np.sum(result.shots * p_i * np.log(pdf_a(a_i, a, m)))

        # y is pretty much an integer, but to map 1.9999 to 2 we must first
        # use round and then int conversion
        y = int(np.round(M * np.arcsin(np.sqrt(qae)) / np.pi))

        # Compute the two intervals in which are candidates for containing
        # the maximum of the log-likelihood function: the two bubbles next to
        # the QAE estimate
        if y == 0:
            right_of_qae = np.sin(np.pi * (y + 1) / M) ** 2
            bubbles = [qae, right_of_qae]

        elif y == int(M / 2):  # remember, M = 2^m is a power of 2
            left_of_qae = np.sin(np.pi * (y - 1) / M) ** 2
            bubbles = [left_of_qae, qae]

        else:
            left_of_qae = np.sin(np.pi * (y - 1) / M) ** 2
            right_of_qae = np.sin(np.pi * (y + 1) / M) ** 2
            bubbles = [left_of_qae, qae, right_of_qae]

        # Find global maximum amongst the two local maxima
        a_opt = qae
        loglik_opt = loglikelihood(a_opt)
        for a, b in zip(bubbles[:-1], bubbles[1:]):
            locmax, val = bisect_max(loglikelihood, a, b, retval=True)
            if val > loglik_opt:
                a_opt = locmax
                loglik_opt = val

        if apply_post_processing:
            return result.post_processing(a_opt)

        return a_opt

    def estimate(self, estimation_problem: EstimationProblem) -> "AmplitudeEstimationResult":
        """Run the amplitude estimation algorithm on provided estimation problem.

        Args:
            estimation_problem: The estimation problem.

        Returns:
            An amplitude estimation results object.

        Raises:
            ValueError: If `state_preparation` or `objective_qubits` are not set in the
                `estimation_problem`.
            ValueError: A quantum instance or sampler must be provided.
            AlgorithmError: Sampler job run error.
        """
        # check if A factory or state_preparation has been set
        if estimation_problem.state_preparation is None:
            raise ValueError(
                "The state_preparation property of the estimation problem must be set."
            )
        if self._quantum_instance is None and self._sampler is None:
            raise ValueError("A quantum instance or sampler must be provided.")

        if estimation_problem.objective_qubits is None:
            raise ValueError("The objective_qubits property of the estimation problem must be set.")

        if estimation_problem.has_good_state:
            warnings.warn(
                "The AmplitudeEstimation class does not support an is_good_state function to "
                "identify good states. For this algorithm, a custom oracle has to be encoded directly "
                "in the grover_operator. If no custom oracle is set, this algorithm identifies good "
                "states as those, where all objective qubits are in state 1."
            )

        result = AmplitudeEstimationResult()
        result.num_evaluation_qubits = self._m
        result.post_processing = estimation_problem.post_processing

        shots = 0
        if self._quantum_instance is not None and self._quantum_instance.is_statevector:
            circuit = self.construct_circuit(estimation_problem, measurement=False)
            # run circuit on statevector simulator
            statevector = self._quantum_instance.execute(circuit).get_statevector()
            result.circuit_results = statevector
            # store number of shots: convention is 1 shot for statevector,
            # needed so that MLE works!
            shots = 1
        else:
            circuit = self.construct_circuit(estimation_problem, measurement=True)
            if self._quantum_instance is not None:
                # run circuit on QASM simulator
                result.circuit_results = self._quantum_instance.execute(circuit).get_counts()
                shots = sum(result.circuit_results.values())
            else:
                try:
                    job = self._sampler.run([circuit])
                    ret = job.result()
                except Exception as exc:
                    raise AlgorithmError("The job was not completed successfully. ") from exc

                shots = ret.metadata[0].get("shots")
                if shots is None:
                    result.circuit_results = ret.quasi_dists[0].binary_probabilities()
                    shots = 1
                else:
                    result.circuit_results = {
                        k: round(v * shots)
                        for k, v in ret.quasi_dists[0].binary_probabilities().items()
                    }

        # store shots
        result.shots = shots
        samples, measurements = self.evaluate_measurements(result.circuit_results)

        result.samples = samples
        result.samples_processed = {
            estimation_problem.post_processing(a): p for a, p in samples.items()
        }
        result.measurements = measurements

        # determine the most likely estimate
        result.max_probability = 0
        for amplitude, (mapped, prob) in zip(samples.keys(), result.samples_processed.items()):
            if prob > result.max_probability:
                result.max_probability = prob
                result.estimation = amplitude
                result.estimation_processed = mapped

        # store the number of oracle queries
        result.num_oracle_queries = result.shots * (self._M - 1)

        # run the MLE post-processing
        mle = self.compute_mle(result)
        result.mle = mle
        result.mle_processed = estimation_problem.post_processing(mle)

        result.confidence_interval = self.compute_confidence_interval(result)
        result.confidence_interval_processed = tuple(
            estimation_problem.post_processing(value) for value in result.confidence_interval
        )

        return result

    @staticmethod
    def compute_confidence_interval(
        result: "AmplitudeEstimationResult", alpha: float = 0.05, kind: str = "likelihood_ratio"
    ) -> tuple[float, float]:
        """Compute the (1 - alpha) confidence interval.

        Args:
            result: An amplitude estimation result for which to compute the confidence interval.
            alpha: Confidence level: compute the (1 - alpha) confidence interval.
            kind: The method to compute the confidence interval, can be 'fisher', 'observed_fisher'
                or 'likelihood_ratio' (default)

        Returns:
            The (1 - alpha) confidence interval of the specified kind.

        Raises:
            AquaError: If 'mle' is not in self._ret.keys() (i.e. `run` was not called yet).
            NotImplementedError: If the confidence interval method `kind` is not implemented.
        """
        # if statevector simulator the estimate is exact
        if isinstance(result.circuit_results, (list, np.ndarray)):
            return (result.mle, result.mle)

        if kind in ["likelihood_ratio", "lr"]:
            return _likelihood_ratio_confint(result, alpha)

        if kind in ["fisher", "fi"]:
            return _fisher_confint(result, alpha, observed=False)

        if kind in ["observed_fisher", "observed_information", "oi"]:
            return _fisher_confint(result, alpha, observed=True)

        raise NotImplementedError(f"CI `{kind}` is not implemented.")


class AmplitudeEstimationResult(AmplitudeEstimatorResult):
    """The ``AmplitudeEstimation`` result object."""

    def __init__(self) -> None:
        super().__init__()
        self._num_evaluation_qubits: int | None = None
        self._mle: float | None = None
        self._mle_processed: float | None = None
        self._samples: dict[float, float] | None = None
        self._samples_processed: dict[float, float] | None = None
        self._y_measurements: dict[int, float] | None = None
        self._max_probability: float | None = None

    @property
    def num_evaluation_qubits(self) -> int:
        """Returns the number of evaluation qubits."""
        return self._num_evaluation_qubits

    @num_evaluation_qubits.setter
    def num_evaluation_qubits(self, num_evaluation_qubits: int) -> None:
        """Set the number of evaluation qubits."""
        self._num_evaluation_qubits = num_evaluation_qubits

    @property
    def mle_processed(self) -> float:
        """Return the post-processed MLE for the amplitude."""
        return self._mle_processed

    @mle_processed.setter
    def mle_processed(self, value: float) -> None:
        """Set the post-processed MLE for the amplitude."""
        self._mle_processed = value

    @property
    def samples_processed(self) -> dict[float, float]:
        """Return the post-processed measurement samples with their measurement probability."""
        return self._samples_processed

    @samples_processed.setter
    def samples_processed(self, value: dict[float, float]) -> None:
        """Set the post-processed measurement samples."""
        self._samples_processed = value

    @property
    def mle(self) -> float:
        r"""Return the MLE for the amplitude, in $[0, 1]$."""
        return self._mle

    @mle.setter
    def mle(self, value: float) -> None:
        r"""Set the MLE for the amplitude, in $[0, 1]$."""
        self._mle = value

    @property
    def samples(self) -> dict[float, float]:
        """Return the measurement samples with their measurement probability."""
        return self._samples

    @samples.setter
    def samples(self, value: dict[float, float]) -> None:
        """Set the measurement samples with their measurement probability."""
        self._samples = value

    @property
    def measurements(self) -> dict[int, float]:
        """Return the measurements as integers with their measurement probability."""
        return self._y_measurements

    @measurements.setter
    def measurements(self, value: dict[int, float]) -> None:
        """Set the measurements as integers with their measurement probability."""
        self._y_measurements = value

    @property
    def max_probability(self) -> float:
        """Return the maximum sampling probability."""
        return self._max_probability

    @max_probability.setter
    def max_probability(self, value: float) -> None:
        """Set the maximum sampling probability."""
        self._max_probability = value


def _compute_fisher_information(result: AmplitudeEstimationResult, observed: bool = False) -> float:
    """Computes the Fisher information for the output of the previous run.

    Args:
        result: An amplitude estimation result for which to compute the confidence interval.
        observed: If True, the observed Fisher information is returned, otherwise
            the expected Fisher information.

    Returns:
        The Fisher information.
    """
    fisher_information = None
    mlv = result.mle  # MLE in [0,1]
    m = result.num_evaluation_qubits
    M = 2**m  # pylint: disable=invalid-name

    if observed:
        a_i = np.asarray(list(result.samples.keys()))
        p_i = np.asarray(list(result.samples.values()))

        # Calculate the observed Fisher information
        fisher_information = sum(p * derivative_log_pdf_a(a, mlv, m) ** 2 for p, a in zip(p_i, a_i))
    else:

        def integrand(x):
            return (derivative_log_pdf_a(x, mlv, m)) ** 2 * pdf_a(x, mlv, m)

        grid = np.sin(np.pi * np.arange(M / 2 + 1) / M) ** 2
        fisher_information = sum(integrand(x) for x in grid)

    return fisher_information


def _fisher_confint(
    result: AmplitudeEstimationResult, alpha: float, observed: bool = False
) -> tuple[float, float]:
    """Compute the Fisher information confidence interval for the MLE of the previous run.

    Args:
        result: An amplitude estimation result for which to compute the confidence interval.
        alpha: Specifies the (1 - alpha) confidence level (0 < alpha < 1).
        observed: If True, the observed Fisher information is used to construct the
            confidence interval, otherwise the expected Fisher information.

    Returns:
        The Fisher information confidence interval.
    """
    # approximate the standard deviation of the MLE and construct the confidence interval
    std = np.sqrt(result.shots * _compute_fisher_information(result, observed))
    confint = result.mle + norm.ppf(1 - alpha / 2) / std * np.array([-1, 1])

    # transform the confidence interval from [0, 1] to the target interval
    return result.post_processing(confint[0]), result.post_processing(confint[1])


def _likelihood_ratio_confint(
    result: AmplitudeEstimationResult, alpha: float
) -> tuple[float, float]:
    """Compute the likelihood ratio confidence interval for the MLE of the previous run.

    Args:
        result: An amplitude estimation result for which to compute the confidence interval.
        alpha: Specifies the (1 - alpha) confidence level (0 < alpha < 1).

    Returns:
        The likelihood ratio confidence interval.
    """
    # Compute the two intervals in which we the look for values above
    # the likelihood ratio: the two bubbles next to the QAE estimate
    m = result.num_evaluation_qubits
    M = 2**m  # pylint: disable=invalid-name
    qae = result.estimation

    y = int(np.round(M * np.arcsin(np.sqrt(qae)) / np.pi))
    if y == 0:
        right_of_qae = np.sin(np.pi * (y + 1) / M) ** 2
        bubbles = [qae, right_of_qae]

    elif y == int(M / 2):  # remember, M = 2^m is a power of 2
        left_of_qae = np.sin(np.pi * (y - 1) / M) ** 2
        bubbles = [left_of_qae, qae]

    else:
        left_of_qae = np.sin(np.pi * (y - 1) / M) ** 2
        right_of_qae = np.sin(np.pi * (y + 1) / M) ** 2
        bubbles = [left_of_qae, qae, right_of_qae]

    # likelihood function
    a_i = np.asarray(list(result.samples.keys()))
    p_i = np.asarray(list(result.samples.values()))

    def loglikelihood(a):
        return np.sum(result.shots * p_i * np.log(pdf_a(a_i, a, m)))

    # The threshold above which the likelihoods are in the
    # confidence interval
    loglik_mle = loglikelihood(result.mle)
    thres = loglik_mle - chi2.ppf(1 - alpha, df=1) / 2

    def cut(x):
        return loglikelihood(x) - thres

    # Store the boundaries of the confidence interval
    # It's valid to start off with the zero-width confidence interval, since the maximum
    # of the likelihood function is guaranteed to be over the threshold, and if alpha = 0
    # that's the valid interval
    lower = upper = result.mle

    # Check the two intervals/bubbles: check if they surpass the
    # threshold and if yes add the part that does to the CI
    for a, b in zip(bubbles[:-1], bubbles[1:]):
        # Compute local maximum and perform a bisect search between
        # the local maximum and the bubble boundaries
        locmax, val = bisect_max(loglikelihood, a, b, retval=True)
        if val >= thres:
            # Bisect pre-condition is that the function has different
            # signs at the boundaries of the interval we search in
            if cut(a) * cut(locmax) < 0:
                left = bisect(cut, a, locmax)
                lower = np.minimum(lower, left)
            if cut(locmax) * cut(b) < 0:
                right = bisect(cut, locmax, b)
                upper = np.maximum(upper, right)

    # Put together CI
    return result.post_processing(lower), result.post_processing(upper)
