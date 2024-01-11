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
"""
Expectation value class
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from itertools import accumulate

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.opflow import PauliSumOp
from qiskit.providers import BackendV1, BackendV2, Options
from qiskit.quantum_info import Pauli, PauliList
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result import Counts, Result
from qiskit.transpiler import PassManager

from .base import BaseEstimator, EstimatorResult
from .primitive_job import PrimitiveJob
from .utils import _circuit_key, _observable_key, init_observable


def _run_circuits(
    circuits: QuantumCircuit | list[QuantumCircuit],
    backend: BackendV1 | BackendV2,
    **run_options,
) -> tuple[Result, list[dict]]:
    """Remove metadata of circuits and run the circuits on a backend.
    Args:
        circuits: The circuits
        backend: The backend
        monitor: Enable job minotor if True
        **run_options: run_options
    Returns:
        The result and the metadata of the circuits
    """
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
    metadata = []
    for circ in circuits:
        metadata.append(circ.metadata)
        circ.metadata = {}
    if isinstance(backend, BackendV1):
        max_circuits = getattr(backend.configuration(), "max_experiments", None)
    elif isinstance(backend, BackendV2):
        max_circuits = backend.max_circuits
    if max_circuits:
        jobs = [
            backend.run(circuits[pos : pos + max_circuits], **run_options)
            for pos in range(0, len(circuits), max_circuits)
        ]
        result = [x.result() for x in jobs]
    else:
        result = [backend.run(circuits, **run_options).result()]
    return result, metadata


def _prepare_counts(results):
    counts = []
    for res in results:
        count = res.get_counts()
        if not isinstance(count, list):
            count = [count]
        counts.extend(count)
    return counts


class BackendEstimator(BaseEstimator):
    """Evaluates expectation value using Pauli rotation gates.

    The :class:`~.BackendEstimator` class is a generic implementation of the
    :class:`~.BaseEstimator` interface that is used to wrap a :class:`~.BackendV2`
    (or :class:`~.BackendV1`) object in the :class:`~.BaseEstimator` API. It
    facilitates using backends that do not provide a native
    :class:`~.BaseEstimator` implementation in places that work with
    :class:`~.BaseEstimator`, such as algorithms in :mod:`qiskit.algorithms`
    including :class:`~.qiskit.algorithms.minimum_eigensolvers.VQE`. However,
    if you're using a provider that has a native implementation of
    :class:`~.BaseEstimator`, it is a better choice to leverage that native
    implementation as it will likely include additional optimizations and be
    a more efficient implementation. The generic nature of this class
    precludes doing any provider- or backend-specific optimizations.
    """

    # pylint: disable=missing-raises-doc
    def __init__(
        self,
        backend: BackendV1 | BackendV2,
        options: dict | None = None,
        abelian_grouping: bool = True,
        bound_pass_manager: PassManager | None = None,
        skip_transpilation: bool = False,
    ):
        """Initalize a new BackendEstimator instance

        Args:
            backend: Required: the backend to run the primitive on
            options: Default options.
            abelian_grouping: Whether the observable should be grouped into
                commuting
            bound_pass_manager: An optional pass manager to run after
                parameter binding.
            skip_transpilation: If this is set to True the internal compilation
                of the input circuits is skipped and the circuit objects
                will be directly executed when this object is called.
        """
        super().__init__(
            circuits=None,
            observables=None,
            parameters=None,
            options=options,
        )

        self._abelian_grouping = abelian_grouping

        self._backend = backend

        self._transpile_options = Options()
        self._bound_pass_manager = bound_pass_manager

        self._preprocessed_circuits: list[tuple[QuantumCircuit, list[QuantumCircuit]]] | None = None
        self._transpiled_circuits: list[QuantumCircuit] | None = None

        self._grouping = list(zip(range(len(self._circuits)), range(len(self._observables))))
        self._skip_transpilation = skip_transpilation

    def __new__(  # pylint: disable=signature-differs
        cls,
        backend: BackendV1 | BackendV2,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ):
        self = super().__new__(cls)
        return self

    @property
    def transpile_options(self) -> Options:
        """Return the transpiler options for transpiling the circuits."""
        return self._transpile_options

    def set_transpile_options(self, **fields):
        """Set the transpiler options for transpiler.
        Args:
            **fields: The fields to update the options
        """
        self._transpiled_circuits = None
        self._transpile_options.update_options(**fields)

    @property
    def preprocessed_circuits(
        self,
    ) -> list[tuple[QuantumCircuit, list[QuantumCircuit]]]:
        """
        Transpiled quantum circuits produced by preprocessing
        Returns:
            List of the transpiled quantum circuit
        """
        self._preprocessed_circuits = self._preprocessing()
        return self._preprocessed_circuits

    @property
    def transpiled_circuits(self) -> list[QuantumCircuit]:
        """
        Transpiled quantum circuits.
        Returns:
            List of the transpiled quantum circuit
        Raises:
            QiskitError: if the instance has been closed.
        """
        self._transpile()
        return self._transpiled_circuits

    @property
    def backend(self) -> BackendV1 | BackendV2:
        """
        Returns:
            The backend which this estimator object based on
        """
        return self._backend

    def _transpile(self):
        """Split Transpile"""
        self._transpiled_circuits = []
        for common_circuit, diff_circuits in self.preprocessed_circuits:
            # 1. transpile a common circuit
            common_circuit = common_circuit.copy()
            num_qubits = common_circuit.num_qubits
            common_circuit.measure_all()
            if not self._skip_transpilation:
                common_circuit = transpile(
                    common_circuit, self.backend, **self.transpile_options.__dict__
                )
            bit_map = {bit: index for index, bit in enumerate(common_circuit.qubits)}
            layout = [bit_map[qr[0]] for _, qr, _ in common_circuit[-num_qubits:]]
            common_circuit.remove_final_measurements()
            # 2. transpile diff circuits
            transpile_opts = copy.copy(self.transpile_options)
            transpile_opts.update_options(initial_layout=layout)
            diff_circuits = transpile(diff_circuits, self.backend, **transpile_opts.__dict__)
            # 3. combine
            transpiled_circuits = []
            for diff_circuit in diff_circuits:
                transpiled_circuit = common_circuit.copy()
                for creg in diff_circuit.cregs:
                    if creg not in transpiled_circuit.cregs:
                        transpiled_circuit.add_register(creg)
                transpiled_circuit.compose(diff_circuit, inplace=True)
                transpiled_circuit.metadata = diff_circuit.metadata
                transpiled_circuits.append(transpiled_circuit)
            self._transpiled_circuits += transpiled_circuits

    def _call(
        self,
        circuits: Sequence[int],
        observables: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:

        # Transpile
        self._grouping = list(zip(circuits, observables))
        transpiled_circuits = self.transpiled_circuits
        num_observables = [len(m) for (_, m) in self.preprocessed_circuits]
        accum = [0] + list(accumulate(num_observables))

        # Bind parameters
        parameter_dicts = [
            dict(zip(self._parameters[i], value)) for i, value in zip(circuits, parameter_values)
        ]
        bound_circuits = [
            transpiled_circuits[circuit_index]
            if len(p) == 0
            else transpiled_circuits[circuit_index].bind_parameters(p)
            for i, (p, n) in enumerate(zip(parameter_dicts, num_observables))
            for circuit_index in range(accum[i], accum[i] + n)
        ]
        bound_circuits = self._bound_pass_manager_run(bound_circuits)

        # Run
        result, metadata = _run_circuits(bound_circuits, self._backend, **run_options)

        return self._postprocessing(result, accum, metadata)

    def _run(
        self,
        circuits: tuple[QuantumCircuit, ...],
        observables: tuple[BaseOperator | PauliSumOp, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ) -> PrimitiveJob:
        circuit_indices = []
        for circuit in circuits:
            index = self._circuit_ids.get(_circuit_key(circuit))
            if index is not None:
                circuit_indices.append(index)
            else:
                circuit_indices.append(len(self._circuits))
                self._circuit_ids[_circuit_key(circuit)] = len(self._circuits)
                self._circuits.append(circuit)
                self._parameters.append(circuit.parameters)
        observable_indices = []
        for observable in observables:
            observable = init_observable(observable)
            index = self._observable_ids.get(_observable_key(observable))
            if index is not None:
                observable_indices.append(index)
            else:
                observable_indices.append(len(self._observables))
                self._observable_ids[_observable_key(observable)] = len(self._observables)
                self._observables.append(observable)
        job = PrimitiveJob(
            self._call, circuit_indices, observable_indices, parameter_values, **run_options
        )
        job.submit()
        return job

    @staticmethod
    def _measurement_circuit(num_qubits: int, pauli: Pauli):
        # Note: if pauli is I for all qubits, this function generates a circuit to measure only
        # the first qubit.
        # Although such an operator can be optimized out by interpreting it as a constant (1),
        # this optimization requires changes in various methods. So it is left as future work.
        qubit_indices = np.arange(pauli.num_qubits)[pauli.z | pauli.x]
        if not np.any(qubit_indices):
            qubit_indices = [0]
        meas_circuit = QuantumCircuit(num_qubits, len(qubit_indices))
        for clbit, i in enumerate(qubit_indices):
            if pauli.x[i]:
                if pauli.z[i]:
                    meas_circuit.sdg(i)
                meas_circuit.h(i)
            meas_circuit.measure(i, clbit)
        return meas_circuit, qubit_indices

    def _preprocessing(self) -> list[tuple[QuantumCircuit, list[QuantumCircuit]]]:
        """
        Preprocessing for evaluation of expectation value using pauli rotation gates.
        """
        preprocessed_circuits = []
        for group in self._grouping:
            circuit = self._circuits[group[0]]
            observable = self._observables[group[1]]
            diff_circuits: list[QuantumCircuit] = []
            if self._abelian_grouping:
                for obs in observable.group_commuting(qubit_wise=True):
                    basis = Pauli(
                        (np.logical_or.reduce(obs.paulis.z), np.logical_or.reduce(obs.paulis.x))
                    )
                    meas_circuit, indices = self._measurement_circuit(circuit.num_qubits, basis)
                    paulis = PauliList.from_symplectic(
                        obs.paulis.z[:, indices],
                        obs.paulis.x[:, indices],
                        obs.paulis.phase,
                    )
                    meas_circuit.metadata = {
                        "paulis": paulis,
                        "coeffs": np.real_if_close(obs.coeffs),
                    }
                    diff_circuits.append(meas_circuit)
            else:
                for basis, obs in zip(observable.paulis, observable):  # type: ignore
                    meas_circuit, indices = self._measurement_circuit(circuit.num_qubits, basis)
                    paulis = PauliList.from_symplectic(
                        obs.paulis.z[:, indices],
                        obs.paulis.x[:, indices],
                        obs.paulis.phase,
                    )
                    meas_circuit.metadata = {
                        "paulis": paulis,
                        "coeffs": np.real_if_close(obs.coeffs),
                    }
                    diff_circuits.append(meas_circuit)

            preprocessed_circuits.append((circuit.copy(), diff_circuits))
        return preprocessed_circuits

    def _postprocessing(
        self, result: Result, accum: list[int], metadata: list[dict]
    ) -> EstimatorResult:
        """
        Postprocessing for evaluation of expectation value using pauli rotation gates.
        """
        counts = _prepare_counts(result)
        expval_list = []
        var_list = []
        shots_list = []

        for i, j in zip(accum, accum[1:]):

            combined_expval = 0.0
            combined_var = 0.0

            for k in range(i, j):
                meta = metadata[k]
                paulis = meta["paulis"]
                coeffs = meta["coeffs"]

                count = counts[k]

                expvals, variances = _pauli_expval_with_variance(count, paulis)

                # Accumulate
                combined_expval += np.dot(expvals, coeffs)
                combined_var += np.dot(variances, coeffs**2)

            expval_list.append(combined_expval)
            var_list.append(combined_var)
            shots_list.append(sum(counts[i].values()))

        metadata = [{"variance": var, "shots": shots} for var, shots in zip(var_list, shots_list)]

        return EstimatorResult(np.real_if_close(expval_list), metadata)

    def _bound_pass_manager_run(self, circuits):
        if self._bound_pass_manager is None:
            return circuits
        else:
            return self._bound_pass_manager.run(circuits)


def _paulis2inds(paulis: PauliList) -> list[int]:
    """Convert PauliList to diagonal integers.
    These are integer representations of the binary string with a
    1 where there are Paulis, and 0 where there are identities.
    """
    # Treat Z, X, Y the same
    nonid = paulis.z | paulis.x

    inds = [0] * paulis.size
    # bits are packed into uint8 in little endian
    # e.g., i-th bit corresponds to coefficient 2^i
    packed_vals = np.packbits(nonid, axis=1, bitorder="little")
    for i, vals in enumerate(packed_vals):
        for j, val in enumerate(vals):
            inds[i] += val.item() * (1 << (8 * j))
    return inds


def _parity(integer: int) -> int:
    """Return the parity of an integer"""
    return bin(integer).count("1") % 2


def _pauli_expval_with_variance(counts: Counts, paulis: PauliList) -> tuple[np.ndarray, np.ndarray]:
    """Return array of expval and variance pairs for input Paulis.
    Note: All non-identity Pauli's are treated as Z-paulis, assuming
    that basis rotations have been applied to convert them to the
    diagonal basis.
    """
    # Diag indices
    size = len(paulis)
    diag_inds = _paulis2inds(paulis)

    expvals = np.zeros(size, dtype=float)
    denom = 0  # Total shots for counts dict
    for bin_outcome, freq in counts.items():
        outcome = int(bin_outcome, 2)
        denom += freq
        for k in range(size):
            coeff = (-1) ** _parity(diag_inds[k] & outcome)
            expvals[k] += freq * coeff

    # Divide by total shots
    expvals /= denom

    # Compute variance
    variances = 1 - expvals**2
    return expvals, variances
