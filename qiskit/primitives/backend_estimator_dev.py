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

"""Estimator class for expectation value calculations based on Backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from functools import reduce
from typing import Any

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.opflow import PauliSumOp
from qiskit.providers import Backend, BackendV2, BackendV2Converter
from qiskit.providers import JobV1 as Job
from qiskit.providers import Options
from qiskit.quantum_info import Pauli, PauliList
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.symplectic.sparse_pauli_op import SparsePauliOp
from qiskit.result import Counts, Result
from qiskit.transpiler import Layout, PassManager

from .base import BaseEstimator, EstimatorResult
from .primitive_job import PrimitiveJob
from .utils import init_observable


################################################################################
## ESTIMATOR
################################################################################
class BackendEstimator(BaseEstimator):
    """Evaluates expectation value using Pauli rotation gates.

    The :class:`~.BackendEstimator` class is a generic implementation of the
    :class:`~.BaseEstimator` interface that is used to wrap a :class:`~.Backend`
    object in the :class:`~.BaseEstimator` API. It facilitates using backends
    that do not provide a native :class:`~.BaseEstimator` implementation in
    places that work with :class:`~.BaseEstimator`, such as algorithms in
    :mod:`qiskit.algorithms` including :class:`~.qiskit.algorithms.minimum_eigensolvers.VQE`.
    However, if you're using a provider that has a native implementation of
    :class:`~.BaseEstimator`, it is a better choice to leverage that native
    implementation as it will likely include additional optimizations and be a
    more efficient implementation. The generic nature of this class precludes
    doing any provider- or backend-specific optimizations.
    """

    # pylint: disable=missing-raises-doc
    def __init__(
        self,
        backend: Backend,
        *,  # TODO: allow backend as positional after removing deprecations
        abelian_grouping: bool = True,
        skip_transpilation: bool = False,
        bound_pass_manager: PassManager | None = None,
        options: dict | None = None,
    ) -> None:
        """Initalize a new BackendEstimator instance.

        Args:
            backend: Required: The backend to run the primitive on.
            abelian_grouping: Whether commuting observable components should be grouped.
            skip_transpilation: If `True`, transpilation of the input circuits is skipped.
            bound_pass_manager: An optional pass manager to run after parameter binding.
            options: Default options.
        """
        # TODO: validation
        self.backend = backend
        self.abelian_grouping = abelian_grouping  # TODO: `group_commuting`
        self.skip_transpilation = skip_transpilation  # TODO: tranpilation level
        self._bound_pass_manager = bound_pass_manager  # TODO: standardize
        super().__init__(
            circuits=None,
            observables=None,
            parameters=None,
            options=options,
        )
        self._transpile_options = Options()

    def __getnewargs__(self) -> tuple:
        return (self._backend,)

    ################################################################################
    ## PROPERTIES
    ################################################################################
    @property
    def backend(self) -> BackendV2:
        """Backend to use for circuit measurements."""
        return self._backend

    @backend.setter
    def backend(self, backend: Backend) -> None:
        if not isinstance(backend, Backend):
            raise TypeError(
                f"Expected `Backend` type for `backend`, got `{type(backend)}` instead."
            )
        # TODO: clear all transpilation caching
        self._backend: BackendV2 = (
            backend if isinstance(backend, BackendV2) else BackendV2Converter(backend)
        )

    @property
    def abelian_grouping(self) -> bool:
        """Groups commuting observable components."""
        return self._abelian_grouping

    @abelian_grouping.setter
    def abelian_grouping(self, abelian_grouping: bool) -> None:
        if not isinstance(abelian_grouping, bool):
            raise TypeError(
                "Expected `bool` type for `abelian_grouping`, "
                f"got `{type(abelian_grouping)}` instead."
            )
        self._abelian_grouping = abelian_grouping

    @property
    def skip_transpilation(self) -> bool:
        """If ``True``, transpilation of the input circuits is skipped."""
        return self._skip_transpilation

    @skip_transpilation.setter
    def skip_transpilation(self, skip_transpilation: bool) -> None:
        if not isinstance(skip_transpilation, bool):
            raise TypeError(
                "Expected `bool` type for `skip_transpilation`, "
                f"got `{type(skip_transpilation)}` instead."
            )
        self._skip_transpilation = skip_transpilation

    @property
    def transpile_options(self) -> Options:
        """Options for transpiling the input circuits."""
        return self._transpile_options

    def set_transpile_options(self, **fields) -> None:
        """Set the transpiler options for transpiler.

        Args:
            **fields: The fields to update the options
        """
        self._transpile_options.update_options(**fields)

    @property
    def _observable_decomposer(self) -> _ObservableDecomposer:
        """Observable decomposer based on object's config."""
        if self.abelian_grouping:
            return _AbelianDecomposer()
        return _NaiveDecomposer()

    @property
    def _expval_reckoner(self) -> _ExpvalReckoner:
        """Strategy for expectation value reckoning."""
        return _SpectralReckoner()

    ################################################################################
    ## IMPLEMENTATION
    ################################################################################
    # TODO: caching
    def _run(
        self,
        circuits: tuple[QuantumCircuit, ...],
        observables: tuple[SparsePauliOp, ...],  # TODO: normalize to `SparsePauliOp`
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ) -> PrimitiveJob:
        job = PrimitiveJob(self._compute, circuits, observables, parameter_values, **run_options)
        job.submit()
        return job

    def _compute(
        self,
        circuits: tuple[QuantumCircuit, ...],
        observables: tuple[SparsePauliOp, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ) -> EstimatorResult:
        """Solve expectation value problem."""
        circuits = self._pre_transpile(circuits)
        circuits = self._bind_parameters(circuits, parameter_values)
        circuits = self._post_transpile(circuits)
        circuits_matrix = self._observe_circuits(circuits, observables)
        counts_matrix = self._execute_matrix(circuits_matrix, **run_options)
        expvals_w_errors = self._reckon_expvals(counts_matrix)
        return self._build_result(expvals_w_errors, counts_matrix)

    def _pre_transpile(self, circuits: Sequence[QuantumCircuit]) -> tuple[QuantumCircuit, ...]:
        """Traspile paramterized quantum circuits to match the estimator's backend.

        The output circuits are annotated with the ``final_layout`` attribute.
        """
        return tuple(self._pre_transpile_single(qc) for qc in circuits)

    def _pre_transpile_single(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Traspile paramterized quantum circuit to match the estimator's backend.

        The output circuit is annotated with the ``final_layout`` attribute.
        """
        # Note: We currently need to use a hacky way to account for the final
        # layout of the transpiled circuit. We insert temporary measurements
        # to keep track of the repositioning of the different qubits.
        original_circuit = circuit.copy()  # To insert measurements
        original_circuit.measure_all()  # To keep track of the final layout
        if self.skip_transpilation:
            transpiled_circuit = original_circuit
        else:
            transpile_options = {**self.transpile_options.__dict__}
            transpiled_circuit = transpile(original_circuit, self.backend, **transpile_options)
        final_layout = self._infer_final_layout(original_circuit, transpiled_circuit)
        transpiled_circuit.remove_final_measurements()
        transpiled_circuit.final_layout = final_layout
        return transpiled_circuit

    def _bind_parameters(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
    ) -> tuple[QuantumCircuit, ...]:
        """Bind circuit parameters.

        Note: for improved performance, this method edits the input circuits in place,
        avoiding costly deepcopy operations but resulting in a side-effect. This is fine
        as long as the input circuits are no longer needed.
        """
        for circuit, values in zip(circuits, parameter_values):
            # TODO: return circuit even if assigning in place
            circuit.assign_parameters(values, inplace=True)
        return tuple(circuits)

    def _post_transpile(self, circuits: Sequence[QuantumCircuit]) -> tuple[QuantumCircuit, ...]:
        """Traspile non-parametrized quantum circuits (i.e. after binding all parameters)."""
        return tuple(self._post_transpile_single(qc) for qc in circuits)

    def _post_transpile_single(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Traspile non-parametrized quantum circuit (i.e. after binding all parameters)."""
        # TODO: rename `_bound_pass_manager`
        if self._bound_pass_manager is not None:
            circuit = self._bound_pass_manager.run(circuit)
        return circuit

    def _observe_circuits(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[SparsePauliOp],
    ) -> tuple[tuple[QuantumCircuit, ...], ...]:
        """For each circuit-observable pair build build all necessary circuits for computation."""
        return tuple(
            self._measure_observable(circuit, observable)
            for circuit, observable in zip(circuits, observables)
        )

    def _execute_matrix(
        self, circuits_matrix: Sequence[Sequence[QuantumCircuit]], **run_options
    ) -> tuple[tuple[Counts]]:
        """Execute circuit matrix and return counts in identical (i.e. one-to-one) arrangement.

        Each :class:`qiskit.result.Counts` object is annotated with the metadata
        from the circuit that produced it.
        """
        circuits = list(qc for group in circuits_matrix for qc in group)  # List for performance
        counts = self._execute(circuits, **run_options)
        counts_iter = iter(counts)
        counts_matrix = tuple(tuple(next(counts_iter) for _ in group) for group in circuits_matrix)
        return counts_matrix

    def _execute(self, circuits: Sequence[QuantumCircuit], **run_options) -> list[Counts]:
        """Execute quantum circuits on backend bypassing max circuits allowed.

        Each :class:`qiskit.result.Counts` object is annotated with the metadata
        from the circuit that produced it.
        """
        # Conversion
        circuits = list(circuits)  # TODO: accept Sequences in `backend.run()`

        # Max circuits
        total_circuits: int = len(circuits)
        max_circuits: int = getattr(self.backend, "max_circuits", None) or total_circuits

        # Raw results
        jobs: tuple[Job] = tuple(
            self.backend.run(circuits[split : split + max_circuits], **run_options)
            for split in range(0, total_circuits, max_circuits)
        )
        raw_results: tuple[Result] = tuple(job.result() for job in jobs)

        # Annotated counts
        job_counts_iter = (
            job_counts if isinstance(job_counts, list) else [job_counts]
            for job_counts in (result.get_counts() for result in raw_results)
        )
        counts_iter = (counts for job_counts in job_counts_iter for counts in job_counts)
        counts_list: list[Counts] = []
        for counts, circuit in zip(counts_iter, circuits):
            counts.metadata = circuit.metadata  # TODO: add `Counts.metadata` attr
            counts_list.append(counts)

        return counts_list

    def _reckon_expvals(
        self, counts_matrix: Sequence[Sequence[Counts]]
    ) -> tuple[tuple[float, float], ...]:
        """Compute expectation values by groups of counts along with their associated std-error.

        One expectation value is computed for every element in each group of counts, according
        to its annotated observable. Then all expectation values are added together on a
        group-by-group basis.
        """
        return tuple(self._reckon_single_expval(counts_group) for counts_group in counts_matrix)

    def _reckon_single_expval(self, counts_group: Sequence[Counts]) -> tuple[float, float]:
        """Compute expectation value and associated std-error for a group of counts.

        One expectation value is computed for every element in the group of counts, according
        to its annotated observable. Then all expectation values are added together.

        Args:
            counts_group: sequence of counts annotated with an observable in their metadata.

        Returns:
            Expectation value and associated std-error.
        """
        expval: float = 0.0
        variance: float = 0.0
        for counts in counts_group:
            observable: SparsePauliOp = counts.metadata["observable"]
            value, std_error = self._expval_reckoner.compute_observable_expval(counts, observable)
            expval += value
            variance += std_error**2
        return expval, np.sqrt(variance)

    def _build_result(
        self,
        expvals_w_errors: Sequence[tuple[float, float]],
        counts_matrix: Sequence[Sequence[Counts]],
    ) -> EstimatorResult:
        """Package results into an :class:`~qiskit.primitives.EstimatorResult` data structure.

        Args:
            expvals_w_errors: a sequence of two-tuples holding expectation values and their
                associated std-errors.
            counts_matrix: the original counts from which the expectation values were derived.
                These will be used for reporting metadata.

        Returns:
            An :class:`~qiskit.primitives.EstimatorResult` object built from the input data.
        """
        expvals, std_errors = tuple(zip(*expvals_w_errors))
        values = np.real_if_close(expvals)
        shots_list = tuple(
            sum(sum(counts.values()) for counts in counts_list) for counts_list in counts_matrix
        )
        num_circuits_list = tuple(len(counts_list) for counts_list in counts_matrix)
        metadata = [
            {
                "variance": (shots / num_circuits) * std_error**2,
                "std_error": std_error,
                "shots": shots,
                "num_circuits": num_circuits,
            }
            for std_error, shots, num_circuits in zip(std_errors, shots_list, num_circuits_list)
        ]
        return EstimatorResult(values, metadata)

    ################################################################################
    ## MEASUREMENT
    ################################################################################
    # TODO: `QuantumCircuit.measure_observable(observable)` once instructions return self
    def _measure_observable(
        self, circuit: QuantumCircuit, observable: SparsePauliOp
    ) -> tuple[QuantumCircuit, ...]:
        """From a base circuit, build all necessary circuits for measuring a given observable.

        Each circuit has its metadata annotated with the observable component
        (i.e. :class:`~qiskit.quantum_info.SparsePauliOp`) that can be directly evaluated.
        """
        measurements = self._build_measurement_circuits(observable)
        circs_w_meas = self._compose_measurements(circuit, measurements)
        return circs_w_meas

    # TODO: caching
    def _build_measurement_circuits(
        self,
        observable: SparsePauliOp,
    ) -> tuple[QuantumCircuit, ...]:
        """Given an observable, build all appendage quantum circuits necessary to measure it.

        This will return one measurement circuit per singly measurable component of the
        observable (i.e. measurable with a single quantum circuit), as retrieved from the
        instance's `observable_decomposer` attribute.
        """
        return tuple(
            self._build_single_measurement_circuit(component)
            for component in self._observable_decomposer.decompose(observable)
        )

    # TODO: pre-transpile gates
    def _build_single_measurement_circuit(self, observable: SparsePauliOp) -> QuantumCircuit:
        """Build measurement circuit for a given observable.

        The input observable can be made out of different components, but they all have to
        share a single common basis in the form of a Pauli operator in order to be measured
        simultaneously (e.g. `ZZ` and `ZI`, or `XI` and `IX`).
        """
        basis: tuple[Pauli] = self._observable_decomposer.extract_pauli_basis(observable)
        if len(basis) != 1:
            raise ValueError("Unable to retrieve a singlet Pauli basis for the given observable.")
        circuit: QuantumCircuit = self._build_pauli_measurement(basis[0])
        # Simplified Paulis (keep only measured qubits)
        measured_qubit_indices = circuit.metadata.get("measured_qubit_indices")
        paulis = PauliList.from_symplectic(
            observable.paulis.z[:, measured_qubit_indices],
            observable.paulis.x[:, measured_qubit_indices],
            observable.paulis.phase,
        )
        # TODO: observable does not need to be hermitian: rename
        circuit.metadata.update({"observable": SparsePauliOp(paulis, observable.coeffs)})
        return circuit

    # TODO: `QuantumCircuit.measure_pauli(pauli)`
    @staticmethod
    def _build_pauli_measurement(pauli: Pauli) -> QuantumCircuit:
        """Build measurement circuit for a given Pauli operator.

        The resulting circuit has its metadata annotated with the indices of the qubits
        that hold measurement gates.
        """
        # TODO: if pauli is I for all qubits, this function generates a circuit to
        # measure only the first qubit. Although such an operator can be optimized out
        # by interpreting it as a constant (1), this optimization requires changes in
        # various methods. So it is left as future work.
        # TODO: insert pre-transpiled gates to avoid re-transpilation.
        # TODO: cache
        measured_qubit_indices = np.arange(pauli.num_qubits)[pauli.z | pauli.x]
        measured_qubit_indices = tuple(measured_qubit_indices.tolist()) or (0,)
        circuit = QuantumCircuit(pauli.num_qubits, len(measured_qubit_indices))
        circuit.metadata = {"measured_qubit_indices": measured_qubit_indices}
        for cbit, qubit in enumerate(measured_qubit_indices):
            if pauli.x[qubit]:
                if pauli.z[qubit]:
                    circuit.sdg(qubit)
                circuit.h(qubit)
            circuit.measure(qubit, cbit)
        return circuit

    def _compose_measurements(
        self,
        base: QuantumCircuit,
        measurements: Sequence[QuantumCircuit] | QuantumCircuit,
    ) -> tuple[QuantumCircuit, ...]:
        """Compose measurement circuits with base circuit considering its final layout."""
        if isinstance(measurements, QuantumCircuit):
            measurements = (measurements,)
        return tuple(self._compose_single_measurement(base, meas) for meas in measurements)

    def _compose_single_measurement(
        self, base: QuantumCircuit, measurement: QuantumCircuit
    ) -> QuantumCircuit:
        """Compose single measurement circuit with base circuit considering its final layout.

        Args:
            base: a quantum circuit with final_layout entry in its metadata
            measurement: a quantum circuit with ... # TODO

        Returns:
            A compsite quantum circuit
        """
        transpile_options = {**self.transpile_options.__dict__}  # TODO: avoid multiple copies
        transpile_options.update({"initial_layout": base.final_layout})
        transpiled_measurement = transpile(measurement, self.backend, **transpile_options)
        circuit = base.compose(transpiled_measurement)
        circuit.metadata = {
            **(base.metadata or {}),  # TODO: default `QuantumCircuit.metadata` to {}
            **(measurement.metadata or {}),
        }
        circuit.metadata.pop("measured_qubit_indices", None)  # TODO: replace with `measured_qubits`
        return circuit

    @classmethod
    def _infer_final_layout(
        cls, original_circuit: QuantumCircuit, transpiled_circuit: QuantumCircuit
    ) -> Layout:
        """Retrieve final layout from original and transpiled circuits (all measured)."""
        physical_qubits = cls._generate_final_layout_intlist(original_circuit, transpiled_circuit)
        layout_dict: dict[int, Any] = dict.fromkeys(range(transpiled_circuit.num_qubits))
        for physical_qubit, virtual_qubit in zip(physical_qubits, original_circuit.qubits):
            layout_dict.update({physical_qubit: virtual_qubit})
        return Layout(layout_dict)

    @staticmethod
    def _generate_final_layout_intlist(
        original_circuit: QuantumCircuit, transpiled_circuit: QuantumCircuit
    ) -> Iterator[int]:
        """Generate final layout intlist of physical qubits.

        Note: Works under the assumption that the original circuit has a `measure_all`
        instruction at its end, and that the transpiler does not affect the classical
        registers.
        """
        # TODO: raise error if assumption in docstring is not met
        qubit_index_map = {qubit: i for i, qubit in enumerate(transpiled_circuit.qubits)}
        num_measurements: int = original_circuit.num_qubits
        for i in range(-num_measurements, 0):
            _, qargs, _ = transpiled_circuit[i]
            physical_qubit = qargs[0]
            yield qubit_index_map[physical_qubit]

    ################################################################################
    ## DEPRECATED
    ################################################################################
    # Note: to allow `backend` as positional argument while deprecated in place
    def __new__(  # pylint: disable=signature-differs
        cls,
        backend: Backend,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ):
        self = super().__new__(cls)
        return self

    def _call(
        self,
        circuits: Sequence[int],
        observables: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:
        raise NotImplementedError("This method has been deprecated, use `run` instead.")


################################################################################
## OBSERVABLE DECOMPOSER
################################################################################
class _ObservableDecomposer(ABC):
    """Strategy interface for decomposing observables and getting associated measurement bases."""

    def decompose(self, observable: BaseOperator | PauliSumOp) -> tuple[SparsePauliOp]:
        """Decomposes a given observable into singly measurable components.

        Note that component decomposition is not unique, for instance, commuting components
        could be grouped together in different ways (i.e. partitioning the set).

        Args:
            observable: the observable to decompose into its core components.

        Returns:
            A list of observables each of which measurable with a single quantum circuit
            (i.e. on a singlet Pauli basis).
        """
        observable = init_observable(observable)
        return self._decompose(observable)

    @abstractmethod
    def _decompose(
        self,
        observable: SparsePauliOp,
    ) -> tuple[SparsePauliOp]:
        ...

    def extract_pauli_basis(self, observable: BaseOperator | PauliSumOp) -> PauliList:
        """Extract Pauli basis for a given observable.

        Note that the resulting basis may be overcomplete depending on the implementation.

        Args:
            observable: an operator for which to obtain a Pauli basis for measurement.

        Returns:
            A `PauliList` of operators serving as a basis for the input observable. Each
            entry conrresponds one-to-one to the components retrieved from `.decompose()`.
        """
        components = self.decompose(observable)
        paulis = tuple(self._extract_singlet_basis(component) for component in components)
        return PauliList(paulis)  # TODO: Allow `PauliList` from generator

    @abstractmethod
    def _extract_singlet_basis(self, observable: SparsePauliOp) -> Pauli:
        """Extract singlet Pauli basis for a given observable.

        The input observable comes from `._decompose()`, and must be singly measurable.
        """
        ...


class _NaiveDecomposer(_ObservableDecomposer):
    """Trivial observable decomposition without grouping components."""

    def _decompose(
        self,
        observable: SparsePauliOp,
    ) -> tuple[SparsePauliOp]:
        return tuple(observable)

    def _extract_singlet_basis(self, observable: SparsePauliOp) -> Pauli:
        return observable.paulis[0]


class _AbelianDecomposer(_ObservableDecomposer):
    """Abelian observable decomposition grouping commuting components."""

    def _decompose(
        self,
        observable: SparsePauliOp,
    ) -> tuple[SparsePauliOp]:
        components = observable.group_commuting(qubit_wise=True)
        return tuple(components)

    def _extract_singlet_basis(self, observable: SparsePauliOp) -> Pauli:
        or_reduce = np.logical_or.reduce
        zx_data_tuple = or_reduce(observable.paulis.z), or_reduce(observable.paulis.x)
        return Pauli(zx_data_tuple)


################################################################################
## EXPECTATION VALUE RECKONING
################################################################################
class _ExpvalReckoner(ABC):
    """Expectation value reckoning interface.

    Classes implementing this interface provide methods for constructing expectation values
    (and associated errors) out of raw Counts and Pauli observables.
    """

    def compute_observable_expval(
        self, counts: Counts, observable: SparsePauliOp
    ) -> tuple[float, float]:
        """Compute expectation value and associated std-error for input observable from counts.

        Note: the input observable needs to be measurable entirely within one circuit
        execution (i.e. resulting in the input counts). Users must ensure that counts
        come from the appropriate circuit execution.

        args:
            counts: a :class:`~qiskti.result.Counts` object from circuit execution.
            observable:

        Returns:
            The expectation value and associated std-error for the input observable.
        """
        expvals, std_errors = np.vstack(
            [self.compute_pauli_expval(counts, pauli) for pauli in observable.paulis]
        ).T
        coeffs = np.array(observable.coeffs)
        expval = np.dot(expvals, coeffs)
        variance = np.dot(std_errors**2, coeffs**2)  # TODO: complex coeffs
        std_error = np.sqrt(variance)
        return expval, std_error

    # TODO: validate num_bits
    @abstractmethod
    def compute_pauli_expval(self, counts: Counts, pauli: Pauli) -> tuple[float, float]:
        """Compute expectation value and associated std-error for input Pauli from counts.

        Args:
            counts: measured by executing a :class``~qiskit.circuit.QuantumCircuit``.
            pauli: the target :class:`~qiskit.quantum_info.Pauli` to observe.

        Returns:
            The expectation value and associated std-error for the input Pauli.
        """


class _SpectralReckoner(_ExpvalReckoner):
    """Expectation value reckoning class based on weighted addition of eigenvalues.

    Note: This class treats X, Y, and Z Paulis identically, assuming that the appropriate
    changes of bases (i.e. rotations) were actively performed in the relevant qubits before
    readout; hence diagonalizing the input Pauli observables.
    """

    def compute_pauli_expval(self, counts: Counts, pauli: Pauli) -> tuple[float, float]:
        shots: int = 0
        expval: float = 0.0
        for bitstring, freq in counts.items():
            observation = self.compute_eigenvalue(bitstring, pauli)
            expval += observation * freq
            shots += freq
        shots = shots or 1  # Avoid division by zero errors if no counts
        expval /= shots
        variance = 1 - expval**2
        std_error = np.sqrt(variance / shots)
        return expval, std_error

    @classmethod
    def compute_eigenvalue(cls, bitstring: str, pauli: Pauli) -> int:
        """Compute eigenvalue for measured bitstring and target Pauli.

        Args:
            bitstring: binary representation of the eigenvector.
            pauli: the target :class:`~qiskit.quantum_info.Pauli` matrix.

        Returns:
            The eigenvalue associated to the bitstring eigenvector and the input Pauli observable.
        """
        measurement = int(bitstring, 2)
        int_mask = cls._pauli_integer_mask(pauli)
        return (-1) ** cls._parity_bit(measurement & int_mask, even=True)

    @staticmethod
    def _pauli_integer_mask(pauli: Pauli) -> int:
        """Build integer mask for input Pauli.

        This is an integer representation of the binary string with a
        1 where there are Paulis, and 0 where there are identities.
        """
        pauli_mask: np.ndarray[Any, np.dtype[bool]] = pauli.z | pauli.x
        packed_mask: list[int] = np.packbits(pauli_mask, bitorder="little").tolist()
        return reduce(lambda value, element: (value << 8) | element, packed_mask)

    @staticmethod
    def _parity_bit(integer: int, even: bool = True) -> int:
        """Return the parity bit for a given integer."""
        even_bit = bin(integer).count("1") % 2
        return even_bit if even else int(not even_bit)
