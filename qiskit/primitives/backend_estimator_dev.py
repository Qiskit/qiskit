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

# from .utils import _circuit_key, _observable_key  # TODO: caching


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
        try:
            return self._abelian_grouping
        except AttributeError:
            return True

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
        """If `True`, transpilation of the input circuits is skipped."""
        try:
            return self._skip_transpilation
        except AttributeError:
            return False

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

    ################################################################################
    ## IMPLEMENTATION
    ################################################################################
    # TODO: caching
    def _run(
        self,
        circuits: tuple[QuantumCircuit, ...],
        observables: tuple[BaseOperator | PauliSumOp, ...],  # TODO: normalize to `SparsePauliOp`
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ) -> PrimitiveJob:
        job = PrimitiveJob(self._compute, circuits, observables, parameter_values, **run_options)
        job.submit()
        return job

    def _compute(
        self,
        circuits: tuple[QuantumCircuit, ...],
        observables: tuple[BaseOperator | PauliSumOp, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ) -> EstimatorResult:
        """Solve expectation value problem on backend."""
        # Pre-processing
        circuit_bundles = self._preprocess(circuits, observables, parameter_values)
        metadata_bundles = tuple(tuple(c.metadata for c in bun) for bun in circuit_bundles)
        job_circuits = list(c for bun in circuit_bundles for c in bun)  # TODO: accept tuple

        # Raw results: counts
        counts_list: list[Counts] = self._run_on_backend(job_circuits, **run_options)

        # Post-processing
        counts_iter = iter(counts_list)
        counts_bundles = tuple(tuple(next(counts_iter) for _ in bun) for bun in circuit_bundles)
        expval_list, var_list, shots_list = self._postprocess(counts_bundles, metadata_bundles)

        # Results
        values = np.real_if_close(expval_list)
        metadata = [
            {"variance": var, "shots": shots, "num_circuits": len(circuit_bundle)}
            for var, shots, circuit_bundle in zip(var_list, shots_list, circuit_bundles)
        ]
        return EstimatorResult(values, metadata)

    def _preprocess(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
    ) -> tuple[tuple[QuantumCircuit, ...], ...]:
        """Preprocess circuit-observable experiments to runnable tuples of circuits, one per pair."""
        return tuple(
            self._preprocess_single(circuit, observable, params)
            for circuit, observable, params in zip(circuits, observables, parameter_values)
        )

    def _preprocess_single(
        self,
        circuit: QuantumCircuit,
        observable: BaseOperator,
        parameter_values: Sequence[float],
    ) -> tuple[QuantumCircuit, ...]:
        """Preprocess single circuit-observable experiment to runnable tuple of circuits."""
        circuit = self._transpile(circuit)  # TODO: Cache (produces a copy)
        circuit.assign_parameters(parameter_values, inplace=True)
        circuit = self._run_bound_pass_manager(circuit)
        measurements = self._build_measurement_circuits(observable)
        circs_w_meas = self._compose_measurements(circuit, measurements)
        return circs_w_meas

    def _run_on_backend(self, circuits: list[QuantumCircuit], **run_options) -> list[Counts]:
        """Run circuits on backend bypassing max circuits allowed."""
        # Max circuits
        total_circuits: int = len(circuits)
        max_circuits: int = getattr(self.backend, "max_circuits", None) or total_circuits

        # Raw results
        jobs: tuple[Job] = tuple(
            self.backend.run(circuits[split : split + max_circuits], **run_options)
            for split in range(0, total_circuits, max_circuits)
        )
        raw_results: tuple[Result] = tuple(job.result() for job in jobs)

        # Counts
        counts_list: list[Counts] = []
        for raw_result in raw_results:
            counts: list[Counts] | Counts = raw_result.get_counts()
            counts_list.extend(counts if isinstance(counts, list) else [counts])
        return counts_list

    def _postprocess(
        self,
        counts_bundles: Sequence[Sequence[Counts]],
        metadata_bundles: Sequence[Sequence[dict[str, Any]]],
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[int, ...]]:
        """Postprocess lists of counts and metadata bundles to expvals, variances, and shots."""
        expval_var_shots_triplets: tuple[tuple[float, float, int], ...] = tuple(
            self._postprocess_single(counts_bundle, meta_bundle)
            for counts_bundle, meta_bundle in zip(counts_bundles, metadata_bundles)
        )
        return tuple(tuple(lst) for lst in zip(*expval_var_shots_triplets))  # type: ignore

    def _postprocess_single(
        self,
        counts_bundle: Sequence[Counts],
        metadata_bundle: Sequence[dict[str, Any]],
    ) -> tuple[float, float, int]:
        """Postprocess single counts and metadata bundles to expval, variance, and shots."""
        expval: float = 0.0
        var: float = 0.0
        for counts, metadata in zip(counts_bundle, metadata_bundle):
            paulis: PauliList = metadata["paulis"]
            coeffs: tuple[float] = metadata["coeffs"]
            expvals, variances = self._compute_expvals_and_variances(counts, paulis)
            expval += np.dot(expvals, coeffs)
            var += np.dot(variances, np.array(coeffs) ** 2)
        shots: int = sum(counts_bundle[0].values())  # TODO: not correct -> counts.shots (?)
        return expval, var, shots

    ################################################################################
    ## COMPUTATION
    ################################################################################
    @classmethod
    def _compute_expvals_and_variances(
        cls, counts: Counts, paulis: PauliList
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return tuples of expvals and variances for input Paulis.

        Note: All non-identity Pauli's are treated as Z-paulis, assuming
        that basis rotations have been applied to convert them to the
        diagonal basis.
        """
        pairs = (cls._compute_expval_variance_pair(counts, pauli) for pauli in paulis)
        return tuple(zip(*pairs))

    @classmethod
    def _compute_expval_variance_pair(cls, counts: Counts, pauli: Pauli) -> tuple[float, float]:
        """Return an expval-variance pair for the given counts and pauli.

        Note: All non-identity Pauli's are treated as Z-paulis, assuming
        that basis rotations have been applied to convert them to the
        diagonal basis.
        """
        shots: int = 0
        expval: float = 0.0
        for bitstring, freq in counts.items():
            observation = cls._observed_value(bitstring, pauli)
            expval += observation * freq
            shots += freq
        expval /= shots or 1  # Avoid division by zero errors if no counts
        variance = 1 - expval**2
        return expval, variance

    @classmethod
    def _observed_value(cls, bitstring: str, pauli: Pauli) -> int:
        """Compute observed eigenvalue from measured bitstring and target Pauli."""
        measurement = int(bitstring, 2)
        int_mask = cls._pauli_integer_mask(pauli)
        return (-1) ** cls._parity_bit(measurement & int_mask, even=True)

    @classmethod
    def _pauli_integer_mask(cls, pauli: Pauli) -> tuple[int]:
        """Build integer masks for input Pauli.

        This is an integer representation of the binary string with a
        1 where there are Paulis, and 0 where there are identities.
        """
        pauli_mask: np.ndarray[bool] = pauli.z | pauli.x
        packed_mask: list[int] = np.packbits(pauli_mask, bitorder="little").tolist()
        return reduce(lambda value, element: (value << 8) + element, packed_mask)

    @staticmethod
    def _parity_bit(integer: int, even: bool = True) -> int:
        """Return the parity bit for a given integer."""
        even_bit = bin(integer).count("1") % 2
        return even_bit if even else int(not even_bit)

    ################################################################################
    ## TRANSPILATION
    ################################################################################
    # TODO: pass backend and run_options
    def _transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Traspile quantum circuit to match the estimator's backend.

        Includes the final layout as metadata.
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
        if transpiled_circuit.metadata is None:  # TODO: QuantumCircuit class should return {}
            transpiled_circuit.metadata = {}
        transpiled_circuit.metadata.update({"final_layout": final_layout})
        return transpiled_circuit

    @classmethod  # TODO: multiple registers
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

    def _run_bound_pass_manager(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Run bound pass manager if set."""
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError(f"Expected `QuantumCircuit`, received {type(circuit)} instead.")
        if self._bound_pass_manager is not None:
            circuit = self._bound_pass_manager.run(circuit)
        return circuit

    ################################################################################
    ## MEASUREMENT
    ################################################################################
    @property
    def _observable_decomposer(self) -> ObservableDecomposer:
        """Observable decomposer based on object's config."""
        if self.abelian_grouping:
            return AbelianDecomposer()
        return NaiveDecomposer()

    # TODO: caching
    def _build_measurement_circuits(
        self,
        observable: BaseOperator | PauliSumOp,
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
    def _build_single_measurement_circuit(
        self, observable: BaseOperator | PauliSumOp
    ) -> QuantumCircuit:
        """Build measurement circuit for a given observable.

        The input observable can be made out of different components, but they all have to
        share a single common basis in the form of a Pauli operator in order to be measured
        simultaneously (e.g. `ZZ` and `ZI`, or `XI` and `IX`).
        """
        basis: tuple[Pauli] = self._observable_decomposer.extract_pauli_basis(observable)
        if len(basis) != 1:
            raise ValueError("Unable to retrieve a singlet Pauli basis for the given observable.")
        circuit: QuantumCircuit = self._build_pauli_measurement(basis[0])
        # Simplified Paulis (removing common identities)
        measured_qubit_indices = circuit.metadata.get("measured_qubit_indices")
        paulis = PauliList.from_symplectic(
            observable.paulis.z[:, measured_qubit_indices],
            observable.paulis.x[:, measured_qubit_indices],
            observable.paulis.phase,
        )
        circuit.metadata = {
            **circuit.metadata,
            "paulis": paulis,
            "coeffs": tuple(np.real_if_close(observable.coeffs)),
        }
        return circuit

    # TODO: `QuantumCircuit.measure_pauli(pauli)`
    @staticmethod
    def _build_pauli_measurement(pauli: Pauli) -> QuantumCircuit:
        """Build measurement circuit for a given Pauli operator."""
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

    ################################################################################
    ## COMPOSITION
    ################################################################################
    def _compose_measurements(
        self,
        base: QuantumCircuit,
        measurements: Sequence[QuantumCircuit] | QuantumCircuit,
    ) -> tuple[QuantumCircuit, ...]:
        """Compose measurement circuits with base circuit considering final layout."""
        if isinstance(measurements, QuantumCircuit):
            measurements = (measurements,)
        return tuple(self._compose_single_measurement(base, meas) for meas in measurements)

    def _compose_single_measurement(
        self, base: QuantumCircuit, measurement: QuantumCircuit
    ) -> QuantumCircuit:
        """Compose single measurement circuit with base circuit considering final layout.

        Args:
            base: a quantum circuit with final_layout entry in its metadata
            measurement: a quantum circuit with

        Returns:
            A compsite quantum circuit
        """
        layout = base.metadata.get("final_layout")
        transpile_options = {**self.transpile_options.__dict__}  # TODO: avoid multiple copies
        transpile_options.update({"initial_layout": layout})
        transpiled_measurement = transpile(measurement, self.backend, **transpile_options)
        circuit = base.compose(transpiled_measurement)
        circuit.metadata = {**base.metadata, **measurement.metadata}
        circuit.metadata.pop("measured_qubit_indices")  # TODO: `measured_qubits`
        return circuit

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
class ObservableDecomposer(ABC):
    """Strategy class for decomposing observables and getting associated measurement bases."""

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


class NaiveDecomposer(ObservableDecomposer):
    """Naive observable decomposition without grouping components."""

    def _decompose(
        self,
        observable: SparsePauliOp,
    ) -> tuple[SparsePauliOp]:
        return tuple(observable)

    def _extract_singlet_basis(self, observable: SparsePauliOp) -> Pauli:
        return observable.paulis[0]


class AbelianDecomposer(ObservableDecomposer):
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
