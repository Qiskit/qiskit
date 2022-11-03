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

"""Estimator class for expectation value calculations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence, Iterator
from typing import Any

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates import HGate, SdgGate
from qiskit.circuit.quantumcircuitdata import CircuitInstruction
from qiskit.compiler import transpile
from qiskit.opflow import PauliSumOp
from qiskit.providers import Backend, Options
from qiskit.quantum_info import Pauli, PauliList
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result import Counts, Result
from qiskit.transpiler import PassManager, Layout

from .base import BaseEstimator, EstimatorResult
from .primitive_job import PrimitiveJob
from .utils import _circuit_key, _observable_key, init_observable


################################################################################
## ESTIMATOR
################################################################################
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
        backend: Backend,
        *,  # TODO: allow backend as positional after removing deprecations
        abelian_grouping: bool = True,
        skip_transpilation: bool = False,
        bound_pass_manager: PassManager | None = None,
        options: dict | None = None,
    ):
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
        self.abelian_grouping = abelian_grouping
        self.skip_transpilation = skip_transpilation  # TODO: tranpilation level
        self._bound_pass_manager = bound_pass_manager  # TODO: standardize
        super().__init__(
            circuits=None,
            observables=None,
            parameters=None,
            options=options,
        )
        self._transpile_options = Options()

    ################################################################################
    ## PROPERTIES
    ################################################################################
    @property
    def backend(self) -> Backend:  # TODO: normalize to one type of Backend
        """Backend to use for circuit measurements."""
        return self._backend

    @backend.setter
    def backend(self, backend: Backend) -> None:
        if not isinstance(backend, Backend):
            raise TypeError(
                "Expected `Backend` type for `backend`, " f"got `{type(backend)}` instead."
            )
        # TODO: clear all transpilation caching
        self._backend = backend

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

    def set_transpile_options(self, **fields):
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
        observables: tuple[BaseOperator | PauliSumOp, ...],
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
        job_circuits = tuple(c for bun in circuit_bundles for c in bun)
        # for circ in job_circuits:
        #     circ.metadata = {}

        # Raw results: counts
        job = self.backend.run(job_circuits, **run_options)
        raw_results: Result = job.result()
        counts: list[Counts] | Counts = raw_results.get_counts()

        # Post-processing
        if not isinstance(counts, list):
            counts = [counts]
        counts_iter = iter(counts)
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
        """Preprocess experiments to runnable lists of circuits: one list per experiment."""
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
        """Preprocess single experiment to runnable list of circuits."""
        circuit = self._transpile(circuit)  # TODO: Cache (produces a copy)
        circuit.assign_parameters(parameter_values, inplace=True)
        circuit = self._run_bound_pass_manager(circuit)
        measurements = self._build_measurement_circuits(observable)
        circs_w_meas = self._compose_measurements(circuit, measurements)
        return circs_w_meas

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
        expval = 0.0
        var = 0.0
        for counts, metadata in zip(counts_bundle, metadata_bundle):
            paulis = metadata["paulis"]
            coeffs = metadata["coeffs"]
            expvals, variances = self._pauli_expvals_with_variance(counts, paulis)
            expval += np.dot(expvals, coeffs)
            var += np.dot(variances, coeffs**2)
        shots = sum(counts_bundle[0].values())  # TODO: not correct -> counts.shots (?)
        return expval, var, shots

    ################################################################################
    ## CALCULATIONS
    ################################################################################
    @classmethod
    def _pauli_expvals_with_variance(
        cls, counts: Counts, paulis: PauliList
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return array of expval and variance pairs for input Paulis.

        Note: All non-identity Pauli's are treated as Z-paulis, assuming
        that basis rotations have been applied to convert them to the
        diagonal basis.
        """
        # Diag indices
        size = len(paulis)
        diag_inds = cls._paulis2inds(paulis)

        expvals = np.zeros(size, dtype=float)
        denom = 0  # Total shots for counts dict
        for bin_outcome, freq in counts.items():
            outcome = int(bin_outcome, 2)
            denom += freq
            for k in range(size):
                coeff = (-1) ** cls._parity_bit(diag_inds[k] & outcome)
                expvals[k] += freq * coeff

        # Divide by total shots
        expvals /= denom

        # Compute variance
        variances = 1 - expvals**2
        return tuple(expvals), tuple(variances)

    @staticmethod
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

    @staticmethod
    def _parity_bit(integer: int) -> int:
        """Return the parity bit of an integer."""
        return bin(integer).count("1") % 2

    ################################################################################
    ## TRANSPILATION
    ################################################################################
    # TODO: pass backend and run_options
    def _transpile(self, circuit: QuantumCircuit):
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
        basis: Pauli = self._observable_decomposer.get_measurement_basis(observable)
        circuit: QuantumCircuit = self._build_pauli_measurement(basis)
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
        base_circuit: QuantumCircuit,
        measurement_circuits: Sequence[QuantumCircuit] | QuantumCircuit,
    ) -> tuple[QuantumCircuit, ...]:
        """Compose measurement circuits with base circuit considering final layout."""
        if isinstance(measurement_circuits, QuantumCircuit):
            measurement_circuits = (measurement_circuits,)
        return tuple(
            self._compose_single_measurement(base_circuit, meas) for meas in measurement_circuits
        )

    def _compose_single_measurement(
        self, base_circuit: QuantumCircuit, measurement_circuit: QuantumCircuit
    ) -> QuantumCircuit:
        """Compose single measurement circuit with base circuit considering final layout."""
        transpile_options = {**self.transpile_options.__dict__}  # TODO: avoid multiple copies
        transpile_options.update({"initial_layout": base_circuit.metadata.get("final_layout")})
        measurement_circuit = transpile(measurement_circuit, self.backend, **transpile_options)
        circuit = base_circuit.compose(measurement_circuit)
        circuit.metadata = {**base_circuit.metadata, **measurement_circuit.metadata}
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

    @classmethod
    def decompose(cls, observable: BaseOperator | PauliSumOp) -> list[BaseOperator | PauliSumOp]:
        """Decomposes a given observable into singly measurable components.

        Note that component decomposition is not unique, for instance, commuting components
        could be grouped together in different ways (i.e. partinioning the set).

        Args:
            obsevable: the observable to decompose into its core components.

        Returns:
            A list of observables each of which measurable with a single quantum circuit
            (i.e. on a single Pauli basis).
        """
        # TODO: validation
        return cls._decompose(observable)

    @staticmethod
    @abstractmethod
    def _decompose(
        observable: BaseOperator | PauliSumOp,
    ) -> list[BaseOperator | PauliSumOp]:
        ...

    @classmethod
    def get_measurement_basis(cls, observable: BaseOperator | PauliSumOp) -> Pauli:
        """Get common Pauli basis for a given observable.

        Args:
            observable: an operator for which to obtain a common Pauli basis for measurement.

        Returns:
            A Pauli operator serving as a common basis for all components of the input observable.

        Raises:
            ValueError: if input observable does not have a common Pauli basis.
        """
        # TODO: validation
        if len(cls.decompose(observable)) != 1:
            raise ValueError("Unable to retrieve a common Pauli basis for the given observable.")
        return cls._get_measurement_basis(observable)

    @staticmethod
    @abstractmethod
    def _get_measurement_basis(observable: BaseOperator | PauliSumOp) -> Pauli:
        ...


class NaiveDecomposer(ObservableDecomposer):
    """Naive observable decomposition without grouping related components."""

    @staticmethod
    def _decompose(
        observable: BaseOperator | PauliSumOp,
    ) -> list[BaseOperator | PauliSumOp]:
        return list(observable)

    @staticmethod
    def _get_measurement_basis(component: BaseOperator | PauliSumOp) -> Pauli:
        return component.paulis[0]


class AbelianDecomposer(ObservableDecomposer):
    """Abelian observable decomposition grouping commuting components."""

    @staticmethod
    def _decompose(
        observable: BaseOperator | PauliSumOp,
    ) -> list[BaseOperator | PauliSumOp]:
        return observable.group_commuting(qubit_wise=True)

    @staticmethod
    def _get_measurement_basis(component: BaseOperator | PauliSumOp) -> Pauli:
        return Pauli(
            (np.logical_or.reduce(component.paulis.z), np.logical_or.reduce(component.paulis.x))
        )
