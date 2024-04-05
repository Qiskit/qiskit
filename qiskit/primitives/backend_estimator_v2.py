# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Estimator V2 implementation for an arbitrary Backend object."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
import math

import numpy as np

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.exceptions import QiskitError
from qiskit.providers import BackendV1, BackendV2
from qiskit.quantum_info import Pauli, PauliList
from qiskit.transpiler import PassManager, PassManagerConfig
from qiskit.transpiler.passes import Optimize1qGatesDecomposition

from .backend_estimator import _pauli_expval_with_variance, _prepare_counts, _run_circuits
from .base import BaseEstimatorV2
from .containers import EstimatorPubLike, PrimitiveResult, PubResult
from .containers.bindings_array import BindingsArray
from .containers.estimator_pub import EstimatorPub
from .primitive_job import PrimitiveJob


@dataclass
class Options:
    """Options for :class:`~.BackendEstimatorV2`."""

    default_precision: float = 0.015625
    """The default precision to use if none are specified in :meth:`~run`.
    Default: 0.015625 (1 / sqrt(4096)).
    """

    abelian_grouping: bool = True
    """Whether the observables should be grouped into sets of qubit-wise commuting observables.
    Default: True.
    """

    seed_simulator: int | None = None
    """The seed to use in the simulator. If None, a random seed will be used.
    Default: None.
    """


class BackendEstimatorV2(BaseEstimatorV2):
    """Evaluates expectation values for provided quantum circuit and observable combinations

    The :class:`~.BackendEstimatorV2` class is a generic implementation of the
    :class:`~.BaseEstimatorV2` interface that is used to wrap a :class:`~.BackendV2`
    (or :class:`~.BackendV1`) object in the :class:`~.BaseEstimatorV2` API. It
    facilitates using backends that do not provide a native
    :class:`~.BaseEstimatorV2` implementation in places that work with
    :class:`~.BaseEstimatorV2`. However,
    if you're using a provider that has a native implementation of
    :class:`~.BaseEstimatorV2`, it is a better choice to leverage that native
    implementation as it will likely include additional optimizations and be
    a more efficient implementation. The generic nature of this class
    precludes doing any provider- or backend-specific optimizations.

    This class does not perform any measurement or gate mitigation, and, presently, is only
    compatible with Pauli-based observables.

    Each tuple of ``(circuit, observables, <optional> parameter values, <optional> precision)``,
    called an estimator primitive unified bloc (PUB), produces its own array-based result. The
    :meth:`~.BackendEstimatorV2.run` method can be given a sequence of pubs to run in one call.

    The options for :class:`~.BackendEstimatorV2` consist of the following items.

    * ``default_precision``: The default precision to use if none are specified in :meth:`~run`.
      Default: 0.015625 (1 / sqrt(4096)).

    * ``abelian_grouping``: Whether the observables should be grouped into sets of qubit-wise
      commuting observables.
      Default: True.

    * ``seed_simulator``: The seed to use in the simulator. If None, a random seed will be used.
      Default: None.
    """

    def __init__(
        self,
        *,
        backend: BackendV1 | BackendV2,
        options: dict | None = None,
    ):
        """
        Args:
            backend: The backend to run the primitive on.
            options: The options to control the default precision (``default_precision``),
                the operator grouping (``abelian_grouping``), and
                the random seed for the simulator (``seed_simulator``).
        """
        self._backend = backend
        self._options = Options(**options) if options else Options()

        basis = PassManagerConfig.from_backend(backend).basis_gates
        if isinstance(backend, BackendV2):
            opt1q = Optimize1qGatesDecomposition(basis=basis, target=backend.target)
        else:
            opt1q = Optimize1qGatesDecomposition(basis=basis)
        self._passmanager = PassManager([opt1q])

    @property
    def options(self) -> Options:
        """Return the options"""
        return self._options

    @property
    def backend(self) -> BackendV1 | BackendV2:
        """Returns the backend which this sampler object based on."""
        return self._backend

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        if precision is None:
            precision = self._options.default_precision
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        self._validate_pubs(coerced_pubs)
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _validate_pubs(self, pubs: list[EstimatorPub]):
        for i, pub in enumerate(pubs):
            if pub.precision <= 0.0:
                raise ValueError(
                    f"The {i}-th pub has precision less than or equal to 0 ({pub.precision}). ",
                    "But precision should be larger than 0.",
                )

    def _run(self, pubs: list[EstimatorPub]) -> PrimitiveResult[PubResult]:
        return PrimitiveResult([self._run_pub(pub) for pub in pubs])

    def _run_pub(self, pub: EstimatorPub) -> PubResult:
        shots = math.ceil(1.0 / pub.precision**2)
        circuit = pub.circuit
        observables = pub.observables
        parameter_values = pub.parameter_values

        # calculate broadcasting of parameters and observables
        param_shape = parameter_values.shape
        param_indices = np.fromiter(np.ndindex(param_shape), dtype=object).reshape(param_shape)
        bc_param_ind, bc_obs = np.broadcast_arrays(param_indices, observables)

        # calculate expectation values for each pair of parameter value set and pauli
        param_obs_map = defaultdict(set)
        for index in np.ndindex(*bc_param_ind.shape):
            param_index = bc_param_ind[index]
            param_obs_map[param_index].update(bc_obs[index].keys())
        expval_map = self._calc_expval_paulis(circuit, parameter_values, param_obs_map, shots)

        # calculate expectation values (evs) and standard errors (stds)
        evs = np.zeros_like(bc_param_ind, dtype=float)
        variances = np.zeros_like(bc_param_ind, dtype=float)
        for index in np.ndindex(*bc_param_ind.shape):
            param_index = bc_param_ind[index]
            for pauli, coeff in bc_obs[index].items():
                expval, variance = expval_map[param_index, pauli]
                evs[index] += expval * coeff
                variances[index] += variance * coeff**2
        stds = np.sqrt(variances / shots)
        data_bin_cls = self._make_data_bin(pub)
        data_bin = data_bin_cls(evs=evs, stds=stds)
        return PubResult(data_bin, metadata={"target_precision": pub.precision})

    def _calc_expval_paulis(
        self,
        circuit: QuantumCircuit,
        parameter_values: BindingsArray,
        param_obs_map: dict[tuple[int, ...], set[str]],
        shots: int,
    ) -> dict[tuple[tuple[int, ...], str], tuple[float, float]]:
        # generate circuits
        circuits = []
        for param_index, pauli_strings in param_obs_map.items():
            bound_circuit = parameter_values.bind(circuit, param_index)
            # sort pauli_strings so that the order is deterministic
            meas_paulis = PauliList(sorted(pauli_strings))
            new_circuits = self._preprocessing(bound_circuit, meas_paulis, param_index)
            circuits.extend(new_circuits)

        # run circuits
        result, metadata = _run_circuits(
            circuits, self._backend, shots=shots, seed_simulator=self._options.seed_simulator
        )

        # postprocessing results
        expval_map: dict[tuple[tuple[int, ...], str], tuple[float, float]] = {}
        counts = _prepare_counts(result)
        for count, meta in zip(counts, metadata):
            orig_paulis = meta["orig_paulis"]
            meas_paulis = meta["meas_paulis"]
            param_index = meta["param_index"]
            expvals, variances = _pauli_expval_with_variance(count, meas_paulis)
            for pauli, expval, variance in zip(orig_paulis, expvals, variances):
                expval_map[param_index, pauli.to_label()] = (expval, variance)
        return expval_map

    def _preprocessing(
        self, circuit: QuantumCircuit, observable: PauliList, param_index: tuple[int, ...]
    ) -> list[QuantumCircuit]:
        # generate measurement circuits with metadata
        meas_circuits: list[QuantumCircuit] = []
        if self._options.abelian_grouping:
            for obs in observable.group_commuting(qubit_wise=True):
                basis = Pauli((np.logical_or.reduce(obs.z), np.logical_or.reduce(obs.x)))
                meas_circuit, indices = _measurement_circuit(circuit.num_qubits, basis)
                paulis = PauliList.from_symplectic(
                    obs.z[:, indices],
                    obs.x[:, indices],
                    obs.phase,
                )
                meas_circuit.metadata = {
                    "orig_paulis": obs,
                    "meas_paulis": paulis,
                    "param_index": param_index,
                }
                meas_circuits.append(meas_circuit)
        else:
            for basis in observable:
                meas_circuit, indices = _measurement_circuit(circuit.num_qubits, basis)
                obs = PauliList(basis)
                paulis = PauliList.from_symplectic(
                    obs.z[:, indices],
                    obs.x[:, indices],
                    obs.phase,
                )
                meas_circuit.metadata = {
                    "orig_paulis": obs,
                    "meas_paulis": paulis,
                    "param_index": param_index,
                }
                meas_circuits.append(meas_circuit)

        # unroll basis gates
        meas_circuits = self._passmanager.run(meas_circuits)

        # combine measurement circuits
        preprocessed_circuits = []
        for meas_circuit in meas_circuits:
            circuit_copy = circuit.copy()
            # meas_circuit is supposed to have a classical register whose name is different from
            # those of the transpiled_circuit
            clbits = meas_circuit.cregs[0]
            for creg in circuit_copy.cregs:
                if clbits.name == creg.name:
                    raise QiskitError(
                        "Classical register for measurements conflict with those of the input "
                        f"circuit: {clbits}. "
                        "Recommended to avoid register names starting with '__'."
                    )
            circuit_copy.add_register(clbits)
            circuit_copy.compose(meas_circuit, clbits=clbits, inplace=True)
            circuit_copy.metadata = meas_circuit.metadata
            preprocessed_circuits.append(circuit_copy)
        return preprocessed_circuits


def _measurement_circuit(num_qubits: int, pauli: Pauli):
    # Note: if pauli is I for all qubits, this function generates a circuit to measure only
    # the first qubit.
    # Although such an operator can be optimized out by interpreting it as a constant (1),
    # this optimization requires changes in various methods. So it is left as future work.
    qubit_indices = np.arange(pauli.num_qubits)[pauli.z | pauli.x]
    if not np.any(qubit_indices):
        qubit_indices = [0]
    meas_circuit = QuantumCircuit(
        QuantumRegister(num_qubits, "q"), ClassicalRegister(len(qubit_indices), f"__c_{pauli}")
    )
    for clbit, i in enumerate(qubit_indices):
        if pauli.x[i]:
            if pauli.z[i]:
                meas_circuit.sdg(i)
            meas_circuit.h(i)
        meas_circuit.measure(i, clbit)
    return meas_circuit, qubit_indices
