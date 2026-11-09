# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Remove the gates that do not affect the outcome of a measurement on a circuit."""

from __future__ import annotations


from qiskit._accelerate.light_cone import light_cone
from qiskit.circuit.commutation_library import SessionCommutationChecker as scc
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import SparseObservable
from qiskit.transpiler.basepasses import TransformationPass


class LightCone(TransformationPass):
    """Remove the gates that do not affect the outcome of a measurement on a circuit.

    Pass for computing the light-cone of an observable or measurement. The Pass can handle
    either an observable one would like to measure or a measurement on a set of qubits.

    The pass only removes gates that commute with every later gate and measurement in the light
    cone, so it may keep some gates that do not affect the outcome.
    """

    def __init__(self, bit_terms: str | None = None, indices: list[int] | None = None) -> None:
        """
        Args:
            bit_terms: If ``None`` the light-cone will be computed for the set of measurements
                in the circuit. If a string is specified, the light-cone will correspond to the
                reduced circuit with the same expectation value for the observable.
            indices: list of non-trivial indices corresponding to the observable in ``bit_terms``.
                As in :meth:`.SparseObservable.to_sparse_list`, ``bit_terms[i]`` acts on qubit
                ``indices[i]``.
        """
        super().__init__()
        valid_characters = {"X", "Y", "Z", "+", "-", "l", "r", "0", "1"}
        self.bit_terms = bit_terms
        self.indices = indices
        if bit_terms is not None:
            if not indices:
                raise ValueError("`indices` must be non-empty when providing `bit_terms`.")
            if not set(bit_terms).issubset(valid_characters):
                raise ValueError(
                    f"`bit_terms` should contain only characters in {valid_characters}."
                )
            if len(bit_terms) != len(indices):
                raise ValueError("`bit_terms` must be the same length as `indices`.")

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the LightCone pass on `dag`.

        Args:
            dag: The DAG to reduce.

        Returns:
            The DAG reduced to the light cone of the observable or of the measurements.

        Raises:
            ValueError: if an observable is given and the circuit contains measurements.
        """
        observable = None
        if self.bit_terms is not None:
            if dag.num_qubits() < max(self.indices) + 1:
                raise ValueError("`indices` contains values outside the qubit range.")
            observable = SparseObservable.from_sparse_list(
                [(self.bit_terms, self.indices, 1.0)], dag.num_qubits()
            )
        light_cone(dag, observable, scc.cc)
        return dag
