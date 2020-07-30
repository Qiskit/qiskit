# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Synthesize UnitaryGates."""

from math import pi
from typing import List

from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import iSwapGate, CXGate, CZGate, RXXGate
from qiskit.extensions.quantum_initializer import isometry
from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitBasisDecomposer


def _choose_kak_gate(basis_gates):
    """Choose the first available 2q gate to use in the KAK decomposition."""

    kak_gate_names = {
        'cx': CXGate(),
        'cz': CZGate(),
        'iswap': iSwapGate(),
        'rxx': RXXGate(pi / 2),
    }

    kak_gate = None
    kak_gates = set(basis_gates or []).intersection(kak_gate_names.keys())
    if kak_gates:
        kak_gate = kak_gate_names[kak_gates.pop()]

    return kak_gate


def _choose_euler_basis(basis_gates):
    """"Choose the first available 1q basis to use in the Euler decomposition."""

    euler_basis_names = {
        'U3': ['u3'],
        'U1X': ['u1', 'rx'],
        'RR': ['r'],
        'ZYZ': ['rz', 'ry'],
        'ZXZ': ['rz', 'rx'],
        'XYX': ['rx', 'ry'],
    }

    basis_set = set(basis_gates or [])

    for basis, gates in euler_basis_names.items():
        if set(gates).issubset(basis_set):
            return basis

    return None


class UnitarySynthesis(TransformationPass):
    """Synthesize gates according to their basis gates."""

    def __init__(self, basis_gates: List[str]):
        """SynthesizeUnitaries initializer.

        Args:
            basis_gates: List of gate names to target.
        """
        super().__init__()
        self._basis_gates = basis_gates

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the UnitarySynthesis pass on `dag`.

        Args:
            dag: input dag.

        Returns:
            Output dag with UnitaryGates synthesized to target basis.
        """

        euler_basis = _choose_euler_basis(self._basis_gates)
        kak_gate = _choose_kak_gate(self._basis_gates)

        decomposer1q, decomposer2q = None, None
        if euler_basis is not None:
            decomposer1q = OneQubitEulerDecomposer(euler_basis)
        if kak_gate is not None:
            decomposer2q = TwoQubitBasisDecomposer(kak_gate, euler_basis=euler_basis)

        for node in dag.named_nodes('unitary'):

            synth_dag = None
            if len(node.qargs) == 1:
                if decomposer1q is None:
                    continue
                synth_dag = circuit_to_dag(decomposer1q(node.op.to_matrix()))
            elif len(node.qargs) == 2:
                if decomposer2q is None:
                    continue
                synth_dag = circuit_to_dag(decomposer2q(node.op.to_matrix()))
            else:
                synth_dag = circuit_to_dag(
                    isometry.Isometry(node.op.to_matrix(), 0, 0).definition)

            dag.substitute_node_with_dag(node, synth_dag)

        return dag
