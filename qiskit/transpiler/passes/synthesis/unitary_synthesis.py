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
from qiskit.exceptions import QiskitError
from qiskit.extensions.quantum_initializer import isometry
from qiskit.quantum_info.synthesis import one_qubit_decompose
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitBasisDecomposer
from qiskit.circuit.library.standard_gates import (iSwapGate, CXGate, CZGate,
                                                   RXXGate, ECRGate)
from qiskit.transpiler.passes.synthesis import plugin


def _choose_kak_gate(basis_gates):
    """Choose the first available 2q gate to use in the KAK decomposition."""

    kak_gate_names = {
        'cx': CXGate(),
        'cz': CZGate(),
        'iswap': iSwapGate(),
        'rxx': RXXGate(pi / 2),
        'ecr': ECRGate()
    }

    kak_gate = None
    kak_gates = set(basis_gates or []).intersection(kak_gate_names.keys())
    if kak_gates:
        kak_gate = kak_gate_names[kak_gates.pop()]

    return kak_gate


def _choose_euler_basis(basis_gates):
    """"Choose the first available 1q basis to use in the Euler decomposition."""
    basis_set = set(basis_gates or [])

    for basis, gates in one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES.items():
        if set(gates).issubset(basis_set):
            return basis

    return None


class UnitarySynthesis(TransformationPass):
    """Synthesize gates according to their basis gates."""

    def __init__(self,
                 basis_gates: List[str],
                 approximation_degree: float = 1,
                 coupling_map=None,
                 method: str = None):
        """
        Synthesize unitaries over some basis gates.

        This pass can approximate 2-qubit unitaries given some approximation
        closeness measure (expressed as approximation_degree). Other unitaries
        are synthesized exactly.

        Args:
            basis_gates: List of gate names to target.
            approximation_degree: closeness of approximation (0: lowest, 1: highest).
        """
        super().__init__()
        self._basis_gates = basis_gates
        self._approximation_degree = approximation_degree
        self.method = method
        self._coupling_map = coupling_map
        self.plugins = plugin.UnitarySynthesisPluginManager()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the UnitarySynthesis pass on `dag`.

        Args:
            dag: input dag.

        Returns:
            Output dag with UnitaryGates synthesized to target basis.
        Raises:
            QiskitError: if a 'method' was specified for the class and is not
                found in the installed plugins list. The list of installed
                plugins can be queried with
                :func:`~qiskit.transpiler.passes.synthesis.plugins.unitary_synthesis_plugin_names`
        """
        if not self.method:
            method = 'default'
        else:
            method = self.method
        if method not in self.plugins.ext_plugins:
            raise QiskitError(
                    'Specified method: %s not found in plugin list' % method)
        plugin_method = self.plugins.ext_plugins[method].obj
        if plugin_method.supports_coupling_map:
            dag_bit_indices = {bit: idx
                               for idx, bit in enumerate(dag.qubits)}
        for node in dag.named_nodes('unitary'):
            synth_dag = None
            kwargs = {}
            if plugin_method.supports_basis_gates:
                kwargs['basis_gates'] = self._basis_gates
            if plugin_method.supports_coupling_map:
                kwargs['coupling_map'] = self._coupling_map
                kwargs['qubits'] = [dag_bit_indices[x] for x in node.qargs]
            if plugin_method.supports_approximation_degree:
                kwargs['approximation_degree'] = self._approximation_degree
            unitary = node.op.to_matrix()
            synth_dag = plugin_method.run(unitary, **kwargs)
            if synth_dag:
                dag.substitute_node_with_dag(node, synth_dag)
        return dag


class DefaultUnitarySynthesis(plugin.UnitarySynthesisPlugin):
    """The default unitary synthesis plugin."""

    @property
    def supports_basis_gates(self):
        return True

    @property
    def supports_coupling_map(self):
        return False

    @property
    def supports_approximation_degree(self):
        return True

    def run(self, unitary, **options):
        basis_gates = options['basis_gates']
        approximation_degree = options['approximation_degree']
        euler_basis = _choose_euler_basis(basis_gates)
        kak_gate = _choose_kak_gate(basis_gates)

        decomposer1q, decomposer2q = None, None
        if euler_basis is not None:
            decomposer1q = one_qubit_decompose.OneQubitEulerDecomposer(euler_basis)
        if kak_gate is not None:
            decomposer2q = TwoQubitBasisDecomposer(kak_gate, euler_basis=euler_basis)
        if unitary.shape == (2, 2):
            if decomposer1q is None:
                return None
            synth_dag = circuit_to_dag(decomposer1q._decompose(unitary))
        elif unitary.shape == (4, 4):
            if decomposer2q is None:
                return None
            synth_dag = circuit_to_dag(decomposer2q(unitary,
                                                    basis_fidelity=approximation_degree))
        else:
            synth_dag = circuit_to_dag(
                isometry.Isometry(unitary.op.to_matrix(), 0, 0).definition)
        return synth_dag
