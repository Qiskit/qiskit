# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Cancel the redundant (self-adjoint) gates through commutation relations."""

from collections import defaultdict

import numpy as np

from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.commutation_checker import CommutationChecker
from qiskit.circuit.commutation_library import standard_gates_commutations
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.library.generalized_gates.pauli import PauliGate
from qiskit.circuit.library.standard_gates.p import PhaseGate
from qiskit.circuit.library.standard_gates.rx import RXGate
from qiskit.circuit.library.standard_gates.rz import RZGate
from qiskit.circuit.library.standard_gates.u1 import U1Gate
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit, DAGInNode, DAGNode, DAGOutNode
from qiskit.quantum_info import Pauli
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.optimization.commutation_analysis import \
    CommutationAnalysis
from qiskit.transpiler.passmanager import PassManager

commutator = CommutationChecker(standard_gates_commutations)


class LightCone(TransformationPass):
    """Remove the gates that do not affect the outcome of a measurement on a circuit.

    Pass for computing the light-cone of an observable or measurement. The Pass can
    handle either a Pauli operator one would like to measure or a measurement on a set
    of qubits.
    """

    def __init__(self, observable: Pauli | None = None):
        """
        LightCone initializer.

        Args:
            observable: If None the lightcone will be computed for the set
            of measurements in the circuit.
            If a Pauli operator is specified, the lightcone will correspond to
            the reduced circuit with the
            same expectation value for the observable.
        """
        super().__init__()
        self.observable = observable

    @staticmethod
    def _find_measurement_qubits(dag: DAGCircuit):
        """For now this method assumes that all the measurements in the circuit are
        final measurements."""
        qubits_measured = set()
        for node in dag.topological_nodes():
            # print(getattr(node, "name", False))
            if getattr(node, "name", False) == "measure":
                qubits_measured |= {x for x in node.qargs}
        print(qubits_measured)
        return qubits_measured

    def _get_initial_lightcone(
        self, dag: DAGCircuit
    ) -> tuple[set[int], tuple[Instruction, list[int]]]:
        """Returns the initial lightcone.
        If obervable is None, the lightcone is a different Z operator in each one of the
        qubits with a measurement. If a Pauli observable is defined, that will be
        the first gate in the circuit.
        """
        if self.observable is None:
            qubits_measured = self._find_measurement_qubits(dag)
            light_cone = qubits_measured
            light_cone_ops = [(PauliGate("Z"), [qubit_index]) for qubit_index in qubits_measured]
        else:
            pauli_string = str(self.observable)
            light_cone = [dag.qubits[i] for i, p in enumerate(pauli_string) if p != "I"]
            light_cone_ops = [(PauliGate(pauli_string.replace("I", "")), light_cone)]

        return set(light_cone), light_cone_ops

    def run(self, dag: DAGCircuit):
        """Run the LightCone pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        light_cone, light_cone_ops = self._get_initial_lightcone(dag)

        print(light_cone, light_cone_ops)
        # iterate over the nodes in reverse topological order
        for node in reversed(list(dag.topological_op_nodes())):
            if not light_cone.intersection(node.qargs):
                dag.remove_op_node(node)
                continue

            commutes_bool = True
            for op in light_cone_ops:
                commute_bool = commutator.commute(op[0], op[1], [], node.op, node.qargs, [])
                if not commute_bool:
                    light_cone.update(node.qargs)
                    light_cone_ops.append((node.op, node.qargs))
                    commutes_bool = False
                    break

            if commutes_bool:
                dag.remove_op_node(node)

        return dag


if __name__ == "__main__":
    """This main will be only for testing purposes and will be removed
    before merging the PR."""
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.converters import circuit_to_dag

    qc = RealAmplitudes(10, entanglement="pairwise").decompose()
    pm = PassManager([LightCone(Pauli("I" * 9 + "Z"))])
    pm_measure = PassManager([LightCone()])
    # print(qc)
    print(pm.run(qc))
    qc.add_register(ClassicalRegister(2))
    qc.measure(1, 0)
    qc.measure(5, 1)
    print(pm_measure.run(qc))
