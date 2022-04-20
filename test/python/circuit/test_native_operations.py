# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""This is a temporary test file to experiment with natively adding operations
to a quantum circuit."""

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.compiler.transpiler import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.dagcircuit.dagnode import DAGOpNode


def create_clifford():
    """Comment so that pylint does not complain."""
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.h(0)
    qc.s(1)
    qc.swap(1, 2)
    # print(qc)
    cliff = Clifford(qc)
    print(cliff)
    return cliff


def show_circuit(circ):
    """Comment so that pylint does not complain."""
    for inst, qargs, cargs in circ.data:
        print(f"  {type(inst)}, {qargs}, {cargs}")


def show_dag(dag):
    """Comment so that pylint does not complain."""
    for node in dag.topological_nodes():
        print(f"-- {type(node)}")
        if isinstance(node, DAGOpNode):
            print(f"---{type(node.op)}, {node.op}")


def run_test():
    """Comment so that pylint does not complain."""
    print("Creating Clifford")
    cliff = create_clifford()
    qc0 = QuantumCircuit(4)
    print("Appending Clifford")
    qc0.append(cliff, [0, 1, 2])
    print(qc0)
    show_circuit(qc0)
    print("Converting to dag")
    dag1 = circuit_to_dag(qc0)
    print(dag1)
    show_dag(dag1)
    print("Running transpiler pass on dag")
    dag2 = Unroll3qOrMore().run(dag1)
    print(dag2)
    print("Converting back to circuit")
    qc3 = dag_to_circuit(dag2)
    print(qc3)


if __name__ == "__main__":
    run_test()
