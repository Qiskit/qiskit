# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-function-docstring

"""EXAMPLE: Translates CX gates to measure and feed-forward controlled-X gates."""

from qiskit.circuit import ClassicalRegister, QuantumRegister
from qiskit.circuit.library import Measure, XGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass

class DynamicReplacement(TransformationPass):

    def run(self, dag):
        mr = ClassicalRegister(1, 'c') # measurement register
        dag.add_clbits(mr) 
        
        for cx_run in dag.collect_runs('cx'):
            for node in cx_run:
                q0, q1 = QuantumRegister(2, 'q')

                mini_dag = DAGCircuit()
                mini_dag.add_qreg(q0.register)
                mini_dag.add_creg(mr)

                mini_dag.apply_operation_back(Measure(), [q0], mr)
                mini_dag.apply_operation_back(XGate().c_if(mr[0], 1), [q1])

                dag.substitute_node_with_dag(node, mini_dag) # mismatch between number of wires

        return dag
