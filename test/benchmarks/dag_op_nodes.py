# This code is part of Qiskit.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Benchmarks for :meth:`.DAGCircuit.op_nodes` type filtering."""

from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister, Measure, Reset
from qiskit.circuit.library import HGate


def _build_mixed_dag(num_ops):
    """Build a DAG with ~80% H gates, ~10% Measure, and ~10% Reset."""
    dag = DAGCircuit()
    qreg = QuantumRegister(1, "q")
    creg = ClassicalRegister(1, "c")
    dag.add_qreg(qreg)
    dag.add_creg(creg)
    qubit = qreg[0]
    clbit = creg[0]
    for index in range(num_ops):
        bucket = index % 10
        if bucket == 0:
            dag.apply_operation_back(Measure(), [qubit], [clbit])
        elif bucket == 1:
            dag.apply_operation_back(Reset(), [qubit], [])
        else:
            dag.apply_operation_back(HGate(), [qubit], [])
    return dag


class OpNodesBenchmark:
    """Compare separate and combined ``op_nodes`` type filters."""

    params = ([1000, 10000],)
    param_names = ["num_ops"]
    timeout = 300

    def setup(self, num_ops):
        self.dag = _build_mixed_dag(num_ops)

    def time_op_nodes_separate_measure_reset(self, _):
        len(self.dag.op_nodes(op=Measure))
        len(self.dag.op_nodes(op=Reset))

    def time_op_nodes_combined_measure_reset(self, _):
        len(self.dag.op_nodes(op={Measure, Reset}))

    def time_op_nodes_single_hgate(self, _):
        len(self.dag.op_nodes(op=HGate))
