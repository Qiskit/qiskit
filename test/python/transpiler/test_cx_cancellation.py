# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for pass cancelling 2 consecutive CNOTs on the same qubits."""

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import PassManager, transpile_dag
from qiskit.transpiler.passes import CXCancellation
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase


class TestCXCancellation(QiskitTestCase):
    """Test the CXCancellation pass."""

    def test_pass_cx_cancellation(self):
        """Test the cx cancellation pass.

        It should cancel consecutive cx pairs on same qubits.
        """
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[1], qr[0])
        dag_circuit = circuit_to_dag(circuit)

        pass_manager = PassManager()
        pass_manager.append(CXCancellation())
        dag_circuit = transpile_dag(dag_circuit, pass_manager=pass_manager)
        resources_after = dag_circuit.count_ops()

        self.assertNotIn('cx', resources_after)
