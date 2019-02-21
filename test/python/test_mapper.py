# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring,redefined-builtin

import unittest
from unittest.mock import patch

from qiskit import compile, execute
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import BasicAer
from qiskit.qobj import Qobj
from qiskit.transpiler._transpiler import transpile_dag
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.mapper.compiling import two_qubit_kak
from qiskit.tools.qi.qi import random_unitary_matrix
from qiskit.mapper.mapping import MapperError
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase, Path
from qiskit.test.mock import FakeTenerife, FakeRueschlikon


class TestMapper(QiskitTestCase):
    """Test the mapper."""


    barrier_pass = BarrierBeforeFinalMeasurements()

    @patch.object(BarrierBeforeFinalMeasurements, 'run', wraps=barrier_pass.run)
    def test_final_measurement_barrier_for_devices(self, mock_pass):
        """Verify BarrierBeforeFinalMeasurements pass is called in default pipeline for devices."""

        circ = QuantumCircuit.from_qasm_file(self._get_resource_path('example.qasm', Path.QASMS))
        dag_circuit = circuit_to_dag(circ)
        transpile_dag(dag_circuit, coupling_map=FakeRueschlikon().configuration().coupling_map)

        self.assertTrue(mock_pass.called)

    @patch.object(BarrierBeforeFinalMeasurements, 'run', wraps=barrier_pass.run)
    def test_final_measurement_barrier_for_simulators(self, mock_pass):
        """Verify BarrierBeforeFinalMeasurements pass is in default pipeline for simulators."""
        circ = QuantumCircuit.from_qasm_file(self._get_resource_path('example.qasm', Path.QASMS))
        dag_circuit = circuit_to_dag(circ)
        transpile_dag(dag_circuit)

        self.assertTrue(mock_pass.called)


if __name__ == '__main__':
    unittest.main()
