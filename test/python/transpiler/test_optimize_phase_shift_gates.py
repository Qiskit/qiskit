# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for pass marging/canceling phase-shift gates and cnot cancellation."""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit import BasicAer, execute
from qiskit.test import QiskitTestCase


class TestOptimizePhaseShiftGates(QiskitTestCase):
    """Test the TestOptimizePhaseShiftGates pass."""

    def test_pass_optimize_phase_shift_gates(self):
        """Test the cx TestOptimizePhaseShiftGates pass.

        It should merge/cancel phase-shift gates and perform cnot cancellation.
        """
        n = 6
        q = QuantumRegister(n, 'q')
        circ = QuantumCircuit(q)
        i = 0
        while i < 150:
            i += 1
            x = np.random.randint(9)
            if x == 0:
                circ.t(q[np.random.randint(n)])
            elif x == 1:
                circ.h(q[np.random.randint(n)])
            elif x == 2:
                circ.tdg(q[np.random.randint(n)])
            elif x == 3:
                circ.s(q[np.random.randint(n)])
            elif x == 4:
                circ.sdg(q[np.random.randint(n)])
            elif x == 5:
                circ.z(q[np.random.randint(n)])
            elif x == 6:
                circ.u1(np.random.random(2)[0], q[np.random.randint(n)])
            elif x == 7:
                y = np.random.randint(n-1)
                circ.cx(q[y], q[y+1])
            elif x == 8:
                y = np.random.randint(n-1)
                circ.cx(q[y+1], q[y])
        circ_dag = qiskit.converters.circuit_to_dag(circ)
        circ_dag = qiskit.transpiler.passes.OptimizePhaseShiftGates.run(circ_dag, circ_dag)
        circ_new = qiskit.converters.dag_to_circuit(circ_dag)
        backend_sim = BasicAer.get_backend('unitary_simulator')
        result_original = execute(circ, backend_sim).result()
        unitary_original = result_original.get_unitary(circ)
        backend_sim = BasicAer.get_backend('unitary_simulator')
        result_optimized = execute(circ_new, backend_sim).result()
        unitary_optimized = result_optimized.get_unitary(circ_new)
        self.assertTrue(np.allclose(unitary_optimized, unitary_original))
