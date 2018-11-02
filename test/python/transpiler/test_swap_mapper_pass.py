# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""SwapMapper pass testing"""

import unittest
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import PassManager, transpile_dag
from qiskit.transpiler.passes import SwapMapper
from ..common import QiskitTestCase


class TestSwapMapperPass(QiskitTestCase):
    """ Tests for SwapMapper pass. """

    def test_pass_swap_mapper(self):
        """Test the swap mapper pass.
         It rewrites based on which qbuits are connected in the coupling map.
        """
        q = QuantumRegister(3)
        circ = QuantumCircuit(q)
        circ.cx(q[0], q[1])
        dag_circuit = DAGCircuit.fromQuantumCircuit(circ)
        coupling_map = [[1, 2], [2, 0]]
        pass_manager = PassManager()

        pass_manager.add_passes(SwapMapper(coupling_map=coupling_map, seed=42))
        dag_circuit = transpile_dag(dag_circuit, pass_manager=pass_manager)
        self.assertEqual('cx q[2],q[1];\n', dag_circuit.qasm(no_decls=True))
        self.assertDictEqual(pass_manager.property_set['layout'],
                             {(q.name, 0): ('q', 2), (q.name, 1): ('q', 1), (q.name, 2): ('q', 0)})

if __name__ == '__main__':
    unittest.main()