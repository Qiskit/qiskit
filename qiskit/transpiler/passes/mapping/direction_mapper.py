# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
A direction mapper rearrenges the direction of the cx nodes to make the circuit
compatible to the directed coupling map.
"""
from copy import copy

from qiskit.transpiler._basepasses import TransformationPass
from qiskit.transpiler import MapperError
from qiskit.dagcircuit import DAGCircuit
from qiskit.mapper import Layout
from qiskit.extensions.standard import HGate


class DirectionMapper(TransformationPass):
    """
     Rearrenges the direction of the cx nodes to make the circuit
     compatible to the directed coupling map.
    """

    def __init__(self, coupling_map, h_gate=None, initial_layout=None):
        """
        Args:
            coupling_map (Coupling): Directed graph represented a coupling map.
            h_gate (Type):  Default: HGate. The Gate class that defines a Hadamard (H) gate.
            initial_layout (Layout): The initial layout of the DAG.
        """

        super().__init__()
        self.coupling_map = coupling_map
        self.initial_layout = initial_layout
        self.swap_gate = h_gate if h_gate is not None else HGate

    def run(self, dag):
        """
        Flips the cx nodes to match the directed coupling map.
        Args:
            dag (DAGCircuit): DAG to map.
        Raises:
            TranspilerError: If the `dag` cannot be mapped just by flipping the cx nodes.
        """
        new_dag = DAGCircuit()

        if self.initial_layout is None:
            # create a one-to-one layout
            self.initial_layout = Layout()
            wire_no = 0
            for qreg in dag.qregs.values():
                for index in range(qreg.size):
                    self.initial_layout[(qreg, index)] = wire_no
                    wire_no += 1
        current_layout = copy(self.initial_layout)

        for layer in dag.serial_layers():
            subdag = layer['graph']

            for cnot in subdag.get_cnot_nodes():
                op = cnot['op']
                physical_q0 = current_layout[op.qargs[0]]
                physical_q1 = current_layout[op.qargs[1]]
                if self.coupling_map.distance(physical_q0, physical_q1) != 1:
                    raise MapperError('The circuit requires a connectiontion between the phsycial '
                                      'qubits %s and %s' % (physical_q0, physical_q1))
            edge_map = current_layout.combine_into_edge_map(self.initial_layout)
            new_dag.extends_at_the_end(subdag, edge_map)

        return new_dag
