# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
The CX direction rearrenges the direction of the cx nodes to make the circuit
compatible with the coupling_map.
"""

from qiskit.transpiler._basepasses import TransformationPass
from qiskit.transpiler import MapperError
from qiskit.dagcircuit import DAGCircuit
from qiskit.mapper import Layout
from qiskit.extensions.standard import HGate


class CXDirection(TransformationPass):
    """
     Rearranges the direction of the cx nodes to make the circuit
     compatible with the directed coupling map.

     It uses this equivalence::

        ---(+)---      --[H]---.---[H]--
            |      =           |
        ----.----      --[H]--(+)--[H]--
    """

    def __init__(self, coupling_map, initial_layout=None):
        """
        Args:
            coupling_map (CouplingMap): Directed graph represented a coupling map.
            initial_layout (Layout): The initial layout of the DAG.
        """

        super().__init__()
        self.coupling_map = coupling_map
        self.layout = initial_layout

    def run(self, dag):
        """
        Flips the cx nodes to match the directed coupling map.
        Args:
            dag (DAGCircuit): DAG to map.
        Returns:
            DAGCircuit: The rearranged dag for the coupling map

        Raises:
            MapperError: If the circuit cannot be mapped just by flipping the
                         cx nodes.
        """
        new_dag = DAGCircuit()

        if self.layout is None:
            # create a one-to-one layout
            self.layout = Layout()
            wire_no = 0
            for qreg in dag.qregs.values():
                for index in range(qreg.size):
                    self.layout[(qreg, index)] = wire_no
                    wire_no += 1

        for layer in dag.serial_layers():
            subdag = layer['graph']

            for cnot in subdag.get_cnot_nodes():

                control = cnot['op'].qargs[0]
                target = cnot['op'].qargs[1]

                physical_q0 = self.layout[control]
                physical_q1 = self.layout[target]
                if self.coupling_map.distance(physical_q0, physical_q1) != 1:
                    raise MapperError('The circuit requires a connectiontion between the phsycial '
                                      'qubits %s and %s' % (physical_q0, physical_q1))

                if (physical_q0, physical_q1) not in self.coupling_map.get_edges():
                    # A flip needs to be done

                    # Create the involved registers
                    if control[0] not in subdag.qregs.values():
                        subdag.add_qreg(control[0])
                    if target[0] not in subdag.qregs.values():
                        subdag.add_qreg(target[0])

                    # Add H gates around
                    subdag.add_basis_element('h', 1, 0, 0)
                    subdag.apply_operation_back(HGate(target))
                    subdag.apply_operation_back(HGate(control))
                    subdag.apply_operation_front(HGate(target))
                    subdag.apply_operation_front(HGate(control))

                    # Flips the CX
                    cnot['op'].qargs[0], cnot['op'].qargs[1] = target, control

            new_dag.extend_back(subdag)

        return new_dag
