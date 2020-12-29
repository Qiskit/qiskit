# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""XY4 is a basic dynamical decoupling (DD) sequence that applies corrective pulses around
both the X and Y axes, to correct multiple components of the system-environment interaction.
The sequence is comprised of an X gate, Y gate, X gate, and Y gate with a fixed delay between
each pulse.

This implementation saturates idle periods of any qubit with XY4 sequences.
"""
from qiskit.circuit.delay import Delay
from qiskit.circuit.library.standard_gates import XGate, YGate
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.dagcircuit import DAGCircuit

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.basis.unroller import Unroller


class XY4(TransformationPass):
    """Pass which inserts XY4 sequences into a scheduled circuit where there are delays
    of sufficent duration.
    """

    def __init__(self, backend_properties, tau_step=10e-9):
        """XY4 pass initializer.

        Args:
            backend_properties (BackendProperties): Properties returned by a
                backend, including information on gate errors, readout errors,
                qubit coherence times, etc.
            tau_step (float): Delay time between pulses in the DD sequence in
                seconds. Default is 10 ns.
        """
        super().__init__()
        self.backend_properties = backend_properties
        self.tau_step = tau_step
        """To do"""
        self.tau_cs = {}

        basis = backend_properties.gate_names()  # TODO
        num_qubits = len(backend_properties.qubits)

        for qubit in range(num_qubits):
            xgate_dag = DAGCircuit()
            xgate_qreg = QuantumRegister(num_qubits, 'q')
            xgate_dag.add_qreg(xgate_qreg)

            xgate_qubit = Qubit(xgate_qreg, qubit)
            xgate_dag.apply_operation_back(XGate(), [xgate_qubit])

            gate_duration = 0
            self.xgate_unroll = Unroller(basis).run(xgate_dag)

            for node in self.xgate_unroll.topological_op_nodes():
                gate_duration += \
                    self.backend_properties.gate_length(node.op.name, qubit)

            ygate_dag = DAGCircuit()
            ygate_qreg = QuantumRegister(num_qubits, 'q')
            ygate_dag.add_qreg(ygate_qreg)
            ygate_qubit = Qubit(ygate_qreg, qubit)
            ygate_dag.apply_operation_back(YGate(), [ygate_qubit])

            self.ygate_unroll = Unroller(basis).run(ygate_dag)

            for node in self.ygate_unroll.topological_op_nodes():
                gate_duration += \
                    self.backend_properties.gate_length(node.op.name, qubit)

            self.tau_cs[qubit] = 4 * self.tau_step + 2 * gate_duration


    def run(self, dag):
        """Run the XY4 pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to new DAG.

        Returns:
            DAGCircuit: A new DAG with XY4 DD sequences inserted.
        """
        new_dag = DAGCircuit()

        new_dag.name = dag.name

        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        for node in dag.topological_op_nodes():

            if not isinstance(node.op, Delay):
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs, node.condition)
                continue

            delay_duration = node.op.duration
            tau_c = self.tau_cs[node.qargs[0].index]

            if tau_c > delay_duration or len(dag.ancestors(node)) <= 1:
                # If a cycle of XY4 can't fit or there isn't at least 1 other operation before.
                new_dag.apply_operation_back(Delay(delay_duration, unit=node.op.unit),
                                                   qargs=node.qargs)
                continue

            count = int(delay_duration // tau_c)
            new_delay = (delay_duration - count * tau_c + self.tau_step) / 2

            if new_delay != 0:
                new_dag.apply_operation_back(Delay(new_delay, unit='s'), qargs=node.qargs)

            first = True

            for _ in range(count):
                if not first:
                    new_dag.apply_operation_back(Delay(self.tau_step, unit='s'), qargs=node.qargs)
                for basis_node in self.xgate_unroll.topological_op_nodes():
                    new_dag.apply_operation_back(basis_node.op, qargs=node.qargs)
                new_dag.apply_operation_back(Delay(self.tau_step, unit='s'), qargs=node.qargs)
                for basis_node in self.ygate_unroll.topological_op_nodes():
                    new_dag.apply_operation_back(basis_node.op, qargs=node.qargs)
                new_dag.apply_operation_back(Delay(self.tau_step, unit='s'), qargs=node.qargs)
                for basis_node in self.xgate_unroll.topological_op_nodes():
                    new_dag.apply_operation_back(basis_node.op, qargs=node.qargs)
                new_dag.apply_operation_back(Delay(self.tau_step, unit='s'), qargs=node.qargs)
                for basis_node in self.ygate_unroll.topological_op_nodes():
                    new_dag.apply_operation_back(basis_node.op, qargs=node.qargs)
                first = False

            if new_delay != 0:
                new_dag.apply_operation_back(Delay(new_delay, unit='s'), qargs=node.qargs)

        return new_dag
