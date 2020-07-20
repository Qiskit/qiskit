# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from qiskit.circuit.library.standard_gates import XGate, YGate
from qiskit.circuit.delay import Delay
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass

class XY4Pass(TransformationPass):

    def run(self, dag):
        """Run the XY4 pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to new DAG.

        Returns:
            DAGCircuit: A new DAG with XY4 DD Sequences inserted in large 
                        enough delays.
        """
        count = 0
        new_delay = 0
        parity = 0
        xy4_duration = 680                            # TODO find backend gate durations to calculate xy4_duration
        first = True
        dd_delay = 10
        new_dag = DAGCircuit()

        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        for node in dag.topological_op_nodes():
            if isinstance(node.op, Delay):
                delay_duration = dag.instruction_durations.get(node.op, node.qargs)

                if xy4_duration <= delay_duration:
                    count = int(delay_duration//xy4_duration)
                    parity = 1 if (delay_duration-count*xy4_duration+dd_delay)%2 else 0
                    new_delay = int((delay_duration-count*xy4_duration+dd_delay)/2)

                    new_dag.apply_operation_back(Delay(new_delay),qargs=node.qargs)

                    for _ in range(count):
                        if not first:
                            new_dag.apply_operation_back(Delay(dd_delay, unit='ns'),qargs=node.qargs)
                        new_dag.apply_operation_back(XGate(),qargs=node.qargs)
                        new_dag.apply_operation_back(Delay(dd_delay, unit='ns'),qargs=node.qargs)
                        new_dag.apply_operation_back(YGate(),qargs=node.qargs)
                        new_dag.apply_operation_back(Delay(dd_delay, unit='ns'),qargs=node.qargs)
                        new_dag.apply_operation_back(XGate(),qargs=node.qargs)
                        new_dag.apply_operation_back(Delay(dd_delay, unit='ns'),qargs=node.qargs)
                        new_dag.apply_operation_back(YGate(),qargs=node.qargs)
                        first = False

                    if parity:
                        new_dag.apply_operation_back(Delay(new_delay+1),qargs=node.qargs)
                    else:
                        new_dag.apply_operation_back(Delay(new_delay),qargs=node.qargs)
                    first = True

                else:
                    new_dag.apply_operation_back(Delay(delay_duration, unit='ns'),qargs=node.qargs)

            else:
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs, node.condition)

        new_dag.name = dag.name
        new_dag.instruction_durations = dag.instruction_durations

        return new_dag