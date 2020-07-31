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

"""XY4 DD Pass"""
from qiskit.circuit.library.standard_gates import XGate, YGate
from qiskit.circuit.delay import Delay
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass


class XY4Pass(TransformationPass):
    """XY4 DD Pass"""

    def __init__(self, backend_properties, dt_in_sec, tau_c=None, tau_step=10):
        """XY4Pass initializer.

        Args:
            backend_properties (BackendProperties): Properties returned by a
                backend, including information on gate errors, readout errors,
                qubit coherence times, etc.
            dt_in_sec (float): Sample duration [sec] used for the conversion.
            tau_c (int): Cycle time of the DD sequence. Default is the sum of gate
                durations of DD sequences with 10 ns delays in between.
            tau_step (int): Delay time between pulses in the DD sequence. Default
                is 10 ns. tau_step calculates tau_c if tau_c is not specified.
        """
        super().__init__()
        self.backend_properties = backend_properties
        self.dt = dt_in_sec

        if tau_c is not None and tau_step != 10:
            raise TranspilerError("Only either tau_c or tau_step can be specified, not both.")

        self.tau_step_dt = int(tau_step * 1e-9 / self.dt)
        self.tau_c = tau_c
        self.approx_tau_cs = {}

        u3_props = self.backend_properties._gates['u3']
        for qubit, props in u3_props.items():
            if 'gate_length' in props:
                gate_length = props['gate_length'][0]
                # TODO: Needs to check if durations of gates exceed cycle time
                # If so, raise error
                if tau_c is not None:
                    self.approx_tau_cs[qubit[0]] = 4 * tau_c // 4
                else:
                    self.approx_tau_cs[qubit[0]] = 4 * self.tau_step_dt + \
                                            round(4 * gate_length / self.dt)

                # Should also raise warning about cycle time estimations

    def run(self, dag):
        """Run the XY4 pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to new DAG.

        Returns:
            DAGCircuit: A new DAG with XY4 DD Sequences inserted in large 
                        enough delays.
        """
        new_dag = DAGCircuit()

        new_dag.name = dag.name
        new_dag.instruction_durations = dag.instruction_durations

        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)
            
        for node in dag.topological_op_nodes():

            if not isinstance(node.op, Delay):

                new_dag.apply_operation_back(node.op, node.qargs, node.cargs, node.condition)

            else:
                delay_duration = node.op.duration
                approx_tau_c = self.approx_tau_cs[node.qargs[0].index] if self.tau_c is None \
                                                                       else self.tau_c

                if approx_tau_c > delay_duration or len(dag.ancestors(node)) <= 1:
                    # If a cycle of XY4 can't fit or there isn't at least 1 other operation before.
                    new_dag.apply_operation_back(Delay(delay_duration), qargs=node.qargs)

                else:
                    count = int(delay_duration // approx_tau_c)
                    parity = 1 if (delay_duration - count * approx_tau_c + self.tau_step_dt) % 2 \
                               else 0
                    new_delay = int((delay_duration - count * approx_tau_c + self.tau_step_dt) / 2)

                    new_dag.apply_operation_back(Delay(new_delay - self.tau_step_dt), \
                                                        qargs=node.qargs)

                    for _ in range(count):
                        new_dag.apply_operation_back(Delay(self.tau_step_dt), qargs=node.qargs)
                        new_dag.apply_operation_back(XGate(), qargs=node.qargs)
                        new_dag.apply_operation_back(Delay(self.tau_step_dt), qargs=node.qargs)
                        new_dag.apply_operation_back(YGate(), qargs=node.qargs)
                        new_dag.apply_operation_back(Delay(self.tau_step_dt), qargs=node.qargs)
                        new_dag.apply_operation_back(XGate(), qargs=node.qargs)
                        new_dag.apply_operation_back(Delay(self.tau_step_dt), qargs=node.qargs)
                        new_dag.apply_operation_back(YGate(), qargs=node.qargs)

                    new_dag.apply_operation_back(Delay(new_delay + parity), qargs=node.qargs)

        return new_dag
