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
from qiskit import circuit
from qiskit.circuit.delay import Delay
from qiskit.converters import circuit_to_dag

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
        self._cycle_times = {}
        self._unrolled_sequence = {}

    def cycle_times(self, qubit):
        """The time [s] it takes for one XY4 sequence on the given qubit."""
        if qubit not in self._cycle_times:
            self._initialize_cycle_time(qubit)
        return self._cycle_times[qubit]

    def unrolled_sequence(self, qubit, gate):
        """The X or Y gate unrolled into the basis.

        Args:
            qubit (int): The qubit unrolling for.
            gate (str): Must be either 'X' or 'Y'.
        """
        if qubit not in self._unrolled_sequence:
            self._initialize_cycle_time(qubit)
        if gate == 'X':
            return self._unrolled_sequence[qubit][0]
        elif gate == 'Y':
            return self._unrolled_sequence[qubit][1]

    def _initialize_cycle_time(self, qubit):
        """Update the time, in seconds, of one XY4 sequence::

            X, tau_step, Y, tau_step, X, tau_step, Y, tau_step

        on qubit labeled ``qubit`` and save the X, Y gates unrolled to the target basis.
        """
        basis = self.backend_properties.gate_names()

        # Duration of an X gate in the target basis
        x_gate = circuit.QuantumCircuit(qubit + 1)
        x_gate.x(qubit)
        xgate_unrolled = Unroller(basis).run(circuit_to_dag(x_gate))

        x_duration = 0
        for node in xgate_unrolled.topological_op_nodes():
            x_duration += \
                self.backend_properties.gate_length(node.op.name, qubit)

        # Duration of a Y gate in the target basis
        y_gate = circuit.QuantumCircuit(qubit + 1)
        y_gate.y(qubit)
        ygate_unrolled = Unroller(basis).run(circuit_to_dag(y_gate))

        y_duration = 0
        for node in ygate_unrolled.topological_op_nodes():
            y_duration += \
                self.backend_properties.gate_length(node.op.name, qubit)

        # Save the unrolled sequences
        self._unrolled_sequence[qubit] = (xgate_unrolled, ygate_unrolled)
        # Save the cycle time
        self._cycle_times[qubit] = 2 * (x_duration + y_duration) + 4 * self.tau_step

    def run(self, dag):
        """Run the XY4 pass on `dag`.

        This replaces Delay instructions (times where qubits are idle) with the XY4 sequence,
        as long as the delay duration is long enough for one sequence.

        Args:
            dag (DAGCircuit): The DAG to be modified.

        Returns:
            DAGCircuit: A new DAG with XY4 DD sequences inserted.
        """
        new_dag = dag._copy_circuit_metadata()

        for node in dag.topological_op_nodes():

            if not isinstance(node.op, Delay):
                # Non-delay operations remain unchanged.
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs, node.condition)
                continue

            delay_duration = node.op.duration
            qubit = node.qargs[0].index
            seq_duration = self.cycle_times(qubit)
            if seq_duration > delay_duration or len(dag.ancestors(node)) <= 1:
                # If the XY4 sequence is longer than the delay, or this is the first operation,
                # then simply add the delay back and continue.
                new_dag.apply_operation_back(Delay(delay_duration, unit=node.op.unit),
                                             qargs=node.qargs)
                continue

            num_sequences = int(delay_duration // seq_duration)
            # Remaining delay, split before and after the sequence. Ideally, this would just be
            # tau_step / 2
            new_delay = (delay_duration - seq_duration*num_sequences + self.tau_step) / 2

            if new_delay > 0:
                new_dag.apply_operation_back(Delay(new_delay, unit='s'), qargs=node.qargs)

            for i in range(2*num_sequences):
                # X
                for basis_node in self.unrolled_sequence(qubit, 'X').topological_op_nodes():
                    new_dag.apply_operation_back(basis_node.op, qargs=node.qargs)
                # tau step
                new_dag.apply_operation_back(Delay(self.tau_step, unit='s'), qargs=node.qargs)
                # Y
                for basis_node in self.unrolled_sequence(qubit, 'Y').topological_op_nodes():
                    new_dag.apply_operation_back(basis_node.op, qargs=node.qargs)
                # tau step
                if i != 2*num_sequences - 1:
                    new_dag.apply_operation_back(Delay(self.tau_step, unit='s'), qargs=node.qargs)

            if new_delay > 0:
                new_dag.apply_operation_back(Delay(new_delay, unit='s'), qargs=node.qargs)

        return new_dag
