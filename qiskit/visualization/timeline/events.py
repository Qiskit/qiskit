# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

r"""
Bit event manager for scheduled circuits.

This module provides a `BitEvents` class that manages a series of instructions for a
circuit bit. Bit-wise filtering of the circuit program makes the arrangement of bit
easier in the core drawer function. The `BitEvents` class is expected to be called
by other programs (not by end-users).

The `BitEvents` class instance is created with the class method ``load_program``:
    ```python
    event = BitEvents.load_program(sched_circuit, inst_durations, QuantumRegister(1)[0])
    ```

The `BitEvents` is created for a specific circuit bit either quantum or classical.
A parsed instruction is saved as ``ScheduledGate``, which is a collection of operand,
associated time, and bits. All parsed gate instructions are returned with `gates` method.
Instruction types specified in `BitEvents._non_gates` are not considered as gates.
If the instruction is associated with multiple bits and the target bit of the instance is
the primary bit of the instruction, the `BitEvents` instance also generates a ``BitLink`` object
that shows a relationship between bits during the multi-bit gates.
"""
from typing import List

from qiskit import circuit
from qiskit.converters import circuit_to_dag
from qiskit.visualization.timeline import types
from qiskit.visualization.exceptions import VisualizationError


class BitEvents:
    """Bit event table."""
    _non_gates = (circuit.Barrier, )

    def __init__(self,
                 bit: types.Bits,
                 instructions: List[types.ScheduledGate]):
        """Create new event for the specified bit.

        Args:
            bit: Bit object associated with this event table.
            instructions: List of scheduled gate object.
        """
        self.bit = bit
        self.instructions = instructions

    @classmethod
    def load_program(cls,
                     scheduled_circuit: circuit.QuantumCircuit,
                     bit: types.Bits):
        """Build new RegisterEvents from scheduled circuit.

        Args:
            scheduled_circuit: Scheduled circuit object to draw.
            bit: Target bit object.

        Returns:
            New `RegisterEvents` object.

        Raises:
            VisualizationError: When the circuit is not transpiled with duration.
        """
        dag = circuit_to_dag(scheduled_circuit)
        nodes = list(dag.topological_op_nodes())

        t0 = 0
        instructions = []
        for node in nodes:
            associated_bits = [qarg for qarg in node.qargs] + [carg for carg in node.cargs]
            if bit not in associated_bits:
                continue

            duration = node.op.duration
            if duration is None:
                raise VisualizationError('Instruction {oper} has no duration. '
                                         'You need to transpile the QuantumCircuit with '
                                         'gate durations before drawing.'.format(oper=node.op))

            instructions.append(types.ScheduledGate(t0=t0,
                                                    operand=node.op,
                                                    duration=duration,
                                                    bits=associated_bits))
            t0 += duration

        return BitEvents(bit, instructions)

    def is_empty(self) -> bool:
        """Return if there is any gate associated with this bit."""
        if any(not isinstance(inst, self._non_gates) for inst in self.instructions):
            return False
        else:
            return True

    def gates(self) -> List[types.ScheduledGate]:
        """Return scheduled gates."""
        gates = []
        for inst in self.instructions:
            if not isinstance(inst.operand, self._non_gates):
                gates.append(inst)
        return gates

    def barriers(self) -> List[types.Barrier]:
        """Return barriers."""
        barriers = []
        for inst in self.instructions:
            if isinstance(inst.operand, circuit.Barrier):
                barrier = types.Barrier(t0=inst.t0,
                                        bits=inst.bits)
                barriers.append(barrier)
        return barriers

    def bit_links(self) -> List[types.GateLink]:
        """Return link between multi-bit gates."""
        links = []
        for inst in self.instructions:
            # generate link iff this is the primary bit.
            if len(inst.bits) > 1 and inst.bits.index(self.bit) == 0:
                t0 = inst.t0 + 0.5 * inst.duration
                link = types.GateLink(t0=t0,
                                      operand=inst.operand,
                                      bits=inst.bits)
                links.append(link)
        return links
