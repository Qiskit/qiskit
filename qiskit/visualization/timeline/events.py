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

"""
Bit event manager for scheduled circuits.

This module provides a :py:class:`BitEvents` class that manages a series of instructions for a
specific circuit bit. Bit-wise filtering of the circuit program makes the arrangement of bits
easier in the core drawer function. The `BitEvents` class is expected to be called
by other programs (not by end-users).

The :py:class:`BitEvents` class instance is created with the class method ``load_program``:
    ```python
    event = BitEvents.load_program(sched_circuit, qregs[0])
    ```

Loaded circuit instructions are saved as ``ScheduledGate``, which is a collection of instruction,
associated time, and bits. All gate instructions are returned by the `.get_gates` method.
Instruction types specified in `BitEvents._non_gates` are not considered as gates.
If an instruction is associated with multiple bits and the target bit of the class instance is
the primary bit of the instruction, the instance also generates a ``GateLink`` object
that shows the relationship between bits during multi-bit gates.
"""
from typing import List, Iterator

from qiskit import circuit
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.timeline import types


class BitEvents:
    """Bit event table."""

    _non_gates = (circuit.Barrier,)

    def __init__(self, bit: types.Bits, instructions: List[types.ScheduledGate], t_stop: int):
        """Create new event for the specified bit.

        Args:
            bit: Bit object associated with this event table.
            instructions: List of scheduled gate object.
            t_stop: Stop time of this bit.
        """
        self.bit = bit
        self.instructions = instructions
        self.stop_time = t_stop

    @classmethod
    def load_program(cls, scheduled_circuit: circuit.QuantumCircuit, bit: types.Bits):
        """Build new BitEvents from scheduled circuit.

        Args:
            scheduled_circuit: Scheduled circuit object to draw.
            bit: Target bit object.

        Returns:
            BitEvents: New `BitEvents` object.

        Raises:
            VisualizationError: When the circuit is not transpiled with duration.
        """
        t0 = 0
        tf = scheduled_circuit.qubit_stop_time(bit)

        instructions = []
        for inst, qargs, cargs in scheduled_circuit.data:
            associated_bits = qargs + cargs
            if bit not in associated_bits:
                continue

            duration = inst.duration
            if duration is None:
                raise VisualizationError(
                    "Instruction {oper} has no duration. "
                    "You need to transpile the QuantumCircuit with "
                    "gate durations before drawing.".format(oper=inst)
                )

            instructions.append(
                types.ScheduledGate(
                    t0=t0,
                    operand=inst,
                    duration=duration,
                    bits=associated_bits,
                    bit_position=associated_bits.index(bit),
                )
            )
            t0 += duration

        return BitEvents(bit, instructions, tf)

    def get_gates(self) -> Iterator[types.ScheduledGate]:
        """Return scheduled gates."""
        for inst in self.instructions:
            if not isinstance(inst.operand, self._non_gates):
                yield inst

    def get_barriers(self) -> Iterator[types.Barrier]:
        """Return barriers."""
        for inst in self.instructions:
            if isinstance(inst.operand, circuit.Barrier):
                barrier = types.Barrier(t0=inst.t0, bits=inst.bits, bit_position=inst.bit_position)
                yield barrier

    def get_gate_links(self) -> Iterator[types.GateLink]:
        """Return link between multi-bit gates."""
        for inst in self.get_gates():
            # generate link iff this is the primary bit.
            if len(inst.bits) > 1 and inst.bit_position == 0:
                t0 = inst.t0 + 0.5 * inst.duration
                link = types.GateLink(t0=t0, opname=inst.operand.name, bits=inst.bits)
                yield link
