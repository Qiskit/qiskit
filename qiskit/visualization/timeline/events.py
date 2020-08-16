# -*- coding: utf-8 -*-

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
circuit bit. Bit-wise filtering of the circuit program makes
the arrangement of bit easier in the core drawer function.
The `BitEvents` class is expected to be called by other programs (not by end-users).

The `BitEvents` class instance is created with the class method ``load_program``:
    ```python
    event = BitEvents.load_program(sched_circuit, inst_durations, QuantumRegister(1)[0])
    ```

The `BitEvents` is created for a specific circuit bit either quantum or classical.
The gate types specified in `BitEvents._filter` are omitted even they are associated with the bit.
A parsed instruction is saved as ``ScheduledGate``, a collection of operand, associated time, and
bits. If the instruction is associated with multiple bits and the target bit of the instance is
the primary bit of the instruction, the `BitEvents` instance also generates a ``BitLink`` object
that shows a relationship between bits during the multi-bit gates.
"""
from typing import Union, List

from qiskit import circuit
from qiskit.converters import circuit_to_dag
from qiskit.visualization.timeline import types


class InstructionDurations:
    # Mock class to detach this PR from Itoko's PR #4555
    # TODO : replace this class
    @staticmethod
    def get(inst_name, qubits):
        if inst_name == 'h':
            return 160
        if inst_name == 'cx':
            return 512


class BitEvents:
    """Bit event table."""
    _filter = (circuit.Barrier, )

    def __init__(self,
                 bit: Union[circuit.Qubit, circuit.Clbit],
                 gates: List[types.ScheduledGate]):
        """Create new event for the specified bit.

        Args:
            bit: Bit object associated with this event table.
            gates: List of scheduled gate object.
        """
        self.bit = bit
        self.gates = gates

    @classmethod
    def load_program(cls,
                     scheduled_circuit: circuit.QuantumCircuit,
                     inst_durations: InstructionDurations,
                     bit: Union[circuit.Qubit, circuit.Clbit]):
        """Build new RegisterEvents from scheduled circuit.

        Args:
            scheduled_circuit: Scheduled circuit object to draw.
            inst_durations: Table of gate lengths.
            bit: Target bit object.

        Returns:
            New `RegisterEvents` object.
        """
        dag = circuit_to_dag(scheduled_circuit)
        nodes = list(dag.topological_op_nodes())

        t0 = 0
        gates = []
        for node in nodes:
            associated_bits = [qarg for qarg in node.qargs] + [carg for carg in node.cargs]
            if bit not in associated_bits or isinstance(node.op, cls._filter):
                continue

            try:
                duration = node.op.duration
            except AttributeError:
                duration = inst_durations.get(inst_name=node.op.name,
                                              qubits=node.qargs)

            gates.append(types.ScheduledGate(t0=t0,
                                             operand=node.op,
                                             duration=duration,
                                             bits=associated_bits))
            t0 += duration

        return BitEvents(bit, gates)

    def bit_links(self) -> List[types.GateLink]:
        """Return link between multi-bit gates."""
        links = []
        for gate in self.gates:
            # generate link iff this is the primary bit.
            if len(gate.bits) > 1 and gate.bits.index(self.bit) == 0:
                t0 = gate.t0 + 0.5 * gate.duration
                link = types.GateLink(t0=t0,
                                      operand=gate.operand,
                                      bits=gate.bits)
                links.append(link)
        return links
