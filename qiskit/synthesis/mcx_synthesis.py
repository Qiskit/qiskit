# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""MCX synthesis."""

from abc import ABC, abstractmethod
from math import ceil
import numpy

from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.library import C3XGate, C4XGate, U2Gate, RCCXGate, U1Gate, CXGate, CCXGate, HGate, MCU1Gate


class MCXSynthesis(ABC):
    """Interface for MCX synthesis algorithms."""

    @abstractmethod
    def synthesize(self, num_cntl_qubits):
        """Synthesize..

        Args:

        Returns:
            QuantumCircuit: A circuit implementing the evolution.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_num_ancilla_qubits(num_ctrl_qubits):
        """Returns the number of ancilla qubits required by the synthesis algorithm."""

        raise NotImplementedError


class MCXSynthesisGrayCode(MCXSynthesis):
    r"""Implement the multi-controlled X gate using the Gray code.

    This delegates the implementation to the MCU1 gate, since :math:`X = H \cdot U1(\pi) \cdot H`.
    """

    def __init__(self, name):
        print(f"MCXSynthesisGrayCode:__init__")
        self.name = name

    @staticmethod
    def get_num_ancilla_qubits(num_ctrl_qubits):
        print(f"MCXSynthesisGrayCode:get_num_ancilla_qubits")
        return 0

    def synthesize(self, num_ctrl_qubits):
        print(f"MCXSynthesisGrayCode:synthesize")

        """Define the MCX gate using the Gray code."""
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        num_qubits = num_ctrl_qubits + self.get_num_ancilla_qubits(num_ctrl_qubits) + 1
        q = QuantumRegister(num_qubits, name="q")
        qc = QuantumCircuit(q, name=self.name)
        qc._append(HGate(), [q[-1]], [])
        qc._append(MCU1Gate(numpy.pi, num_ctrl_qubits=num_ctrl_qubits), q[:], [])
        qc._append(HGate(), [q[-1]], [])

        return qc


class MCXSynthesisRecursive(MCXSynthesis):
    """Implement the multi-controlled X gate using recursion.

    Using a single ancilla qubit, the multi-controlled X gate is recursively split onto
    four sub-registers. This is done until we reach the 3- or 4-controlled X gate since
    for these we have a concrete implementation that do not require ancillas.
    """

    def __init__(self, name):
        print(f"MCXSynthesisRecursive:__init__")
        self.name = name

    @staticmethod
    def get_num_ancilla_qubits(num_ctrl_qubits):
        print(f"MCXSynthesisRecursive:get_num_ancilla_qubits")
        return int(num_ctrl_qubits > 4)

    def synthesize(self, num_ctrl_qubits):
        """Define the MCX gate using recursion."""
        print(f"MCXSynthesisRecursive:synthesize")

        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        num_qubits = num_ctrl_qubits + self.get_num_ancilla_qubits(num_ctrl_qubits) + 1

        q = QuantumRegister(num_qubits, name="q")
        qc = QuantumCircuit(q, name=self.name)
        if num_qubits == 4:
            qc._append(C3XGate(), q[:], [])
        elif num_qubits == 5:
            qc._append(C4XGate(), q[:], [])
        else:
            for instr, qargs, cargs in self._recurse(q[:-1], q_ancilla=q[-1]):
                qc._append(instr, qargs, cargs)
        return qc

    def _recurse(self, q, q_ancilla=None):
        # recursion stop
        if len(q) == 4:
            return [(C3XGate(), q[:], [])]
        if len(q) == 5:
            return [(C4XGate(), q[:], [])]
        if len(q) < 4:
            raise AttributeError("Something went wrong in the recursion, have less than 4 qubits.")

        # recurse
        num_ctrl_qubits = len(q) - 1
        middle = ceil(num_ctrl_qubits / 2)
        first_half = [*q[:middle], q_ancilla]
        second_half = [*q[middle:num_ctrl_qubits], q_ancilla, q[num_ctrl_qubits]]

        rule = []
        rule += self._recurse(first_half, q_ancilla=q[middle])
        rule += self._recurse(second_half, q_ancilla=q[middle - 1])
        rule += self._recurse(first_half, q_ancilla=q[middle])
        rule += self._recurse(second_half, q_ancilla=q[middle - 1])

        return rule

class MCXSynthesisVChain(MCXSynthesis):
    """Implement the multi-controlled X gate using a V-chain of CX gates."""

    def __init__(self, name, dirty_ancillas):
        print(f"MCXSynthesisVChain:__init__")
        self.name = name
        self.dirty_ancillas = dirty_ancillas

    @staticmethod
    def get_num_ancilla_qubits(num_ctrl_qubits):
        print(f"MCXSynthesisVChain:get_num_ancilla_qubits")
        return max(0, num_ctrl_qubits - 2)

    def synthesize(self, num_ctrl_qubits):
        print(f"MCXSynthesisVChain:synthesize")

        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        num_qubits = num_ctrl_qubits + self.get_num_ancilla_qubits(num_ctrl_qubits) + 1

        q = QuantumRegister(num_qubits, name="q")
        qc = QuantumCircuit(q, name=self.name)
        q_controls = q[: num_ctrl_qubits]
        q_target = q[num_ctrl_qubits]
        q_ancillas = q[num_ctrl_qubits + 1 :]

        definition = []

        if self.dirty_ancillas:
            i = num_ctrl_qubits - 3
            ancilla_pre_rule = [
                (U2Gate(0, numpy.pi), [q_target], []),
                (CXGate(), [q_target, q_ancillas[i]], []),
                (U1Gate(-numpy.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_controls[-1], q_ancillas[i]], []),
                (U1Gate(numpy.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_target, q_ancillas[i]], []),
                (U1Gate(-numpy.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_controls[-1], q_ancillas[i]], []),
                (U1Gate(numpy.pi / 4), [q_ancillas[i]], []),
            ]
            for inst in ancilla_pre_rule:
                definition.append(inst)

            for j in reversed(range(2, num_ctrl_qubits - 1)):
                definition.append(
                    (RCCXGate(), [q_controls[j], q_ancillas[i - 1], q_ancillas[i]], [])
                )
                i -= 1

        definition.append((RCCXGate(), [q_controls[0], q_controls[1], q_ancillas[0]], []))
        i = 0
        for j in range(2, num_ctrl_qubits - 1):
            definition.append((RCCXGate(), [q_controls[j], q_ancillas[i], q_ancillas[i + 1]], []))
            i += 1

        if self.dirty_ancillas:
            ancilla_post_rule = [
                (U1Gate(-numpy.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_controls[-1], q_ancillas[i]], []),
                (U1Gate(numpy.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_target, q_ancillas[i]], []),
                (U1Gate(-numpy.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_controls[-1], q_ancillas[i]], []),
                (U1Gate(numpy.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_target, q_ancillas[i]], []),
                (U2Gate(0, numpy.pi), [q_target], []),
            ]
            for inst in ancilla_post_rule:
                definition.append(inst)
        else:
            definition.append((CCXGate(), [q_controls[-1], q_ancillas[i], q_target], []))

        for j in reversed(range(2, num_ctrl_qubits - 1)):
            definition.append((RCCXGate(), [q_controls[j], q_ancillas[i - 1], q_ancillas[i]], []))
            i -= 1
        definition.append((RCCXGate(), [q_controls[0], q_controls[1], q_ancillas[i]], []))

        if self.dirty_ancillas:
            for i, j in enumerate(list(range(2, num_ctrl_qubits - 1))):
                definition.append(
                    (RCCXGate(), [q_controls[j], q_ancillas[i], q_ancillas[i + 1]], [])
                )

        for instr, qargs, cargs in definition:
            qc._append(instr, qargs, cargs)

        return qc


def mcx_mode_to_synthesis(mcx_mode):
    mcx_mode_to_synthesis_map = {
        "noancilla":     MCXSynthesisGrayCode("mcx_gray"),
        "recursion":     MCXSynthesisRecursive("mcx_recursive"),
        "v-chain":       MCXSynthesisVChain("mcx_vchain", dirty_ancillas=False),
        "v-chain-dirty": MCXSynthesisVChain("mcx_vchain", dirty_ancillas=True),

        # outdated, previous names
        "advanced":     MCXSynthesisRecursive("mcx_recursive"),
        "basic":        MCXSynthesisVChain("mcx_vchain", dirty_ancillas=False),
        "basic-dirty-ancilla": MCXSynthesisVChain("mcx_vchain", dirty_ancillas=True),
    }

    return mcx_mode_to_synthesis_map[mcx_mode]


