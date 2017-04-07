"""
Fundamental controlled-NOT gate.

Author: Andrew Cross
"""
from qiskit import Instruction
from qiskit import Gate
from qiskit import QuantumCircuit
from qiskit import CompositeGate


class CXBase(Gate):
    """Fundamental controlled-NOT gate."""

    def __init__(self, ctl, tgt, circ=None):
        """Create new CX instruction."""
        super(Instruction, self).__init__("CX", [], [ctl, tgt], circ)

    def qasm(self):
        """Return OPENQASM string."""
        ctl = self.arg[0]
        tgt = self.arg[1]
        return self._qasmif("CX %s[%d],%s[%d];" % (ctl[0].name, ctl[1],
                                                   tgt[0].name, tgt[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.cxbase(self.arg[0], self.arg[1]))


def cx_base(self, ctl, tgt):
    """Apply CX ctl, tgt."""
    self._check_qubit(ctl)
    self._check_qubit(tgt)
    self._check_dups([ctl, tgt])
    return self._attach(CXBase(ctl, tgt, self))


QuantumCircuit.cx_base = cx_base
CompositeGate.cx_base = cx_base
