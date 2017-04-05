"""
controlled-Phase gate.

Author: Andrew Cross
"""
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
from qiskit import Gate
from qiskit import CompositeGate
from qiskit import InstructionSet
from qiskit.extensions.standard import header


class CzGate(Gate):
    """controlled-Z gate."""

    def __init__(self, ctl, tgt):
        """Create new CZ gate."""
        super(CzGate, self).__init__("cz", [], [ctl, tgt])

    def qasm(self):
        """Return OPENQASM string."""
        ctl = self.arg[0]
        tgt = self.arg[1]
        return self._qasmif("cz %s[%d],%s[%d];" % (ctl[0].name, ctl[1],
                                                   tgt[0].name, tgt[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """
        Reapply this gate to corresponding qubits in circ.

        Register index bounds checked by the gate method.
        """
        rearg = self._remap_arg(circ)
        self._modifiers(circ.cz(rearg[0], rearg[1]))


def cz(self, i, j):
    """Apply CZ from i to j in this register."""
    self._check_bound()
    gs = InstructionSet()
    self.check_range(i)
    self.check_range(j)
    for p in self.bound_to:
        gs.add(p.cz((self, i), (self, j)))
    return gs


QuantumRegister.cz = cz


def cz(self, ctl, tgt):
    """Apply CZ to circuit."""
    self._check_qreg(ctl[0])
    ctl[0].check_range(ctl[1])
    self._check_qreg(tgt[0])
    tgt[0].check_range(tgt[1])
    self._check_dups([ctl, tgt])
    return self._attach(CzGate(ctl, tgt))


QuantumCircuit.cz = cz


def cz(self, ctl, tgt):
    """Apply CZ to composite."""
    self._check_qubit(ctl)
    ctl[0].check_range(ctl[1])
    self._check_qubit(tgt)
    tgt[0].check_range(tgt[1])
    self._check_dups([ctl, tgt])
    return self._attach(CzGate(ctl, tgt))


CompositeGate.cz = cz
