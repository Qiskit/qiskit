"""
Fundamental controlled-NOT gate.

Author: Andrew Cross
"""
from ._instruction import Instruction
from ._gate import Gate


class CXBase(Gate):
    """Fundamental controlled-NOT gate."""

    def __init__(self, ctl, tgt):
        """Create new CX instruction."""
        super(Instruction, self).__init__("CX", [], [ctl, tgt])

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
        """
        Reapply this gate to corresponding qubits in circ.

        Register index bounds checked by the gate method.
        """
        rearg = self._remap_arg(circ)
        self._modifiers(circ.cxbase(rearg[0], rearg[1]))
