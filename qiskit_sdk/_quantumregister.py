"""
Quantum register reference object.

Calling an instruction member function applies the gate to all
program instances the register is bound to.

Author: Andrew Cross
"""
from ._register import Register
from ._instructionset import InstructionSet


class QuantumRegister(Register):
    """Implement a quantum register."""

    def reset(self, j=-1):
        """Reset the jth qubit of this register (or all)."""
        self._check_bound()
        gs = InstructionSet()
        if j == -1:
            for p in self.bound_to:
                for k in range(self.sz):
                    gs.add(p.reset((self, k)))
        else:
            self.check_range(j)
            for p in self.bound_to:
                gs.add(p.reset((self, j)))
        return gs

    def u_base(self, tpl, j=-1):
        """Apply U to the jth qubit of this register (or all)."""
        self._check_bound()
        gs = InstructionSet()
        if j == -1:
            for p in self.bound_to:
                for k in range(self.sz):
                    gs.add(p.u_base(tpl, (self, k)))
        else:
            self.check_range(j)
            for p in self.bound_to:
                gs.add(p.u_base(tpl, (self, j)))
        return gs

    def cx_base(self, ctl, tgt):
        """Apply CX from the ctl to tgt qubit of this register."""
        self._check_bound()
        self.check_range(ctl)
        self.check_range(tgt)
        gs = InstructionSet()
        for p in self.bound_to:
            gs.add(p.cx_base((self, ctl), (self, tgt)))
        return gs

    def barrier(self, *idx):
        """Apply barrier to indices (or all)."""
        self._check_bound()
        bl = []
        if len(idx) == 0:
            for j in range(self.sz):
                bl.append((self, j))
        else:
            for j in idx:
                self.check_range(j)
                bl.append((self, j))
        gs = InstructionSet()
        for p in self.bound_to:
            gs.add(p.barrier(bl))
        return gs
