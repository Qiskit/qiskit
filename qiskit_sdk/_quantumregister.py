"""
Quantum register object.

Author: Andrew Cross
"""
from ._register import Register
from ._gateset import GateSet
from ._reset import Reset
from ._ubase import UBase
from ._cxbase import CXBase
from ._barrier import Barrier


class QuantumRegister(Register):
    """Implement a quantum register."""

    def _attach(self, g):
        """Attach a gate."""
        self.data.append(g)
        for p in self.bound_to:
            p.data.append(g)
        return g

    def reset(self, j=-1):
        """Reset the jth qubit of this register (or all)."""
        if j == -1:
            gs = GateSet()
            for k in range(self.sz):
                gs.add(self.reset(k))
            return gs
        else:
            self._check_range(j)
            return self._attach(Reset((self, j)))

    def u_base(self, tpl, j=-1):
        """Apply U to the jth qubit of this register (or all)."""
        if j == -1:
            gs = GateSet()
            for k in range(self.sz):
                gs.add(self.u_base(tpl, k))
            return gs
        else:
            self._check_range(j)
            return self._attach(UBase(tpl, (self, j)))

    def cx_base(self, ctl, tgt):
        """Apply CX from the ctl to tgt qubit of this register."""
        self._check_range(ctl)
        self._check_range(tgt)
        return self._attach(CXBase((self, ctl), (self, tgt)))

    def barrier(self, *idx):
        """Apply barrier to indices (or all)."""
        if len(idx) == 0:
            pass
        else:
            bl = []
            for j in idx:
                self._check_range(j)
                bl.append((self, j))
            return self._attach(Barrier(bl))
