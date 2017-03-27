"""
Composite gate, a container for a sequence of unitary gates.

Author: Andrew Cross
"""
from ._gate import Gate
from ._quantumregister import QuantumRegister
from ._qiskitexception import QISKitException
from ._ubase import UBase
from ._cxbase import CXBase
from ._barrier import Barrier


class CompositeGate(Gate):
    """Composite gate, a sequence of unitary gates."""

    def __init__(self, name, param, arg):
        """Create a new composite gate.

        name = instruction name string
        param = list of real parameters
        arg = list of pairs (Register, index)
        """
        for a in arg:
            if not isinstance(a[0], QuantumRegister):
                raise QISKitException("argument not (QuantumRegister, int) "
                                      + "tuple")
        self.name = name
        self.param = param
        self.arg = arg
        self.data = []  # gate sequence defining the composite unitary
        self.inverse_flag = False

    def _attach(self, g):
        """Attach a gate."""
        self.data.append(g)
        return g

    def _check_qubit(self, r):
        """Raise exception if r is not an argument or not qreg."""
        if type(r[0]) is not QuantumRegister:
            raise QISKitException("expected quantum register")
        if r not in self.arg:
            raise QISKitException("qubit '%s[%d]' not argument of gate"
                                  % (r[0].name, r[1]))

    def qasm(self):
        """Return OPENQASM string."""
        return "\n".join([g.qasm() for g in self.data])

    def inverse(self):
        """Invert this gate."""
        self.seq = [g.inverse() for g in reversed(self.data)]
        self.inverse_flag = not self.inverse_flag
        return self

    def control(self, *qregs):
        """Add controls to this gate."""
        self.seq = [g.control(qregs) for g in self.data]
        return self

    def doif(self, c, val):
        """Add classical control register."""
        self.seq = [g.doif(c, val) for g in self.data]
        return self

    def u_base(self, tpl, q):
        """Apply U to q in this composite gate."""
        self._check_qubit(q)
        q[0].check_range(q[1])
        return self._attach(UBase(tpl, q))

    def cx_base(self, ctl, tgt):
        """Apply CX ctl, tgt."""
        self._check_qubit(ctl)
        self._check_qubit(tgt)
        ctl[0].check_range(ctl[1])
        tgt[0].check_range(tgt[1])
        # self._check_dups([ctl, tgt])
        return self._attach(CXBase(ctl, tgt))

    def barrier(self, *tup):
        """Apply barrier to tuples (reg, idx)."""
        for t in tup:
            self._check_qubit(t)
            t[0].check_range(t[1])
        # self._check_dups(tup)
        return self._attach(Barrier(*tup))
