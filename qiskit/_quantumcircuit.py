"""
Quantum circuit object.

Author: Andrew Cross
"""
from ._qiskitexception import QISKitException
from ._quantumregister import QuantumRegister
from ._classicalregister import ClassicalRegister
from ._barrier import Barrier
from ._measure import Measure
from ._reset import Reset
from ._ubase import UBase
from ._cxbase import CXBase


class QuantumCircuit(object):
    """Quantum circuit."""

    # Class variable OPENQASM header
    header = "OPENQASM 2.0;"

    def __init__(self, *regs):
        """Create a new circuit."""
        # Data contains a list of instructions in the order they were applied.
        self.data = []
        # This is a list of register references bound to this circuit.
        self.regs = list(regs)
        for r in regs:
            r.bind_to(self)

    def _attach(self, g):
        """Attach a gate."""
        g.set_circuit(self)
        self.data.append(g)
        return g

    def add(self, *regs):
        """Add registers."""
        for r in regs:
            if type(r) is not QuantumRegister and \
               type(r) is not ClassicalRegister:
                raise QISKitException("expected a register")
            if r not in self.regs:
                self.regs.append(r)
                r.bind_to(self)

    def _check_qreg(self, r):
        """Raise exception if r is not bound to this circuit or not qreg."""
        if type(r) is not QuantumRegister:
            raise QISKitException("expected quantum register")
        if r not in self.regs:
            raise QISKitException("register '%s' not bound to this circuit"
                                  % r.name)

    def _check_creg(self, r):
        """Raise exception if r is not bound to this circuit or not creg."""
        if type(r) is not ClassicalRegister:
            raise QISKitException("expected classical register")
        if r not in self.regs:
            raise QISKitException("register '%s' not bound to this circuit"
                                  % r.name)

    def qasm(self):
        """Return OPENQASM string."""
        s = self.header + "\n"
        for r in self.regs:
            s += r.qasm() + "\n"
        for i in self.data:
            s += i.qasm() + "\n"
        return s

    def measure(self, q, c):
        """Measure q into c (tuples)."""
        self._check_qreg(q[0])
        self._check_creg(c[0])
        q[0].check_range(q[1])
        c[0].check_range(c[1])
        return self._attach(Measure(q, c))

    def reset(self, q):
        """Reset q."""
        self._check_qreg(q[0])
        q[0].check_range(q[1])
        return self._attach(Reset(q))

    def u_base(self, tpl, q):
        """Apply U to q."""
        self._check_qreg(q[0])
        q[0].check_range(q[1])
        return self._attach(UBase(tpl, q))

    def cx_base(self, ctl, tgt):
        """Apply CX ctl, tgt."""
        self._check_qreg(ctl[0])
        self._check_qreg(tgt[0])
        ctl[0].check_range(ctl[1])
        tgt[0].check_range(tgt[1])
        # self._check_dups([ctl, tgt])
        return self._attach(CXBase(ctl, tgt))

    def barrier(self, *tup):
        """Apply barrier to tuples (reg, idx)."""
        for t in tup:
            self._check_qreg(t[0])
            t[0].check_range(t[1])
        # self._check_dups(tup)
        return self._attach(Barrier(*tup))
