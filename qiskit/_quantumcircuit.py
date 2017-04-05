"""
Quantum circuit object.

Author: Andrew Cross
"""
import itertools
from ._qiskitexception import QISKitException
from ._register import Register
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
        # This is a map of register references bound to this circuit, by name.
        self.regs = {}
        self.add(*regs)

    def map_register(self, r):
        """
        Lookup a register in this circuit with the properties of r.

        Return the register if this circuit has one with the same name,
        size, and type as r's name, size, and type. Otherwise return None.
        """
        if r.name in self.regs:
            s = self.regs[r.name]
            if s.sz == r.sz:
                if ((isinstance(r, QuantumRegister) and
                   isinstance(s, QuantumRegister)) or
                   (isinstance(r, ClassicalRegister) and
                   isinstance(s, ClassicalRegister))):
                    return s
        return None

    def __add__(self, rhs):
        """
        Append rhs to self if self contains rhs's registers.

        Return self + rhs as a new object.
        """
        for r in rhs.regs.values():
            if self.map_register(r) is None:
                raise QISKitException("circuits are not compatible")
        p = QuantumCircuit(*[r.unbound_copy() for r in self.regs.values()])
        for g in itertools.chain(self.data, rhs.data):
            g.reapply(p)
        return p

    def __iadd__(self, rhs):
        """
        Append rhs to self if self contains rhs's registers.

        Modify and return self.
        """
        for r in rhs.regs.values():
            if self.map_register(r) is None:
                raise QISKitException("circuits are not compatible")
        for g in rhs.data:
            g.reapply(self)
        return self

    def _attach(self, g):
        """Attach a gate."""
        g.set_circuit(self)
        self.data.append(g)
        return g

    def add(self, *regs):
        """Add registers."""
        for r in regs:
            if not isinstance(r, Register):
                raise QISKitException("expected a register")
            if r.name not in self.regs:
                self.regs[r.name] = r
                r.bind_to(self)
            else:
                raise QISKitException("register name \"%s\" already exists"
                                      % r.name)

    def _check_qreg(self, r):
        """Raise exception if r is not bound to this circuit or not qreg."""
        if type(r) is not QuantumRegister:
            raise QISKitException("expected quantum register")
        if r not in self.regs.values():
            raise QISKitException("register '%s' not bound to this circuit"
                                  % r.name)

    def _check_creg(self, r):
        """Raise exception if r is not bound to this circuit or not creg."""
        if type(r) is not ClassicalRegister:
            raise QISKitException("expected classical register")
        if r not in self.regs.values():
            raise QISKitException("register '%s' not bound to this circuit"
                                  % r.name)

    def _check_dups(self, qubits):
        """Raise exception if list of qubits contains duplicates."""
        squbits = set(qubits)
        if len(squbits) != len(qubits):
            raise QISKitException("duplicate qubit arguments")

    def qasm(self):
        """Return OPENQASM string."""
        s = self.header + "\n"
        for r in self.regs.values():
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
        self._check_dups([ctl, tgt])
        return self._attach(CXBase(ctl, tgt))

    def barrier(self, *tup):
        """Apply barrier to tuples (reg, idx)."""
        for t in tup:
            self._check_qreg(t[0])
            t[0].check_range(t[1])
        self._check_dups(tup)
        return self._attach(Barrier(*tup))
