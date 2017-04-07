"""
Quantum circuit object.

Author: Andrew Cross
"""
import itertools
from ._qiskitexception import QISKitException
from ._register import Register
from ._quantumregister import QuantumRegister
from ._classicalregister import ClassicalRegister
from ._measure import Measure
from ._reset import Reset
from ._instructionset import InstructionSet


class QuantumCircuit(object):
    """Quantum circuit."""

    # Class variable OPENQASM header
    header = "OPENQASM 2.0;"

    def __init__(self, *regs):
        """Create a new circuit."""
        # Data contains a list of instructions in the order they were applied.
        self.data = []
        # This is a map of registers bound to this circuit, by name.
        self.regs = {}
        self.add(*regs)

    def has_register(self, r):
        """
        Test if this circuit has the register r.

        Return True or False.
        """
        if r.name in self.regs:
            s = self.regs[r.name]
            if s.sz == r.sz:
                if ((isinstance(r, QuantumRegister) and
                   isinstance(s, QuantumRegister)) or
                   (isinstance(r, ClassicalRegister) and
                   isinstance(s, ClassicalRegister))):
                    return True
        return False

    def combine(self, rhs):
        """
        Append rhs to self if self contains rhs's registers.

        Return self + rhs as a  new object.
        """
        for r in rhs.regs.values():
            if not self.has_register(r):
                raise QISKitException("circuits are not compatible")
        p = QuantumCircuit(*[r for r in self.regs.values()])
        for g in itertools.chain(self.data, rhs.data):
            g.reapply(p)
        return p

    def extend(self, rhs):
        """
        Append rhs to self if self contains rhs's registers.

        Modify and return self.
        """
        for r in rhs.regs.values():
            if not self.has_register(r):
                raise QISKitException("circuits are not compatible")
        for g in rhs.data:
            g.reapply(self)
        return self

    def __add__(self, rhs):
        """Overload + to implement self.concatenate."""
        return self.combine(rhs)

    def __iadd__(self, rhs):
        """Overload += to implement self.extend."""
        return self.extend(rhs)

    def _attach(self, g):
        """Attach a gate."""
        self.data.append(g)
        return g

    def add(self, *regs):
        """Add registers."""
        for r in regs:
            if not isinstance(r, Register):
                raise QISKitException("expected a register")
            if r.name not in self.regs:
                self.regs[r.name] = r
            else:
                raise QISKitException("register name \"%s\" already exists"
                                      % r.name)

    def _check_qreg(self, r):
        """Raise exception if r is not in this circuit or not qreg."""
        if not isinstance(r, QuantumRegister):
            raise QISKitException("expected quantum register")
        if not self.has_register(r):
            raise QISKitException("register '%s' not in this circuit" % r.name)

    def _check_qubit(self, q):
        """Raise exception if q is not in this circuit or invalid format."""
        self._check_qreg(q[0])
        q[0].check_range(q[1])

    def _check_creg(self, r):
        """Raise exception if r is not in this circuit or not creg."""
        if not isinstance(r, ClassicalRegister):
            raise QISKitException("expected classical register")
        if not self.has_register(r):
            raise QISKitException("register '%s' not in this circuit" % r.name)

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
        self._check_qubit(q)
        self._check_creg(c[0])
        c[0].check_range(c[1])
        return self._attach(Measure(q, c, self))

    def reset(self, q):
        """Reset q."""
        if isinstance(q, QuantumRegister):
            gs = InstructionSet()
            for j in range(q.sz):
                gs.add(self.reset((q, j)))
            return gs
        else:
            self._check_qubit(q)
            return self._attach(Reset(q, self))
