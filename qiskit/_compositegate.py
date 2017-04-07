"""
Composite gate, a container for a sequence of unitary gates.

Author: Andrew Cross
"""
from ._gate import Gate
from ._qiskitexception import QISKitException


class CompositeGate(Gate):
    """Composite gate, a sequence of unitary gates."""

    def __init__(self, name, param, arg, circ=None):
        """Create a new composite gate.

        name = instruction name string
        param = list of real parameters
        arg = list of pairs (Register, index)
        circ = QuantumCircuit or CompositeGate containing this gate
        """
        super(Gate, self).__init__(name, param, arg, circ)
        self.data = []  # gate sequence defining the composite unitary
        self.inverse_flag = False

    def has_register(self, r):
        """Test if this gate's circuit has the register r."""
        self.check_circuit()
        return self.circuit.has_register(r)

    def _modifiers(self, g):
        """Apply any modifiers of this gate to another composite g."""
        if self.inverse_flag:
            g.inverse()
        super(Gate, self)._modifiers(g)

    def _attach(self, g):
        """Attach a gate."""
        self.data.append(g)
        return g

    def _check_qubit(self, q):
        """Raise exception if q is not an argument or not qreg in circuit."""
        self.check_circuit()
        self.circuit._check_qubit(q)
        if (q[0].name, q[1]) not in map(lambda x: (x[0].name, x[1]), self.arg):
            raise QISKitException("qubit '%s[%d]' not argument of gate"
                                  % (q[0].name, q[1]))

    def _check_qreg(self, r):
        """Raise exception if r is not in this gate's circuit."""
        self.check_circuit()
        self.circuit._check_qreg(r)

    def _check_creg(self, r):
        """Raise exception if r is not in this gate's circuit."""
        self.check_circuit()
        self.circuit._check_creg(r)

    def _check_dups(self, qubits):
        """Raise exception if list of qubits contains duplicates."""
        squbits = set(qubits)
        if len(squbits) != len(qubits):
            raise QISKitException("duplicate qubit arguments")

    def qasm(self):
        """Return OPENQASM string."""
        return "\n".join([g.qasm() for g in self.data])

    def inverse(self):
        """Invert this gate."""
        self.seq = [g.inverse() for g in reversed(self.data)]
        self.inverse_flag = not self.inverse_flag
        return self

    def q_if(self, *qregs):
        """Add controls to this gate."""
        self.seq = [g.q_if(qregs) for g in self.data]
        return self

    def c_if(self, c, val):
        """Add classical control register."""
        self.seq = [g.c_if(c, val) for g in self.data]
        return self
