"""
Quantum computer instruction.

Author: Andrew Cross
"""
from ._register import Register
from ._qiskitexception import QISKitException


class Instruction(object):
    """Generic quantum computer instruction."""

    def __init__(self, name, param, arg):
        """Create a new instruction.

        name = instruction name string
        param = list of real parameters
        arg = list of pairs (Register, index)
        """
        for a in arg:
            if not isinstance(a[0], Register):
                raise QISKitException("argument not (Register, int) tuple")
        self.name = name
        self.param = param
        self.arg = arg
        self.control = None
        self.circuit = None

    def set_circuit(self, p):
        """Point back to the circuit that contains this instruction."""
        self.circuit = p

    def c_if(self, c, val):
        """Add classical control on register c and value val."""
        if self.circuit is None:
            raise QISKitException("gate is not associated to a circuit")
        self.circuit._check_creg(c)
        if val < 0:
            raise QISKitException("control value should be non-negative")
        self.control = (c, val)

    def _modifiers(self, g):
        """Apply any modifiers of this instruction to another one."""
        if self.control is not None:
            cc = g.circuit.map_register(self.control[0])
            if cc is None:
                raise QISKitException("could not map control register %s"
                                      % self.control[0].name)
            g.c_if(cc, self.control[1])

    def _qasmif(self, s):
        """Print an if statement if needed."""
        if self.control is None:
            return s
        else:
            return "if(%s==%d) " % (self.control[0].name, self.control[1]) + s

    def _remap_arg(self, circ):
        """Remap self.arg to corresponding qubits in circ."""
        rearg = []
        for r in self.arg:
            rp = circ.map_register(r[0])
            if rp is None:
                raise QISKitException("could not map register %s" % r[0].name)
            rearg.append((rp, r[1]))
        return rearg
