"""
Quantum computer instruction.

Author: Andrew Cross
"""
from ._register import Register
from ._qiskitexception import QISKitException


class Instruction(object):
    """Generic quantum computer instruction."""

    def __init__(self, name, param, arg, circ=None):
        """Create a new instruction.

        name = instruction name string
        param = list of real parameters
        arg = list of pairs (Register, index)
        circ = QuantumCircuit or CompositeGate containing this instruction
        """
        for a in arg:
            if not isinstance(a[0], Register):
                raise QISKitException("argument not (Register, int) tuple")
        self.name = name
        self.param = param
        self.arg = arg
        self.control = None  # tuple (ClassicalRegister, int) for "if"
        self.circuit = circ

    def check_circuit(self):
        """Raise exception if self.circuit is None."""
        if self.circuit is None:
            raise QISKitException("Instruction's circuit not assigned")

    def c_if(self, c, val):
        """Add classical control on register c and value val."""
        self.check_circuit()
        self.circuit._check_creg(c)
        if val < 0:
            raise QISKitException("control value should be non-negative")
        self.control = (c, val)

    def _modifiers(self, g):
        """Apply any modifiers of this instruction to another one."""
        if self.control is not None:
            self.check_circuit()
            if not g.circuit.has_register(self.control[0]):
                raise QISKitException("control register %s not found"
                                      % self.control[0].name)
            g.c_if(self.control[0], self.control[1])

    def _qasmif(self, s):
        """Print an if statement if needed."""
        if self.control is None:
            return s
        else:
            return "if(%s==%d) " % (self.control[0].name, self.control[1]) + s
