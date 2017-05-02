"""
Quantum computer instruction.

Author: Andrew Cross
"""
from ._register import Register
from ._qiskitexception import QISKitException


class Instruction(object):
    """Generic quantum computer instruction."""

    def __init__(self, name, param, arg, circuit=None):
        """Create a new instruction.

        name = instruction name string
        param = list of real parameters
        arg = list of pairs (Register, index)
        circuit = QuantumCircuit or CompositeGate containing this instruction
        """
        for a in arg:
            if not isinstance(a[0], Register):
                raise QISKitException("argument not (Register, int) tuple")
        self.name = name
        self.param = param
        self.arg = arg
        self.control = None  # tuple (ClassicalRegister, int) for "if"
        self.circuit = circuit

    def check_circuit(self):
        """Raise exception if self.circuit is None."""
        if self.circuit is None:
            raise QISKitException("Instruction's circuit not assigned")

    def c_if(self, classical, val):
        """Add classical control on register clasical and value val."""
        self.check_circuit()
        self.circuit._check_creg(classical)
        if val < 0:
            raise QISKitException("control value should be non-negative")
        self.control = (classical, val)

    def _modifiers(self, gate):
        """Apply any modifiers of this instruction to another one."""
        if self.control is not None:
            self.check_circuit()
            if not gate.circuit.has_register(self.control[0]):
                raise QISKitException("control register %s not found"
                                      % self.control[0].name)
            gate.c_if(self.control[0], self.control[1])

    def _qasmif(self, string):
        """Print an if statement if needed."""
        # TODO: validate is the var String is correct
        if self.control is None:
            return string
        else:
            return "if(%s==%d) " % (self.control[0].name, self.control[1]) + string
