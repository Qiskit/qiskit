"""
Unitary gate.

Author: Andrew Cross
"""
from ._instruction import Instruction
from ._quantumregister import QuantumRegister
from ._qiskitexception import QISKitException


class Gate(Instruction):
    """Unitary gate."""

    def __init__(self, name, param, args, circuit=None):
        """Create a new composite gate.

        name = instruction name string
        param = list of real parameters
        arg = list of pairs (Register, index)
        circuit = QuantumCircuit or CompositeGate containing this gate
        """
        for argument in args:
            if not isinstance(argument[0], QuantumRegister):
                raise QISKitException("argument not (QuantumRegister, int) "
                                      + "tuple")
        super(Gate, self).__init__(name, param, args, circuit)

    def inverse(self):
        """Invert this gate."""
        raise QISKitException("inverse not implemented")

    def q_if(self, *qregs):
        """Add controls to this gate."""
        raise QISKitException("control not implemented")
