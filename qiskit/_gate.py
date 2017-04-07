"""
Unitary gate.

Author: Andrew Cross
"""
from ._instruction import Instruction
from ._quantumregister import QuantumRegister
from ._qiskitexception import QISKitException


class Gate(Instruction):
    """Unitary gate."""

    def __init__(self, name, param, arg, circ=None):
        """Create a new composite gate.

        name = instruction name string
        param = list of real parameters
        arg = list of pairs (Register, index)
        circ = QuantumCircuit or CompositeGate containing this gate
        """
        for a in arg:
            if not isinstance(a[0], QuantumRegister):
                raise QISKitException("argument not (QuantumRegister, int) "
                                      + "tuple")
        super(Gate, self).__init__(name, param, arg, circ)

    def inverse(self):
        """Invert this gate."""
        raise QISKitException("inverse not implemented")

    def q_if(self, *qregs):
        """Add controls to this gate."""
        raise QISKitException("control not implemented")
