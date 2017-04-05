"""
Element of SU(2).

Author: Andrew Cross
"""
from ._gate import Gate
from ._qiskitexception import QISKitException


class UBase(Gate):
    """Element of SU(2)."""

    def __init__(self, param, qubit):
        """Create new reset instruction."""
        if len(param) != 3:
            raise QISKitException("expected 3 parameters")
        super(UBase, self).__init__("U", param, [qubit])

    def qasm(self):
        """Return OPENQASM string."""
        theta = self.param[0]
        phi = self.param[1]
        lamb = self.param[2]
        qubit = self.arg[0]
        return self._qasmif("U(%.15f,%.15f,%.15f) %s[%d];" % (theta, phi,
                            lamb, qubit[0].name, qubit[1]))

    def inverse(self):
        """Invert this gate.

        U(theta,phi,lambda)^dagger = U(-theta,-lambda,-phi)
        """
        self.param[0] = -self.param[0]
        phi = self.param[1]
        self.param[1] = -self.param[2]
        self.param[2] = -phi
        return self

    def reapply(self, circ):
        """
        Reapply this gate to corresponding qubits in circ.

        Register index bounds checked by the gate method.
        """
        rearg = self._remap_arg(circ)
        self._modifiers(circ.ubase(rearg[0]))
