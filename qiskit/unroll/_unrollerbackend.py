"""
Base backend object for the unroller that raises BackendException.

Author: Andrew Cross
"""
from ._backendexception import BackendException


class UnrollerBackend(object):
    """Backend for the unroller that raises BackendException.

    This backend also serves as a base class for other unroller backends.
    """

    def __init__(self, basis=None):
        """Setup this backend.

        basis is a list of operation name strings.
        """
        if basis:
            basis = []
        raise BackendException("Backend __init__ unimplemented")

    def set_basis(self, basis):
        """Declare the set of user-defined gates to emit.

        basis is a list of operation name strings.
        """
        raise BackendException("Backend set_basis unimplemented")

    def version(self, version):
        """Print the version string.

        v is a version number.
        """
        raise BackendException("Backend version unimplemented")

    def new_qreg(self, name, size):
        """Create a new quantum register.

        name = name of the register
        sz = size of the register
        """
        raise BackendException("Backend new_qreg unimplemented")

    def new_creg(self, name, size):
        """Create a new classical register.

        name = name of the register
        sz = size of the register
        """
        raise BackendException("Backend new_creg unimplemented")

    def define_gate(self, name, gatedata):
        """Define a new quantum gate.

        name is a string.
        gatedata is the AST node for the gate.
        """
        raise BackendException("Backend define_gate unimplemented")

    def u(self, arg, qubit):
        """Fundamental single qubit gate.

        arg is 3-tuple of float parameters.
        qubit is (regname,idx) tuple.
        """
        raise BackendException("Backend u unimplemented")

    def cx(self, qubit0, qubit1):
        """Fundamental two qubit gate.

        qubit0 is (regname,idx) tuple for the control qubit.
        qubit1 is (regname,idx) tuple for the target qubit.
        """
        raise BackendException("Backend cx unimplemented")

    def measure(self, qubit, bit):
        """Measurement operation.

        qubit is (regname, idx) tuple for the input qubit.
        bit is (regname, idx) tuple for the output bit.
        """
        raise BackendException("Backend measure unimplemented")

    def barrier(self, qubitlists):
        """Barrier instruction.

        qubitlists is a list of lists of (regname, idx) tuples.
        """
        raise BackendException("Backend barrier unimplemented")

    def reset(self, qubit):
        """Reset instruction.

        qubit is a (regname, idx) tuple.
        """
        raise BackendException("Backend reset unimplemented")

    def set_condition(self, creg, cval):
        """Attach a current condition.

        creg is a name string.
        cval is the integer value for the test.
        """
        raise BackendException("Backend set_condition unimplemented")

    def drop_condition(self):
        """Drop the current condition."""
        raise BackendException("Backend drop_condition unimplemented")

    def start_gate(self, name, args, qubits):
        """Begin a custom gate.

        name is name string.
        args is list of floating point parameters.
        qubits is list of (regname, idx) tuples.
        """
        raise BackendException("Backend start_gate unimplemented")

    def end_gate(self, name, args, qubits):
        """End a custom gate.

        name is name string.
        args is list of floating point parameters.
        qubits is list of (regname, idx) tuples.
        """
        raise BackendException("Backend end_gate unimplemented")
