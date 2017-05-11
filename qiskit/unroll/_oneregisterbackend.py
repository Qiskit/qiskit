"""
Backend that creates a Circuit object with one "q" qreg.

Author: Andrew Cross
"""
from ._unrollerbackend import UnrollerBackend
from ._backendexception import BackendException
from ..circuit import Circuit


class OneRegisterBackend(UnrollerBackend):
    """Backend that merges qregs in "q"."""

    def __init__(self, basis=None):
        """Setup this backend.

        basis is a list of operation name strings.
        """
        self.prec = 15
        self.creg = None
        self.cval = None
        self.circuit = Circuit()
        if basis:
            self.basis = basis
        else:
            self.basis = []
        self.listen = True
        self.in_gate = ""
        self.gates = {}
        self.qubit_map = {}
        self.index_sum = 0
        self.created_qreg = False

    def initialize_qreg(self):
        """Add the qreg to the Circuit."""
        if not self.created_qreg:
            self.circuit.add_qreg("q", self.index_sum)
            self.created_qreg = True

    def set_basis(self, basis):
        """Declare the set of user-defined gates to emit."""
        self.basis = basis

    def define_gate(self, name, gatedata):
        """Record and pass down the data for this gate."""
        self.gates[name] = gatedata
        self.circuit.add_gate_data(name, gatedata)

    def version(self, version):
        """Accept the version string.

        v is a version number.
        """
        pass

    def new_qreg(self, name, size):
        """Create a new quantum register.

        name = name of the register
        sz = size of the register
        """
        if self.created_qreg:
            # This is not ideal. Come back to it later.
            raise BackendException("sorry, already added the qreg; please ",
                                   "declare all qregs before applying a gate")
        # Add qubits to the map but don't add qreg
        for j in range(size):
            dest = ("q", j + self.index_sum)
            self.qubit_map[(name, j)] = dest
        self.index_sum += size

    def new_creg(self, name, size):
        """Create a new classical register.

        name = name of the register
        sz = size of the register
        """
        self.circuit.add_creg(name, size)

    def u(self, arg, qubit):
        """Fundamental single qubit gate.

        arg is 3-tuple of float parameters.
        qubit is (regname,idx) tuple.
        """
        self.initialize_qreg()
        qubit = self.qubit_map[qubit]
        if self.listen:
            if self.creg is not None:
                condition = (self.creg, self.cval)
            else:
                condition = None
            if "U" not in self.basis:
                self.basis.append("U")
                self.circuit.add_basis_element("U", 1, 0, 3)
            self.circuit.apply_operation_back("U", [qubit], [], list(arg), condition)

    def cx(self, qubit0, qubit1):
        """Fundamental two qubit gate.

        qubit0 is (regname,idx) tuple for the control qubit.
        qubit1 is (regname,idx) tuple for the target qubit.
        """
        self.initialize_qreg()
        qubit0 = self.qubit_map[qubit0]
        qubit1 = self.qubit_map[qubit1]
        if self.listen:
            if self.creg is not None:
                condition = (self.creg, self.cval)
            else:
                condition = None
            if "CX" not in self.basis:
                self.basis.append("CX")
                self.circuit.add_basis_element("CX", 2)
            self.circuit.apply_operation_back("CX", [qubit0, qubit1], [],
                                              [], condition)

    def measure(self, qubit, bit):
        """Measurement operation.

        qubit is (regname, idx) tuple for the input qubit.
        bit is (regname, idx) tuple for the output bit.
        """
        self.initialize_qreg()
        qubit = self.qubit_map[qubit]
        if self.creg is not None:
            condition = (self.creg, self.cval)
        else:
            condition = None
        if "measure" not in self.basis:
            self.basis.append("measure")
            self.circuit.add_basis_element("measure", 1, 1)
        self.circuit.apply_operation_back("measure", [qubit], [bit], [], condition)

    def barrier(self, qubitlists):
        """Barrier instruction.

        qubitlists is a list of lists of (regname, idx) tuples.
        """
        self.initialize_qreg()
        if self.listen:
            names = []
            for x in qubitlists:
                for j in range(len(x)):
                    names.append(self.qubit_map[x[j]])
            if "barrier" not in self.basis:
                self.basis.append("barrier")
                self.circuit.add_basis_element("barrier", -1)
            self.circuit.apply_operation_back("barrier", names)

    def reset(self, qubit):
        """Reset instruction.

        qubit is a (regname, idx) tuple.
        """
        self.initialize_qreg()
        qubit = self.qubit_map[qubit]
        if self.creg is not None:
            condition = (self.creg, self.cval)
        else:
            condition = None
        if "reset" not in self.basis:
            self.basis.append("reset")
            self.circuit.add_basis_element("reset", 1)
        self.circuit.apply_operation_back("reset", [qubit], [], [], condition)

    def set_condition(self, creg, cval):
        """Attach a current condition.

        creg is a name string.
        cval is the integer value for the test.
        """
        self.creg = creg
        self.cval = cval

    def drop_condition(self):
        """Drop the current condition."""
        self.creg = None
        self.cval = None

    def start_gate(self, name, args, qubits):
        """Begin a custom gate.

        name is name string.
        args is list of floating point parameters.
        qubits is list of (regname, idx) tuples.
        """
        self.initialize_qreg()
        qubits = [self.qubit_map[x] for x in qubits]
        if self.listen and name not in self.basis \
           and self.gates[name]["opaque"]:
            raise BackendException("opaque gate %s not in basis" % name)
        if self.listen and name in self.basis:
            if self.creg is not None:
                condition = (self.creg, self.cval)
            else:
                condition = None
            self.in_gate = name
            self.listen = False
            self.circuit.add_basis_element(name, len(qubits), 0, len(args))
            self.circuit.apply_operation_back(name, qubits, [], args,
                                              condition)

    def end_gate(self, name, args, qubits):
        """End a custom gate.

        name is name string.
        args is list of floating point parameters.
        qubits is list of (regname, idx) tuples.
        """
        if name == self.in_gate:
            self.in_gate = ""
            self.listen = True
