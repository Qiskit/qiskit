"""
Backend for the unroller that prints OPENQASM.

Author: Andrew Cross
"""
from ._backendexception import BackendException
from ._unrollerbackend import UnrollerBackend


class PrinterBackend(UnrollerBackend):
    """Backend for the unroller that prints OPENQASM.

    This backend also serves as a base class for other unroller backends.
    """

    def __init__(self, basis=None):
        """Setup this backend.

        basis is a list of operation name strings.
        """
        self.prec = 15
        self.creg = None
        self.cval = None
        self.gates = {}
        self.comments = False
        # TODO: control basis elements
        if basis:
            self.basis = basis
        else:
            self.basis = []
        self.listen = True
        self.in_gate = ""
        self.printed_gates = []

    def set_comments(self, comments):
        """Set comments to True to enable."""
        self.comments = comments

    def set_basis(self, basis):
        """Declare the set of user-defined gates to emit.

        basis is a list of operation name strings.
        """
        self.basis = basis

    def _fs(self, f):
        """Format a floating point number as a string.

        Uses self.prec to determine the precision.
        """
        fmt = "{0:0.%sf}" % self.prec
        return fmt.format(f)

    def version(self, version):
        """Print the version string.

        v is a version number.
        """
        print("OPENQASM %s;" % version)

    def new_qreg(self, name, size):
        """Create a new quantum register.

        name = name of the register
        sz = size of the register
        """
        assert size >= 0, "invalid qreg size"
        print("qreg %s[%d];" % (name, size))

    def new_creg(self, name, size):
        """Create a new classical register.

        name = name of the register
        sz = size of the register
        """
        print("creg %s[%d];" % (name, size))

    def _gate_string(self, name):
        """Print OPENQASM for the named gate."""
        out = ""
        if self.gates[name]["opaque"]:
            out = "opaque " + name
        else:
            out = "gate " + name
        if self.gates[name]["n_args"] > 0:
            out += "(" + ",".join(self.gates[name]["args"]) + ")"
            out += " " + ",".join(self.gates[name]["bits"])
        if self.gates[name]["opaque"]:
            out += ";"
        else:
            out += "\n{\n" + self.gates[name]["body"].qasm() + "}"
        return out

    def define_gate(self, name, gatedata):
        """Define a new quantum gate.

        name is a string.
        gatedata is the AST node for the gate.
        """
        atomics = ["U", "CX", "measure", "reset", "barrier"]
        self.gates[name] = gatedata
        # Print out the gate definition if it is in self.basis
        if name in self.basis and name not in atomics:
            # Print the hierarchy of gates this gate calls
            if not self.gates[name]["opaque"]:
                calls = self.gates[name]["body"].calls()
                for call in calls:
                    if call not in self.printed_gates:
                        print(self._gate_string(call))
                        self.printed_gates.append(call)
            # Print the gate itself
            if name not in self.printed_gates:
                print(self._gate_string(name))
                self.printed_gates.append(name)

    def u(self, arg, qubit):
        """Fundamental single qubit gate.

        arg is 3-tuple of float parameters.
        qubit is (regname,idx) tuple.
        """
        if self.listen:
            if "U" not in self.basis:
                self.basis.append("U")
            if self.creg is not None:
                print("if(%s==%d) " % (self.creg, self.cval), end="")
            print("U(%s,%s,%s) %s[%d];" % (self._fs(arg[0]), self._fs(arg[1]),
                                           self._fs(arg[2]), qubit[0],
                                           qubit[1]))

    def cx(self, qubit0, qubit1):
        """Fundamental two qubit gate.

        qubit0 is (regname,idx) tuple for the control qubit.
        qubit1 is (regname,idx) tuple for the target qubit.
        """
        if self.listen:
            if "CX" not in self.basis:
                self.basis.append("CX")
            if self.creg is not None:
                print("if(%s==%d) " % (self.creg, self.cval), end="")
            print("CX %s[%d],%s[%d];" % (qubit0[0], qubit0[1],
                                         qubit1[0], qubit1[1]))

    def measure(self, qubit, bit):
        """Measurement operation.

        qubit is (regname, idx) tuple for the input qubit.
        bit is (regname, idx) tuple for the output bit.
        """
        if "measure" not in self.basis:
            self.basis.append("measure")
        if self.creg is not None:
            print("if(%s==%d) " % (self.creg, self.cval), end="")
        print("measure %s[%d] -> %s[%d];" % (qubit[0], qubit[1],
                                             bit[0], bit[1]))

    def barrier(self, qubitlists):
        """Barrier instruction.

        qubitlists is a list of lists of (regname, idx) tuples.
        """
        if self.listen:
            if "barrier" not in self.basis:
                self.basis.append("barrier")
            names = []
            for qubitlist in qubitlists:
                if len(qubitlist) == 1:
                    names.append("%s[%d]" % (qubitlist[0][0], qubitlist[0][1]))
                else:
                    names.append("%s" % qubitlist[0][0])
            print("barrier %s;" % ",".join(names))

    def reset(self, qubit):
        """Reset instruction.

        qubit is a (regname, idx) tuple.
        """
        if "reset" not in self.basis:
            self.basis.append("reset")
        if self.creg is not None:
            print("if(%s==%d) " % (self.creg, self.cval), end="")
        print("reset %s[%d];" % (qubit[0], qubit[1]))

    def set_condition(self, creg, cval):
        """Attach a current condition.

        creg is a name string.
        cval is the integer value for the test.
        """
        self.creg = creg
        self.cval = cval
        if self.comments:
            print("// set condition %s, %s" % (creg, cval))

    def drop_condition(self):
        """Drop the current condition."""
        self.creg = None
        self.cval = None
        if self.comments:
            print("// drop condition")

    def start_gate(self, name, args, qubits):
        """Begin a custom gate.

        name is name string.
        args is list of floating point parameters.
        qubits is list of (regname, idx) tuples.
        """
        condition = None
        if self.listen and self.comments:
            print("// start %s, %s, %s" % (name, list(map(self._fs, args)),
                                           qubits))
        if self.listen and name not in self.basis \
           and self.gates[name]["opaque"]:
            raise BackendException("opaque gate %s not in basis" % name)
        if self.listen and name in self.basis:
            # TODO: check this - bug?
            if self.creg is not None:
                # TODO: for Andrew - check the condition var scope.
                condition = (self.creg, self.cval)
            else:
                condition = None
            self.in_gate = name
            self.listen = False
            squbits = ["%s[%d]" % (x[0], x[1]) for x in qubits]
            print(name, end="")
            if len(args) > 0:
                print("(%s)" % ",".join(map(self._fs, args)), end="")
            print(" %s;" % ",".join(squbits))

    def end_gate(self, name, args, qubits):
        """End a custom gate.

        name is name string.
        args is list of floating point parameters.
        qubits is list of (regname, idx) tuples.
        """
        if name == self.in_gate:
            self.in_gate = ""
            self.listen = True
        if self.listen and self.comments:
            print("// end %s, %s, %s" % (name, list(map(self._fs, args)),
                                         qubits))
