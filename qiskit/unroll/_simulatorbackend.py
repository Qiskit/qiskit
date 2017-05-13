"""
Backend for the unroller that composes unitary matrices to simulate circuit.

"""
import numpy as np
from qiskit.unroll import BackendException
from qiskit.unroll import UnrollerBackend


class SimulatorBackend(UnrollerBackend):
    """Backend for the unroller that composes unitary matrices."""

    def __init__(self, basis=None):
        """Setup this backend.

        basis is a list of operation name strings.
        """
        self.unitary_gates = {}
        self.unitary_gates['quantum_circuit'] = []
        self.number_of_qubits = 0
        self.number_of_cbits = 0
        self.qubit_order = {}
        self.cbit_order = {}
        self.gate_order = 0
        self.prec = 15
        self.creg = None
        self.cval = None
        self.gates = {}
        self.trace = False
        if basis:
            self.basis = basis
        else:
            self.basis = []  # default, unroll to U, CX
        self.listen = True
        self.in_gate = ""
        self.printed_gates = []

    def set_trace(self, trace):
        """Set trace to True to enable."""
        self.trace = trace

    def set_basis(self, basis):
        """Declare the set of user-defined gates to emit.

        basis is a list of operation name strings.
        """
        self.basis = basis

    def _fs(self, number):
        """Format a floating point number as a string.

        Uses self.prec to determine the precision.
        """
        fmt = "{0:0.%sf}" % self.prec
        return fmt.format(number)

    def version(self, version):
        """Print the version string.

        v is a version number.
        """
        pass

    def new_qreg(self, name, size):
        """Create a new quantum register.

        name = name of the register
        sz = size of the register
        """
        assert size >= 0, "invalid qreg size"

        for j in range(size):
            self.qubit_order[(name, j)] = self.number_of_qubits + j
        self.number_of_qubits += size
        # print(self.qubit_order)
        self.unitary_gates['number_of_qubits'] = self.number_of_qubits
        self.unitary_gates['qubit_order'] = self.qubit_order
        if self.trace:
            print("added %d qubits from qreg %s giving a total of %d qubits" %
                  (size, name, self.number_of_qubits))

    def new_creg(self, name, size):
        """Create a new classical register.

        name = name of the register
        sz = size of the register
        """
        assert size >= 0, "invalid creg size"

        for j in range(size):
            self.cbit_order[(name, j)] = self.number_of_cbits + j
        self.number_of_cbits += size
        self.unitary_gates['number_of_cbits'] = self.number_of_cbits
        self.unitary_gates['cbit_order'] = self.cbit_order
        if self.trace:
            print("added %d cbits from creg %s giving a total of %d qubits" %
                  (size, name, self.number_of_cbits))

    def define_gate(self, name, gatedata):
        """Define a new quantum gate.

        name is a string.
        gatedata is the AST node for the gate.
        """
        self.gates[name] = gatedata
        # Jay: you don't need to do anything here,
        #      this is called when the unroller sees "gate blah() blah {}"

    def u(self, arg, qubit):
        """Fundamental single qubit gate.

        arg is 3-tuple of float parameters.
        qubit is (regname,idx) tuple.
        """
        if self.listen:
            if "U" not in self.basis:
                self.basis.append("U")
            if self.trace:
                if self.creg is not None:
                    print("if(%s==%d) " % (self.creg, self.cval), end="")
                print("U(%s,%s,%s) %s[%d];" % (self._fs(arg[0]),
                                               self._fs(arg[1]),
                                               self._fs(arg[2]), qubit[0],
                                               qubit[1]))
            if self.creg is not None:
                raise BackendException("UnitarySimulator does not support if")
            indices = [self.qubit_order.get(qubit)]
            self.gate_order += 1
            # print(np.exp(1j*arg[2])*np.sin(arg[0]/2.0))
            gate = np.array([[np.cos(arg[0]/2.0),
                            -np.exp(1j*arg[2])*np.sin(arg[0]/2.0)],
                            [np.exp(1j*arg[1])*np.sin(arg[0]/2.0),
                            np.exp(1j*arg[1]+1j*arg[2])*np.cos(arg[0]/2.0)]])
            self.unitary_gates['number_of_gates'] = self.gate_order
            self.unitary_gates['quantum_circuit'].append({
                        # 'order': self.gate_order,
                        'gate_size': 1,
                        'matrix': gate,
                        'name': "U(%s,%s,%s)" % (self._fs(arg[0]),
                                                 self._fs(arg[1]),
                                                 self._fs(arg[2])),
                        'indices': indices
                        })

            # print(self.unitary_gates)

    def cx(self, qubit0, qubit1):
        """Fundamental two qubit gate.

        qubit0 is (regname,idx) tuple for the control qubit.
        qubit1 is (regname,idx) tuple for the target qubit.
        """
        if self.listen:
            if "CX" not in self.basis:
                self.basis.append("CX")
            if self.trace:
                if self.creg is not None:
                    print("if(%s==%d) " % (self.creg, self.cval), end="")
                print("CX %s[%d],%s[%d];" % (qubit0[0], qubit0[1],
                                             qubit1[0], qubit1[1]))
            if self.creg is not None:
                raise BackendException("UnitarySimulator does not support if")
            indices = [self.qubit_order.get(qubit0),
                       self.qubit_order.get(qubit1)]
            self.gate_order += 1
            cx = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1],
                          [0, 0, 1, 0]])
            self.unitary_gates['number_of_gates'] = self.gate_order
            self.unitary_gates['quantum_circuit'].append({
                        # 'order': self.gate_order,
                        'gate_size': 2,
                        'matrix': cx,
                        'name': 'CX',
                        'indices': indices,
                        })

    def measure(self, qubit, bit):
        """Measurement operation.

        qubit is (regname, idx) tuple for the input qubit.
        bit is (regname, idx) tuple for the output bit.
        """
        raise BackendException("UnitarySimulator does not support measurement")

    def barrier(self, qubitlists):
        """Barrier instruction.

        qubitlists is a list of lists of (regname, idx) tuples.
        """
        pass  # ignore barriers

    def reset(self, qubit):
        """Reset instruction.

        qubit is a (regname, idx) tuple.
        """
        raise BackendException("UnitarySimulator does not support reset")

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
        if self.listen and self.trace and name not in self.basis:
            print("// start %s, %s, %s" % (name, list(map(self._fs, args)),
                                           qubits))
        if self.listen and name not in self.basis \
           and self.gates[name]["opaque"]:
            raise BackendException("opaque gate %s not in basis" % name)
        if self.listen and name in self.basis:
            self.in_gate = name
            self.listen = False
            squbits = ["%s[%d]" % (x[0], x[1]) for x in qubits]
            if self.trace:
                if self.creg is not None:
                    print("if(%s==%d) " % (self.creg, self.cval), end="")
                print(name, end="")
                if len(args) > 0:
                    print("(%s)" % ",".join(map(self._fs, args)), end="")
                print(" %s;" % ",".join(squbits))
            if self.creg is not None:
                raise BackendException("UnitarySimulator does not support if")
            # Jay: update here down with your code for any other gates,
            #      like h, u1, u2, u3, if you want to treat those specially.
            # Otherwise, set self.basis = [] and just implement U and CX.

    def end_gate(self, name, args, qubits):
        """End a custom gate.

        name is name string.
        args is list of floating point parameters.
        qubits is list of (regname, idx) tuples.
        """
        if name == self.in_gate:
            self.in_gate = ""
            self.listen = True
        if self.listen and self.trace and name not in self.basis:
            print("// end %s, %s, %s" % (name, list(map(self._fs, args)),
                                         qubits))
