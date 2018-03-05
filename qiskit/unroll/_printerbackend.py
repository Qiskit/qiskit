# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Backend for the unroller that prints OpenQASM.
"""
from ._backenderror import BackendError
from ._unrollerbackend import UnrollerBackend


class PrinterBackend(UnrollerBackend):
    """Backend for the unroller that prints OpenQASM.

    This backend also serves as an example class for other unroller backends.
    """

    def __init__(self, basis=None):
        """Setup this backend.

        basis is a list of operation name strings.
        """
        super().__init__(basis)
        self.prec = 15
        self.creg = None
        self.cval = None
        self.gates = {}
        self.comments = False
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

    def u(self, arg, qubit, nested_scope=None):
        """Fundamental single qubit gate.

        arg is 3-tuple of Node expression objects.
        qubit is (regname,idx) tuple.
        nested_scope is a list of dictionaries mapping expression variables
        to Node expression objects in order of increasing nesting depth.
        """
        if self.listen:
            if "U" not in self.basis:
                self.basis.append("U")
            if self.creg is not None:
                print("if(%s==%d) " % (self.creg, self.cval), end="")
            print("U(%s,%s,%s) %s[%d];" % (arg[0].sym(nested_scope),
                                           arg[1].sym(nested_scope),
                                           arg[2].sym(nested_scope),
                                           qubit[0],
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

    def start_gate(self, name, args, qubits, nested_scope=None):
        """Begin a custom gate.

        name is name string.
        args is list of Node expression objects.
        qubits is list of (regname, idx) tuples.
        nested_scope is a list of dictionaries mapping expression variables
        to Node expression objects in order of increasing nesting depth.
        """
        if self.listen and self.comments:
            print("// start %s, %s, %s" % (name,
                                           list(map(lambda x:
                                                    str(x.sym(nested_scope)),
                                                    args)),
                                           qubits))
        if self.listen and name not in self.basis \
           and self.gates[name]["opaque"]:
            raise BackendError("opaque gate %s not in basis" % name)
        if self.listen and name in self.basis:
            self.in_gate = name
            self.listen = False
            squbits = ["%s[%d]" % (x[0], x[1]) for x in qubits]
            if self.creg is not None:
                print("if(%s==%d) " % (self.creg, self.cval), end="")
            print(name, end="")
            if args:
                print("(%s)" % ",".join(map(lambda x:
                                            str(x.sym(nested_scope)),
                                            args)), end="")
            print(" %s;" % ",".join(squbits))

    def end_gate(self, name, args, qubits, nested_scope=None):
        """End a custom gate.

        name is name string.
        args is list of Node expression objects.
        qubits is list of (regname, idx) tuples.
        nested_scope is a list of dictionaries mapping expression variables
        to Node expression objects in order of increasing nesting depth.
        """
        if name == self.in_gate:
            self.in_gate = ""
            self.listen = True
        if self.listen and self.comments:
            print("// end %s, %s, %s" % (name,
                                         list(map(lambda x:
                                                  str(x.sym(nested_scope)),
                                                  args)),
                                         qubits))

    def get_output(self):
        """This backend will return nothing, as the output has been directly
        written to screen"""
        pass
