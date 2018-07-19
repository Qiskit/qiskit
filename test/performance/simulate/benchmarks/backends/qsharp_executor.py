"""
Q# Backend
"""
import subprocess

from qiskit.unroll._backenderror import BackendError
from qiskit.unroll._unrollerbackend import UnrollerBackend
from qiskit.unroll import DagUnroller, DAGBackend
from qiskit.dagcircuit import DAGCircuit


class PrintQsharp(UnrollerBackend):
    """Backend for the unroller that prints Q# code.
    """

    def __init__(self, file, basis=None):
        super().__init__(basis)
        self.gate_val = {"gates": {}, "in_gate": "", "basis": []}
        if basis:
            self.gate_val["basis"] = basis
        else:
            self.gate_val["basis"] = []
        self.flags = {"comments": False, "listen": True,
                      "measure_flag": False, "print_header": False}
        self.file = file
        self.level = 0
        self.qreg_attr = {"name": None, "size": 0}
        self.creg_attr = {"creg": None, "cval": None, "name": None, "size": 0}

    def _print_code(self, line, end="\n", with_indent=True):
        """
        Print Indent and Code line
        """
        space = ""
        if with_indent:
            for _ in range(self.level):
                space += "    "
        self.file.write(space + line + end)
        # print (space + line, end=end)

    def set_comments(self, comments):
        """
        set comments
        """
        pass

    def set_basis(self, basis):
        """Declare the set of user-defined gates to emit.

        basis is a list of operation name strings.
        """
        self.gate_val["basis"] = basis

    def _print_header_if_necessary(self):
        if (not self.flags["print_header"] and self.qreg_attr["size"] != 0 and
                self.creg_attr["size"] != 0):
            self._print_code("mutable %s = new Result[%d];" %
                             (self.creg_attr["name"], self.creg_attr["size"]))
            self._print_code("using (%s = Qubit[%d]) {" %
                             (self.qreg_attr["name"], self.qreg_attr["size"]))
            self.level += 1
            self.flags["print_header"] = True

    def version(self, version):
        """Print the version string.

        v is a version number.
        """
        pass

    def print_header(self):
        """
        Print Header
        """
        self._print_code("namespace QasmBench." + "Bench" + " {")
        self.level += 1

        self._print_code("open Microsoft.Quantum.Primitive;")
        self._print_code("open Microsoft.Quantum.Canon;")
        self._print_code("open Microsoft.Quantum.Extensions.Math;")

        self._print_code("operation " + "Bench" + "() : Result[] {")
        self.level += 1

        self._print_code("body {")
        self.level += 1

    def new_qreg(self, name, size):
        if self.qreg_attr["size"] != 0:
            raise Exception("support only one quantum register array")

        self.qreg_attr["name"] = name
        self.qreg_attr["size"] = size
        self._print_header_if_necessary()

    def new_creg(self, name, size):
        if self.creg_attr["size"] != 0:
            raise Exception("support only one classical register array")
        self.creg_attr["name"] = name
        self.creg_attr["size"] = size
        self._print_header_if_necessary()

    def print_footer(self):
        """
        Print Footer
        """
        self._print_code("ResetAll(%s);" % self.qreg_attr["name"])
        self.level -= 1
        self._print_code("}")
        self._print_code("return %s;" % self.creg_attr["name"])
        self.level -= 1
        self._print_code("}")
        self.level -= 1
        self._print_code("}")
        self.level -= 1
        self._print_code("}")

    def define_gate(self, name, gatedata):
        """Define a new quantum gate.

        name is a string.
        gatedata is the AST node for the gate.
        """
        atomics = ["U", "CX", "measure", "reset", "barrier"]
        self.gate_val["gates"][name] = gatedata
        # Print out the gate definition if it is in self.basis
        basis_list = self.gate_val["basis"]
        if name in basis_list and name not in atomics:
            raise Exception("support only U, CX, MEASURE, RESET, and BARRIER: unsupport=" + name)

    @classmethod
    def _resolve(cls, line):
        return line.replace("pi", "PI()")

    def _print_start_if(self):
        for i in range(self.creg_attr["size"]):
            if i == 0:
                self._print_code("if (", end="")
            else:
                self._print_code(" && ", end="", with_indent=False)
            if 1 << i & self.creg_attr["cval"] == 0:
                self._print_code("(%s[%d]==Zero)" %
                                 (self.creg_attr["name"], i), end="", with_indent=False)
            else:
                self._print_code("(%s[%d]==One)" %
                                 (self.creg_attr["name"], i), end="", with_indent=False)
        self._print_code(") {", with_indent=False)
        self.level += 1

    def _print_end_if(self):
        self.level -= 1
        self._print_code("}")

    def u(self, arg, qubit, nested_scope=None):
        if self.flags["listen"]:
            if "U" not in self.gate_val["basis"]:
                self.gate_val["basis"].append("U")

            if self.creg_attr["creg"] is not None:
                self._print_start_if()

            self._print_code("//U(%s,%s,%s) %s[%d];" % (arg[0].sym(nested_scope),
                                                        arg[1].sym(nested_scope),
                                                        arg[2].sym(nested_scope),
                                                        qubit[0],
                                                        qubit[1]))
            if arg[0].sym(nested_scope) != 0:
                self._print_code("Rz(%s, %s[%d]);" %
                                 (self._resolve(str(arg[0].sym(nested_scope))), qubit[0], qubit[1]))
            if arg[1].sym(nested_scope) != 0:
                self._print_code("Ry(%s, %s[%d]);" %
                                 (self._resolve(str(arg[1].sym(nested_scope))), qubit[0], qubit[1]))
            if arg[2].sym(nested_scope) != 0:
                self._print_code("Rz(%s, %s[%d]);" %
                                 (self._resolve(str(arg[2].sym(nested_scope))), qubit[0], qubit[1]))

            if self.creg_attr["creg"] is not None:
                self._print_end_if()

    def cx(self, qubit0, qubit1):
        """Fundamental two qubit gate.

        qubit0 is (regname,idx) tuple for the control qubit.
        qubit1 is (regname,idx) tuple for the target qubit.
        """
        if self.flags["listen"]:
            if "CX" not in self.gate_val["basis"]:
                self.gate_val["basis"].append("CX")

            if self.creg_attr["creg"] is not None:
                self._print_start_if()

            self._print_code("//CX %s[%d],%s[%d];" % (qubit0[0], qubit0[1], qubit1[0], qubit1[1]))
            self._print_code("CNOT (%s[%d], %s[%d]);" %
                             (qubit0[0], qubit0[1], qubit1[0], qubit1[1]))

            if self.creg_attr["creg"] is not None:
                self._print_end_if()

    def measure(self, qubit, bit):
        """Measurement operation.

        qubit is (regname, idx) tuple for the input qubit.
        bit is (regname, idx) tuple for the output bit.
        """
        if "measure" not in self.gate_val["basis"]:
            self.gate_val["basis"].append("measure")

        if self.creg_attr["creg"] is not None:
            self._print_code("//if(%s==%d) " %
                             (self.creg_attr["creg"], self.creg_attr["cval"]), end="")
            self._print_start_if()

        self._print_code("//measure %s[%d] -> %s[%d];" % (qubit[0], qubit[1], bit[0], bit[1]))
        self._print_code("set %s[%d] = M(%s[%d]);" % (bit[0], bit[1], qubit[0], qubit[1]))

        if self.creg_attr["creg"] is not None:
            self._print_end_if()

    def barrier(self, qubitlists):
        """Barrier instruction.

        qubitlists is a list of lists of (regname, idx) tuples.
        """
        if self.flags["listen"]:
            if "barrier" not in self.gate_val["basis"]:
                self.gate_val["basis"].append("barrier")
            names = []
            for qubitlist in qubitlists:
                if len(qubitlist) == 1:
                    names.append("%s[%d]" % (qubitlist[0][0], qubitlist[0][1]))
                else:
                    names.append("%s" % qubitlist[0][0])
            self._print_code("//barrier %s;" % ",".join(names))

    def reset(self, qubit):
        """Reset instruction.

        qubit is a (regname, idx) tuple.
        """
        if "reset" not in self.gate_val["basis"]:
            self.gate_val["basis"].append("reset")
        if self.creg_attr["creg"] is not None:
            self._print_code("if(%s==%d) " %
                             (self.creg_attr["creg"], self.creg_attr["cval"]), end="")
        self._print_code("reset %s[%d];" % (qubit[0], qubit[1]))

    def set_condition(self, creg, cval):
        """Attach a current condition.

        creg is a name string.
        cval is the integer value for the test.
        """
        self.creg_attr["creg"] = creg
        self.creg_attr["cval"] = cval
        if self.flags["comments"]:
            self._print_code("// set condition %s, %s" % (creg, cval))

    def drop_condition(self):
        """Drop the current condition."""
        self.creg_attr["creg"] = None
        self.creg_attr["cval"] = None
        if self.flags["comments"]:
            self._print_code("// drop condition")

    def start_gate(self, name, args, qubits, nested_scope=None):
        """Begin a custom gate.

        name is name string.
        args is list of Node expression objects.
        qubits is list of (regname, idx) tuples.
        nested_scope is a list of dictionaries mapping expression variables
        to Node expression objects in order of increasing nesting depth.
        """
        if self.flags["listen"] and self.flags["comments"]:
            self._print_code("// start %s, %s, %s" %
                             (name, list(map(lambda x:
                                             str(x.sym(nested_scope)), args)),
                              qubits))
        if self.flags["listen"] and name not in self.gate_val["basis"] \
           and self.gate_val["gates"][name]["opaque"]:
            raise BackendError("opaque gate %s not in basis" % name)
        if self.flags["listen"] and name in self.gate_val["basis"]:
            raise Exception("custom gate is not supported.")

    def end_gate(self, name, args, qubits, nested_scope=None):
        """End a custom gate.

        name is name string.
        args is list of Node expression objects.
        qubits is list of (regname, idx) tuples.
        nested_scope is a list of dictionaries mapping expression variables
        to Node expression objects in order of increasing nesting depth.
        """
        if name == self.gate_val["in_gate"]:
            self.gate_val["in_gate"] = ""
            self.flags["listen"] = True
        if self.flags["listen"] and self.flags["comments"]:
            self._print_code("// end %s, %s, %s" %
                             (name, list(map(lambda x:
                                             str(x.sym(nested_scope)), args)),
                              qubits))

    def get_output(self):
        """This backend will return nothing, as the output has been directly
        written to screen"""
        pass


class QsharpExecutor(object):
    """
    Executor for Q#
    """
    def __init__(self, executor):
        self.name = ["Qsharp"]
        self.seed = executor.seed
        self.application = executor.name
        self.backend_name = executor.backend_name
        self.result = None

    @classmethod
    def generator(cls, circ):
        """
        generate projecq code from QuantumProgram
        """
        dagcirc = DAGCircuit.fromQuantumCircuit(circ)

        expand_circ = DagUnroller(dagcirc, DAGBackend())
        expand_circ.expand_gates()
        qs_circ = expand_circ.execute()

        q_file = open("backends/workspace/Qsharp/benchmark/benchmark.qs", 'w')

        print_qs = PrintQsharp(q_file)
        qs_circ = DagUnroller(qs_circ, print_qs)
        print_qs.print_header()
        qs_circ.execute()
        print_qs.print_footer()
        q_file.close()

    @classmethod
    def run_simulation(cls, q_prog):
        """
        run simulation on ProjectQ
        """
        cls.generator(q_prog)
        ret = subprocess.check_output("dotnet run -c Release",
                                      shell=True, cwd="backends/workspace/Qsharp/benchmark")
        return float(ret)

    @classmethod
    def verify_result(cls):
        """
        Verify simulation results
        """
        raise Exception("Not supported")
