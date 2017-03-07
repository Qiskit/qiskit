"""
OPENQASM interpreter.

Author: Andrew Cross
"""
import math
import copy
from ._unrollerexception import UnrollerException


class Unroller(object):
    """OPENQASM interpreter object that unrolls subroutines and loops."""

    def __init__(self, ast, be=None):
        """Initialize interpreter's data."""
        # Abstract syntax tree from parser
        self.ast = ast
        # Backend object
        self.be = be
        # OPENQASM version number
        self.version = 0.0
        # Dict of qreg names and sizes
        self.qregs = {}
        # Dict of creg names and sizes
        self.cregs = {}
        # Dict of gates names and properties
        self.gates = {}
        # List of dictionaries mapping local parameter ids to real values
        self.arg_stack = [{}]
        # List of dictionaries mapping local bit ids to global ids (name,idx)
        self.bit_stack = [{}]

    def _process_bit_id(self, n):
        """Process an Id or IndexedId node as a bit or register type.

        Return a list of tuples (name,index).
        """
        if n.type == "indexed_id":
            # An indexed bit or qubit
            return [(n.name, n.index)]
        elif n.type == "id":
            # A qubit or qreg or creg
            if len(self.bit_stack[-1]) == 0:
                # Global scope
                if n.name in self.qregs:
                    return [(n.name, j) for j in range(self.qregs[n.name])]
                elif n.name in self.cregs:
                    return [(n.name, j) for j in range(self.cregs[n.name])]
                else:
                    raise UnrollerException("expected qreg or creg name:",
                                            "line=%s" % n.line,
                                            "file=%s" % n.file)
            else:
                # local scope
                if n.name in self.bit_stack[-1]:
                    return [self.bit_stack[-1][n.name]]
                else:
                    raise UnrollerException("excepted local bit name:",
                                            "line=%s" % n.line,
                                            "file=%s" % n.file)

    def _process_local_id(self, n):
        """Process an Id node n as a local id."""
        # The id must be in arg_stack i.e. the id is inside a gate_body
        id_dict = self.arg_stack[-1]
        if n.name in id_dict:
            return float(id_dict[n.name])
        else:
            raise UnrollerException("expected local parameter name:",
                                    "line=%s" % n.line,
                                    "file=%s" % n.file)

    def _process_custom_unitary(self, n):
        """Process a custom unitary node n."""
        name = n.name
        if n.arguments is not None:
            args = self._process_node(n.arguments)
        else:
            args = []
        bits = [self._process_bit_id(m) for m in n.bitlist.children]
        if name in self.gates:
            gargs = self.gates[name]["args"]
            gbits = self.gates[name]["bits"]
            gbody = self.gates[name]["body"]
            # Loop over register arguments, if any.
            maxidx = max(map(len, bits))
            for idx in range(maxidx):
                self.arg_stack.append({gargs[j]: args[j]
                                       for j in range(len(gargs))})
                # Only index into register arguments.
                f = list(map(lambda x: idx*x,
                             [len(bits[j]) > 1 for j in range(len(bits))]))
                self.bit_stack.append({gbits[j]: bits[j][f[j]]
                                       for j in range(len(gbits))})
                self.be.start_gate(name,
                                   [self.arg_stack[-1][s] for s in gargs],
                                   [self.bit_stack[-1][s] for s in gbits])
                if not self.gates[name]["opaque"]:
                    self._process_children(gbody)
                self.be.end_gate(name,
                                 [self.arg_stack[-1][s] for s in gargs],
                                 [self.bit_stack[-1][s] for s in gbits])
                self.arg_stack.pop()
                self.bit_stack.pop()
            else:
                raise UnrollerException("internal error undefined gate:",
                                        "line=%s" % n.line, "file=%s" % n.file)

    def _process_gate(self, n, opaque=False):
        """Process a gate node n.

        If opaque is True, process the node as an opaque gate node.
        """
        self.gates[n.name] = {}
        de = self.gates[n.name]
        de["opaque"] = opaque
        de["n_args"] = n.n_args()
        de["n_bits"] = n.n_bits()
        if n.n_args() > 0:
            de["args"] = [c.name for c in n.arguments.children]
        else:
            de["args"] = []
        de["bits"] = [c.name for c in n.bitlist.children]
        if opaque:
            de["body"] = None
        else:
            de["body"] = n.body
        self.be.define_gate(n.name, copy.deepcopy(de))

    def _process_cnot(self, n):
        """Process a CNOT gate node n."""
        id0 = self._process_bit_id(n.children[0])
        id1 = self._process_bit_id(n.children[1])
        if not(len(id0) == len(id1) or len(id0) == 1 or len(id1) == 1):
            raise UnrollerException("internal error: qreg size mismatch",
                                    "line=%s" % n.line, "file=%s" % n.file)
        maxidx = max([len(id0), len(id1)])
        for idx in range(maxidx):
            if len(id0) > 1 and len(id1) > 1:
                self.be.cx(id0[idx], id1[idx])
            elif len(id0) > 1:
                self.be.cx(id0[idx], id1[0])
            else:
                self.be.cx(id0[0], id1[idx])

    def _process_binop(self, n):
        """Process a binary operation node n."""
        op = n.children[0]
        lexpr = n.children[1]
        rexpr = n.children[2]
        if op == '+':
            return self._process_node(lexpr) + self._process_node(rexpr)
        elif op == '-':
            return self._process_node(lexpr) - self._process_node(rexpr)
        elif op == '*':
            return self._process_node(lexpr) * self._process_node(rexpr)
        elif op == '/':
            return self._process_node(lexpr) / self._process_node(rexpr)
        elif op == '^':
            return self._process_node(lexpr) ** self._process_node(rexpr)
        else:
            raise UnrollerException("internal error: undefined binop",
                                    "line=%s" % n.line, "file=%s" % n.file)

    def _process_prefix(self, n):
        """Process a prefix node n."""
        op = n.children[0]
        expr = n.children[1]
        if op == '+':
            return self._process_node(expr)
        elif op == '-':
            return -self._process_node(expr)
        else:
            raise UnrollerException("internal error: undefined prefix",
                                    "line=%s" % n.line, "file=%s" % n.file)

    def _process_measure(self, n):
        """Process a measurement node n."""
        id0 = self._process_bit_id(n.children[0])
        id1 = self._process_bit_id(n.children[1])
        if len(id0) != len(id1):
            raise UnrollerException("internal error: reg size mismatch",
                                    "line=%s" % n.line, "file=%s" % n.file)
        for idx in range(len(id0)):
            self.be.measure(id0[idx], id1[idx])

    def _process_if(self, n):
        """Process an if node n."""
        creg = n.children[0].name
        cval = n.children[1]
        self.be.set_condition(creg, cval)
        self._process_node(n.children[2])
        self.be.drop_condition()

    def _process_external(self, n):
        """Process an external function node n."""
        op = n.children[0].name
        expr = n.children[1]
        dispatch = {
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'exp': math.exp,
            'ln': math.ln,
            'sqrt': math.sqrt
        }
        if op in dispatch:
            return dispatch[op](self._process_node(expr))
        else:
            raise UnrollerException("internal error: undefined external",
                                    "line=%s" % n.line, "file=%s" % n.file)

    def _process_children(self, n):
        """Call process_node for all children of node n."""
        for c in n.children:
            self._process_node(c)

    def _process_node(self, n):
        """Carry out the action associated with node n."""
        if n.type == "program":
            self._process_children(n)

        elif n.type == "qreg":
            self.qregs[n.name] = int(n.index)
            self.be.new_qreg(n.name, int(n.index))

        elif n.type == "creg":
            self.cregs[n.name] = int(n.index)
            self.be.new_creg(n.name, int(n.index))

        elif n.type == "id":
            return self._process_local_id(n)

        elif n.type == "int":
            # We process int nodes when they are leaves of expressions
            # and cast them to float to avoid, for example, 3/2 = 1.
            return float(n.value)

        elif n.type == "real":
            return float(n.value)

        elif n.type == "indexed_id":
            # We should not get here.
            raise UnrollerException("internal error n.type == indexed_id:",
                                    "line=%s" % n.line,
                                    "file=%s" % n.file)

        elif n.type == "id_list":
            # We process id_list nodes when they are leaves of barriers.
            return [self._process_bit_id(m) for m in n.children]

        elif n.type == "primary_list":
            # We should only be called for a barrier.
            return [self._process_bit_id(m) for m in n.children]

        elif n.type == "gate":
            self._process_gate(n)

        elif n.type == "custom_unitary":
            self._process_custom_unitary(n)

        elif n.type == "universal_unitary":
            args = tuple(self._process_node(n.children[0]))
            qid = self._process_bit_id(n.children[1])
            for idx in range(len(qid)):
                self.be.u(args, qid[idx])

        elif n.type == "cnot":
            self._process_cnot(n)

        elif n.type == "expression_list":
            return [self._process_node(m) for m in n.children]

        elif n.type == "binop":
            self._process_binop(n)

        elif n.type == "prefix":
            self._process_prefix(n)

        elif n.type == "measure":
            self._process_measure(n)

        elif n.type == "magic":
            self.version = float(n.children[0])
            self.be.version(n.children[0])

        elif n.type == "barrier":
            ids = self._process_node(n.children[0])
            self.be.barrier(ids)

        elif n.type == "reset":
            id0 = self._process_bit_id(n.children[0])
            for idx in range(len(id0)):
                self.be.reset(id0[idx])

        elif n.type == "if":
            self._process_if(n)

        elif n.type == "opaque":
            self._process_gate(n, opaque=True)

        elif n.type == "external":
            self._process_external(n)

        else:
            raise UnrollerException("internal error: undefined node type",
                                    n.type, "line=%s" % n.line,
                                    "file=%s" % n.file)

    def set_backend(self, be):
        """Set the backend object."""
        self.be = be

    def execute(self):
        """Interpret OPENQASM and make appropriate backend calls."""
        if self.be is not None:
            self._process_node(self.ast)
        else:
            raise UnrollerException("backend not attached")
