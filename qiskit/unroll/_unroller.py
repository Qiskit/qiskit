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
OPENQASM interpreter.

Author: Andrew Cross
"""
import math
import copy
from ._unrollerexception import UnrollerException


class Unroller(object):
    """OPENQASM interpreter object that unrolls subroutines and loops."""

    def __init__(self, ast, backend=None):
        """Initialize interpreter's data."""
        # Abstract syntax tree from parser
        self.ast = ast
        # Backend object
        self.backend = backend
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

    def _process_bit_id(self, node):
        """Process an Id or IndexedId node as a bit or register type.

        Return a list of tuples (name,index).
        """
        if node.type == "indexed_id":
            # An indexed bit or qubit
            return [(node.name, node.index)]
        elif node.type == "id":
            # A qubit or qreg or creg
            if len(self.bit_stack[-1]) == 0:
                # Global scope
                if node.name in self.qregs:
                    return [(node.name, j)
                            for j in range(self.qregs[node.name])]
                elif node.name in self.cregs:
                    return [(node.name, j)
                            for j in range(self.cregs[node.name])]
                else:
                    raise UnrollerException("expected qreg or creg name:",
                                            "line=%s" % node.line,
                                            "file=%s" % node.file)
            else:
                # local scope
                if node.name in self.bit_stack[-1]:
                    return [self.bit_stack[-1][node.name]]
                else:
                    raise UnrollerException("excepted local bit name:",
                                            "line=%s" % node.line,
                                            "file=%s" % node.file)

    def _process_local_id(self, node):
        """Process an Id node as a local id."""
        # The id must be in arg_stack i.e. the id is inside a gate_body
        id_dict = self.arg_stack[-1]
        if node.name in id_dict:
            return float(id_dict[node.name])
        else:
            raise UnrollerException("expected local parameter name:",
                                    "line=%s" % node.line,
                                    "file=%s" % node.file)

    def _process_custom_unitary(self, node):
        """Process a custom unitary node."""
        name = node.name
        if node.arguments is not None:
            args = self._process_node(node.arguments)
        else:
            args = []
        bits = [self._process_bit_id(node_element)
                for node_element in node.bitlist.children]
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
                element = list(map(lambda x: idx * x,
                                   [len(bits[j]) > 1 for j in range(len(bits))]))
                self.bit_stack.append({gbits[j]: bits[j][element[j]]
                                       for j in range(len(gbits))})
                self.backend.start_gate(name,
                                        [self.arg_stack[-1][s] for s in gargs],
                                        [self.bit_stack[-1][s] for s in gbits])
                if not self.gates[name]["opaque"]:
                    self._process_children(gbody)
                self.backend.end_gate(name,
                                      [self.arg_stack[-1][s] for s in gargs],
                                      [self.bit_stack[-1][s] for s in gbits])
                self.arg_stack.pop()
                self.bit_stack.pop()
        else:
            raise UnrollerException("internal error undefined gate:",
                                    "line=%s" % node.line, "file=%s" % node.file)

    def _process_gate(self, node, opaque=False):
        """Process a gate node.

        If opaque is True, process the node as an opaque gate node.
        """
        self.gates[node.name] = {}
        de = self.gates[node.name]
        de["opaque"] = opaque
        de["n_args"] = node.n_args()
        de["n_bits"] = node.n_bits()
        if node.n_args() > 0:
            de["args"] = [element.name for element in node.arguments.children]
        else:
            de["args"] = []
        de["bits"] = [c.name for c in node.bitlist.children]
        if opaque:
            de["body"] = None
        else:
            de["body"] = node.body
        self.backend.define_gate(node.name, copy.deepcopy(de))

    def _process_cnot(self, node):
        """Process a CNOT gate node."""
        id0 = self._process_bit_id(node.children[0])
        id1 = self._process_bit_id(node.children[1])
        if not(len(id0) == len(id1) or len(id0) == 1 or len(id1) == 1):
            raise UnrollerException("internal error: qreg size mismatch",
                                    "line=%s" % node.line, "file=%s" % node.file)
        maxidx = max([len(id0), len(id1)])
        for idx in range(maxidx):
            if len(id0) > 1 and len(id1) > 1:
                self.backend.cx(id0[idx], id1[idx])
            elif len(id0) > 1:
                self.backend.cx(id0[idx], id1[0])
            else:
                self.backend.cx(id0[0], id1[idx])

    def _process_binop(self, node):
        """Process a binary operation node."""
        operation = node.children[0]
        lexpr = node.children[1]
        rexpr = node.children[2]
        if operation == '+':
            return self._process_node(lexpr) + self._process_node(rexpr)
        elif operation == '-':
            return self._process_node(lexpr) - self._process_node(rexpr)
        elif operation == '*':
            return self._process_node(lexpr) * self._process_node(rexpr)
        elif operation == '/':
            return self._process_node(lexpr) / self._process_node(rexpr)
        elif operation == '^':
            return self._process_node(lexpr) ** self._process_node(rexpr)
        else:
            raise UnrollerException("internal error: undefined binop",
                                    "line=%s" % node.line, "file=%s" % node.file)

    def _process_prefix(self, node):
        """Process a prefix node."""
        operation = node.children[0]
        expr = node.children[1]
        if operation == '+':
            return self._process_node(expr)
        elif operation == '-':
            return -self._process_node(expr)
        else:
            raise UnrollerException("internal error: undefined prefix",
                                    "line=%s" % node.line, "file=%s" % node.file)

    def _process_measure(self, node):
        """Process a measurement node."""
        id0 = self._process_bit_id(node.children[0])
        id1 = self._process_bit_id(node.children[1])
        if len(id0) != len(id1):
            raise UnrollerException("internal error: reg size mismatch",
                                    "line=%s" % node.line, "file=%s" % node.file)
        for idx, idy in zip(id0, id1):
            self.backend.measure(idx, idy)

    def _process_if(self, node):
        """Process an if node."""
        creg = node.children[0].name
        cval = node.children[1]
        self.backend.set_condition(creg, cval)
        self._process_node(node.children[2])
        self.backend.drop_condition()

    def _process_external(self, n):
        """Process an external function node n."""
        op = n.children[0].name
        expr = n.children[1]
        dispatch = {
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'exp': math.exp,
            'ln': math.log,
            'sqrt': math.sqrt
        }
        if op in dispatch:
            return dispatch[op](self._process_node(expr))
        else:
            raise UnrollerException("internal error: undefined external",
                                    "line=%s" % n.line, "file=%s" % n.file)

    def _process_children(self, node):
        """Call process_node for all children of node."""
        for c in node.children:
            self._process_node(c)

    def _process_node(self, node):
        """Carry out the action associated with node n."""
        if node.type == "program":
            self._process_children(node)

        elif node.type == "qreg":
            self.qregs[node.name] = int(node.index)
            self.backend.new_qreg(node.name, int(node.index))

        elif node.type == "creg":
            self.cregs[node.name] = int(node.index)
            self.backend.new_creg(node.name, int(node.index))

        elif node.type == "id":
            return self._process_local_id(node)

        elif node.type == "int":
            # We process int nodes when they are leaves of expressions
            # and cast them to float to avoid, for example, 3/2 = 1.
            return float(node.value)

        elif node.type == "real":
            return float(node.value)

        elif node.type == "indexed_id":
            # We should not get here.
            raise UnrollerException("internal error n.type == indexed_id:",
                                    "line=%s" % node.line,
                                    "file=%s" % node.file)

        elif node.type == "id_list":
            # We process id_list nodes when they are leaves of barriers.
            return [self._process_bit_id(node_children)
                    for node_children in node.children]

        elif node.type == "primary_list":
            # We should only be called for a barrier.
            return [self._process_bit_id(m) for m in node.children]

        elif node.type == "gate":
            self._process_gate(node)

        elif node.type == "custom_unitary":
            self._process_custom_unitary(node)

        elif node.type == "universal_unitary":
            args = tuple(self._process_node(node.children[0]))
            qid = self._process_bit_id(node.children[1])
            for element in qid:
                self.backend.u(args, element)

        elif node.type == "cnot":
            self._process_cnot(node)

        elif node.type == "expression_list":
            return [self._process_node(node_children)
                    for node_children in node.children]

        elif node.type == "binop":
            return self._process_binop(node)

        elif node.type == "prefix":
            return self._process_prefix(node)

        elif node.type == "measure":
            self._process_measure(node)

        elif node.type == "magic":
            self.version = float(node.children[0])
            self.backend.version(node.children[0])

        elif node.type == "barrier":
            ids = self._process_node(node.children[0])
            self.backend.barrier(ids)

        elif node.type == "reset":
            id0 = self._process_bit_id(node.children[0])
            for idx in range(len(id0)):
                self.backend.reset(id0[idx])

        elif node.type == "if":
            self._process_if(node)

        elif node.type == "opaque":
            self._process_gate(node, opaque=True)

        elif node.type == "external":
            return self._process_external(node)

        else:
            raise UnrollerException("internal error: undefined node type",
                                    node.type, "line=%s" % node.line,
                                    "file=%s" % node.file)

    def set_backend(self, backend):
        """Set the backend object."""
        self.backend = backend

    def execute(self):
        """Interpret OPENQASM and make appropriate backend calls."""
        if self.backend is not None:
            self._process_node(self.ast)
        else:
            raise UnrollerException("backend not attached")
