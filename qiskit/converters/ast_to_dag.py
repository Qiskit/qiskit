# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
AST (abstract syntax tree) to DAG (directed acyclic graph) converter.

Acts as an OpenQASM interpreter.
"""
from collections import OrderedDict
from qiskit.dagcircuit import DAGCircuit
from qiskit.exceptions import QiskitError

from qiskit.circuit import QuantumRegister, ClassicalRegister, Gate, QuantumCircuit
from qiskit.qasm.node.real import Real
from qiskit.circuit.measure import Measure
from qiskit.circuit.reset import Reset
from qiskit.circuit.barrier import Barrier
from qiskit.circuit.delay import Delay
from qiskit.circuit.library import standard_gates as std


def ast_to_dag(ast):
    """Build a ``DAGCircuit`` object from an AST ``Node`` object.

    Args:
        ast (Program): a Program Node of an AST (parser's output)

    Return:
        DAGCircuit: the DAG representing an OpenQASM's AST

    Raises:
        QiskitError: if the AST is malformed.

    Example:
        .. code-block::

            from qiskit.converters import ast_to_dag
            from qiskit import qasm, QuantumCircuit, ClassicalRegister, QuantumRegister

            q = QuantumRegister(3, 'q')
            c = ClassicalRegister(3, 'c')
            circ = QuantumCircuit(q, c)
            circ.h(q[0])
            circ.cx(q[0], q[1])
            circ.measure(q[0], c[0])
            circ.rz(0.5, q[1]).c_if(c, 2)
            qasm_str = circ.qasm()
            ast = qasm.Qasm(data=qasm_str).parse()
            dag = ast_to_dag(ast)
    """
    dag = DAGCircuit()
    AstInterpreter(dag)._process_node(ast)

    return dag


class AstInterpreter:
    """Interprets an OpenQASM by expanding subroutines and unrolling loops."""

    standard_extension = {
        "u1": std.U1Gate,
        "u2": std.U2Gate,
        "u3": std.U3Gate,
        "u": std.UGate,
        "p": std.PhaseGate,
        "x": std.XGate,
        "y": std.YGate,
        "z": std.ZGate,
        "t": std.TGate,
        "tdg": std.TdgGate,
        "s": std.SGate,
        "sdg": std.SdgGate,
        "sx": std.SXGate,
        "sxdg": std.SXdgGate,
        "swap": std.SwapGate,
        "rx": std.RXGate,
        "rxx": std.RXXGate,
        "ry": std.RYGate,
        "rz": std.RZGate,
        "rzz": std.RZZGate,
        "id": std.IGate,
        "h": std.HGate,
        "cx": std.CXGate,
        "cy": std.CYGate,
        "cz": std.CZGate,
        "ch": std.CHGate,
        "crx": std.CRXGate,
        "cry": std.CRYGate,
        "crz": std.CRZGate,
        "csx": std.CSXGate,
        "cu1": std.CU1Gate,
        "cp": std.CPhaseGate,
        "cu": std.CUGate,
        "cu3": std.CU3Gate,
        "ccx": std.CCXGate,
        "cswap": std.CSwapGate,
        "delay": Delay,
        "rccx": std.RCCXGate,
        "rc3x": std.RC3XGate,
        "c3x": std.C3XGate,
        "c3sqrtx": std.C3SXGate,
        "c4x": std.C4XGate,
    }

    def __init__(self, dag):
        """Initialize interpreter's data."""
        # DAG object to populate
        self.dag = dag
        # OPENQASM version number (ignored for now)
        self.version = 0.0
        # Dict of gates names and properties
        self.gates = OrderedDict()
        # Keeping track of conditional gates
        self.condition = None
        # List of dictionaries mapping local parameter ids to expression Nodes
        self.arg_stack = [{}]
        # List of dictionaries mapping local bit ids to global ids (name, idx)
        self.bit_stack = [{}]

    def _process_bit_id(self, node):
        """Process an Id or IndexedId node as a bit or register type.

        Return a list of tuples (Register,index).
        """
        reg = None

        if node.name in self.dag.qregs:
            reg = self.dag.qregs[node.name]
        elif node.name in self.dag.cregs:
            reg = self.dag.cregs[node.name]
        else:
            raise QiskitError(
                "expected qreg or creg name:", "line=%s" % node.line, "file=%s" % node.file
            )

        if node.type == "indexed_id":
            # An indexed bit or qubit
            return [reg[node.index]]
        elif node.type == "id":
            # A qubit or qreg or creg
            if not self.bit_stack[-1]:
                # Global scope
                return list(reg)
            else:
                # local scope
                if node.name in self.bit_stack[-1]:
                    return [self.bit_stack[-1][node.name]]
                raise QiskitError(
                    "expected local bit name:", "line=%s" % node.line, "file=%s" % node.file
                )
        return None

    def _process_custom_unitary(self, node):
        """Process a custom unitary node."""
        name = node.name
        if node.arguments is not None:
            args = self._process_node(node.arguments)
        else:
            args = []
        bits = [self._process_bit_id(node_element) for node_element in node.bitlist.children]

        if name in self.gates:
            self._arguments(name, bits, args)
        else:
            raise QiskitError(
                "internal error undefined gate:", "line=%s" % node.line, "file=%s" % node.file
            )

    def _process_u(self, node):
        """Process a U gate node."""
        args = self._process_node(node.arguments)
        bits = [self._process_bit_id(node.bitlist)]

        self._arguments("u", bits, args)

    def _arguments(self, name, bits, args):
        gargs = self.gates[name]["args"]
        gbits = self.gates[name]["bits"]

        maxidx = max(map(len, bits))
        for idx in range(maxidx):
            self.arg_stack.append({gargs[j]: args[j] for j in range(len(gargs))})
            # Only index into register arguments.
            element = [idx * x for x in [len(bits[j]) > 1 for j in range(len(bits))]]
            self.bit_stack.append({gbits[j]: bits[j][element[j]] for j in range(len(gbits))})
            self._create_dag_op(
                name,
                [self.arg_stack[-1][s].sym() for s in gargs],
                [self.bit_stack[-1][s] for s in gbits],
            )
            self.arg_stack.pop()
            self.bit_stack.pop()

    def _process_gate(self, node, opaque=False):
        """Process a gate node.

        If opaque is True, process the node as an opaque gate node.
        """
        self.gates[node.name] = {}
        de_gate = self.gates[node.name]
        de_gate["print"] = True  # default
        de_gate["opaque"] = opaque
        de_gate["n_args"] = node.n_args()
        de_gate["n_bits"] = node.n_bits()
        if node.n_args() > 0:
            de_gate["args"] = [element.name for element in node.arguments.children]
        else:
            de_gate["args"] = []
        de_gate["bits"] = [c.name for c in node.bitlist.children]
        if node.name in self.standard_extension:
            return
        if opaque:
            de_gate["body"] = None
        else:
            de_gate["body"] = node.body

    def _process_cnot(self, node):
        """Process a CNOT gate node."""
        id0 = self._process_bit_id(node.children[0])
        id1 = self._process_bit_id(node.children[1])
        if not (len(id0) == len(id1) or len(id0) == 1 or len(id1) == 1):
            raise QiskitError(
                "internal error: qreg size mismatch", "line=%s" % node.line, "file=%s" % node.file
            )
        maxidx = max([len(id0), len(id1)])
        for idx in range(maxidx):
            cx_gate = std.CXGate()
            cx_gate.condition = self.condition
            if len(id0) > 1 and len(id1) > 1:
                self.dag.apply_operation_back(cx_gate, [id0[idx], id1[idx]], [])
            elif len(id0) > 1:
                self.dag.apply_operation_back(cx_gate, [id0[idx], id1[0]], [])
            else:
                self.dag.apply_operation_back(cx_gate, [id0[0], id1[idx]], [])

    def _process_measure(self, node):
        """Process a measurement node."""
        id0 = self._process_bit_id(node.children[0])
        id1 = self._process_bit_id(node.children[1])
        if len(id0) != len(id1):
            raise QiskitError(
                "internal error: reg size mismatch", "line=%s" % node.line, "file=%s" % node.file
            )
        for idx, idy in zip(id0, id1):
            meas_gate = Measure()
            meas_gate.condition = self.condition
            self.dag.apply_operation_back(meas_gate, [idx], [idy])

    def _process_if(self, node):
        """Process an if node."""
        creg_name = node.children[0].name
        creg = self.dag.cregs[creg_name]
        cval = node.children[1].value
        self.condition = (creg, cval)
        self._process_node(node.children[2])
        self.condition = None

    def _process_children(self, node):
        """Call process_node for all children of node."""
        for kid in node.children:
            self._process_node(kid)

    def _process_node(self, node):
        """Carry out the action associated with a node."""
        if node.type == "program":
            self._process_children(node)

        elif node.type == "qreg":
            qreg = QuantumRegister(node.index, node.name)
            self.dag.add_qreg(qreg)

        elif node.type == "creg":
            creg = ClassicalRegister(node.index, node.name)
            self.dag.add_creg(creg)

        elif node.type == "id":
            raise QiskitError("internal error: _process_node on id")

        elif node.type == "int":
            raise QiskitError("internal error: _process_node on int")

        elif node.type == "real":
            raise QiskitError("internal error: _process_node on real")

        elif node.type == "indexed_id":
            raise QiskitError("internal error: _process_node on indexed_id")

        elif node.type == "id_list":
            # We process id_list nodes when they are leaves of barriers.
            return [self._process_bit_id(node_children) for node_children in node.children]

        elif node.type == "primary_list":
            # We should only be called for a barrier.
            return [self._process_bit_id(m) for m in node.children]

        elif node.type == "gate":
            self._process_gate(node)

        elif node.type == "custom_unitary":
            self._process_custom_unitary(node)

        elif node.type == "universal_unitary":
            self._process_u(node)

        elif node.type == "cnot":
            self._process_cnot(node)

        elif node.type == "expression_list":
            return node.children

        elif node.type == "binop":
            raise QiskitError("internal error: _process_node on binop")

        elif node.type == "prefix":
            raise QiskitError("internal error: _process_node on prefix")

        elif node.type == "measure":
            self._process_measure(node)

        elif node.type == "format":
            self.version = node.version()

        elif node.type == "barrier":
            ids = self._process_node(node.children[0])
            qubits = []
            for qubit in ids:
                for j, _ in enumerate(qubit):
                    qubits.append(qubit[j])
            self.dag.apply_operation_back(Barrier(len(qubits)), qubits, [])

        elif node.type == "reset":
            id0 = self._process_bit_id(node.children[0])
            for i, _ in enumerate(id0):
                reset = Reset()
                reset.condition = self.condition
                self.dag.apply_operation_back(reset, [id0[i]], [])

        elif node.type == "if":
            self._process_if(node)

        elif node.type == "opaque":
            self._process_gate(node, opaque=True)

        elif node.type == "external":
            raise QiskitError("internal error: _process_node on external")

        else:
            raise QiskitError(
                "internal error: undefined node type",
                node.type,
                "line=%s" % node.line,
                "file=%s" % node.file,
            )
        return None

    def _gate_rules_to_qiskit_circuit(self, node, params):
        """From a gate definition in qasm, to a QuantumCircuit format."""
        rules = []
        qreg = QuantumRegister(node["n_bits"])
        bit_args = {node["bits"][i]: q for i, q in enumerate(qreg)}
        exp_args = {node["args"][i]: Real(q) for i, q in enumerate(params)}

        for child_op in node["body"].children:
            qparams = []
            eparams = []
            for param_list in child_op.children[1:]:
                if param_list.type == "id_list":
                    qparams = [bit_args[param.name] for param in param_list.children]
                elif param_list.type == "expression_list":
                    for param in param_list.children:
                        eparams.append(param.sym(nested_scope=[exp_args]))
            op = self._create_op(child_op.name, params=eparams)
            rules.append((op, qparams, []))
        circ = QuantumCircuit(qreg)
        for instr, qargs, cargs in rules:
            circ._append(instr, qargs, cargs)
        return circ

    def _create_dag_op(self, name, params, qargs):
        """
        Create a DAG node out of a parsed AST op node.

        Args:
            name (str): operation name to apply to the DAG
            params (list): op parameters
            qargs (list(Qubit)): qubits to attach to

        Raises:
            QiskitError: if encountering a non-basis opaque gate
        """
        op = self._create_op(name, params)
        op.condition = self.condition
        self.dag.apply_operation_back(op, qargs, [])

    def _create_op(self, name, params):
        if name in self.standard_extension:
            op = self.standard_extension[name](*params)
        elif name in self.gates:
            op = Gate(name=name, num_qubits=self.gates[name]["n_bits"], params=params)
            if not self.gates[name]["opaque"]:
                # call a custom gate (otherwise, opaque)
                op.definition = self._gate_rules_to_qiskit_circuit(self.gates[name], params=params)
        else:
            raise QiskitError("unknown operation for ast node name %s" % name)
        return op
