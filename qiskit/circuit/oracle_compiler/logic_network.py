# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
import ast
import _ast
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import ZGate, TGate, SGate, TdgGate, SdgGate, U1Gate, \
    XGate, HGate, U3Gate
import tweedledum


class ParseError(Exception):
    pass


class LogicNetwork(ast.NodeVisitor):
    bitops = {_ast.BitAnd: 'create_and',
              _ast.BitOr: 'create_or',
              _ast.BitXor: 'create_xor',
              _ast.And: 'create_and',
              _ast.Or: 'create_or',
              _ast.Not: 'create_not'
              }

    def __init__(self, source):
        self.source = source
        node = ast.parse(source)
        self.scopes = []
        self.args = []
        self._network = None
        self.visit(node)
        super().__init__()

    @staticmethod
    def tweedledum2qiskit(tweedledum_circuit, qregs=None):
        """
        {'num_qubits': 2, 'gates': [{'gate': 'X',
        'qubits': [1], 'control_qubits': [0], 'control_state': '1'}]}"""
        gates = {'z': ZGate, 't': TGate, 's': SGate, 'tdg': TdgGate, 'sdg': SdgGate, 'u1': U1Gate,
                 'x': XGate, 'h': HGate, 'u3': U3Gate}
        if qregs:
            circuit = QuantumCircuit(*qregs)
        else:
            circuit = QuantumCircuit(tweedledum_circuit['num_qubits'])
        for gate in tweedledum_circuit['gates']:
            basegate = gates.get(gate['gate'].lower())
            if basegate is None:
                raise Exception('TODO')

            ctrl_qubits = gate.get('control_qubits', [])
            trgt_qubits = gate.get('qubits', [])

            if ctrl_qubits:
                gate = basegate().control(len(ctrl_qubits), ctrl_state=gate.get('control_state'))
            else:
                gate = basegate()
            circuit.append(gate, ctrl_qubits + trgt_qubits)
        return circuit

    @property
    def types(self):
        """returns vars ands types in the scope"""
        ret = []
        for scope in self.scopes:
            ret.append({k: v[0] for k, v in scope.items()})
        return ret

    def simulate(self):
        return tweedledum.simulate(self._network)

    def synth(self, arg_regs=False) -> QuantumCircuit:
        if arg_regs:
            qregs = [QuantumRegister(1, name=arg) for arg in self.args
                     if self.types[0][arg] == 'Bit']
            qregs.reverse()
            if self.types[0]['return'] == 'Bit':
                qregs.append(QuantumRegister(1, name='return'))
        else:
            qregs = None
        return LogicNetwork.tweedledum2qiskit(tweedledum.synthesize_xag(self._network), qregs=qregs)

    def visit_Module(self, node):
        if len(node.body) != 1 and not isinstance(node.body[0], ast.FunctionDef):
            raise ParseError("just functions, sorry!")
        self.visit(node.body[0])

    def visit_FunctionDef(self, node):
        if node.returns is None:
            raise ParseError("return type is needed")
        self.scopes.append({'return': (node.returns.id, None),
                            node.returns.id: ('type', None)})
        self._network = tweedledum.xag_network()
        self.extend_scope(node.args)
        return super().generic_visit(node)

    def visit_Return(self, node):
        _type, signal = self.visit(node.value)
        if _type != self.scopes[-1]['return'][0]:
            raise ParseError("return type error")
        self._network.create_po(signal)
        return

    def visit_Assign(self, node):
        type_value, signal_value = self.visit(node.value)
        for target in node.targets:
            self.scopes[-1][target.id] = (type_value, signal_value)
        return (type_value, signal_value)

    def bit_binop(self, op, values):
        bitop = LogicNetwork.bitops.get(type(op))
        if not bitop:
            raise ParseError("Unknown binop.op %s" % op)
        binop = getattr(self._network, bitop)

        left_type, left_signal = values[0]
        if left_type != 'Bit':
            raise ParseError("binop type error")

        for right_type, right_signal in values[1:]:
            if right_type != 'Bit':
                raise ParseError("binop type error")
            left_signal = binop(left_signal, right_signal)

        return 'Bit', left_signal

    def visit_BoolOp(self, node):
        """ node.left=Bit and node.right=Bit return Bit """
        return self.bit_binop(node.op, [self.visit(value) for value in node.values])

    def visit_BinOp(self, node):
        """ node.left=Bit and node.right=Bit return Bit """
        return self.bit_binop(node.op, [self.visit(node.left), self.visit(node.right)])

    def visit_UnaryOp(self, node):
        operand_type, operand_signal = self.visit(node.operand)
        if operand_type == 'Bit':
            bitop = LogicNetwork.bitops.get(type(node.op))
            if bitop:
                return 'Bit', getattr(self._network, bitop)(operand_signal)
            else:
                raise TypeError("UntaryOp.op %s does not operate with Bit type " % node.op)

    def visit_Name(self, node):
        if node.id not in self.scopes[-1]:
            raise ParseError('out of scope: %s' % node.id)
        return self.scopes[-1][node.id]

    def generic_visit(self, node):
        if isinstance(node, (_ast.arguments, _ast.arg, _ast.Load, _ast.BitAnd,
                             _ast.BitOr, _ast.BitXor, _ast.BoolOp, _ast.Or)):
            return super().generic_visit(node)
        raise ParseError("Unknown node: %s" % type(node))

    def extend_scope(self, args_node: _ast.arguments) -> None:
        for arg in args_node.args:
            if arg.annotation is None:
                raise ParseError("argument type is needed")
            self.args.append(arg.arg)
            self.scopes[-1][arg.annotation.id] = ('type', None)
            self.scopes[-1][arg.arg] = (arg.annotation.id, self._network.create_pi())
