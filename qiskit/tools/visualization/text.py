# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=

"""
A module for drawing circuits in ascii art or some other text representation
"""

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import transpile


class DrawElement():
    def __init__(self, instruction):
        params = ""
        if 'params' in instruction:
            if len(instruction['params']):
                params += "(%s)" % ','.join([str(i) for i in instruction['params']])
        self.label = "%s%s" % (instruction['name'], params)
        self.width = len(self.label)

    @property
    def length(self):
        return max(len(self.top),len(self.mid),len(self.bot))

    @property
    def label_size(self):
        return len(self.label)

class MeasureTo(DrawElement):
    def __init__(self, instruction):
        super().__init__(instruction)
        self.top = " ║ "
        self.mid = "═╩═"
        self.bot = "   "


class MeasureFrom(DrawElement):
    def __init__(self, instruction):
        super().__init__(instruction)
        self.top = "┌─┐"
        self.mid = "┤M├"
        self.bot = "└╥┘"


class DrawElementMultiBit(DrawElement):
    @property
    def top(self):
        return self._top % self._top_connector.center(self.width, self._top_border)

    @property
    def mid(self):
        return self._mid % self._mid_content.center(self.width)

    @property
    def bot(self):
        return self._bot % self._bot_connector.center(self.width, self._bot_border)


class MultiQubitGateTop(DrawElementMultiBit):
    def __init__(self, instruction):
        super().__init__(instruction)
        self.label = instruction['name']
        self._mid_content = "" # The label will be put by some other part of the box.
        self._top = "┌─%s─┐"
        self._mid = "┤ %s ├"
        self._bot = "│ %s │"
        self._top_connector = self._top_border = '─'
        self._bot_connector = self._bot_border = " "

class MultiQubitGateMid(DrawElementMultiBit):
    def __init__(self, instruction, input_length, order):
        self.label = instruction['name']
        self._top = "│ %s │"
        self._mid = "┤ %s ├"
        self._bot = "│ %s │"
        self._top_border = self._bot_border = ' '

        # TODO logic about centering vertically, using input_length and order
        # '*'.center((input_lenght*3)-1)
        self._top_connector = self._bot_connector = self._mid_content = ''
        # TODO for now, force it in every Mid in the middle
        self._mid_content = self.label

class MultiQubitGateBot(DrawElementMultiBit):
    def __init__(self, instruction, input_length):
        super().__init__(instruction)
        self.label = instruction['name']
        self._top = "│ %s │"
        self._mid = "┤ %s ├"
        self._bot = "└─%s─┘"
        self._top_border = " "
        self._bot_border = '─'

        self._mid_content = self._bot_connector = self._top_connector = ""
        if input_length <= 2:
            self._top_connector = self.label

class ConditionalTo(DrawElementMultiBit):
    def __init__(self, instruction):
        self.label = self._mid_content = instruction['name']
        self._top = "┌─%s─┐"
        self._mid = "┤ %s ├"
        self._bot = "└─%s─┘"
        self._bot_border = self._top_connector = self._top_border = '─'
        self._bot_connector = '┬'


class ConditionalFrom(DrawElementMultiBit):
    def __init__(self, instruction):
        self.label = self._mid_content = "%s %s" % ('=', instruction['conditional']['val'])
        self._top = "┌─%s─┐"
        self._mid = "╡ %s ╞"
        self._bot = "└─%s─┘"
        self._top_connector = '┴'
        self._top_border = self._bot_connector = self._bot_border = '─'


class Barrier(DrawElement):
    def __init__(self, instruction):
        super().__init__(instruction)
        self.top = " ¦ "
        self.mid = "─¦─"
        self.bot = " ¦ "


class CXcontrol(DrawElement):
    def __init__(self, instruction):
        super().__init__(instruction)
        self.top = "   "
        self.mid = "─■─"
        self.bot = " │ "


class CXtarget(DrawElement):
    def __init__(self, instruction):
        super().__init__(instruction)
        self.top = " │ "
        self.mid = "(+)"
        self.bot = "   "


class UnitaryGate(DrawElement):
    def __init__(self, instruction):
        super().__init__(instruction)
        label_size = len(self.label)
        self.top = "┌─%s─┐" % ('─' * label_size)
        self.mid = "┤ %s ├" % self.label
        self.bot = "└─%s─┘" % ('─' * label_size)


class EmptyWire(DrawElement):
    def __init__(self, length):
        self._length = length
        self.top = " " * length
        self.bot = " " * length

    @staticmethod
    def fillup_layer(layer, first_clbit):
        max_length = max([i.length for i in filter(lambda x: x is not None, layer)])
        for nones in [i for i, x in enumerate(layer) if x == None]:
            layer[nones] = EmptyClbitWire(max_length) if nones >= first_clbit else EmptyQubitWire(
                max_length)
        return layer


class EmptyQubitWire(EmptyWire):
    def __init__(self, length):
        super().__init__(length)
        self.mid = '─' * length


class EmptyClbitWire(EmptyWire):
    def __init__(self, length):
        super().__init__(length)
        self.mid = '═' * length


class InputWire(EmptyWire):
    def __init__(self, label):
        self.label = label
        self.top = " " * len(self.label)
        self.mid = label
        self.bot = " " * len(self.label)


class TextDrawing():
    def __init__(self, json_circuit, encoding='cp437'):
        self.json_circuit = json_circuit
        self.encoding = encoding

    def lines(self):
        return TextDrawing.drawWires(self.build_wires())

    def wire_names(self):
        ret = []
        header = self.json_circuit['header']
        for qubit in header['qubit_labels']:
            ret.append("%s%s: |0>" % (qubit[0], qubit[1]))
        for qubit in header['clbit_labels']:
            ret.append("%s%s: 0 " % (qubit[0], qubit[1]))
        return ret

    @staticmethod
    def drawWires(wires):
        lines = []
        bot_line = None
        for wire in wires:
            # TOP
            top_line = ""
            for instruction in wire:
                top_line += instruction.top

            if bot_line is None:
                lines.append(top_line)
            else:
                lines.append(TextDrawing.merge_lines(lines.pop(), top_line))

            # MID
            mid_line = ""
            for instruction in wire:
                mid_line += instruction.mid

            lines.append(TextDrawing.merge_lines(lines[-1], mid_line, icod="bot"))

            # BOT
            bot_line = ""
            for instruction in wire:
                bot_line += instruction.bot
            lines.append(TextDrawing.merge_lines(lines[-1], bot_line, icod="bot"))

        return lines

    @staticmethod
    def merge_lines(top, bot, icod="top"):
        """

        Args:
            top:
            bot:
            icod: in case of doubts
        Returns:
        """
        ret = ""
        for topc, botc in zip(top, bot):
            if topc == botc:
                ret += topc
            elif topc in '┼╪':
                ret += "│"
            elif topc == " ":
                ret += botc
            elif topc in '┬╥' and botc == " ":
                ret += topc
            elif topc in '┬│' and botc == "═":
                ret += '╪'
            elif topc in '┬│' and botc == "─":
                ret += '┼'
            elif topc in '└┘║│¦' and botc == " ":
                ret += topc
            elif topc in '─═' and botc == " " and icod == "top":
                ret += topc
            elif topc in '─═' and botc == " " and icod == "bot":
                ret += botc
            elif topc == "║" and botc in "═":
                ret += "╬"
            elif topc in "║╥" and botc in "─":
                ret += "╫"
            elif topc in '╫╬' and botc in " ":
                ret += "║"
            else:
                ret += botc
        return ret

    def build_wires(self):
        layers = []
        noqubits = self.json_circuit['header']['number_of_qubits']
        noclbits = self.json_circuit['header']['number_of_clbits']

        names = self.wire_names()
        longest = len(max(names))
        inputs_wires = []
        for name in names:
            inputs_wires.append(InputWire(name.rjust(longest)))
        layers.append(inputs_wires)

        for no, instruction in enumerate(self.json_circuit['instructions']):
            qubit_layer = [None] * noqubits
            clbit_layer = [None] * noclbits

            if instruction['name'] == 'measure':
                qubit_layer[instruction['qubits'][0]] = MeasureFrom(instruction)
                clbit_layer[instruction['clbits'][0]] = MeasureTo(instruction)

            elif instruction['name'] == 'barrier':
                for qubit in instruction['qubits']:
                    qubit_layer[qubit] = Barrier(instruction)

            elif 'conditional' in instruction:
                # conditional
                mask = int(instruction['conditional']['mask'], 16)

                clbit_layer[mask] = ConditionalFrom(instruction)  # TODO
                qubit_layer[instruction['qubits'][0]] = ConditionalTo(instruction)

                longest = max(clbit_layer[mask].label_size,
                              qubit_layer[instruction['qubits'][0]].label_size)
                clbit_layer[mask].width=longest
                qubit_layer[instruction['qubits'][0]].width=longest

            elif instruction['name'] in ['cx', 'CX']:
                # cx
                control = instruction['qubits'][0]
                target = instruction['qubits'][1]

                qubit_layer[control] = CXcontrol(instruction)
                qubit_layer[target] = CXtarget(instruction)

            elif len(instruction['qubits']) == 1 and 'clbits' not in instruction:
                # unitary gate
                qubit_layer[instruction['qubits'][0]] = UnitaryGate(instruction)

            elif len(instruction['qubits']) >= 2 and 'clbits' not in instruction:
                # multiple qubit gate
                qubits = sorted(instruction['qubits'])

                # Checks if qubits are consecutive
                if qubits != [i for i in range(qubits[0], qubits[-1] + 1)]:
                    raise Exception(
                        "I don't know how to build a gate with multiple qubits when they are not adjacent to each other",
                        instruction)

                qubit_layer[qubits[0]] = MultiQubitGateTop(instruction)
                for order,qubit in enumerate(qubits[1:-1],1):
                    qubit_layer[qubit] = MultiQubitGateMid(instruction, len(qubits), order)
                qubit_layer[qubits[-1]] = MultiQubitGateBot(instruction, len(qubits))

                # Adjust width
                affected_part_of_the_layer = qubit_layer[qubits[0]:qubits[-1]+1]
                longest = max([qubit.label_size for qubit in affected_part_of_the_layer])
                for qubit in affected_part_of_the_layer:
                    qubit.width=longest

            else:
                raise Exception("I don't know how to handle this instruction", instruction)

            layers.append(qubit_layer + clbit_layer)

        # TODO compress (layers)

        # Replace the Nones with EmptyWire
        for layerno, layer in enumerate(layers):
            layers[layerno] = EmptyWire.fillup_layer(layer, noqubits)

        return [i for i in zip(*layers)]


def text_draw(circuit, filename=None,
              basis="id,u0,u1,u2,u3,x,y,z,h,s,sdg,t,tdg,rx,ry,rz,"
                    "cx,cy,cz,ch,crz,cu1,cu3,swap,ccx,cswap", line_length=80, encoding="cp437"):
    dag_circuit = DAGCircuit.fromQuantumCircuit(circuit, expand_gates=False)
    json_circuit = transpile(dag_circuit, basis_gates=basis, format='json')

    return TextDrawing(json_circuit, encoding).lines()
