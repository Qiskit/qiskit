# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
A module for drawing circuits in ascii art or some other text representation
"""

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import transpile


class DrawElement():
    """ An element is an instruction or an operation that need to be drawn."""

    def __init__(self, instruction=None):
        if instruction:
            params = ""
            if 'params' in instruction:
                if instruction['params']:
                    params += "(%s)" % ','.join(['%.5g' % i for i in instruction['params']])
            self.label = "%s%s" % (instruction['name'].upper(), params)
        else:
            self.label = ""
        self._width = None
        self._top = self._mid = self._bot = ""
        self._top_connector = self._mid_content = self._bot_connector = " "
        self._top_border = self._bot_border = " "

    @property
    def top(self):
        """ Constructs the top line of the element"""
        if "%s" in self._top:
            return self._top % self._top_connector.center(self.width, self._top_border)
        return self._top

    @property
    def mid(self):
        """ Constructs the middle line of the element"""
        if "%s" in self._mid:
            return self._mid % self._mid_content.center(self.width)
        return self._mid

    @property
    def bot(self):
        """ Constructs the bottom line of the element"""
        if "%s" in self._bot:
            return self._bot % self._bot_connector.center(self.width, self._bot_border)
        return self._bot

    @property
    def length(self):
        """ Returns the length of the element, including the box around."""
        return max(len(self.top), len(self.mid), len(self.bot))

    @property
    def width(self):
        if self._width:
            return self._width
        return len(self.label)

    @width.setter
    def width(self, value):
        self._width = value


class MeasureTo(DrawElement):
    def __init__(self, instruction):
        super().__init__(instruction)
        self._top = " ║ "
        self._mid = "═╩═"
        self._bot = "   "


class MeasureFrom(DrawElement):
    def __init__(self, instruction):
        super().__init__(instruction)
        self._top = "┌─┐"
        self._mid = "┤M├"
        self._bot = "└╥┘"


class DrawElementMultiBit(DrawElement):

    def center_label(self, input_length, order):
        location_in_the_box = '*'.center(input_length * 2 - 1).index('*') + 1
        top_limit = (order - 1) * 2 + 2
        bot_limit = top_limit + 2
        if top_limit <= location_in_the_box < bot_limit:
            if location_in_the_box == top_limit:
                self._top_connector = self.label
            elif location_in_the_box == top_limit + 1:
                self._mid_content = self.label
            else:
                self._bot_connector = self.label


class MultiQubitGateTop(DrawElementMultiBit):
    def __init__(self, instruction):
        super().__init__(instruction)
        self._mid_content = ""  # The label will be put by some other part of the box.
        self._top = "┌─%s─┐"
        self._mid = "┤ %s ├"
        self._bot = "│ %s │"
        self._top_connector = self._top_border = '─'
        self._bot_connector = self._bot_border = " "


class MultiQubitGateMid(DrawElementMultiBit):
    def __init__(self, instruction, input_length, order):
        super().__init__(instruction)
        self._top = "│ %s │"
        self._mid = "┤ %s ├"
        self._bot = "│ %s │"
        self._top_border = self._bot_border = ' '
        self._top_connector = self._bot_connector = self._mid_content = ''
        self.center_label(input_length, order)


class MultiQubitGateBot(DrawElementMultiBit):
    def __init__(self, instruction, input_length):
        super().__init__(instruction)
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
        super().__init__(instruction)
        self._mid_content = self.label
        self._top = "┌─%s─┐"
        self._mid = "┤ %s ├"
        self._bot = "└─%s─┘"
        self._bot_border = self._top_connector = self._top_border = '─'
        self._bot_connector = '┬'


class ConditionalFrom(DrawElementMultiBit):
    def __init__(self, instruction):
        super().__init__(instruction)
        self.label = self._mid_content = "%s %s" % ('=', instruction['conditional']['val'])
        self._top = "┌─%s─┐"
        self._mid = "╡ %s ╞"
        self._bot = "└─%s─┘"
        self._top_connector = '┴'
        self._top_border = self._bot_connector = self._bot_border = '─'


class ConditionalFromTop(DrawElementMultiBit):
    def __init__(self, instruction):
        super().__init__(instruction)
        self._mid_content = ""  # The label will be put by some other part of the box.
        self._top = "┌─%s─┐"
        self._mid = "╡ %s ╞"
        self._bot = "│ %s │"
        self._top_connector = '┴'
        self._top_border = '─'
        self._bot_connector = self._bot_border = " "


class ConditionalFromMid(DrawElementMultiBit):
    def __init__(self, instruction, input_length, order):
        super().__init__(instruction)
        self.label = self._mid_content = "%s %s" % ('=', instruction['conditional']['val'])
        self._top = "│ %s │"
        self._mid = "╡ %s ╞"
        self._bot = "│ %s │"
        self._top_border = self._bot_border = ' '
        self._top_connector = self._bot_connector = self._mid_content = ''
        self.center_label(input_length, order)


class ConditionalFromBot(DrawElementMultiBit):
    def __init__(self, instruction, input_length):
        super().__init__(instruction)
        self.label = self._mid_content = "%s %s" % ('=', instruction['conditional']['val'])
        self._top = "│ %s │"
        self._mid = "╡ %s ╞"
        self._bot = "└─%s─┘"
        self._top_border = " "
        self._bot_border = '─'

        self._mid_content = self._bot_connector = self._top_connector = ""
        if input_length <= 2:
            self._top_connector = self.label


class Barrier(DrawElement):
    def __init__(self, instruction):
        super().__init__(instruction)
        self._top = " ¦ "
        self._mid = "─¦─"
        self._bot = " ¦ "


class SwapTop(DrawElement):
    def __init__(self, instruction):
        super().__init__(instruction)
        self._top = "   "
        self._mid = "─X─"
        self._bot = " │ "


class SwapBot(DrawElement):
    def __init__(self, instruction):
        super().__init__(instruction)
        self._top = " │ "
        self._mid = "─X─"
        self._bot = "   "


class Reset(DrawElement):
    def __init__(self, instruction):
        super().__init__(instruction)
        self._top = "     "
        self._mid = "─|0>─"
        self._bot = "     "


class CXcontrol(DrawElement):
    def __init__(self, instruction):
        super().__init__(instruction)
        self._top = "   "
        self._mid = "─■─"
        self._bot = " │ "


class CXtarget(DrawElement):
    def __init__(self, instruction):
        super().__init__(instruction)
        self._top = " │ "
        self._mid = "(+)"
        self._bot = "   "


class UnitaryGate(DrawElement):
    def __init__(self, instruction):
        super().__init__(instruction)
        label_size = len(self.label)
        self._top = "┌─%s─┐" % ('─' * label_size)
        self._mid = "┤ %s ├" % self.label
        self._bot = "└─%s─┘" % ('─' * label_size)


class EmptyWire(DrawElement):
    def __init__(self, length=0):
        super().__init__()
        self._length = length
        self._top = " " * length
        self._bot = " " * length

    @staticmethod
    def fillup_layer(layer, first_clbit):
        max_length = max([i.length for i in filter(lambda x: x is not None, layer)])
        for nones in [i for i, x in enumerate(layer) if x is None]:
            layer[nones] = EmptyClbitWire(max_length) if nones >= first_clbit else EmptyQubitWire(
                max_length)
        return layer


class BreakWire(DrawElement):
    def __init__(self, arrow_char):
        super().__init__()
        self._top = arrow_char
        self._mid = arrow_char
        self._bot = arrow_char

    @staticmethod
    def fillup_layer(layer, arrow_char):
        layer_length = len(layer)
        breakwire_layer = []
        for _ in range(layer_length):
            breakwire_layer.append(BreakWire(arrow_char))
        return breakwire_layer


class EmptyQubitWire(EmptyWire):
    def __init__(self, length):
        super().__init__(length)
        self._mid = '─' * length


class EmptyClbitWire(EmptyWire):
    def __init__(self, length):
        super().__init__(length)
        self._mid = '═' * length


class InputWire(EmptyWire):
    def __init__(self, label):
        super().__init__()
        self.label = label
        self._top = " " * len(self.label)
        self._mid = label
        self._bot = " " * len(self.label)

    @staticmethod
    def fillup_layer(names):  # pylint: disable=arguments-differ
        longest = max([len(name) for name in names])
        inputs_wires = []
        for name in names:
            inputs_wires.append(InputWire(name.rjust(longest)))
        return inputs_wires


class TextDrawing():
    def __init__(self, json_circuit):
        self.json_circuit = json_circuit

    def lines(self, line_length=None):

        noqubits = self.json_circuit['header']['number_of_qubits']
        layers = self.build_layers()

        # TODO compress layers
        # -| H |----------
        # --------| H |---
        # should be
        # -| H |---
        # -| H |---

        layer_groups = [[]]
        rest_of_the_line = line_length
        for layerno, layer in enumerate(layers):
            # Replace the Nones with EmptyWire
            layers[layerno] = EmptyWire.fillup_layer(layer, noqubits)

            if line_length is None:
                # does not page
                layer_groups[-1].append(layer)
                continue

            # chop the layer to the line_length (pager)
            layer_length = layers[layerno][0].length

            if layer_length < rest_of_the_line:
                layer_groups[-1].append(layer)
                rest_of_the_line -= layer_length
            else:
                layer_groups[-1].append(BreakWire.fillup_layer(layer, '»'))

                # New group
                layer_groups.append([BreakWire.fillup_layer(layer, '«')])
                rest_of_the_line = line_length - layer_groups[-1][-1][0].length

                layer_groups[-1].append(
                    InputWire.fillup_layer(self.wire_names(with_initial_value=False)))
                rest_of_the_line -= layer_groups[-1][-1][0].length

                layer_groups[-1].append(layer)
                rest_of_the_line -= layer_groups[-1][-1][0].length

        lines = []
        for layer_group in layer_groups:
            wires = [i for i in zip(*layer_group)]
            lines += TextDrawing.draw_wires(wires)

        return lines

    def wire_names(self, with_initial_value=True):
        ret = []

        if with_initial_value:
            initial_value = {'qubit': '|0>', 'clbit': '0 '}
        else:
            initial_value = {'qubit': '', 'clbit': ''}

        header = self.json_circuit['header']
        for qubit in header['qubit_labels']:
            ret.append("%s_%s: %s" % (qubit[0], qubit[1], initial_value['qubit']))
        for creg in header['clbit_labels']:
            for clbit in range(creg[1]):
                ret.append("%s_%s: %s" % (creg[0], clbit, initial_value['clbit']))
        return ret

    @staticmethod
    def draw_wires(wires):
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
        Merges two lines (top and bot) in the way that the overlapping make senses.
        Args:
            top (str): the top line
            bot (str): the bottom line
            icod (top or bot): in case of doubt, which line should have priority? Default: "top".
        Returns:
            str: The merge of both lines.
        """
        ret = ""
        for topc, botc in zip(top, bot):
            if topc == botc:
                ret += topc
            elif topc in '┼╪' and botc == " ":
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
            elif topc in "║╥" and botc in "═":
                ret += "╬"
            elif topc in "║╥" and botc in "─":
                ret += "╫"
            elif topc in '╫╬' and botc in " ":
                ret += "║"
            elif topc == '└' and botc == "┌":
                ret += "├"
            elif topc == '┘' and botc == "┐":
                ret += "┤"
            else:
                ret += botc
        return ret

    def clbit_index_from_mask(self, mask):
        clbit_len = self.json_circuit['header']['number_of_clbits']
        bit_mask = [bool(mask & (1 << n)) for n in range(clbit_len)]
        return [i for i, x in enumerate(bit_mask) if x]

    @staticmethod
    def normalize_width(layer):
        instructions = [instruction for instruction in filter(lambda x: x is not None, layer)]
        longest = max([instruction.width for instruction in instructions])
        for instruction in instructions:
            instruction.width = longest

    def build_layers(self):
        layers = []
        noqubits = self.json_circuit['header']['number_of_qubits']
        noclbits = self.json_circuit['header']['number_of_clbits']

        layers.append(InputWire.fillup_layer(self.wire_names(with_initial_value=True)))

        for instruction in self.json_circuit['instructions']:
            qubit_layer = [None] * noqubits
            clbit_layer = [None] * noclbits

            if instruction['name'] == 'measure':
                qubit_layer[instruction['qubits'][0]] = MeasureFrom(instruction)
                clbit_layer[instruction['clbits'][0]] = MeasureTo(instruction)

            elif instruction['name'] == 'barrier':
                # barrier
                for qubit in instruction['qubits']:
                    qubit_layer[qubit] = Barrier(instruction)

            elif instruction['name'] == 'swap':
                # swap
                qubit_layer[instruction['qubits'][0]] = SwapTop(instruction)
                qubit_layer[instruction['qubits'][1]] = SwapBot(instruction)

            elif instruction['name'] == 'cswap':
                # cswap
                qubit_layer[instruction['qubits'][0]] = CXcontrol(instruction)
                qubit_layer[instruction['qubits'][1]] = SwapBot(instruction)
                qubit_layer[instruction['qubits'][2]] = SwapBot(instruction)

            elif instruction['name'] == 'reset':
                qubit_layer[instruction['qubits'][0]] = Reset(instruction)

            elif 'conditional' in instruction:
                # conditional
                clbits = self.clbit_index_from_mask(int(instruction['conditional']['mask'], 16))
                if len(clbits) == 1:
                    clbit_layer[clbits[0]] = ConditionalFrom(instruction)
                else:
                    clbit_layer[clbits[0]] = ConditionalFromTop(instruction)
                    for order, clbit in enumerate(clbits[1:-1], 1):
                        clbit_layer[clbit] = ConditionalFromMid(instruction, len(clbits), order)
                    clbit_layer[clbits[-1]] = ConditionalFromBot(instruction, len(clbits))

                qubit_layer[instruction['qubits'][0]] = ConditionalTo(instruction)

                TextDrawing.normalize_width(clbit_layer + qubit_layer)

            elif instruction['name'] in ['cx', 'CX', 'ccx']:
                # cx/ccx
                control = [qubit for qubit in instruction['qubits'][:-1]]
                target = instruction['qubits'][-1]

                for qubit in control:
                    qubit_layer[qubit] = CXcontrol(instruction)
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
                        ("I don't know how to build a gate with multiple qubits when"
                         "they are not adjacent to each other"),
                        instruction)

                qubit_layer[qubits[0]] = MultiQubitGateTop(instruction)
                for order, qubit in enumerate(qubits[1:-1], 1):
                    qubit_layer[qubit] = MultiQubitGateMid(instruction, len(qubits), order)
                qubit_layer[qubits[-1]] = MultiQubitGateBot(instruction, len(qubits))

                TextDrawing.normalize_width(qubit_layer)

            else:
                raise Exception("I don't know how to handle this instruction", instruction)

            layers.append(qubit_layer + clbit_layer)

        return layers


def text_drawer(circuit, basis="id,u0,u1,u2,u3,x,y,z,h,s,sdg,t,tdg,rx,ry,rz,"
                               "cx,cy,cz,ch,crz,cu1,cu3,swap,ccx,cswap", line_length=None):
    dag_circuit = DAGCircuit.fromQuantumCircuit(circuit, expand_gates=False)
    json_circuit = transpile(dag_circuit, basis_gates=basis, format='json')

    return "\n".join(TextDrawing(json_circuit).lines(line_length))
