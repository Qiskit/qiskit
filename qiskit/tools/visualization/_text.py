# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
A module for drawing circuits in ascii art or some other text representation
"""

from itertools import groupby


class DrawElement():
    """ An element is an instruction or an operation that need to be drawn."""

    def __init__(self, label=None):
        self._width = None
        self.label = self.mid_content = label
        self.top_format = self.mid_format = self.bot_format = "%s"
        self.top_connect = self.bot_connect = " "
        self.top_pad = self._mid_padding = self.bot_pad = " "
        self.bot_connector = {}
        self.top_connector = {}
        self.right_fill = self.left_fill = 0

    @property
    def top(self):
        """ Constructs the top line of the element"""
        ret = self.top_format % self.top_connect.center(
            self.width - self.left_fill - self.right_fill, self.top_pad)
        if self.right_fill:
            ret = ret.ljust(self.right_fill, self.top_pad)
        if self.left_fill:
            ret = ret.rjust(self.left_fill, self.top_pad)
        return ret

    @property
    def mid(self):
        """ Constructs the middle line of the element"""
        ret = self.mid_format % self.mid_content.center(
            self.width - self.left_fill - self.right_fill, self._mid_padding)
        if self.right_fill:
            ret = ret.ljust(self.right_fill, self._mid_padding)
        if self.left_fill:
            ret = ret.rjust(self.left_fill, self._mid_padding)
        return ret

    @property
    def bot(self):
        """ Constructs the bottom line of the element"""
        ret = self.bot_format % self.bot_connect.center(
            self.width - self.left_fill - self.right_fill, self.bot_pad)
        if self.right_fill:
            ret = ret.ljust(self.right_fill, self.bot_pad)
        if self.left_fill:
            ret = ret.rjust(self.left_fill, self.bot_pad)
        return ret

    @property
    def length(self):
        """ Returns the length of the element, including the box around."""
        return max(len(self.top), len(self.mid), len(self.bot))

    @length.setter
    def length(self, value):
        """ Adjusts width so the length fits."""
        self.width = value - max(
            [len(getattr(self, i) % '') for i in ["bot_format", "mid_format", "top_format"]])

    @property
    def width(self):
        """ Returns the width of the label, including padding"""
        if self._width:
            return self._width
        return len(self.mid_content)

    @width.setter
    def width(self, value):
        self._width = value

    def connect(self, wire_char, where, label=None):
        """
        Connects boxes and elements using wire_char and setting proper connectors.
        Args:
            wire_char (char): For example '║' or '│'.
            where (list["top", "bot"]): Where the connector should be set.
            label (string): Some connectors have a label (see cu1, for example).
        """
        if 'top' in where and self.top_connector:
            self.top_connect = self.top_connector[wire_char]

        if 'bot' in where and self.bot_connector:
            self.bot_connect = self.bot_connector[wire_char]

        if label:
            self.top_format = self.top_format[:-1] + (label if label else "")


class BoxOnClWire(DrawElement):
    """ Draws a box on the classical wire
        top: ┌───┐ ┌───────┐
        mid: ╡ A ╞ ╡   A   ╞
        bot: └───┘ └───────┘
    """

    def __init__(self, label="", top_connect='─', bot_connect='─'):
        super().__init__(label)
        self.top_format = "┌─%s─┐"
        self.mid_format = "╡ %s ╞"
        self.bot_format = "└─%s─┘"
        self.top_pad = self.bot_pad = '─'
        self.top_connect = top_connect
        self.bot_connect = bot_connect
        self.mid_content = label


class BoxOnQuWire(DrawElement):
    """ Draws a box on the quantum wire
        top: ┌───┐ ┌───────┐
        mid: ┤ A ├ ┤   A   ├
        bot: └───┘ └───────┘
    """

    def __init__(self, label="", top_connect='─', bot_connect='─'):
        super().__init__(label)
        self.top_format = "┌─%s─┐"
        self.mid_format = "┤ %s ├"
        self.bot_format = "└─%s─┘"
        self.top_pad = self.bot_pad = '─'
        self.top_connect = top_connect
        self.bot_connect = bot_connect
        self.mid_content = label
        self.top_connector = {"│": '┴'}
        self.bot_connector = {"│": '┬'}


class MeasureTo(DrawElement):
    """ The element on the classic wire to which the measure is performed
    top:  ║     ║
    mid: ═╩═ ═══╩═══
    bot:
    """

    def __init__(self):
        super().__init__()
        self.top_connect = " ║ "
        self.mid_content = "═╩═"
        self.bot_connect = "   "
        self._mid_padding = "═"


class MeasureFrom(BoxOnQuWire):
    """ The element on the quantum wire in which the measure is performed
        top: ┌─┐    ┌─┐
        mid: ┤M├ ───┤M├───
        bot: └╥┘    └╥┘
    """

    def __init__(self):
        super().__init__()
        self.top_format = self.mid_format = self.bot_format = "%s"
        self.top_connect = "┌─┐"
        self.mid_content = "┤M├"
        self.bot_connect = "└╥┘"


class MultiBox(DrawElement):
    """Elements that is draw on over multiple wires."""

    def center_label(self, input_length, order):
        """
        In multi-bit elements, the label is centered vertically.
        Args:
            input_length (int): Rhe amount of wires affected.
            order (int): Which middle element is this one?
        """
        location_in_the_box = '*'.center(input_length * 2 - 1).index('*') + 1
        top_limit = (order - 1) * 2 + 2
        bot_limit = top_limit + 2
        if top_limit <= location_in_the_box < bot_limit:
            if location_in_the_box == top_limit:
                self.top_connect = self.label
            elif location_in_the_box == top_limit + 1:
                self.mid_content = self.label
            else:
                self.bot_connect = self.label


class BoxOnQuWireTop(MultiBox, BoxOnQuWire):
    """ Draws the top part of a box that affects more than one quantum wire"""

    def __init__(self, label="", top_connect=None):
        super().__init__(label)
        self.mid_content = ""  # The label will be put by some other part of the box.
        self.bot_format = "│ %s │"
        self.top_connect = top_connect if top_connect else '─'
        self.bot_connect = self.bot_pad = " "


class BoxOnQuWireMid(MultiBox, BoxOnQuWire):
    """ Draws the middle part of a box that affects more than one quantum wire"""

    def __init__(self, label, input_length, order):
        super().__init__(label)
        self.top_format = "│ %s │"
        self.bot_format = "│ %s │"
        self.center_label(input_length, order)


class BoxOnQuWireBot(MultiBox, BoxOnQuWire):
    """ Draws the bottom part of a box that affects more than one quantum wire"""

    def __init__(self, label, input_length):
        super().__init__(label)
        self.top_format = "│ %s │"

        self.mid_content = self.bot_connect = self.top_connect = ""
        if input_length <= 2:
            self.top_connect = label


class BoxOnClWireTop(MultiBox, BoxOnClWire):
    """ Draws the top part of a conditional box that affects more than one classical wire"""

    def __init__(self, label="", top_connect=None):
        super().__init__(label)
        self.mid_content = ""  # The label will be put by some other part of the box.
        self.bot_format = "│ %s │"
        self.top_connect = top_connect if top_connect else '─'
        self.bot_connect = self.bot_pad = " "


class BoxOnClWireMid(MultiBox, BoxOnClWire):
    """ Draws the middle part of a conditional box that affects more than one classical wire"""

    def __init__(self, label, input_length, order):
        super().__init__(label)
        self.mid_content = label
        self.top_format = "│ %s │"
        self.bot_format = "│ %s │"
        self.top_pad = self.bot_pad = ' '
        self.top_connect = self.bot_connect = self.mid_content = ''
        self.center_label(input_length, order)


class BoxOnClWireBot(MultiBox, BoxOnClWire):
    """ Draws the bottom part of a conditional box that affects more than one classical wire"""

    def __init__(self, label, input_length, bot_connect='─'):
        super().__init__(label)
        self.top_format = "│ %s │"
        self.top_pad = " "
        self.bot_connect = bot_connect

        self.mid_content = self.top_connect = ""
        if input_length <= 2:
            self.top_connect = label


class DirectOnQuWire(DrawElement):
    """
    Element to the wire (without the box).
    """

    def __init__(self, label=""):
        super().__init__(label)
        self.top_format = ' %s '
        self.mid_format = '─%s─'
        self.bot_format = ' %s '
        self._mid_padding = '─'
        self.top_connector = {"│": '│'}
        self.bot_connector = {"│": '│'}


class Barrier(DirectOnQuWire):
    """ Draws a barrier.
        top:  ░     ░
        mid: ─░─ ───░───
        bot:  ░     ░
    """

    def __init__(self, label=""):
        super().__init__("░")
        self.top_connect = "░"
        self.bot_connect = "░"
        self.top_connector = {}
        self.bot_connector = {}


class Ex(DirectOnQuWire):
    """ Draws an X (usually with a connector). E.g. the top part of a swap gate
    top:
    mid: ─X─ ───X───
    bot:  │     │
    """

    def __init__(self, bot_connect=" ", top_connect=" "):
        super().__init__("X")
        self.bot_connect = bot_connect
        self.top_connect = top_connect


class Reset(DirectOnQuWire):
    """ Draws a reset gate"""

    def __init__(self):
        super().__init__("|0>")


class Bullet(DirectOnQuWire):
    """ Draws a bullet (usually with a connector). E.g. the top part of a CX gate.
    top:
    mid: ─■─  ───■───
    bot:  │      │
    """

    def __init__(self, top_connect=" ", bot_connect=" "):
        super().__init__('■')
        self.top_connect = top_connect
        self.bot_connect = bot_connect


class EmptyWire(DrawElement):
    """ This element is just the wire, with no instructions nor operations."""

    def __init__(self, wire):
        super().__init__(wire)
        self._mid_padding = wire

    @staticmethod
    def fillup_layer(layer, first_clbit):
        """
        Given a layer, replace the Nones in it with EmptyWire elements.
        Args:
            layer (list): The layer that contains Nones.
            first_clbit (int): The first wire that is classic.

        Returns:
            list: The new layer, with no Nones.
        """
        for nones in [i for i, x in enumerate(layer) if x is None]:
            layer[nones] = EmptyWire('═') if nones >= first_clbit else EmptyWire('─')
        return layer


class BreakWire(DrawElement):
    """ This element is used to break the drawing in several pages."""

    def __init__(self, arrow_char):
        super().__init__()
        self.top_format = self.mid_format = self.bot_format = "%s"
        self.top_connect = arrow_char
        self.mid_content = arrow_char
        self.bot_connect = arrow_char

    @staticmethod
    def fillup_layer(layer_length, arrow_char):
        """
        Creates a layer with BreakWire elements.
        Args:
            layer_length (int): The length of the layer to create
            arrow_char (char): The char used to create the BreakWire element.

        Returns:
            list: The new layer.
        """
        breakwire_layer = []
        for _ in range(layer_length):
            breakwire_layer.append(BreakWire(arrow_char))
        return breakwire_layer


class InputWire(DrawElement):
    """ This element is the label and the initial value of a wire."""

    def __init__(self, label):
        super().__init__(label)

    @staticmethod
    def fillup_layer(names):  # pylint: disable=arguments-differ
        """
        Creates a layer with InputWire elements.
        Args:
            names (list): List of names for the wires.

        Returns:
            list: The new layer
        """
        longest = max([len(name) for name in names])
        inputs_wires = []
        for name in names:
            inputs_wires.append(InputWire(name.rjust(longest)))
        return inputs_wires


class TextDrawing():
    """ The text drawing"""

    def __init__(self, json_circuit, reversebits=False, plotbarriers=True):
        self.json_circuit = json_circuit
        self.reversebits = reversebits
        self.plotbarriers = plotbarriers
        self.qubitorder = self._get_qubitorder()
        self.clbitorder = self._get_clbitorder()

    def _get_qubit_labels(self):
        qubits = []
        for qubit in self.json_circuit['header']['qubit_labels']:
            qubits.append("%s_%s" % (qubit[0], qubit[1]))
        return qubits

    def _get_clbit_labels(self):
        clbits = []
        for creg in self.json_circuit['header']['clbit_labels']:
            for clbit in range(creg[1]):
                clbits.append("%s_%s" % (creg[0], clbit))
        return clbits

    def _get_qubitorder(self):
        header = self.json_circuit['header']

        if not self.reversebits:
            return [qub for qub in range(header['number_of_qubits'])]

        qreg_dest_order = []
        for _, qubits in groupby(header['qubit_labels'], lambda x: x[0]):
            qubits = [qubit for qubit in qubits]
            qubits.reverse()
            for qubit in qubits:
                qreg_dest_order.append("%s_%s" % (qubit[0], qubit[1]))
        return [qreg_dest_order.index(ind) for ind in self._get_qubit_labels()]

    def _get_clbitorder(self):
        header = self.json_circuit['header']

        if not self.reversebits:
            return [clb for clb in range(header['number_of_clbits'])]

        creg_dest_order = []
        for creg in self.json_circuit['header']['clbit_labels']:
            bit_nos = [bit for bit in range(creg[1])]
            bit_nos.reverse()
            for clbit in bit_nos:
                creg_dest_order.append("%s_%s" % (creg[0], clbit))
        return [creg_dest_order.index(ind) for ind in self._get_clbit_labels()]

    def lines(self, line_length=None):
        """
        Generates a list with lines. These lines form the text drawing.
        Args:
            line_length (int): Optional. When given, breaks the circuit drawing to this length. This
                               useful when the drawing does not fit in the console.

        Returns:
            list: A list of lines with the text drawing.
        """
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

            TextDrawing.normalize_width(layer)

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
                layer_groups[-1].append(BreakWire.fillup_layer(len(layer), '»'))

                # New group
                layer_groups.append([BreakWire.fillup_layer(len(layer), '«')])
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
        """
        Returns a list of names for each wire.
        Args:
            with_initial_value (bool): Optional (Default: True). If true, adds the initial value to
                                       the name.

        Returns:
            List: The list of wire names.
        """
        qubit_labels = self._get_qubit_labels()
        clbit_labels = self._get_clbit_labels()

        if with_initial_value:
            qubit_labels = ['%s: |0>' % qubit for qubit in qubit_labels]
            clbit_labels = ['%s: 0 ' % clbit for clbit in clbit_labels]
        else:
            qubit_labels = ['%s: ' % qubit for qubit in qubit_labels]
            clbit_labels = ['%s: ' % clbit for clbit in clbit_labels]

        qubit_wires = [None] * self.json_circuit['header']['number_of_qubits']
        clbit_wires = [None] * self.json_circuit['header']['number_of_clbits']

        for order, label in enumerate(qubit_labels):
            qubit_wires[self.qubitorder[order]] = label

        for order, label in enumerate(clbit_labels):
            clbit_wires[self.clbitorder[order]] = label

        return qubit_wires + clbit_wires

    @staticmethod
    def draw_wires(wires):
        """
        Given a list of wires, creates a list of lines with the text drawing.
        Args:
            wires (list): A list of wires with instructions.

        Returns:
            list: A list of lines with the text drawing.
        """
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
    def label_for_conditional(instruction):
        """ Creates the label for a conditional instruction."""
        return "%s %s" % ('=', instruction['conditional']['val'])

    @staticmethod
    def params_for_label(instruction):
        """Get the params and format them to add them to a label. None of there is no param."""
        if 'params' in instruction and instruction['params']:
            return ['%.5g' % i for i in instruction['params']]
        return None

    @staticmethod
    def label_for_box(instruction):
        """ Creates the label for a box."""
        label = instruction['name'].capitalize()
        params = TextDrawing.params_for_label(instruction)
        if params:
            label += "(%s)" % ','.join(params)
        return label

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
            elif topc in '┬╥' and botc in " ║│":
                ret += topc
            elif topc in '┬│' and botc == "═":
                ret += '╪'
            elif topc in '┬│' and botc == "─":
                ret += '┼'
            elif topc in '└┘║│░' and botc == " ":
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
        """
        Given a mask, returns a list of classical indexes affected by that mask.
        Args:
            mask (int): The mask.

        Returns:
            list: A list of indexes affected by the mask.
        """
        clbit_len = self.json_circuit['header']['number_of_clbits']
        bit_mask = [bool(mask & (1 << n)) for n in range(clbit_len)]
        return [i for i, x in enumerate(bit_mask) if x]

    @staticmethod
    def normalize_width(layer):
        """
        When the elements of the layer have different widths, sets the width to the max elements.
        Args:
            layer (list): A list of elements.
        """
        instructions = [instruction for instruction in filter(lambda x: x is not None, layer)]
        longest = max([instruction.length for instruction in instructions])
        for instruction in instructions:
            instruction.length = longest

    def build_layers(self):
        """
        Constructs layers.
        Returns:
            list: List of DrawElements.
        Raises:
            Exception: When the drawing is, for some reason, impossible to be drawn.
        """
        layers = []

        layers.append(InputWire.fillup_layer(self.wire_names(with_initial_value=True)))

        for instruction in self.json_circuit['instructions']:
            layer = Layer(self.qubitorder, self.clbitorder)
            connector_label = None

            if instruction['name'] == 'measure':
                layer.set_qubit(instruction['qubits'][0], MeasureFrom())
                layer.set_clbit(instruction['clbits'][0], MeasureTo())

            elif instruction['name'] == 'barrier':
                # barrier
                if not self.plotbarriers:
                    continue

                for qubit in instruction['qubits']:
                    layer.set_qubit(qubit, Barrier())

            elif instruction['name'] == 'swap':
                # swap
                for qubit in instruction['qubits']:
                    layer.set_qubit(qubit, Ex())

            elif instruction['name'] == 'cswap':
                # cswap
                layer.set_qubit(instruction['qubits'][0], Bullet())
                layer.set_qubit(instruction['qubits'][1], Ex())
                layer.set_qubit(instruction['qubits'][2], Ex())

            elif instruction['name'] == 'reset':
                layer.set_qubit(instruction['qubits'][0], Reset())

            elif 'conditional' in instruction:
                # conditional
                clbits = self.clbit_index_from_mask(int(instruction['conditional']['mask'], 16))
                cllabel = TextDrawing.label_for_conditional(instruction)
                qulabel = TextDrawing.label_for_box(instruction)

                layer.set_cl_multibox(clbits, cllabel, top_connect='┴')
                layer.set_qubit(instruction['qubits'][0], BoxOnQuWire(qulabel, bot_connect='┬'))

            elif instruction['name'] in ['cx', 'CX', 'ccx']:
                # cx/ccx
                for qubit in [qubit for qubit in instruction['qubits'][:-1]]:
                    layer.set_qubit(qubit, Bullet())
                layer.set_qubit(instruction['qubits'][-1], BoxOnQuWire('X'))

            elif instruction['name'] == 'cy':
                # cy
                layer.set_qubit(instruction['qubits'][0], Bullet())
                layer.set_qubit(instruction['qubits'][1], BoxOnQuWire('Y'))

            elif instruction['name'] == 'cz':
                # cz
                layer.set_qubit(instruction['qubits'][0], Bullet())
                layer.set_qubit(instruction['qubits'][1], Bullet())

            elif instruction['name'] == 'ch':
                # ch
                layer.set_qubit(instruction['qubits'][0], Bullet())
                layer.set_qubit(instruction['qubits'][1], BoxOnQuWire('H'))

            elif instruction['name'] == 'cu1':
                # cu1
                connector_label = TextDrawing.params_for_label(instruction)[0]
                layer.set_qubit(instruction['qubits'][0], Bullet())
                layer.set_qubit(instruction['qubits'][1], Bullet())

            elif instruction['name'] == 'cu3':
                # cu3
                params = TextDrawing.params_for_label(instruction)
                layer.set_qubit(instruction['qubits'][0], Bullet())
                layer.set_qubit(instruction['qubits'][1],
                                BoxOnQuWire("U3(%s)" % ','.join(params)))

            elif instruction['name'] == 'crz':
                # crz
                label = "Rz(%s)" % TextDrawing.params_for_label(instruction)[0]
                layer.set_qubit(instruction['qubits'][0], Bullet())
                layer.set_qubit(instruction['qubits'][1],
                                BoxOnQuWire(label))

            elif len(instruction['qubits']) == 1 and 'clbits' not in instruction:
                # unitary gate
                layer.set_qubit(instruction['qubits'][0],
                                BoxOnQuWire(TextDrawing.label_for_box(instruction)))

            elif len(instruction['qubits']) >= 2 and 'clbits' not in instruction:
                # multiple qubit gate
                layer.set_qu_multibox(instruction['qubits'], TextDrawing.label_for_box(instruction))

            else:
                raise Exception("I don't know how to handle this instruction", instruction)

            layer.connect_with("│", connector_label)

            layers.append(layer.full_layer)

        return layers


class Layer:
    """ A layer is the "column" of the circuit. """

    def __init__(self, qubitorder, clbitorder):
        self.qubitorder = qubitorder
        self.clbitorder = clbitorder
        self.qubit_layer = [None] * len(qubitorder)
        self.clbit_layer = [None] * len(clbitorder)

    @property
    def full_layer(self):
        """
        Returns the composition of qubits and classic wires.
        Returns:
            String: self.qubit_layer + self.clbit_layer
        """
        return self.qubit_layer + self.clbit_layer

    def set_qubit(self, qubit, element):
        """
        Sets the qubit to the element
        Args:
            qubit (int): Qubit index.
            element (DrawElement): Element to set in the qubit
        """
        self.qubit_layer[self.qubitorder[qubit]] = element

    def set_clbit(self, clbit, element):
        """
        Sets the clbit to the element
        Args:
            clbit (int): Clbit index.
            element (DrawElement): Element to set in the clbit
        """
        self.clbit_layer[self.clbitorder[clbit]] = element

    def _set_multibox(self, wire_type, bits, label, top_connect=None):
        # pylint: disable=invalid-name
        bits = sorted(bits)
        if wire_type == "cl":
            set_bit = self.set_clbit
            BoxOnWire = BoxOnClWire
            BoxOnWireTop = BoxOnClWireTop
            BoxOnWireMid = BoxOnClWireMid
            BoxOnWireBot = BoxOnClWireBot
        elif wire_type == "qu":
            set_bit = self.set_qubit
            BoxOnWire = BoxOnQuWire
            BoxOnWireTop = BoxOnQuWireTop
            BoxOnWireMid = BoxOnQuWireMid
            BoxOnWireBot = BoxOnQuWireBot

        # Checks if qubits are consecutive
        if bits != [i for i in range(bits[0], bits[-1] + 1)]:
            raise Exception("I don't know how to build a gate with multiple bits when"
                            "they are not adjacent to each other")

        if len(bits) == 1:
            set_bit(bits[0], BoxOnWire(label, top_connect=top_connect))
        else:
            set_bit(bits[0], BoxOnWireTop(label, top_connect=top_connect))
            for order, bit in enumerate(bits[1:-1], 1):
                set_bit(bit, BoxOnWireMid(label, len(bits), order))
            set_bit(bits[-1], BoxOnWireBot(label, len(bits)))

    def set_cl_multibox(self, bits, label, top_connect='┴'):
        """
        Sets the multi clbit box.
        Args:
            bits (list[int]): A list of affected bits.
            label (string): The label for the multi clbit box.
            top_connect (char): The char to connect the box on the top.
        """
        self._set_multibox("cl", bits, label, top_connect=top_connect)

    def set_qu_multibox(self, bits, label):
        """
        Sets the multi qubit box.
        Args:
            bits (list[int]): A list of affected bits.
            label (string): The label for the multi qubit box.
        """
        self._set_multibox("qu", bits, label)

    def connect_with(self, wire_char, label=None):
        """
        Connects the elements in the layer using wire_char.
        Args:
            wire_char (char): For example '║' or '│'.
            label (string): Some connectors have a label (see cu1, for example).
        """
        affected_bits = [bit for bit in self.full_layer if bit is not None]

        if len([qbit for qbit in self.qubit_layer if qbit is not None]) == 1:
            # Nothing to connect
            return

        affected_bits[0].connect(wire_char, ['bot'])
        for affected_bit in affected_bits[1:-1]:
            affected_bit.connect(wire_char, ['bot', 'top'])
        affected_bits[-1].connect(wire_char, ['top'], label)

        if label:
            for affected_bit in affected_bits:
                affected_bit.right_fill = len(label) + len(affected_bit.mid)
