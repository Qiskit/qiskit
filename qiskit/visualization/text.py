# -*- coding: utf-8 -*-

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
A module for drawing circuits in ascii art or some other text representation
"""

from shutil import get_terminal_size
import sys
import sympy
from numpy import ndarray

from .exceptions import VisualizationError


class DrawElement():
    """ An element is an instruction or an operation that need to be drawn."""

    def __init__(self, label=None):
        self._width = None
        self.label = self.mid_content = label
        self.top_format = self.mid_format = self.bot_format = "%s"
        self.top_connect = self.bot_connect = " "
        self.top_pad = self._mid_padding = self.bot_pad = " "
        self.mid_bck = self.top_bck = self.bot_bck = " "
        self.bot_connector = {}
        self.top_connector = {}
        self.right_fill = self.left_fill = self.layer_width = 0
        self.wire_label = ""

    @property
    def top(self):
        """ Constructs the top line of the element"""
        ret = self.top_format % self.top_connect.center(
            self.width, self.top_pad)
        if self.right_fill:
            ret = ret.ljust(self.right_fill, self.top_pad)
        if self.left_fill:
            ret = ret.rjust(self.left_fill, self.top_pad)
        ret = ret.center(self.layer_width, self.top_bck)
        return ret

    @property
    def mid(self):
        """ Constructs the middle line of the element"""
        ret = self.mid_format % self.mid_content.center(
            self.width, self._mid_padding)
        if self.right_fill:
            ret = ret.ljust(self.right_fill, self._mid_padding)
        if self.left_fill:
            ret = ret.rjust(self.left_fill, self._mid_padding)
        ret = ret.center(self.layer_width, self.mid_bck)
        return ret

    @property
    def bot(self):
        """ Constructs the bottom line of the element"""
        ret = self.bot_format % self.bot_connect.center(
            self.width, self.bot_pad)
        if self.right_fill:
            ret = ret.ljust(self.right_fill, self.bot_pad)
        if self.left_fill:
            ret = ret.rjust(self.left_fill, self.bot_pad)
        ret = ret.center(self.layer_width, self.bot_bck)
        return ret

    @property
    def length(self):
        """ Returns the length of the element, including the box around."""
        return max(len(self.top), len(self.mid), len(self.bot))

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
        top: ┌───┐   ┌───┐
        mid: ╡ A ╞ ══╡ A ╞══
        bot: └───┘   └───┘
    """

    def __init__(self, label="", top_connect='─', bot_connect='─'):
        super().__init__(label)
        self.top_format = "┌─%s─┐"
        self.mid_format = "╡ %s ╞"
        self.bot_format = "└─%s─┘"
        self.top_pad = self.bot_pad = '─'
        self.mid_bck = '═'
        self.top_connect = top_connect
        self.bot_connect = bot_connect
        self.mid_content = label


class BoxOnQuWire(DrawElement):
    """ Draws a box on the quantum wire
        top: ┌───┐   ┌───┐
        mid: ┤ A ├ ──┤ A ├──
        bot: └───┘   └───┘
    """

    def __init__(self, label="", top_connect='─', bot_connect='─'):
        super().__init__(label)
        self.top_format = "┌─%s─┐"
        self.mid_format = "┤ %s ├"
        self.bot_format = "└─%s─┘"
        self.top_pad = self.bot_pad = self.mid_bck = '─'
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
        self.mid_bck = "═"


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

        self.top_pad = self.bot_pad = " "
        self._mid_padding = '─'


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
        top_limit = order * 2 + 2
        bot_limit = top_limit + 2
        if top_limit <= location_in_the_box < bot_limit:
            if location_in_the_box == top_limit:
                self.top_connect = self.label
            elif location_in_the_box == top_limit + 1:
                self.mid_content = self.label
            else:
                self.bot_connect = self.label

    @property
    def width(self):
        """ Returns the width of the label, including padding"""
        if self._width:
            return self._width
        return len(self.label)


class BoxOnQuWireTop(MultiBox, BoxOnQuWire):
    """ Draws the top part of a box that affects more than one quantum wire"""

    def __init__(self, label="", top_connect=None, wire_label=''):
        super().__init__(label)
        self.wire_label = wire_label
        self.bot_connect = self.bot_pad = " "
        self.mid_content = ""  # The label will be put by some other part of the box.
        self.left_fill = len(self.wire_label)
        self.top_format = "┌{}─%s─┐".format(self.top_pad * self.left_fill)
        self.mid_format = "┤{} %s ├".format(self.wire_label)
        self.bot_format = "│{} %s │".format(self.bot_pad * self.left_fill)
        self.top_connect = top_connect if top_connect else '─'


class BoxOnQuWireMid(MultiBox, BoxOnQuWire):
    """ Draws the middle part of a box that affects more than one quantum wire"""

    def __init__(self, label, input_length, order, wire_label=''):
        super().__init__(label)
        self.top_pad = self.bot_pad = self.top_connect = self.bot_connect = " "
        self.wire_label = wire_label
        self.left_fill = len(self.wire_label)
        self.top_format = "│{} %s │".format(self.top_pad * self.left_fill)
        self.mid_format = "┤{} %s ├".format(self.wire_label)
        self.bot_format = "│{} %s │".format(self.bot_pad * self.left_fill)
        self.top_connect = self.bot_connect = self.mid_content = ''
        self.center_label(input_length, order)


class BoxOnQuWireBot(MultiBox, BoxOnQuWire):
    """ Draws the bottom part of a box that affects more than one quantum wire"""

    def __init__(self, label, input_length, bot_connect='─', wire_label=''):
        super().__init__(label)
        self.wire_label = wire_label
        self.top_pad = " "
        self.left_fill = len(self.wire_label)
        self.top_format = "│{} %s │".format(self.top_pad * self.left_fill)
        self.mid_format = "┤{} %s ├".format(self.wire_label)
        self.bot_format = "└{}─%s─┘".format(self.bot_pad * self.left_fill)
        self.bot_connect = bot_connect

        self.mid_content = self.top_connect = ""
        if input_length <= 2:
            self.top_connect = label


class BoxOnClWireTop(MultiBox, BoxOnClWire):
    """ Draws the top part of a conditional box that affects more than one classical wire"""

    def __init__(self, label="", top_connect=None, wire_label=''):
        super().__init__(label)
        self.wire_label = wire_label
        self.mid_content = ""  # The label will be put by some other part of the box.
        self.bot_format = "│ %s │"
        self.top_connect = top_connect if top_connect else '─'
        self.bot_connect = self.bot_pad = " "


class BoxOnClWireMid(MultiBox, BoxOnClWire):
    """ Draws the middle part of a conditional box that affects more than one classical wire"""

    def __init__(self, label, input_length, order, wire_label=''):
        super().__init__(label)
        self.wire_label = wire_label
        self.top_format = "│ %s │"
        self.bot_format = "│ %s │"
        self.top_pad = self.bot_pad = ' '
        self.top_connect = self.bot_connect = self.mid_content = ''
        self.center_label(input_length, order)


class BoxOnClWireBot(MultiBox, BoxOnClWire):
    """ Draws the bottom part of a conditional box that affects more than one classical wire"""

    def __init__(self, label, input_length, bot_connect='─', wire_label=''):
        super().__init__(label)
        self.wire_label = wire_label
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
        self._mid_padding = self.mid_bck = '─'
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

    def __init__(self, top_connect="", bot_connect=""):
        super().__init__('■')
        self.top_connect = top_connect
        self.bot_connect = bot_connect
        self.mid_bck = '─'


class EmptyWire(DrawElement):
    """ This element is just the wire, with no instructions nor operations."""

    def __init__(self, wire):
        super().__init__(wire)
        self._mid_padding = self.mid_bck = wire

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

    def __init__(self, qregs, cregs, instructions, plotbarriers=True,
                 line_length=None, vertically_compressed=True):
        self.qregs = qregs
        self.cregs = cregs
        self.instructions = instructions

        self.plotbarriers = plotbarriers
        self.line_length = line_length
        self.vertically_compressed = vertically_compressed

    def __str__(self):
        return self.single_string()

    def _repr_html_(self):
        return '<pre style="word-wrap: normal;' \
               'white-space: pre;' \
               'line-height: 15px;">%s</pre>' % self.single_string()

    def _get_qubit_labels(self):
        qubits = []
        for qubit in self.qregs:
            qubits.append("%s_%s" % (qubit[0].name, qubit[1]))
        return qubits

    def _get_clbit_labels(self):
        clbits = []
        for clbit in self.cregs:
            clbits.append("%s_%s" % (clbit[0].name, clbit[1]))
        return clbits

    def single_string(self):
        """
        Creates a long string with the ascii art
        Returns:
            str: The lines joined by '\n'
        """
        return "\n".join(self.lines())

    def dump(self, filename, encoding="utf8"):
        """
        Dumps the ascii art in the file.
        Args:
            filename (str): File to dump the ascii art.
            encoding (str): Optional. Default "utf-8".
        """
        with open(filename, mode='w', encoding=encoding) as text_file:
            text_file.write(self.single_string())

    def lines(self, line_length=None):
        """
        Generates a list with lines. These lines form the text drawing.
        Args:
            line_length (int): Optional. Breaks the circuit drawing to this length. This
                               useful when the drawing does not fit in the console. If
                               None (default), it will try to guess the console width using
                               shutil.get_terminal_size(). If you don't want pagination
                               at all, set line_length=-1.

        Returns:
            list: A list of lines with the text drawing.
        """
        if line_length is None:
            line_length = self.line_length
        if not line_length:
            if ('ipykernel' in sys.modules) and ('spyder' not in sys.modules):
                line_length = 80
            else:
                line_length, _ = get_terminal_size()

        noqubits = len(self.qregs)

        layers = self.build_layers()

        layer_groups = [[]]
        rest_of_the_line = line_length
        for layerno, layer in enumerate(layers):
            # Replace the Nones with EmptyWire
            layers[layerno] = EmptyWire.fillup_layer(layer, noqubits)

            TextDrawing.normalize_width(layer)

            if line_length == -1:
                # Do not use pagination (aka line breaking. aka ignore line_length).
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
            lines += TextDrawing.draw_wires(wires, self.vertically_compressed)

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

        return qubit_labels + clbit_labels

    @staticmethod
    def draw_wires(wires, vertically_compressed=True):
        """
        Given a list of wires, creates a list of lines with the text drawing.
        Args:
            wires (list): A list of wires with instructions.
            vertically_compressed (bool): Default is `True`. It merges the lines
                                     so the drawing will take less vertical room.
        Returns:
            list: A list of lines with the text drawing.
        """
        lines = []
        bot_line = None
        for wire in wires:
            # TOP
            top_line = ''
            for instruction in wire:
                top_line += instruction.top

            if bot_line is None:
                lines.append(top_line)
            else:
                if vertically_compressed:
                    lines.append(TextDrawing.merge_lines(lines.pop(), top_line))
                else:
                    lines.append(TextDrawing.merge_lines(lines[-1], top_line, icod="bot"))

            # MID
            mid_line = ''
            for instruction in wire:
                mid_line += instruction.mid
            lines.append(TextDrawing.merge_lines(lines[-1], mid_line, icod="bot"))

            # BOT
            bot_line = ''
            for instruction in wire:
                bot_line += instruction.bot
            lines.append(TextDrawing.merge_lines(lines[-1], bot_line, icod="bot"))

        return lines

    @staticmethod
    def label_for_conditional(instruction):
        """ Creates the label for a conditional instruction."""
        return "%s %s" % ('=', instruction.condition[1])

    @staticmethod
    def params_for_label(instruction):
        """Get the params and format them to add them to a label. None if there
         are no params of if the params are numpy.ndarrays."""

        if not hasattr(instruction.op, 'params'):
            return None
        if all([isinstance(param, ndarray) for param in instruction.op.params]):
            return None

        ret = []
        for param in instruction.op.params:
            if isinstance(param, (sympy.Number, float)):
                ret.append('%.5g' % param)
            else:
                ret.append('%s' % param)
        return ret

    @staticmethod
    def label_for_box(instruction):
        """ Creates the label for a box."""
        label = instruction.name.capitalize()
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
            elif topc in '┬╥' and botc in " ║│" and icod == "top":
                ret += topc
            elif topc in '┬' and botc == " " and icod == "bot":
                ret += '│'
            elif topc in '╥' and botc == " " and icod == "bot":
                ret += '║'
            elif topc in '┬│' and botc == "═":
                ret += '╪'
            elif topc in '┬│' and botc == "─":
                ret += '┼'
            elif topc in '└┘║│░' and botc == " " and icod == "top":
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
            elif botc in "┐┌" and icod == 'top':
                ret += "┬"
            elif topc in "┘└" and botc in "─" and icod == 'top':
                ret += "┴"
            else:
                ret += botc
        return ret

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
            instruction.layer_width = longest

    def _instruction_to_gate(self, instruction, layer):
        """ Convert an instruction into its corresponding Gate object, and establish
        any connections it introduces between qubits"""

        current_cons = []
        connection_label = None

        # add in a gate that operates over multiple qubits
        def add_connected_gate(instruction, gates, layer, current_cons):
            for i, gate in enumerate(gates):
                layer.set_qubit(instruction.qargs[i], gate)
                actual_index = self.qregs.index(instruction.qargs[i])
                current_cons.append((actual_index, gate))

        if instruction.name == 'measure':
            gate = MeasureFrom()
            layer.set_qubit(instruction.qargs[0], gate)
            layer.set_clbit(instruction.cargs[0], MeasureTo())

        elif instruction.name in ['barrier', 'snapshot', 'save', 'load',
                                  'noise']:
            # barrier
            if not self.plotbarriers:
                return layer, current_cons, connection_label

            for qubit in instruction.qargs:
                layer.set_qubit(qubit, Barrier())

        elif instruction.name == 'swap':
            # swap
            gates = [Ex() for _ in range(len(instruction.qargs))]
            add_connected_gate(instruction, gates, layer, current_cons)

        elif instruction.name == 'cswap':
            # cswap
            gates = [Bullet(), Ex(), Ex()]
            add_connected_gate(instruction, gates, layer, current_cons)

        elif instruction.name == 'reset':
            layer.set_qubit(instruction.qargs[0], Reset())

        elif instruction.condition is not None:
            # conditional
            cllabel = TextDrawing.label_for_conditional(instruction)
            qulabel = TextDrawing.label_for_box(instruction)

            layer.set_cl_multibox(instruction.condition[0], cllabel, top_connect='┴')
            layer.set_qubit(instruction.qargs[0], BoxOnQuWire(qulabel, bot_connect='┬'))

        elif instruction.name in ['cx', 'CX', 'ccx']:
            # cx/ccx
            gates = [Bullet() for _ in range(len(instruction.qargs) - 1)]
            gates.append(BoxOnQuWire('X'))
            add_connected_gate(instruction, gates, layer, current_cons)

        elif instruction.name == 'cy':
            # cy
            gates = [Bullet(), BoxOnQuWire('Y')]
            add_connected_gate(instruction, gates, layer, current_cons)

        elif instruction.name == 'cz':
            # cz
            gates = [Bullet(), Bullet()]
            add_connected_gate(instruction, gates, layer, current_cons)

        elif instruction.name == 'ch':
            # ch
            gates = [Bullet(), BoxOnQuWire('H')]
            add_connected_gate(instruction, gates, layer, current_cons)

        elif instruction.name == 'cu1':
            # cu1
            connection_label = TextDrawing.params_for_label(instruction)[0]
            gates = [Bullet(), Bullet()]
            add_connected_gate(instruction, gates, layer, current_cons)

        elif instruction.name == 'rzz':
            # rzz
            connection_label = "zz(%s)" % TextDrawing.params_for_label(instruction)[0]
            gates = [Bullet(), Bullet()]
            add_connected_gate(instruction, gates, layer, current_cons)

        elif instruction.name == 'cu3':
            # cu3
            params = TextDrawing.params_for_label(instruction)
            gates = [Bullet(), BoxOnQuWire("U3(%s)" % ','.join(params))]
            add_connected_gate(instruction, gates, layer, current_cons)

        elif instruction.name == 'crz':
            # crz
            label = "Rz(%s)" % TextDrawing.params_for_label(instruction)[0]

            gates = [Bullet(), BoxOnQuWire(label)]
            add_connected_gate(instruction, gates, layer, current_cons)

        elif len(instruction.qargs) == 1 and not instruction.cargs:
            # unitary gate
            layer.set_qubit(instruction.qargs[0],
                            BoxOnQuWire(TextDrawing.label_for_box(instruction)))

        elif len(instruction.qargs) >= 2 and not instruction.cargs:
            # multiple qubit gate
            label = instruction.name
            params = TextDrawing.params_for_label(instruction)
            if params:
                label += "(%s)" % ','.join(params)
            layer.set_qu_multibox(instruction.qargs, label)

        else:
            raise VisualizationError(
                "Text visualizer does not know how to handle this instruction", instruction)

        # sort into the order they were declared in
        # this ensures that connected boxes have lines in the right direction
        current_cons.sort(key=lambda tup: tup[0])
        current_cons = [g for q, g in current_cons]

        return layer, current_cons, connection_label

    def build_layers(self):
        """
        Constructs layers.
        Returns:
            list: List of DrawElements.
        Raises:
            VisualizationError: When the drawing is, for some reason, impossible to be drawn.
        """
        wire_names = self.wire_names(with_initial_value=True)
        if not wire_names:
            return []

        layers = [InputWire.fillup_layer(wire_names)]

        for instruction_layer in self.instructions:
            layer = Layer(self.qregs, self.cregs)

            for instruction in instruction_layer:
                layer, current_connections, connection_label = \
                    self._instruction_to_gate(instruction, layer)

                layer.connections.append((connection_label, current_connections))
                layer.connect_with("│")
            layers.append(layer.full_layer)

        return layers


class Layer:
    """ A layer is the "column" of the circuit. """

    def __init__(self, qregs, cregs):
        self.qregs = qregs
        self.cregs = cregs
        self.qubit_layer = [None] * len(qregs)
        self.connections = []
        self.clbit_layer = [None] * len(cregs)

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
            qubit (qbit): Element of self.qregs.
            element (DrawElement): Element to set in the qubit
        """
        self.qubit_layer[self.qregs.index(qubit)] = element

    def set_clbit(self, clbit, element):
        """
        Sets the clbit to the element
        Args:
            clbit (cbit): Element of self.cregs.
            element (DrawElement): Element to set in the clbit
        """
        self.clbit_layer[self.cregs.index(clbit)] = element

    def _set_multibox(self, wire_type, bits, label, top_connect=None):
        # pylint: disable=invalid-name
        bits = list(bits)
        if wire_type == "cl":
            bit_index = sorted([i for i, x in enumerate(self.cregs) if x in bits])
            bits.sort(key=self.cregs.index)
            qargs = [''] * len(bits)
            set_bit = self.set_clbit
            BoxOnWire = BoxOnClWire
            BoxOnWireTop = BoxOnClWireTop
            BoxOnWireMid = BoxOnClWireMid
            BoxOnWireBot = BoxOnClWireBot
        elif wire_type == "qu":
            bit_index = sorted([i for i, x in enumerate(self.qregs) if x in bits])
            qargs = [str(bits.index(qbit)) for qbit in self.qregs if qbit in bits]
            bits.sort(key=self.qregs.index)
            set_bit = self.set_qubit
            BoxOnWire = BoxOnQuWire
            BoxOnWireTop = BoxOnQuWireTop
            BoxOnWireMid = BoxOnQuWireMid
            BoxOnWireBot = BoxOnQuWireBot
        else:
            raise VisualizationError("_set_multibox only supports 'cl' and 'qu' as wire types.")

        if len(bit_index) == 1:
            set_bit(bits[0], BoxOnWire(label, top_connect=top_connect))
        else:
            box_height = max(bit_index) - min(bit_index) + 1
            set_bit(bits.pop(0),
                    BoxOnWireTop(label, top_connect=top_connect, wire_label=qargs.pop(0)))
            for order, bit_i in enumerate(range(min(bit_index) + 1, max(bit_index))):
                if bit_i in bit_index:
                    named_bit = bits.pop(0)
                    wire_label = qargs.pop(0)
                else:
                    named_bit = (self.qregs + self.cregs)[bit_i]
                    wire_label = ' ' * len(qargs[0])
                set_bit(named_bit, BoxOnWireMid(label, box_height, order, wire_label=wire_label))
            set_bit(bits.pop(0), BoxOnWireBot(label, box_height, wire_label=qargs.pop(0)))

    def set_cl_multibox(self, creg, label, top_connect='┴'):
        """
        Sets the multi clbit box.
        Args:
            creg (string): The affected classical register.
            label (string): The label for the multi clbit box.
            top_connect (char): The char to connect the box on the top.
        """
        clbit = [bit for bit in self.cregs if bit[0] == creg]
        self._set_multibox("cl", clbit, label, top_connect=top_connect)

    def set_qu_multibox(self, bits, label):
        """
        Sets the multi qubit box.
        Args:
            bits (list[int]): A list of affected bits.
            label (string): The label for the multi qubit box.
        """
        self._set_multibox("qu", bits, label)

    def connect_with(self, wire_char):
        """
        Connects the elements in the layer using wire_char.
        Args:
            wire_char (char): For example '║' or '│'.
        """

        if len([qbit for qbit in self.qubit_layer if qbit is not None]) == 1:
            # Nothing to connect
            return

        for label, affected_bits in self.connections:

            if not affected_bits:
                continue

            affected_bits[0].connect(wire_char, ['bot'])
            for affected_bit in affected_bits[1:-1]:
                affected_bit.connect(wire_char, ['bot', 'top'])

            affected_bits[-1].connect(wire_char, ['top'], label)

            if label:
                for affected_bit in affected_bits:
                    affected_bit.right_fill = len(label) + len(affected_bit.mid)
