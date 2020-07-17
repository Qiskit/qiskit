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

from warnings import warn
from shutil import get_terminal_size
import sys
from numpy import ndarray

from qiskit.circuit import ControlledGate, Gate, Instruction
from qiskit.circuit import Reset as ResetInstruction
from qiskit.circuit import Measure as MeasureInstruction
from qiskit.circuit import Barrier as BarrierInstruction
from qiskit.circuit.library.standard_gates import IGate, RZZGate, SwapGate
from qiskit.extensions import UnitaryGate, HamiltonianGate, Snapshot
from qiskit.extensions.quantum_initializer.initializer import Initialize
from qiskit.circuit.tools.pi_check import pi_check
from .exceptions import VisualizationError


class TextDrawerCregBundle(VisualizationError):
    """The parameter "cregbundle" was set to True in an imposible situation. For example, an
    instruction needs to refer to individual classical wires'"""
    pass


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
        """Connects boxes and elements using wire_char and setting proper connectors.

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
    """Draws a box on the classical wire.

    ::

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
    """Draws a box on the quantum wire.

    ::

        top: ┌───┐   ┌───┐
        mid: ┤ A ├ ──┤ A ├──
        bot: └───┘   └───┘
    """

    def __init__(self, label="", top_connect='─', conditional=False):
        super().__init__(label)
        self.top_format = "┌─%s─┐"
        self.mid_format = "┤ %s ├"
        self.bot_format = "└─%s─┘"
        self.top_pad = self.bot_pad = self.mid_bck = '─'
        self.top_connect = top_connect
        self.bot_connect = '┬' if conditional else '─'
        self.mid_content = label
        self.top_connector = {"│": '┴'}
        self.bot_connector = {"│": '┬'}


class MeasureTo(DrawElement):
    """The element on the classic wire to which the measure is performed.

    ::

        top:  ║     ║
        mid: ═╩═ ═══╩═══
        bot:
    """

    def __init__(self, label=''):
        super().__init__()
        self.top_connect = " ║ "
        self.mid_content = "═╩═"
        self.bot_connect = label
        self.mid_bck = "═"


class MeasureFrom(BoxOnQuWire):
    """The element on the quantum wire in which the measure is performed.

    ::

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
        """In multi-bit elements, the label is centered vertically.

        Args:
            input_length (int): Rhe amount of wires affected.
            order (int): Which middle element is this one?
        """
        if input_length == order == 0:
            self.top_connect = self.label
            return
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
        self.top_format = "┌{}%s──┐".format(self.top_pad * self.left_fill)
        self.mid_format = "┤{} %s ├".format(self.wire_label)
        self.bot_format = "│{} %s │".format(self.bot_pad * self.left_fill)
        self.top_connect = top_connect if top_connect else '─'


class BoxOnWireMid(MultiBox):
    """ A generic middle box"""

    def __init__(self, label, input_length, order, wire_label=''):
        super().__init__(label, input_length, order)
        self.top_pad = self.bot_pad = self.top_connect = self.bot_connect = " "
        self.wire_label = wire_label
        self.left_fill = len(self.wire_label)
        self.top_format = "│{} %s │".format(self.top_pad * self.left_fill)
        self.bot_format = "│{} %s │".format(self.bot_pad * self.left_fill)
        self.top_connect = self.bot_connect = self.mid_content = ''
        self.center_label(input_length, order)


class BoxOnQuWireMid(BoxOnWireMid, BoxOnQuWire):
    """ Draws the middle part of a box that affects more than one quantum wire"""

    def __init__(self, label, input_length, order, wire_label='', control_label=None):
        super().__init__(label, input_length, order, wire_label=wire_label)
        if control_label:
            self.mid_format = "{}{} %s ├".format(control_label, self.wire_label)
        else:
            self.mid_format = "┤{} %s ├".format(self.wire_label)


class BoxOnQuWireBot(MultiBox, BoxOnQuWire):
    """ Draws the bottom part of a box that affects more than one quantum wire"""

    def __init__(self, label, input_length, bot_connect=None, wire_label='', conditional=False):
        super().__init__(label)
        self.wire_label = wire_label
        self.top_pad = " "
        self.left_fill = len(self.wire_label)
        self.top_format = "│{} %s │".format(self.top_pad * self.left_fill)
        self.mid_format = "┤{} %s ├".format(self.wire_label)
        self.bot_format = "└{}%s──┘".format(self.bot_pad * self.left_fill)
        bot_connect = bot_connect if bot_connect else '─'
        self.bot_connect = '┬' if conditional else bot_connect

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


class BoxOnClWireMid(BoxOnWireMid, BoxOnClWire):
    """ Draws the middle part of a conditional box that affects more than one classical wire"""

    def __init__(self, label, input_length, order, wire_label='', **_):
        super().__init__(label, input_length, order, wire_label=wire_label)
        self.mid_format = "╡{} %s ╞".format(self.wire_label)


class BoxOnClWireBot(MultiBox, BoxOnClWire):
    """ Draws the bottom part of a conditional box that affects more than one classical wire"""

    def __init__(self, label, input_length, bot_connect='─', wire_label='', **_):
        super().__init__(label)
        self.wire_label = wire_label
        self.left_fill = len(self.wire_label)
        self.top_pad = ' '
        self.bot_pad = '─'
        self.top_format = "│{} %s │".format(self.top_pad * self.left_fill)
        self.mid_format = "╡{} %s ╞".format(self.wire_label)
        self.bot_format = "└{}%s──┘".format(self.bot_pad * self.left_fill)
        bot_connect = bot_connect if bot_connect else '─'
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
    """Draws a barrier.

    ::

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
    """Draws an X (usually with a connector). E.g. the top part of a swap gate.

    ::

        top:
        mid: ─X─ ───X───
        bot:  │     │
    """

    def __init__(self, bot_connect=" ", top_connect=" ", conditional=False):
        super().__init__("X")
        self.bot_connect = "│" if conditional else bot_connect
        self.top_connect = top_connect


class Reset(DirectOnQuWire):
    """ Draws a reset gate"""

    def __init__(self, conditional=False):
        super().__init__("|0>")
        if conditional:
            self.bot_connect = "│"


class Bullet(DirectOnQuWire):
    """ Draws a bullet (usually with a connector). E.g. the top part of a CX gate.

    ::

        top:
        mid: ─■─  ───■───
        bot:  │      │
    """

    def __init__(self, top_connect="", bot_connect="", conditional=False,
                 label=None, bottom=False):
        super().__init__('■')
        self.top_connect = top_connect
        self.bot_connect = '│' if conditional else bot_connect
        if label and bottom:
            self.bot_connect = label
        elif label:
            self.top_connect = label
        self.mid_bck = '─'


class OpenBullet(DirectOnQuWire):
    """Draws an open bullet (usually with a connector). E.g. the top part of a CX gate.

    ::

        top:
        mid: ─o─  ───o───
        bot:  │      │
    """

    def __init__(self, top_connect="", bot_connect="", conditional=False,
                 label=None, bottom=False):
        super().__init__('o')
        self.top_connect = top_connect
        self.bot_connect = '│' if conditional else bot_connect
        if label and bottom:
            self.bot_connect = label
        elif label:
            self.top_connect = label
        self.mid_bck = '─'


class EmptyWire(DrawElement):
    """This element is just the wire, with no instructions nor operations."""

    def __init__(self, wire):
        super().__init__(wire)
        self._mid_padding = self.mid_bck = wire

    @staticmethod
    def fillup_layer(layer, first_clbit):
        """Given a layer, replace the Nones in it with EmptyWire elements.

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
        """Creates a layer with BreakWire elements.

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
    def fillup_layer(names):
        """Creates a layer with InputWire elements.

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
                 line_length=None, vertical_compression='high', layout=None, initial_state=True,
                 cregbundle=False):
        self.qregs = qregs
        self.cregs = cregs
        self.instructions = instructions
        self.layout = layout
        self.initial_state = initial_state

        self.cregbundle = cregbundle
        self.plotbarriers = plotbarriers
        self.line_length = line_length
        if vertical_compression not in ['high', 'medium', 'low']:
            raise ValueError("Vertical compression can only be 'high', 'medium', or 'low'")
        self.vertical_compression = vertical_compression

    def __str__(self):
        return self.single_string()

    def _repr_html_(self):
        return '<pre style="word-wrap: normal;' \
               'white-space: pre;' \
               'background: #fff0;' \
               'line-height: 1.1;' \
               'font-family: &quot;Courier New&quot;,Courier,monospace">' \
               '%s</pre>' % self.single_string()

    def __repr__(self):
        return self.single_string()

    def single_string(self):
        """Creates a long string with the ascii art.

        Returns:
            str: The lines joined by a newline (``\\n``)
        """
        return "\n".join(self.lines())

    def dump(self, filename, encoding="utf8"):
        """Dumps the ascii art in the file.

        Args:
            filename (str): File to dump the ascii art.
            encoding (str): Optional. Default "utf-8".
        """
        with open(filename, mode='w', encoding=encoding) as text_file:
            text_file.write(self.single_string())

    def lines(self, line_length=None):
        """Generates a list with lines. These lines form the text drawing.

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

        try:
            layers = self.build_layers()
        except TextDrawerCregBundle:
            self.cregbundle = False
            warn('The parameter "cregbundle" was disable, since an instruction needs to refer to '
                 'individual classical wires', RuntimeWarning, 2)
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
                    InputWire.fillup_layer(self.wire_names(with_initial_state=False)))
                rest_of_the_line -= layer_groups[-1][-1][0].length

                layer_groups[-1].append(layer)
                rest_of_the_line -= layer_groups[-1][-1][0].length

        lines = []
        for layer_group in layer_groups:
            wires = list(zip(*layer_group))
            lines += self.draw_wires(wires)

        return lines

    def wire_names(self, with_initial_state=False):
        """Returns a list of names for each wire.

        Args:
            with_initial_state (bool): Optional (Default: False). If true, adds
                the initial value to the name.

        Returns:
            List: The list of wire names.
        """
        if with_initial_state:
            initial_qubit_value = '|0>'
            initial_clbit_value = '0 '
        else:
            initial_qubit_value = ''
            initial_clbit_value = ''

        qubit_labels = []
        if self.layout is None:
            for bit in self.qregs:
                label = '{name}_{index}: ' + initial_qubit_value
                qubit_labels.append(label.format(name=bit.register.name,
                                                 index=bit.index,
                                                 physical=''))
        else:
            for bit in self.qregs:
                label = '{name}_{index} -> {physical} ' + initial_qubit_value
                qubit_labels.append(label.format(name=self.layout[bit.index].register.name,
                                                 index=self.layout[bit.index].index,
                                                 physical=bit.index))
        clbit_labels = []
        previous_creg = None
        for bit in self.cregs:
            if self.cregbundle:
                if previous_creg == bit.register:
                    continue
                previous_creg = bit.register
                label = '{name}: {initial_value}{size}/'
                clbit_labels.append(label.format(name=bit.register.name,
                                                 initial_value=initial_clbit_value,
                                                 size=bit.register.size))
            else:
                label = '{name}_{index}: ' + initial_clbit_value
                clbit_labels.append(label.format(name=bit.register.name, index=bit.index))
        return qubit_labels + clbit_labels

    def should_compress(self, top_line, bot_line):
        """Decides if the top_line and bot_line should be merged,
        based on `self.vertical_compression`."""
        if self.vertical_compression == 'high':
            return True
        if self.vertical_compression == 'low':
            return False
        for top, bot in zip(top_line, bot_line):
            if top == '┴' and bot == '┬':
                return False
        for line in (bot_line, top_line):
            no_spaces = line.replace(' ', '')
            if len(no_spaces) > 0 and all(c.isalpha() or c.isnumeric() for c in no_spaces):
                return False
        return True

    def draw_wires(self, wires):
        """Given a list of wires, creates a list of lines with the text drawing.

        Args:
            wires (list): A list of wires with instructions.
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
                if self.should_compress(top_line, bot_line):
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
        return "= %s" % instruction.condition[1]

    @staticmethod
    def params_for_label(instruction):
        """Get the params and format them to add them to a label. None if there
         are no params or if the params are numpy.ndarrays."""
        op = instruction.op
        if not hasattr(op, 'params'):
            return None
        if any([isinstance(param, ndarray) for param in op.params]):
            return None

        ret = []
        for param in op.params:
            try:
                str_param = pi_check(param, ndigits=5)
                ret.append('%s' % str_param)
            except TypeError:
                ret.append('%s' % param)
        return ret

    @staticmethod
    def special_label(instruction):
        """Some instructions have special labels"""
        labels = {IGate: 'I',
                  Initialize: 'initialize',
                  UnitaryGate: 'unitary',
                  HamiltonianGate: 'Hamiltonian'}
        instruction_type = type(instruction)
        if instruction_type in {Gate, Instruction}:
            return instruction.name
        return labels.get(instruction_type, None)

    @staticmethod
    def label_for_box(instruction, controlled=False):
        """ Creates the label for a box."""
        if controlled:
            if getattr(instruction.op.base_gate, 'label', None) is not None:
                return instruction.op.base_gate.label
            label = TextDrawing.special_label(
                instruction.op.base_gate) or instruction.op.base_gate.name.upper()
        else:
            if getattr(instruction.op, 'label', None) is not None:
                return instruction.op.label
            label = TextDrawing.special_label(instruction.op) or instruction.name.upper()
        params = TextDrawing.params_for_label(instruction)

        if params:
            label += "(%s)" % ','.join(params)
        return label

    @staticmethod
    def merge_lines(top, bot, icod="top"):
        """Merges two lines (top and bot) in the way that the overlapping make senses.

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
            elif topc in '║╫╬' and botc in " ":
                ret += "║"
            elif topc == '└' and botc == "┌" and icod == 'top':
                ret += "├"
            elif topc == '┘' and botc == "┐":
                ret += "┤"
            elif botc in "┐┌" and icod == 'top':
                ret += "┬"
            elif topc in "┘└" and botc in "─" and icod == 'top':
                ret += "┴"
            elif botc == " " and icod == 'top':
                ret += topc
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
        instructions = list(filter(lambda x: x is not None, layer))
        longest = max([instruction.length for instruction in instructions])
        for instruction in instructions:
            instruction.layer_width = longest

    @staticmethod
    def controlled_wires(instruction, layer):
        """
        Analyzes the instruction in the layer and checks if the controlled arguments are in
        the box or out of the box.

        Args:
            instruction (Instruction): instruction to analyse
            layer (Layer): The layer in which the instruction is inserted.

        Returns:
            Tuple(list, list, list):
              - tuple: controlled arguments on top of the "instruction box", and its status
              - tuple: controlled arguments on bottom of the "instruction box", and its status
              - tuple: controlled arguments in the "instruction box", and its status
              - the rest of the arguments
        """
        num_ctrl_qubits = instruction.op.num_ctrl_qubits
        ctrl_qubits = instruction.qargs[:num_ctrl_qubits]
        args_qubits = instruction.qargs[num_ctrl_qubits:]
        ctrl_state = "{0:b}".format(instruction.op.ctrl_state).rjust(num_ctrl_qubits, '0')[::-1]

        in_box = list()
        top_box = list()
        bot_box = list()

        qubit_index = sorted([i for i, x in enumerate(layer.qregs) if x in args_qubits])

        for ctrl_qubit in zip(ctrl_qubits, ctrl_state):
            if min(qubit_index) > layer.qregs.index(ctrl_qubit[0]):
                top_box.append(ctrl_qubit)
            elif max(qubit_index) < layer.qregs.index(ctrl_qubit[0]):
                bot_box.append(ctrl_qubit)
            else:
                in_box.append(ctrl_qubit)
        return (top_box, bot_box, in_box, args_qubits)

    def _set_ctrl_state(self, instruction, conditional, ctrl_label, bottom):
        """ Takes the ctrl_state from instruction and appends Bullet or OpenBullet
        to gates depending on whether the bit in ctrl_state is 1 or 0. Returns gates"""

        gates = []
        num_ctrl_qubits = instruction.op.num_ctrl_qubits
        ctrl_qubits = instruction.qargs[:num_ctrl_qubits]
        cstate = "{0:b}".format(instruction.op.ctrl_state).rjust(num_ctrl_qubits, '0')[::-1]
        for i in range(len(ctrl_qubits)):
            if cstate[i] == '1':
                gates.append(Bullet(conditional=conditional, label=ctrl_label,
                                    bottom=bottom))
            else:
                gates.append(OpenBullet(conditional=conditional, label=ctrl_label,
                                        bottom=bottom))
        return gates

    def _instruction_to_gate(self, instruction, layer):
        """ Convert an instruction into its corresponding Gate object, and establish
        any connections it introduces between qubits"""

        current_cons = []
        connection_label = None
        ctrl_label = None
        box_label = None
        conditional = False
        multi_qubit_instruction = len(instruction.qargs) >= 2 and not instruction.cargs
        label_multibox = False
        base_gate = getattr(instruction.op, 'base_gate', None)

        if instruction.condition is not None:
            # conditional
            cllabel = TextDrawing.label_for_conditional(instruction)
            layer.set_cl_multibox(instruction.condition[0], cllabel, top_connect='┴')
            conditional = True

        # add in a gate that operates over multiple qubits
        def add_connected_gate(instruction, gates, layer, current_cons):
            for i, gate in enumerate(gates):
                actual_index = self.qregs.index(instruction.qargs[i])
                if actual_index not in [i for i, j in current_cons]:
                    layer.set_qubit(instruction.qargs[i], gate)
                    current_cons.append((actual_index, gate))

        if multi_qubit_instruction and \
                getattr(instruction.op, 'label', None) is not None and \
                getattr(base_gate, 'label', None) is not None:
            # If a multi qubit instruction has a label, and the base gate has a
            # label, the label is applied to the bullet instead of the box.
            ctrl_label = getattr(instruction.op, 'label', None)
            box_label = getattr(base_gate, 'label', None)

        elif multi_qubit_instruction and \
                getattr(instruction.op, 'label', None) is not None:
            # If a multi qubit instruction has a label, it is a box
            label_multibox = True
            layer._set_multibox(instruction.op.label, qubits=instruction.qargs,
                                conditional=conditional)

        if isinstance(instruction.op, MeasureInstruction):
            gate = MeasureFrom()
            layer.set_qubit(instruction.qargs[0], gate)
            if self.cregbundle:
                layer.set_clbit(instruction.cargs[0], MeasureTo(str(instruction.cargs[0].index)))
            else:
                layer.set_clbit(instruction.cargs[0], MeasureTo())

        elif isinstance(instruction.op, (BarrierInstruction, Snapshot)):
            # barrier
            if not self.plotbarriers:
                return layer, current_cons, connection_label

            for qubit in instruction.qargs:
                if qubit in self.qregs:
                    layer.set_qubit(qubit, Barrier())
        elif isinstance(instruction.op, SwapGate):
            # swap
            gates = [Ex(conditional=conditional) for _ in range(len(instruction.qargs))]
            add_connected_gate(instruction, gates, layer, current_cons)

        elif isinstance(instruction.op, ResetInstruction):
            # reset
            layer.set_qubit(instruction.qargs[0], Reset(conditional=conditional))

        elif isinstance(instruction.op, RZZGate):
            # rzz
            connection_label = "zz(%s)" % TextDrawing.params_for_label(instruction)[0]
            gates = [Bullet(conditional=conditional), Bullet(conditional=conditional)]
            add_connected_gate(instruction, gates, layer, current_cons)

        elif len(instruction.qargs) == 1 and not instruction.cargs:
            # unitary gate
            layer.set_qubit(instruction.qargs[0],
                            BoxOnQuWire(TextDrawing.label_for_box(instruction),
                                        conditional=conditional))

        elif isinstance(instruction.op, ControlledGate) and not label_multibox:
            label = box_label if box_label is not None \
                else TextDrawing.label_for_box(instruction, controlled=True)
            params_array = TextDrawing.controlled_wires(instruction, layer)
            controlled_top, controlled_bot, controlled_edge, rest = params_array
            gates = self._set_ctrl_state(instruction, conditional, ctrl_label,
                                         bool(controlled_bot))
            if base_gate.name == 'z':
                # cz
                gates.append(Bullet(conditional=conditional))
            elif base_gate.name == 'u1':
                # cu1
                connection_label = TextDrawing.params_for_label(instruction)[0]
                gates.append(Bullet(conditional=conditional))
            elif base_gate.name == 'swap':
                # cswap
                gates += [Ex(conditional=conditional), Ex(conditional=conditional)]
                add_connected_gate(instruction, gates, layer, current_cons)
            elif base_gate.name == 'rzz':
                # crzz
                connection_label = "zz(%s)" % TextDrawing.params_for_label(instruction)[0]
                gates += [Bullet(conditional=conditional), Bullet(conditional=conditional)]
            elif len(rest) > 1:
                top_connect = '┴' if controlled_top else None
                bot_connect = '┬' if controlled_bot else None
                indexes = layer.set_qu_multibox(rest, label,
                                                conditional=conditional,
                                                controlled_edge=controlled_edge,
                                                top_connect=top_connect, bot_connect=bot_connect)
                for index in range(min(indexes), max(indexes) + 1):
                    # Dummy element to connect the multibox with the bullets
                    current_cons.append((index, DrawElement('')))
            elif base_gate.name == 'z':
                gates.append(Bullet(conditional=conditional))
            else:
                gates.append(BoxOnQuWire(label, conditional=conditional))
            add_connected_gate(instruction, gates, layer, current_cons)

        elif len(instruction.qargs) >= 2 and not instruction.cargs:
            # multiple qubit gate
            label = TextDrawing.label_for_box(instruction)
            layer.set_qu_multibox(instruction.qargs, label, conditional=conditional)

        elif instruction.qargs and instruction.cargs:
            # multiple gate, involving both qargs AND cargs
            label = TextDrawing.label_for_box(instruction)
            if self.cregbundle and instruction.cargs:
                raise TextDrawerCregBundle('TODO')
            layer._set_multibox(label, qubits=instruction.qargs, clbits=instruction.cargs,
                                conditional=conditional)
        else:
            raise VisualizationError(
                "Text visualizer does not know how to handle this instruction: ", instruction.name)

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
        wire_names = self.wire_names(with_initial_state=self.initial_state)
        if not wire_names:
            return []

        layers = [InputWire.fillup_layer(wire_names)]

        for instruction_layer in self.instructions:
            layer = Layer(self.qregs, self.cregs, self.cregbundle)

            for instruction in instruction_layer:
                layer, current_connections, connection_label = \
                    self._instruction_to_gate(instruction, layer)

                layer.connections.append((connection_label, current_connections))
                layer.connect_with("│")
            layers.append(layer.full_layer)

        return layers


class Layer:
    """ A layer is the "column" of the circuit. """

    def __init__(self, qregs, cregs, cregbundle=False):
        self.qregs = qregs
        if cregbundle:
            self.cregs = []
            previous_creg = None
            for bit in cregs:
                if previous_creg == bit.register:
                    continue
                previous_creg = bit.register
                self.cregs.append(bit.register)
        else:
            self.cregs = cregs
        self.qubit_layer = [None] * len(qregs)
        self.connections = []
        self.clbit_layer = [None] * len(cregs)
        self.cregbundle = cregbundle

    @property
    def full_layer(self):
        """
        Returns the composition of qubits and classic wires.
        Returns:
            String: self.qubit_layer + self.clbit_layer
        """
        return self.qubit_layer + self.clbit_layer

    def set_qubit(self, qubit, element):
        """Sets the qubit to the element.

        Args:
            qubit (qbit): Element of self.qregs.
            element (DrawElement): Element to set in the qubit
        """
        self.qubit_layer[self.qregs.index(qubit)] = element

    def set_clbit(self, clbit, element):
        """Sets the clbit to the element.

        Args:
            clbit (cbit): Element of self.cregs.
            element (DrawElement): Element to set in the clbit
        """
        if self.cregbundle:
            self.clbit_layer[self.cregs.index(clbit.register)] = element
        else:
            self.clbit_layer[self.cregs.index(clbit)] = element

    def _set_multibox(self, label, qubits=None, clbits=None, top_connect=None,
                      bot_connect=None, conditional=False, controlled_edge=None):
        if qubits is not None and clbits is not None:
            qubits = list(qubits)
            clbits = list(clbits)
            cbit_index = sorted([i for i, x in enumerate(self.cregs) if x in clbits])
            qbit_index = sorted([i for i, x in enumerate(self.qregs) if x in qubits])

            # Further below, indices are used as wire labels. Here, get the length of
            # the longest label, and pad all labels with spaces to this length.
            wire_label_len = max(len(str(len(qubits) - 1)),
                                 len(str(len(clbits) - 1)))
            qargs = [str(qubits.index(qbit)).ljust(wire_label_len, ' ')
                     for qbit in self.qregs if qbit in qubits]
            cargs = [str(clbits.index(cbit)).ljust(wire_label_len, ' ')
                     for cbit in self.cregs if cbit in clbits]

            box_height = len(self.qregs) - min(qbit_index) + max(cbit_index) + 1

            self.set_qubit(qubits.pop(0), BoxOnQuWireTop(label, wire_label=qargs.pop(0)))
            order = 0
            for order, bit_i in enumerate(range(min(qbit_index) + 1, len(self.qregs))):
                if bit_i in qbit_index:
                    named_bit = qubits.pop(0)
                    wire_label = qargs.pop(0)
                else:
                    named_bit = self.qregs[bit_i]
                    wire_label = ' ' * wire_label_len
                self.set_qubit(named_bit, BoxOnQuWireMid(label, box_height, order,
                                                         wire_label=wire_label))
            for order, bit_i in enumerate(range(max(cbit_index)), order + 1):
                if bit_i in cbit_index:
                    named_bit = clbits.pop(0)
                    wire_label = cargs.pop(0)
                else:
                    named_bit = self.cregs[bit_i]
                    wire_label = ' ' * wire_label_len
                self.set_clbit(named_bit, BoxOnClWireMid(label, box_height, order,
                                                         wire_label=wire_label))
            self.set_clbit(clbits.pop(0),
                           BoxOnClWireBot(label, box_height, wire_label=cargs.pop(0)))
            return cbit_index
        if qubits is None and clbits is not None:
            bits = list(clbits)
            bit_index = sorted([i for i, x in enumerate(self.cregs) if x in bits])
            wire_label_len = len(str(len(bits) - 1))
            bits.sort(key=self.cregs.index)
            qargs = [''] * len(bits)
            set_bit = self.set_clbit
            OnWire = BoxOnClWire
            OnWireTop = BoxOnClWireTop
            OnWireMid = BoxOnClWireMid
            OnWireBot = BoxOnClWireBot
        elif clbits is None and qubits is not None:
            bits = list(qubits)
            bit_index = sorted([i for i, x in enumerate(self.qregs) if x in bits])
            wire_label_len = len(str(len(bits) - 1))
            qargs = [str(bits.index(qbit)).ljust(wire_label_len, ' ')
                     for qbit in self.qregs if qbit in bits]
            bits.sort(key=self.qregs.index)
            set_bit = self.set_qubit
            OnWire = BoxOnQuWire
            OnWireTop = BoxOnQuWireTop
            OnWireMid = BoxOnQuWireMid
            OnWireBot = BoxOnQuWireBot
        else:
            raise VisualizationError("_set_multibox error!.")

        control_index = {}
        if controlled_edge:
            for index, qubit in enumerate(self.qregs):
                for qubit_in_edge, value in controlled_edge:
                    if qubit == qubit_in_edge:
                        control_index[index] = '■' if value == '1' else 'o'
        if len(bit_index) == 1:
            set_bit(bits[0], OnWire(label, top_connect=top_connect))
        else:
            box_height = max(bit_index) - min(bit_index) + 1
            set_bit(bits.pop(0), OnWireTop(label, top_connect=top_connect, wire_label=qargs.pop(0)))
            for order, bit_i in enumerate(range(min(bit_index) + 1, max(bit_index))):
                if bit_i in bit_index:
                    named_bit = bits.pop(0)
                    wire_label = qargs.pop(0)
                else:
                    named_bit = (self.qregs + self.cregs)[bit_i]
                    wire_label = ' ' * wire_label_len

                control_label = control_index.get(bit_i)
                set_bit(named_bit, OnWireMid(label, box_height, order, wire_label=wire_label,
                                             control_label=control_label))
            set_bit(bits.pop(0), OnWireBot(label, box_height, bot_connect=bot_connect,
                                           wire_label=qargs.pop(0), conditional=conditional))
        return bit_index

    def set_cl_multibox(self, creg, label, top_connect='┴'):
        """Sets the multi clbit box.

        Args:
            creg (string): The affected classical register.
            label (string): The label for the multi clbit box.
            top_connect (char): The char to connect the box on the top.
        """
        if self.cregbundle:
            self.set_clbit(creg[0], BoxOnClWire(label=label, top_connect=top_connect))
        else:
            clbit = [bit for bit in self.cregs if bit.register == creg]
            self._set_multibox(label, clbits=clbit, top_connect=top_connect)

    def set_qu_multibox(self, bits, label, top_connect=None, bot_connect=None,
                        conditional=False, controlled_edge=None):
        """Sets the multi qubit box.

        Args:
            bits (list[int]): A list of affected bits.
            label (string): The label for the multi qubit box.
            top_connect (char): None or a char connector on the top
            bot_connect (char): None or a char connector on the bottom
            conditional (bool): If the box has a conditional
            controlled_edge (list): A list of bit that are controlled (to draw them at the edge)
        Return:
            List: A list of indexes of the box.
        """
        return self._set_multibox(label, qubits=bits, top_connect=top_connect,
                                  bot_connect=bot_connect,
                                  conditional=conditional, controlled_edge=controlled_edge)

    def connect_with(self, wire_char):
        """Connects the elements in the layer using wire_char.

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
