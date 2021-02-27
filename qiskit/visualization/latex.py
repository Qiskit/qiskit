# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name,consider-using-enumerate

"""latex visualization backends."""

import collections
import io
import math
import re

import numpy as np
from qiskit.circuit import Gate, Instruction
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.visualization.qcstyle import DefaultStyle
from qiskit.visualization import exceptions
from qiskit.circuit.tools.pi_check import pi_check
from .utils import generate_latex_label


class QCircuitImage:
    """This class contains methods to create \\LaTeX circuit images.

    The class targets the \\LaTeX package Q-circuit
    (https://arxiv.org/pdf/quant-ph/0406003).

    Thanks to Eric Sabo for the initial implementation for Qiskit.
    """

    def __init__(self, qubits, clbits, ops, scale,
                 plot_barriers=True, layout=None, initial_state=False,
                 cregbundle=False, global_phase=None):
        """QCircuitImage initializer.

        Args:
            qubits (list[Qubit]): list of qubits
            clbits (list[Clbit]): list of clbits
            ops (list[list[DAGNode]]): list of circuit instructions, grouped by layer
            scale (float): image scaling
            plot_barriers (bool): Enable/disable drawing barriers in the output
               circuit. Defaults to True.
            layout (Layout or None): If present, the layout information will be
               included.
            initial_state (bool): Optional. Adds |0> in the beginning of the line. Default: `False`.
            cregbundle (bool): Optional. If set True bundle classical registers. Default: `False`.
            global_phase (float): Optional, the global phase for the circuit.
        Raises:
            ImportError: If pylatexenc is not installed
        """
        # list of lists corresponding to layers of the circuit
        self.ops = ops

        # image scaling
        self.scale = 0.7 if scale is None else scale

        # Map of qregs to sizes
        self.qregs = {}

        # Map of cregs to sizes
        self.cregs = {}

        # List of qregs and cregs in order of appearance in code and image
        self.ordered_regs = []

        # Map from registers to the list they appear in the image
        self.img_regs = {}

        # Array to hold the \\LaTeX commands to generate a circuit image.
        self._latex = []

        # Variable to hold image depth (width)
        self.img_depth = 0

        # Variable to hold image width (height)
        self.img_width = 0

        # Variable to hold total circuit depth
        self.sum_column_widths = 0

        # Variable to hold total circuit width
        self.sum_wire_heights = 0

        # em points of separation between circuit columns
        self.column_separation = 1

        # em points of separation between circuit wire
        self.wire_separation = 0

        # presence of "box" or "target" determines wire spacing
        self.has_box = False
        self.has_target = False
        self.layout = layout
        self.initial_state = initial_state
        self.plot_barriers = plot_barriers

        #################################
        self.qregs = self._get_register_specs(qubits)
        self.qubit_list = qubits
        self.ordered_regs = qubits + clbits
        self.cregs = self._get_register_specs(clbits)
        self.clbit_list = clbits
        self.img_regs = {bit: ind for ind, bit in
                         enumerate(self.ordered_regs)}
        if cregbundle:
            self.img_width = len(qubits) + len(self.cregs)
        else:
            self.img_width = len(self.img_regs)
        self.wire_type = {}
        for bit in self.ordered_regs:
            self.wire_type[bit] = bit.register in self.cregs.keys()
        self.cregbundle = cregbundle
        self.global_phase = global_phase

        self._style = DefaultStyle().style

    def latex(self):
        """Return LaTeX string representation of circuit."""

        self._initialize_latex_array()
        self._build_latex_array()
        header_1 = r"""% \documentclass[preview]{standalone}
% If the image is too large to fit on this documentclass use
\documentclass[draft]{beamer}
"""
        beamer_line = "\\usepackage[size=custom,height=%d,width=%d,scale=%.1f]{beamerposter}\n"
        header_2 = r"""% instead and customize the height and width (in cm) to fit.
% Large images may run out of memory quickly.
% To fix this use the LuaLaTeX compiler, which dynamically
% allocates memory.
\usepackage[braket, qm]{qcircuit}
\usepackage{amsmath}
\pdfmapfile{+sansmathaccent.map}
% \usepackage[landscape]{geometry}
% Comment out the above line if using the beamer documentclass.
\begin{document}
"""
        qcircuit_line = r"""
\begin{equation*}
    \Qcircuit @C=%.1fem @R=%.1fem @!R {
"""
        output = io.StringIO()
        output.write(header_1)
        output.write('%% img_width = %d, img_depth = %d\n' % (self.img_width, self.img_depth))
        output.write(beamer_line % self._get_beamer_page())
        output.write(header_2)
        if self.global_phase:
            output.write(r"""{$\mathrm{%s} \mathrm{%s}$}"""
                         % ('global\\,phase:\\,', pi_check(self.global_phase, output='latex')))
        output.write(qcircuit_line %
                     (self.column_separation, self.wire_separation))
        for i in range(self.img_width):
            output.write("\t \t")
            for j in range(self.img_depth + 1):
                output.write(self._latex[i][j])
                if j != self.img_depth:
                    output.write(" & ")
                else:
                    output.write(r'\\' + '\n')
        output.write('\t }\n')
        output.write('\\end{equation*}\n\n')
        output.write('\\end{document}')
        contents = output.getvalue()
        output.close()
        return contents

    def _initialize_latex_array(self):
        """Initialize qubit and clbit labels and set wire separation"""
        self.img_depth, self.sum_column_widths = self._get_image_depth()
        self.sum_wire_heights = self.img_width
        # choose the most compact wire spacing, while not squashing them
        if self.has_box:
            self.wire_separation = 0.2
        elif self.has_target:
            self.wire_separation = 0.8
        else:
            self.wire_separation = 1.0
        self._latex = [
            ["\\cw" if self.wire_type[self.ordered_regs[j]]
             else "\\qw" for _ in range(self.img_depth + 1)]
            for j in range(self.img_width)]
        self._latex.append([" "] * (self.img_depth + 1))
        if self.cregbundle:
            offset = 0
        for i in range(self.img_width):
            if self.wire_type[self.ordered_regs[i]]:
                if self.cregbundle:
                    self._latex[i][0] = \
                        "\\lstick{" + self.ordered_regs[i + offset].register.name + ":"
                    clbitsize = self.cregs[self.ordered_regs[i + offset].register]
                    self._latex[i][1] = "\\lstick{/_{_{" + str(clbitsize) + "}}} \\cw"
                    offset += clbitsize - 1
                else:
                    self._latex[i][0] = "\\lstick{" + self.ordered_regs[i].register.name + \
                                            "_{" + str(self.ordered_regs[i].index) + "}:"
                if self.initial_state:
                    self._latex[i][0] += "0"
                self._latex[i][0] += "}"
            else:
                if self.layout is None:
                    label = "\\lstick{{ {{{}}}_{{{}}} : ".format(
                        self.ordered_regs[i].register.name, self.ordered_regs[i].index)
                else:
                    if self.layout[self.ordered_regs[i].index]:
                        label = "\\lstick{{ {{{}}}_{{{}}}\\mapsto{{{}}} : ".format(
                            self.layout[self.ordered_regs[i].index].register.name,
                            self.layout[self.ordered_regs[i].index].index,
                            self.ordered_regs[i].index)
                    else:
                        label = "\\lstick{{ {{{}}} : ".format(self.ordered_regs[i].index)
                if self.initial_state:
                    label += "\\ket{{0}}"
                label += " }"
                self._latex[i][0] = label

    def _get_image_depth(self):
        """Get depth information for the circuit."""
        max_column_widths = []
        # Determine wire spacing before image depth
        for layer in self.ops:
            for op in layer:
                # useful information for determining wire spacing
                boxed_gates = ['u1', 'u2', 'u3', 'u', 'p', 'x', 'y', 'z', 'h',
                               's', 'sdg', 't', 'tdg', 'sx', 'sxdg', 'rx', 'ry', 'rz',
                               'ch', 'cy', 'crz', 'cu2', 'cu3', 'cu', 'id']
                target_gates = ['cx', 'ccx', 'cu1', 'cp', 'rzz']
                if op.name in boxed_gates:
                    self.has_box = True
                elif op.name in target_gates:
                    self.has_target = True
                elif isinstance(op.op, ControlledGate):
                    self.has_box = True

        for layer in self.ops:

            current_max = 0
            for op in layer:
                arg_str_len = 0

                # the wide gates
                for arg in op.op.params:
                    if not any([isinstance(param, np.ndarray) for param in op.op.params]):
                        arg_str = re.sub(r'[-+]?\d*\.\d{2,}|\d{2,}',
                                         self._truncate_float, str(arg))
                        arg_str_len += len(arg_str)

                # the width of the column is the max of all the gates in the column
                current_max = max(arg_str_len, current_max)

            max_column_widths.append(current_max)

        # wires in the beginning and end
        columns = 2
        if self.cregbundle and (self.ops and self.ops[0] and
                                (self.ops[0][0].name == "measure" or self.ops[0][0].condition)):
            columns += 1

        # all gates take up 1 column except from those with labels (ie cu1, cp, rzz)
        # which take 3 columns
        for layer in self.ops:
            column_width = 1
            for op in layer:
                base_name = None if not hasattr(op.op, 'base_gate') else op.op.base_gate.name
                if op.name == 'rzz' or base_name in ['u1', 'p', 'rzz']:
                    column_width = 4
            columns += column_width

        # every 3 characters is roughly one extra 'unit' of width in the cell
        # the gate name is 1 extra 'unit'
        # the qubit/cbit labels plus initial states is 2 more
        # the wires poking out at the ends is 2 more
        sum_column_widths = sum(1 + v / 3 for v in max_column_widths)

        max_reg_name = 3
        for reg in self.ordered_regs:
            max_reg_name = max(max_reg_name,
                               len(reg.register.name))
        sum_column_widths += 5 + max_reg_name / 3

        # could be a fraction so ceil
        return columns, math.ceil(sum_column_widths)

    def _get_beamer_page(self):
        """Get height, width & scale attributes for the beamer page."""

        # PIL python package limits image size to around a quarter gigabyte
        # this means the beamer image should be limited to < 50000
        # if you want to avoid a "warning" too, set it to < 25000
        PIL_limit = 40000

        # the beamer latex template limits each dimension to < 19 feet
        # (i.e. 575cm)
        beamer_limit = 550

        # columns are roughly twice as big as wires
        aspect_ratio = self.sum_wire_heights / self.sum_column_widths

        # choose a page margin so circuit is not cropped
        margin_factor = 1.5
        height = min(self.sum_wire_heights * margin_factor, beamer_limit)
        width = min(self.sum_column_widths * margin_factor, beamer_limit)

        # if too large, make it fit
        if height * width > PIL_limit:
            height = min(np.sqrt(PIL_limit * aspect_ratio), beamer_limit)
            width = min(np.sqrt(PIL_limit / aspect_ratio), beamer_limit)

        # if too small, give it a minimum size
        height = max(height, 10)
        width = max(width, 10)

        return (height, width, self.scale)

    def _get_gate_ctrl_text(self, op):
        """Load the gate_text and ctrl_text strings based on names and labels"""
        op_label = getattr(op.op, 'label', None)
        op_type = type(op.op)
        if hasattr(op.op, 'base_gate'):
            base_name = op.op.base_gate.name
            base_label = op.op.base_gate.label
            base_type = type(op.op.base_gate)
        else:
            base_name = base_label = base_type = None
        ctrl_text = None

        if base_label:
            gate_text = base_label
            ctrl_text = op_label
        elif op_label and isinstance(op.op, ControlledGate):
            gate_text = base_name
            ctrl_text = op_label
        elif op_label:
            gate_text = op_label
        elif base_name:
            gate_text = base_name
        else:
            gate_text = op.name

        if gate_text in self._style['disptex']:
            gate_text = self._style['disptex'][gate_text]
            # Only add mathmode formatting if not already mathmode in disptex
            if gate_text[0] != '$' and gate_text[-1] != '$':
                gate_text = "$\\mathrm{{{}}}$".format(gate_text)

        # Only captitalize internally-created gate or instruction names
        elif ((gate_text == op.name and op_type not in (Gate, Instruction))
              or (gate_text == base_name and base_type not in (Gate, Instruction))):
            gate_text = "$\\mathrm{{{}}}$".format(gate_text.capitalize())
        else:
            gate_text = "$\\mathrm{{{}}}$".format(gate_text)
            # Remove mathmode _, ^, and - formatting from user names and labels
            gate_text = gate_text.replace('_', '\\_').replace('^', '\\string^')
            gate_text = gate_text.replace('-', '\\mbox{-}')

        ctrl_text = "$\\mathrm{{{}}}$".format(ctrl_text)
        return gate_text, ctrl_text

    def _build_latex_array(self):
        """Returns an array of strings containing \\LaTeX for this circuit.
        """
        column = 1
        # Leave a column to display number of classical registers if needed
        if self.cregbundle and (self.ops and self.ops[0] and
                                (self.ops[0][0].name == "measure" or self.ops[0][0].condition)):
            column += 1

        for layer in self.ops:
            num_cols_used = 1

            for op in layer:
                base_name = None if not hasattr(op.op, 'base_gate') else op.op.base_gate.name
                if op.name == "measure":
                    self._build_measure(op, column)

                elif op.name in ['barrier', 'snapshot', 'load', 'save', 'noise']:
                    self._build_barrier(op, column)

                else:
                    gate_text, _ = self._get_gate_ctrl_text(op)
                    gate_text = self._add_params_to_gate_text(op, gate_text)
                    gate_text = generate_latex_label(gate_text).replace(" ", "\\,")

                    wire_list = []
                    for wire in op.qargs:
                        wire_list.append(self.img_regs[wire])

                    if op.condition:
                        self._add_condition(op, wire_list, column)

                    if len(wire_list) == 1:
                        self._latex[wire_list[0]][column] = "\\gate{%s}" % gate_text

                    elif len(wire_list) == 2:
                        if op.name == "swap":
                            self._build_swap(wire_list, column)

                        elif op.name == 'rzz' or base_name in ['u1', 'p', 'rzz']:
                            symm_name = 'rzz' if op.name == 'rzz' else base_name
                            num_cols_used = self._build_symmetric_gate(op, symm_name, gate_text,
                                                                       wire_list, column,
                                                                       num_cols_used)
                        elif isinstance(op.op, ControlledGate):
                            self._build_single_ctrl_gate(op, gate_text, wire_list, column)

                        else:
                            self._build_multi_gate(gate_text, wire_list, column)

                    elif isinstance(op.op, ControlledGate):
                        num_cols_used = self._build_multi_ctrl_gate(op, gate_text,
                                                                    wire_list, column,
                                                                    num_cols_used)
                    else:
                        self._build_multi_gate(gate_text, wire_list, column)

            column += num_cols_used

    def _build_multi_gate(self, gate_text, wire_list, col):
        """Add a multiple wire gate to the _latex list"""
        wire1 = min(wire_list)
        wire2 = max(wire_list)
        self._latex[wire1][col] = "\\multigate{%s}{%s}" % \
            (wire2 - wire1, gate_text)
        for wire in range(wire1 + 1, wire2 + 1):
            self._latex[wire][col] = "\\ghost{%s}" % gate_text

    def _build_single_ctrl_gate(self, op, gate_text, wire_list, col):
        """Add a gate with a single control to the _latex list"""
        wire1 = min(wire_list)
        wire2 = max(wire_list)
        cond = "{:b}".format(op.op.ctrl_state).rjust(1, '0')[::-1]
        if cond == '0':
            self._latex[wire1][col] = "\\ctrlo{" + str(wire2 - wire1) + "}"
        elif cond == '1':
            self._latex[wire1][col] = "\\ctrl{" + str(wire2 - wire1) + "}"

        if op.op.base_gate.name == 'x':
            self._latex[wire2][col] = "\\targ"
        elif op.op.base_gate.name == 'z':
            self._latex[wire2][col] = "\\control\\qw"
        else:
            self._latex[wire2][col] = "\\gate{%s}" % gate_text

    def _build_multi_ctrl_gate(self, op, gate_text, wire_list, col, num_cols_used):
        """Add a gate with multiple controls to the _latex list"""
        num_ctrl_qubits = op.op.num_ctrl_qubits
        wireqargs = wire_list[num_ctrl_qubits:]
        ctrlqargs = wire_list[:num_ctrl_qubits]
        wire1 = min(wireqargs)
        wire2 = max(wireqargs)
        ctrl_state = "{:b}".format(op.op.ctrl_state).rjust(num_ctrl_qubits, '0')[::-1]

        # First do single qubit gates
        if len(wireqargs) == 1:
            self._add_multi_controls(wire_list, ctrlqargs, ctrl_state, col)
            if op.op.base_gate.name == 'x':
                self._latex[wireqargs[0]][col] = "\\targ"
            else:
                self._latex[wireqargs[0]][col] = "\\gate{%s}" % gate_text
        else:
            if op.op.base_gate.name in ['swap', 'rzz']:
                self._add_multi_controls(wire_list, ctrlqargs, ctrl_state, col)
                if op.op.base_gate.name == 'swap':
                    self._build_swap(wireqargs, col)
                elif op.op.base_gate.name == 'rzz':
                    num_cols_used = self._build_symmetric_gate(op, 'rzz', gate_text,
                                                               wire_list, col, num_cols_used)
            else:
                # If any controls appear in the span of the multiqubit
                # gate just treat the whole thing as a big gate
                for ctrl in ctrlqargs:
                    if ctrl in range(wire1, wire2):
                        wireqargs = wire_list
                        break
                else:
                    self._add_multi_controls(wire_list, ctrlqargs, ctrl_state, col)

                self._build_multi_gate(gate_text, wireqargs, col)
        return num_cols_used

    def _build_swap(self, wire_list, col):
        """Add a swap gate"""
        wire1 = min(wire_list)
        wire2 = max(wire_list)
        self._latex[wire1][col] = "\\qswap"
        self._latex[wire2][col] = "\\qswap \\qwx[" + str(wire1 - wire2) + "]"

    def _build_symmetric_gate(self, op, symm_name, gate_text, wire_list, col, num_cols_used):
        """Add symmetric gates for cu1, cp, and rzz"""
        wire1 = min(wire_list)
        wire2 = max(wire_list)
        cond = '1' if symm_name == 'rzz' else "{:b}".format(op.op.ctrl_state).rjust(1, '0')[::-1]

        if cond == '0':
            self._latex[wire1][col] = "\\ctrlo{" + str(wire2 - wire1) + "}"
        elif cond == '1':
            self._latex[wire1][col] = "\\ctrl{" + str(wire2 - wire1) + "}"

        self._latex[wire2][col] = "\\control \\qw"
        self._latex[wire1][col+1] = "\\dstick{\\hspace{2.0em}%s} \\qw" % gate_text
        return max(num_cols_used, 4)

    def _build_measure(self, op, col):
        """Build a meter and the lines to the creg"""
        if op.condition:
            raise exceptions.VisualizationError(
                "If controlled measures currently not supported.")

        wire1 = self.img_regs[op.qargs[0]]
        if self.cregbundle:
            wire2 = self.img_regs[self.clbit_list[0]]
            cregindex = self.img_regs[op.cargs[0]] - wire2
            for creg_size in self.cregs.values():
                if cregindex >= creg_size:
                    cregindex -= creg_size
                    wire2 += 1
                else:
                    break
        else:
            wire2 = self.img_regs[op.cargs[0]]

        self._latex[wire1][col] = "\\meter"
        if self.cregbundle:
            self._latex[wire2][col] = \
                "\\dstick{_{_{%s}}} \\cw \\cwx[-%s]" % \
                (str(cregindex), str(wire2-wire1))
        else:
            self._latex[wire2][col] = \
                "\\control \\cw \\cwx[-" + str(wire2 - wire1) + "]"

    def _build_barrier(self, op, col):
        """Build a partial or full barrier if plot_barriers set"""
        if self.plot_barriers:
            indexes = [self._get_qubit_index(x) for x in op.qargs]
            indexes.sort()
            first = last = indexes[0]
            for index in indexes[1:]:
                if index - 1 == last:
                    last = index
                else:
                    pos = self.img_regs[self.qubit_list[first]]
                    self._latex[pos][col - 1] += " \\barrier[0em]{" + str(
                        last - first) + "}"
                    self._latex[pos][col] = "\\qw"
                    first = last = index
            pos = self.img_regs[self.qubit_list[first]]
            self._latex[pos][col - 1] += " \\barrier[0em]{" + str(
                last - first) + "}"
            self._latex[pos][col] = "\\qw"

    def _add_multi_controls(self, wire_list, ctrlqargs, ctrl_state, col):
        """Add more than one control to a gate"""
        for index, ctrl_item in enumerate(zip(ctrlqargs, ctrl_state)):
            pos = ctrl_item[0]
            cond = ctrl_item[1]
            nxt = wire_list[index]
            if wire_list[index] > wire_list[-1]:
                nxt -= 1
                while nxt not in wire_list:
                    nxt -= 1
            else:
                nxt += 1
                while nxt not in wire_list:
                    nxt += 1
            if cond == '0':
                self._latex[pos][col] = "\\ctrlo{" + str(
                    nxt - wire_list[index]) + "}"
            elif cond == '1':
                self._latex[pos][col] = "\\ctrl{" + str(
                    nxt - wire_list[index]) + "}"

    def _add_params_to_gate_text(self, op, gate_text):
        """Add the params to the end of the current gate_text"""
        # Must limit to 4 params or may get dimension too large error
        # from xy-pic xymatrix command
        if (len(op.op.params) > 0 and not any(
                [isinstance(param, np.ndarray) for param in op.op.params])):
            gate_text += "\\,\\mathrm{(}"
            if len(op.op.params) > 4:
                gate_text += '\\mathrm{>4\\,parameters}'
            else:
                for param in op.op.params:
                    gate_text += "\\mathrm{%s}," % pi_check(param, output='latex', ndigits=4)
                gate_text = gate_text[:-1]
            gate_text += "\\mathrm{)}"
        return gate_text

    def _add_condition(self, op, wire_list, col):
        """Add a condition to the _latex list"""
        # cwire - the wire number for the first wire for the condition register
        #      or if cregbundle, wire number of the condition register itself
        # if_value - a bit string for the condition
        # gap - the number of wires from cwire to the gate qubit
        mask = self._get_mask(op.condition[0])
        cl_reg = self.clbit_list[self._ffs(mask)]
        if_reg = cl_reg.register

        # for cregbundle, start at first register and increment cwire until it hits
        # the conditional register
        if self.cregbundle:
            cwire = len(self.qubit_list)
            for creg in self.cregs:
                if if_reg == creg[0].register:
                    break
                cwire += 1
        else:
            # otherwise select the first bit wire for the condition register
            # instead of the register wire
            cwire = self.img_regs[if_reg[cl_reg.index]]

        if_value = format(op.condition[1], 'b').zfill(self.cregs[if_reg])[::-1]
        temp = wire_list + [cwire]
        temp.sort(key=int)
        bottom = temp[len(wire_list) - 1]
        gap = cwire - bottom
        creg_rng = 1 if self.cregbundle else self.cregs[if_reg]

        for i in range(creg_rng):
            if self.cregbundle:
                # Print the condition value at the bottom
                self._latex[cwire + i][col] = \
                    "\\dstick{_{_{=%s}}} \\cw \\cwx[-%s]" % (str(op.condition[1]), str(gap))
            else:
                if if_value[i] == '1':
                    self._latex[cwire + i][col] = \
                        "\\control \\cw \\cwx[-" + str(gap) + "]"
                    gap = 1
                else:
                    self._latex[cwire + i][col] = \
                        "\\controlo \\cw \\cwx[-" + str(gap) + "]"
                    gap = 1

    def _get_mask(self, creg_name):
        """Get the clbit bit mask"""
        mask = 0
        for index, cbit in enumerate(self.clbit_list):
            if creg_name == cbit.register:
                mask |= (1 << index)
        return mask

    def _get_qubit_index(self, qubit):
        """Get the index number for a quantum bit."""
        for i, bit in enumerate(self.qubit_list):
            if qubit == bit:
                qindex = i
                break
        else:
            raise exceptions.VisualizationError("unable to find bit for operation")
        return qindex

    def _ffs(self, mask):
        """Find index of first set bit."""
        origin = (mask & (-mask)).bit_length()
        return origin - 1

    def _get_register_specs(self, bits):
        """Get the number and size of unique registers from bits list."""
        regs = collections.OrderedDict([(bit.register, bit.register.size) for bit in bits])
        return regs

    def _truncate_float(self, matchobj, ndigits=4):
        """Truncate long floats."""
        if matchobj.group(0):
            return '%.{}g'.format(ndigits) % float(matchobj.group(0))
        return ''
