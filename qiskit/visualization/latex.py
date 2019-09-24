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

# pylint: disable=invalid-name

"""latex circuit visualization backends."""

import collections
import io
import json
import math
import re

try:
    from pylatexenc.latexencode import utf8tolatex

    HAS_PYLATEX = True
except ImportError:
    HAS_PYLATEX = False

import numpy as np
from qiskit.visualization import qcstyle as _qcstyle
from qiskit.visualization import exceptions
from .tools.pi_check import pi_check


class QCircuitImage:
    """This class contains methods to create \\LaTeX circuit images.

    The class targets the \\LaTeX package Q-circuit
    (https://arxiv.org/pdf/quant-ph/0406003).

    Thanks to Eric Sabo for the initial implementation for Qiskit.
    """

    def __init__(self, qubits, clbits, ops, scale, style=None,
                 plot_barriers=True, reverse_bits=False, layout=None):
        """
        Args:
            qubits (list[Qubit]): list of qubits
            clbits (list[Clbit]): list of clbits
            ops (list[list[DAGNode]]): list of circuit instructions, grouped by layer
            scale (float): image scaling
            style (dict or str): dictionary of style or file name of style file
            reverse_bits (bool): When set to True reverse the bit order inside
               registers for the output visualization.
            plot_barriers (bool): Enable/disable drawing barriers in the output
               circuit. Defaults to True.
            layout (Layout or None): If present, the layout information will be
               included.
        Raises:
            ImportError: If pylatexenc is not installed
        """
        if not HAS_PYLATEX:
            raise ImportError('The latex and latex_source drawers need '
                              'pylatexenc installed. Run "pip install '
                              'pylatexenc" before using the latex or '
                              'latex_source drawers.')
        # style sheet
        self._style = _qcstyle.BWStyle()
        if style:
            if isinstance(style, dict):
                self._style.set_style(style)
            elif isinstance(style, str):
                with open(style, 'r') as infile:
                    dic = json.load(infile)
                self._style.set_style(dic)

        # list of lists corresponding to layers of the circuit
        self.ops = ops

        # image scaling
        self.scale = scale

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
        self.sum_row_heights = 0

        # em points of separation between circuit columns
        self.column_separation = 1

        # em points of separation between circuit row
        self.row_separation = 0

        # presence of "box" or "target" determines row spacing
        self.has_box = False
        self.has_target = False
        self.reverse_bits = reverse_bits
        self.layout = layout
        self.plot_barriers = plot_barriers

        #################################
        self.qregs = _get_register_specs(qubits)
        self.qubit_list = qubits
        self.ordered_regs = qubits + clbits
        self.cregs = _get_register_specs(clbits)
        self.clbit_list = clbits
        self.img_regs = {bit: ind for ind, bit in
                         enumerate(self.ordered_regs)}
        self.img_width = len(self.img_regs)
        self.wire_type = {}
        for bit in self.ordered_regs:
            self.wire_type[bit] = bit.register in self.cregs.keys()

    def latex(self, aliases=None):
        """Return LaTeX string representation of circuit.

        This method uses the LaTeX Qconfig package to create a graphical
        representation of the circuit.

        Returns:
            string: for writing to a LaTeX file.
        """
        self._initialize_latex_array(aliases)
        self._build_latex_array(aliases)
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
\begin{equation*}"""
        qcircuit_line = r"""
    \Qcircuit @C=%.1fem @R=%.1fem @!R {
"""
        output = io.StringIO()
        output.write(header_1)
        output.write('%% img_width = %d, img_depth = %d\n' % (self.img_width, self.img_depth))
        output.write(beamer_line % self._get_beamer_page())
        output.write(header_2)
        output.write(qcircuit_line %
                     (self.column_separation, self.row_separation))
        for i in range(self.img_width):
            output.write("\t \t")
            for j in range(self.img_depth + 1):
                cell_str = self._latex[i][j]
                # Don't truncate offset float if drawing a barrier
                if 'barrier' in cell_str:
                    output.write(cell_str)
                else:
                    # floats can cause "Dimension too large" latex error in
                    # xymatrix this truncates floats to avoid issue.
                    cell_str = re.sub(r'[-+]?\d*\.\d{2,}|\d{2,}',
                                      _truncate_float,
                                      cell_str)
                    output.write(cell_str)
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

    def _initialize_latex_array(self, aliases=None):
        del aliases  # unused
        self.img_depth, self.sum_column_widths = self._get_image_depth()
        self.sum_row_heights = self.img_width
        # choose the most compact row spacing, while not squashing them
        if self.has_box:
            self.row_separation = 0.0
        elif self.has_target:
            self.row_separation = 0.2
        else:
            self.row_separation = 1.0
        self._latex = [
            ["\\cw" if self.wire_type[self.ordered_regs[j]]
             else "\\qw" for _ in range(self.img_depth + 1)]
            for j in range(self.img_width)]
        self._latex.append([" "] * (self.img_depth + 1))
        for i in range(self.img_width):
            if self.wire_type[self.ordered_regs[i]]:
                self._latex[i][0] = "\\lstick{" + self.ordered_regs[i].register.name + \
                                    "_{" + str(self.ordered_regs[i].index) + "}" + \
                                    ": 0}"
            else:
                if self.layout is None:
                    self._latex[i][0] = "\\lstick{{ {}_{} : \\ket{{0}} }}".format(
                        self.ordered_regs[i].register.name, self.ordered_regs[i].index)
                else:
                    self._latex[i][0] = "\\lstick{{({}_{{{}}})~q_{{{}}} : \\ket{{0}} }}".format(
                        self.layout[self.ordered_regs[i].index].register.name,
                        self.layout[self.ordered_regs[i].index].index,
                        self.ordered_regs[i].index)

    def _get_image_depth(self):
        """Get depth information for the circuit.

        Returns:
            int: number of columns in the circuit
            int: total size of columns in the circuit
        """

        max_column_widths = []
        # Determine row spacing before image depth
        for layer in self.ops:
            for op in layer:
                # useful information for determining row spacing
                boxed_gates = ['u0', 'u1', 'u2', 'u3', 'x', 'y', 'z', 'h', 's',
                               'sdg', 't', 'tdg', 'rx', 'ry', 'rz', 'ch', 'cy',
                               'crz', 'cu3', 'id']
                target_gates = ['cx', 'ccx']
                if op.name in boxed_gates:
                    self.has_box = True
                if op.name in target_gates:
                    self.has_target = True

        for layer in self.ops:

            # store the max width for the layer
            current_max = 0

            for op in layer:

                # update current op width
                arg_str_len = 0

                # the wide gates
                for arg in op.op.params:
                    arg_str = re.sub(r'[-+]?\d*\.\d{2,}|\d{2,}',
                                     _truncate_float, str(arg))
                    arg_str_len += len(arg_str)

                # the width of the column is the max of all the gates in the column
                current_max = max(arg_str_len, current_max)

            max_column_widths.append(current_max)

        # wires in the beginning and end
        columns = 2

        # all gates take up 1 column except from those with labels (cu1) which take 2
        columns += sum([2 if nd.name == 'cu1' else 1 for layer in self.ops for nd in layer])

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
        """Get height, width & scale attributes for the beamer page.

        Returns:
            tuple: (height, width, scale) desirable page attributes
        """
        # PIL python package limits image size to around a quarter gigabyte
        # this means the beamer image should be limited to < 50000
        # if you want to avoid a "warning" too, set it to < 25000
        PIL_limit = 40000

        # the beamer latex template limits each dimension to < 19 feet
        # (i.e. 575cm)
        beamer_limit = 550

        # columns are roughly twice as big as rows
        aspect_ratio = self.sum_row_heights / self.sum_column_widths

        # choose a page margin so circuit is not cropped
        margin_factor = 1.5
        height = min(self.sum_row_heights * margin_factor, beamer_limit)
        width = min(self.sum_column_widths * margin_factor, beamer_limit)

        # if too large, make it fit
        if height * width > PIL_limit:
            height = min(np.sqrt(PIL_limit * aspect_ratio), beamer_limit)
            width = min(np.sqrt(PIL_limit / aspect_ratio), beamer_limit)

        # if too small, give it a minimum size
        height = max(height, 10)
        width = max(width, 10)

        return (height, width, self.scale)

    def _get_mask(self, creg_name):
        mask = 0
        for index, cbit in enumerate(self.clbit_list):
            if creg_name == cbit.register:
                mask |= (1 << index)
        return mask

    def _build_latex_array(self, aliases=None):
        """Returns an array of strings containing \\LaTeX for this circuit.

        If aliases is not None, aliases contains a dict mapping
        the current qubits in the circuit to new qubit names.
        We will deduce the register names and sizes from aliases.
        """

        # Rename qregs if necessary
        if aliases:
            qregdata = {}
            for q in aliases.values():
                if q[0] not in qregdata:
                    qregdata[q[0]] = q[1] + 1
                elif qregdata[q[0]] < q[1] + 1:
                    qregdata[q[0]] = q[1] + 1
        else:
            qregdata = self.qregs

        column = 1
        for layer in self.ops:
            num_cols_used = 1

            for op in layer:
                if op.condition:
                    mask = self._get_mask(op.condition[0])
                    cl_reg = self.clbit_list[self._ffs(mask)]
                    if_reg = cl_reg.register
                    pos_2 = self.img_regs[cl_reg]
                    if_value = format(op.condition[1],
                                      'b').zfill(self.cregs[if_reg])[::-1]
                if op.name not in ['measure', 'barrier', 'snapshot', 'load',
                                   'save', 'noise']:
                    nm = utf8tolatex(op.name).replace(" ", "\\,")
                    qarglist = op.qargs
                    if aliases is not None:
                        qarglist = map(lambda x: aliases[x], qarglist)
                    if len(qarglist) == 1:
                        pos_1 = self.img_regs[qarglist[0]]

                        if op.condition:
                            mask = self._get_mask(op.condition[0])
                            cl_reg = self.clbit_list[self._ffs(mask)]
                            if_reg = cl_reg.register
                            pos_2 = self.img_regs[cl_reg]

                            if nm == "x":
                                self._latex[pos_1][column] = "\\gate{X}"
                            elif nm == "y":
                                self._latex[pos_1][column] = "\\gate{Y}"
                            elif nm == "z":
                                self._latex[pos_1][column] = "\\gate{Z}"
                            elif nm == "h":
                                self._latex[pos_1][column] = "\\gate{H}"
                            elif nm == "s":
                                self._latex[pos_1][column] = "\\gate{S}"
                            elif nm == "sdg":
                                self._latex[pos_1][column] = "\\gate{S^\\dag}"
                            elif nm == "t":
                                self._latex[pos_1][column] = "\\gate{T}"
                            elif nm == "tdg":
                                self._latex[pos_1][column] = "\\gate{T^\\dag}"
                            elif nm == "u0":
                                self._latex[pos_1][column] = "\\gate{U_0(%s)}" % (
                                    pi_check(op.op.params[0], output='latex'))
                            elif nm == "u1":
                                self._latex[pos_1][column] = "\\gate{U_1(%s)}" % (
                                    pi_check(op.op.params[0], output='latex'))
                            elif nm == "u2":
                                self._latex[pos_1][column] = \
                                    "\\gate{U_2\\left(%s,%s\\right)}" % (
                                        pi_check(op.op.params[0], output='latex'),
                                        pi_check(op.op.params[1], output='latex'))
                            elif nm == "u3":
                                self._latex[pos_1][column] = ("\\gate{U_3(%s,%s,%s)}" % (
                                    pi_check(op.op.params[0], output='latex'),
                                    pi_check(op.op.params[1], output='latex'),
                                    pi_check(op.op.params[2], output='latex')))
                            elif nm == "rx":
                                self._latex[pos_1][column] = "\\gate{R_x(%s)}" % (
                                    pi_check(op.op.params[0], output='latex'))
                            elif nm == "ry":
                                self._latex[pos_1][column] = "\\gate{R_y(%s)}" % (
                                    pi_check(op.op.params[0], output='latex'))
                            elif nm == "rz":
                                self._latex[pos_1][column] = "\\gate{R_z(%s)}" % (
                                    pi_check(op.op.params[0], output='latex'))
                            else:
                                self._latex[pos_1][column] = ("\\gate{%s}" % nm)

                            gap = pos_2 - pos_1
                            for i in range(self.cregs[if_reg]):
                                if if_value[i] == '1':
                                    self._latex[pos_2 + i][column] = \
                                        "\\control \\cw \\cwx[-" + str(gap) + "]"
                                    gap = 1
                                else:
                                    self._latex[pos_2 + i][column] = \
                                        "\\controlo \\cw \\cwx[-" + str(gap) + "]"
                                    gap = 1

                        else:
                            if nm == "x":
                                self._latex[pos_1][column] = "\\gate{X}"
                            elif nm == "y":
                                self._latex[pos_1][column] = "\\gate{Y}"
                            elif nm == "z":
                                self._latex[pos_1][column] = "\\gate{Z}"
                            elif nm == "h":
                                self._latex[pos_1][column] = "\\gate{H}"
                            elif nm == "s":
                                self._latex[pos_1][column] = "\\gate{S}"
                            elif nm == "sdg":
                                self._latex[pos_1][column] = "\\gate{S^\\dag}"
                            elif nm == "t":
                                self._latex[pos_1][column] = "\\gate{T}"
                            elif nm == "tdg":
                                self._latex[pos_1][column] = "\\gate{T^\\dag}"
                            elif nm == "u0":
                                self._latex[pos_1][column] = "\\gate{U_0(%s)}" % (
                                    pi_check(op.op.params[0], output='latex'))
                            elif nm == "u1":
                                self._latex[pos_1][column] = "\\gate{U_1(%s)}" % (
                                    pi_check(op.op.params[0], output='latex'))
                            elif nm == "u2":
                                self._latex[pos_1][column] = \
                                    "\\gate{U_2\\left(%s,%s\\right)}" % (
                                        pi_check(op.op.params[0], output='latex'),
                                        pi_check(op.op.params[1], output='latex'))
                            elif nm == "u3":
                                self._latex[pos_1][column] = ("\\gate{U_3(%s,%s,%s)}" % (
                                    pi_check(op.op.params[0], output='latex'),
                                    pi_check(op.op.params[1], output='latex'),
                                    pi_check(op.op.params[2], output='latex')))
                            elif nm == "rx":
                                self._latex[pos_1][column] = "\\gate{R_x(%s)}" % (
                                    pi_check(op.op.params[0], output='latex'))
                            elif nm == "ry":
                                self._latex[pos_1][column] = "\\gate{R_y(%s)}" % (
                                    pi_check(op.op.params[0], output='latex'))
                            elif nm == "rz":
                                self._latex[pos_1][column] = "\\gate{R_z(%s)}" % (
                                    pi_check(op.op.params[0], output='latex'))
                            elif nm == "reset":
                                self._latex[pos_1][column] = (
                                    "\\push{\\rule{.6em}{0em}\\ket{0}\\"
                                    "rule{.2em}{0em}} \\qw")
                            else:
                                self._latex[pos_1][column] = ("\\gate{%s}" % nm)

                    elif len(qarglist) == 2:
                        pos_1 = self.img_regs[qarglist[0]]
                        pos_2 = self.img_regs[qarglist[1]]

                        if op.condition:
                            pos_3 = self.img_regs[(if_reg, 0)]
                            temp = [pos_1, pos_2, pos_3]
                            temp.sort(key=int)
                            bottom = temp[1]

                            gap = pos_3 - bottom
                            for i in range(self.cregs[if_reg]):
                                if if_value[i] == '1':
                                    self._latex[pos_3 + i][column] = \
                                        "\\control \\cw \\cwx[-" + str(gap) + "]"
                                    gap = 1
                                else:
                                    self._latex[pos_3 + i][column] = \
                                        "\\controlo \\cw \\cwx[-" + str(gap) + "]"
                                    gap = 1

                            if nm == "cx":
                                self._latex[pos_1][column] = \
                                    "\\ctrl{" + str(pos_2 - pos_1) + "}"
                                self._latex[pos_2][column] = "\\targ"
                            elif nm == "cz":
                                self._latex[pos_1][column] = \
                                    "\\ctrl{" + str(pos_2 - pos_1) + "}"
                                self._latex[pos_2][column] = "\\control\\qw"
                            elif nm == "cy":
                                self._latex[pos_1][column] = \
                                    "\\ctrl{" + str(pos_2 - pos_1) + "}"
                                self._latex[pos_2][column] = "\\gate{Y}"
                            elif nm == "ch":
                                self._latex[pos_1][column] = \
                                    "\\ctrl{" + str(pos_2 - pos_1) + "}"
                                self._latex[pos_2][column] = "\\gate{H}"
                            elif nm == "swap":
                                self._latex[pos_1][column] = "\\qswap"
                                self._latex[pos_2][column] = \
                                    "\\qswap \\qwx[" + str(pos_1 - pos_2) + "]"
                            elif nm == "crz":
                                self._latex[pos_1][column] = \
                                    "\\ctrl{" + str(pos_2 - pos_1) + "}"
                                self._latex[pos_2][column] = \
                                    "\\gate{R_z(%s)}" % (pi_check(op.op.params[0], output='latex'))
                            elif nm == "cu1":
                                self._latex[pos_1][column] = "\\ctrl{" + str(
                                    pos_2 - pos_1) + "}"
                                self._latex[pos_2][column] = "\\control \\qw"
                                self._latex[min(pos_1, pos_2)][column + 1] = \
                                    "\\dstick{%s}\\qw" % (pi_check(op.op.params[0], output='latex'))
                                self._latex[max(pos_1, pos_2)][column + 1] = "\\qw"
                                # this is because this gate takes up 2 columns,
                                # and we have just written to the next column
                                num_cols_used = 2
                            elif nm == "cu3":
                                self._latex[pos_1][column] = \
                                    "\\ctrl{" + str(pos_2 - pos_1) + "}"
                                self._latex[pos_2][column] = \
                                    "\\gate{U_3(%s,%s,%s)}" % \
                                    (pi_check(op.op.params[0], output='latex'),
                                     pi_check(op.op.params[1], output='latex'),
                                     pi_check(op.op.params[2], output='latex'))
                        else:
                            temp = [pos_1, pos_2]
                            temp.sort(key=int)

                            if nm == "cx":
                                self._latex[pos_1][column] = "\\ctrl{" + str(
                                    pos_2 - pos_1) + "}"
                                self._latex[pos_2][column] = "\\targ"
                            elif nm == "cz":
                                self._latex[pos_1][column] = "\\ctrl{" + str(
                                    pos_2 - pos_1) + "}"
                                self._latex[pos_2][column] = "\\control\\qw"
                            elif nm == "cy":
                                self._latex[pos_1][column] = "\\ctrl{" + str(
                                    pos_2 - pos_1) + "}"
                                self._latex[pos_2][column] = "\\gate{Y}"
                            elif nm == "ch":
                                self._latex[pos_1][column] = "\\ctrl{" + str(
                                    pos_2 - pos_1) + "}"
                                self._latex[pos_2][column] = "\\gate{H}"
                            elif nm == "swap":
                                self._latex[pos_1][column] = "\\qswap"
                                self._latex[pos_2][column] = \
                                    "\\qswap \\qwx[" + str(pos_1 - pos_2) + "]"
                            elif nm == "crz":
                                self._latex[pos_1][column] = "\\ctrl{" + str(
                                    pos_2 - pos_1) + "}"
                                self._latex[pos_2][column] = \
                                    "\\gate{R_z(%s)}" % (pi_check(op.op.params[0], output='latex'))
                            elif nm == "cu1":
                                self._latex[pos_1][column] = "\\ctrl{" + str(
                                    pos_2 - pos_1) + "}"
                                self._latex[pos_2][column] = "\\control \\qw"
                                self._latex[min(pos_1, pos_2)][column + 1] = \
                                    "\\dstick{%s}\\qw" % (pi_check(op.op.params[0], output='latex'))
                                self._latex[max(pos_1, pos_2)][column + 1] = "\\qw"
                                num_cols_used = 2
                            elif nm == "cu3":
                                self._latex[pos_1][column] = "\\ctrl{" + str(
                                    pos_2 - pos_1) + "}"
                                self._latex[pos_2][column] = \
                                    ("\\gate{U_3(%s,%s,%s)}" %
                                     (pi_check(op.op.params[0], output='latex'),
                                      pi_check(op.op.params[1], output='latex'),
                                      pi_check(op.op.params[2], output='latex')))
                            else:
                                start_pos = min([pos_1, pos_2])
                                stop_pos = max([pos_1, pos_2])
                                if stop_pos - start_pos >= 2:
                                    delta = stop_pos - start_pos
                                    self._latex[start_pos][column] = ("\\multigate{%s}{%s}"
                                                                      % (delta, nm))
                                    for i_pos in range(start_pos + 1, stop_pos + 1):
                                        self._latex[i_pos][column] = ("\\ghost{%s}"
                                                                      % nm)
                                else:
                                    self._latex[start_pos][column] = ("\\multigate{1}{%s}"
                                                                      % nm)
                                    self._latex[stop_pos][column] = ("\\ghost{%s}" %
                                                                     nm)

                    elif len(qarglist) == 3:
                        pos_1 = self.img_regs[qarglist[0]]
                        pos_2 = self.img_regs[qarglist[1]]
                        pos_3 = self.img_regs[qarglist[2]]

                        if op.condition:
                            pos_4 = self.img_regs[(if_reg, 0)]
                            temp = [pos_1, pos_2, pos_3, pos_4]
                            temp.sort(key=int)
                            bottom = temp[2]

                            gap = pos_4 - bottom
                            for i in range(self.cregs[if_reg]):
                                if if_value[i] == '1':
                                    self._latex[pos_4 + i][column] = \
                                        "\\control \\cw \\cwx[-" + str(gap) + "]"
                                    gap = 1
                                else:
                                    self._latex[pos_4 + i][column] = \
                                        "\\controlo \\cw \\cwx[-" + str(gap) + "]"
                                    gap = 1

                            if nm == "ccx":
                                self._latex[pos_1][column] = "\\ctrl{" + str(
                                    pos_2 - pos_1) + "}"
                                self._latex[pos_2][column] = "\\ctrl{" + str(
                                    pos_3 - pos_2) + "}"
                                self._latex[pos_3][column] = "\\targ"

                            if nm == "cswap":
                                self._latex[pos_1][column] = "\\ctrl{" + str(
                                    pos_2 - pos_1) + "}"
                                self._latex[pos_2][column] = "\\qswap"
                                self._latex[pos_3][column] = \
                                    "\\qswap \\qwx[" + str(pos_2 - pos_3) + "]"
                        else:
                            if nm == "ccx":
                                self._latex[pos_1][column] = "\\ctrl{" + str(
                                    pos_2 - pos_1) + "}"
                                self._latex[pos_2][column] = "\\ctrl{" + str(
                                    pos_3 - pos_2) + "}"
                                self._latex[pos_3][column] = "\\targ"

                            elif nm == "cswap":
                                self._latex[pos_1][column] = "\\ctrl{" + str(
                                    pos_2 - pos_1) + "}"
                                self._latex[pos_2][column] = "\\qswap"
                                self._latex[pos_3][column] = \
                                    "\\qswap \\qwx[" + str(pos_2 - pos_3) + "]"
                            else:
                                start_pos = min([pos_1, pos_2, pos_3])
                                stop_pos = max([pos_1, pos_2, pos_3])
                                if stop_pos - start_pos >= 3:
                                    delta = stop_pos - start_pos
                                    self._latex[start_pos][column] = ("\\multigate{%s}{%s}" %
                                                                      (delta, nm))
                                    for i_pos in range(start_pos + 1, stop_pos + 1):
                                        self._latex[i_pos][column] = ("\\ghost{%s}" %
                                                                      nm)
                                else:
                                    self._latex[pos_1][column] = ("\\multigate{2}{%s}" %
                                                                  nm)
                                    self._latex[pos_2][column] = ("\\ghost{%s}" %
                                                                  nm)
                                    self._latex[pos_3][column] = ("\\ghost{%s}" %
                                                                  nm)

                    elif len(qarglist) > 3:
                        nbits = len(qarglist)
                        pos_array = [self.img_regs[(qarglist[0].register,
                                                    qarglist[0].index)]]
                        for i in range(1, nbits):
                            pos_array.append(self.img_regs[(qarglist[i].register,
                                                            qarglist[i].index)])
                        pos_start = min(pos_array)
                        pos_stop = max(pos_array)
                        self._latex[pos_start][column] = ("\\multigate{%s}{%s}" %
                                                          (nbits - 1, nm))
                        for pos in range(pos_start + 1, pos_stop + 1):
                            self._latex[pos][column] = ("\\ghost{%s}" % nm)

                elif op.name == "measure":
                    if (len(op.cargs) != 1
                            or len(op.qargs) != 1
                            or op.op.params):
                        raise exceptions.VisualizationError("bad operation record")

                    if op.condition:
                        raise exceptions.VisualizationError(
                            "If controlled measures currently not supported.")

                    if aliases:
                        newq = aliases[(qname, qindex)]
                        qname = newq[0]
                        qindex = newq[1]

                    pos_1 = self.img_regs[op.qargs[0]]
                    pos_2 = self.img_regs[op.cargs[0]]

                    try:
                        self._latex[pos_1][column] = "\\meter"
                        self._latex[pos_2][column] = \
                            "\\cw \\cwx[-" + str(pos_2 - pos_1) + "]"
                    except Exception as e:
                        raise exceptions.VisualizationError(
                            'Error during Latex building: %s' % str(e))

                elif op.name in ['barrier', 'snapshot', 'load', 'save',
                                 'noise']:
                    if self.plot_barriers:
                        qarglist = op.qargs
                        indexes = [self._get_qubit_index(x) for x in qarglist]
                        start_bit = self.qubit_list[min(indexes)]
                        if aliases is not None:
                            qarglist = map(lambda x: aliases[x], qarglist)
                        start = self.img_regs[start_bit]
                        span = len(op.qargs) - 1

                        self._latex[start][column - 1] += " \\barrier[0em]{" + str(
                            span) + "}"
                        self._latex[start][column] = "\\qw"
                else:
                    raise exceptions.VisualizationError("bad node data")

            # increase the number of columns by the number of columns this layer used
            column += num_cols_used

    def _get_qubit_index(self, qubit):
        """Get the index number for a quantum bit
        Args:
            qubit (tuple): The tuple of the bit of the form
                (register_name, bit_number)
        Returns:
            int: The index in the bit list
        Raises:
            VisualizationError: If the bit isn't found
        """
        for i, bit in enumerate(self.qubit_list):
            if qubit == bit:
                qindex = i
                break
        else:
            raise exceptions.VisualizationError("unable to find bit for operation")
        return qindex

    def _ffs(self, mask):
        """Find index of first set bit.

        Args:
            mask (int): integer to search
        Returns:
            int: index of the first set bit.
        """
        origin = (mask & (-mask)).bit_length()
        return origin - 1


def _get_register_specs(bits):
    """Get the number and size of unique registers from bits list.

    Args:
        bits (list[Bit]): this list is of the form::
            [Qubit(v0, 0), Qubit(v0, 1), Qubit(v0, 2), Qubit(v0, 3), Qubit(v1, 0)]
            which indicates a size-4 register and a size-1 register

    Returns:
        OrderedDict: ordered map of Registers to their sizes
    """
    regs = collections.OrderedDict([(bit.register, bit.register.size) for bit in bits])
    return regs


def _truncate_float(matchobj, ndigits=3):
    """Truncate long floats

    Args:
        matchobj (re.Match): contains original float
        ndigits (int): Number of digits to print
    Returns:
       str: returns truncated float
    """
    if matchobj.group(0):
        return '%.{}g'.format(ndigits) % float(matchobj.group(0))
    return ''
