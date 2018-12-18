# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,anomalous-backslash-in-string,missing-docstring

"""latex circuit visualization backends."""

import collections
import io
import itertools
import json
import math
import operator
import re

import numpy as np

from qiskit.tools.visualization import _error
from qiskit.tools.visualization import _qcstyle


class QCircuitImage(object):
    """This class contains methods to create \\LaTeX circuit images.

    The class targets the \\LaTeX package Q-circuit
    (https://arxiv.org/pdf/quant-ph/0406003).

    Thanks to Eric Sabo for the initial implementation for Qiskit.
    """

    def __init__(self, qregs, cregs, ops, scale, style=None,
                 plot_barriers=True, reverse_bits=False):
        """
        Args:
            qregs (list): A list of tuples for the quantum registers
            cregs (list): A list of tuples for the classical registers
            ops (list): A list of dicts where each entry is a operation from
                the circuit.
            scale (float): image scaling
            style (dict or str): dictionary of style or file name of style file
            reverse_bits (bool): When set to True reverse the bit order inside
               registers for the output visualization.
            plot_barriers (bool): Enable/disable drawing barriers in the output
               circuit. Defaults to True.
        """
        # style sheet
        self._style = _qcstyle.QCStyle()
        if style:
            if isinstance(style, dict):
                self._style.set_style(style)
            elif isinstance(style, str):
                with open(style, 'r') as infile:
                    dic = json.load(infile)
                self._style.set_style(dic)

        # compiled qobj circuit
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
        self.column_separation = 0.5

        # em points of separation between circuit row
        self.row_separation = 0.0

        # presence of "box" or "target" determines row spacing
        self.has_box = False
        self.has_target = False
        self.reverse_bits = reverse_bits
        self.plot_barriers = plot_barriers

        #################################
        self.qregs = collections.OrderedDict(_get_register_specs(qregs))
        self.qubit_list = qregs
        self.ordered_regs = qregs + cregs
        self.cregs = collections.OrderedDict(_get_register_specs(cregs))
        self.clbit_list = cregs
        self.img_regs = {bit: ind for ind, bit in
                         enumerate(self.ordered_regs)}
        self.img_width = len(self.img_regs)
        self.wire_type = {}
        for key, value in self.ordered_regs:
            self.wire_type[(key, value)] = key in self.cregs.keys()

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
        # pylint: disable=unused-argument
        self.img_depth, self.sum_column_widths = self._get_image_depth(aliases)
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
             else "\\qw" for i in range(self.img_depth + 1)]
            for j in range(self.img_width)]
        self._latex.append([" "] * (self.img_depth + 1))
        for i in range(self.img_width):
            if self.wire_type[self.ordered_regs[i]]:
                self._latex[i][0] = "\\lstick{" + self.ordered_regs[i][0].name + \
                                    "_{" + str(self.ordered_regs[i][1]) + "}" + \
                                    ": 0}"
            else:
                self._latex[i][0] = "\\lstick{" + \
                                    self.ordered_regs[i][0].name + "_{" + \
                                    str(self.ordered_regs[i][1]) + "}" + \
                                    ": \\ket{0}}"

    def _get_image_depth(self, aliases=None):
        """Get depth information for the circuit.

        Args:
            aliases (dict): dict mapping the current qubits in the circuit to
                new qubit names.

        Returns:
            int: number of columns in the circuit
            int: total size of columns in the circuit

        Raises:
            VisualizationError: if trying to draw unsupported gates
        """
        columns = 2  # wires in the beginning and end
        is_occupied = [False] * self.img_width
        max_column_width = {}
        for op in self.ops:
            # useful information for determining row spacing
            boxed_gates = ['u0', 'u1', 'u2', 'u3', 'x', 'y', 'z', 'h', 's',
                           'sdg', 't', 'tdg', 'rx', 'ry', 'rz', 'ch', 'cy',
                           'crz', 'cu3']
            target_gates = ['cx', 'ccx']
            if op['name'] in boxed_gates:
                self.has_box = True
            if op['name'] in target_gates:
                self.has_target = True

            # useful information for determining column widths and final image
            # scaling
            if op['name'] not in ['measure', 'reset', 'barrier']:
                qarglist = op['qargs']
                if aliases is not None:
                    qarglist = map(lambda x: aliases[x], qarglist)
                if len(qarglist) == 1:
                    pos_1 = self.img_regs[(qarglist[0][0],
                                           qarglist[0][1])]
                    if 'condition' in op and op['condition']:
                        mask = self._get_mask(op['condition'][0])
                        cl_reg = self.clbit_list[self._ffs(mask)]
                        if_reg = cl_reg[0]
                        pos_2 = self.img_regs[cl_reg]
                        for i in range(pos_1, pos_2 + self.cregs[if_reg]):
                            if is_occupied[i] is False:
                                is_occupied[i] = True
                            else:
                                columns += 1
                                is_occupied = [False] * self.img_width
                                for j in range(pos_1, pos_2 + 1):
                                    is_occupied[j] = True
                                break
                    else:
                        if is_occupied[pos_1] is False:
                            is_occupied[pos_1] = True
                        else:
                            columns += 1
                            is_occupied = [False] * self.img_width
                            is_occupied[pos_1] = True
                elif len(qarglist) == 2:
                    pos_1 = self.img_regs[(qarglist[0][0], qarglist[0][1])]
                    pos_2 = self.img_regs[(qarglist[1][0], qarglist[1][1])]

                    if 'condition' in op and op['condition']:
                        mask = self._get_mask(op['condition'][0])
                        cl_reg = self.clbit_list[self._ffs(mask)]
                        if_reg = cl_reg[0]
                        pos_3 = self.img_regs[(if_reg, 0)]
                        if pos_1 > pos_2:
                            for i in range(pos_2, pos_3 + self.cregs[if_reg]):
                                if is_occupied[i] is False:
                                    is_occupied[i] = True
                                else:
                                    columns += 1
                                    is_occupied = [False] * self.img_width
                                    for j in range(pos_2, pos_3 + 1):
                                        is_occupied[j] = True
                                    break
                        else:
                            for i in range(pos_1, pos_3 + self.cregs[if_reg]):
                                if is_occupied[i] is False:
                                    is_occupied[i] = True
                                else:
                                    columns += 1
                                    is_occupied = [False] * self.img_width
                                    for j in range(pos_1, pos_3 + 1):
                                        is_occupied[j] = True
                                    break
                        # symetric gates have angle labels
                        if op['name'] in ['cu1']:
                            columns += 1
                            is_occupied = [False] * self.img_width
                            is_occupied[max(pos_1, pos_2)] = True
                    else:
                        temp = [pos_1, pos_2]
                        temp.sort(key=int)
                        top = temp[0]
                        bottom = temp[1]

                        for i in range(top, bottom + 1):
                            if is_occupied[i] is False:
                                is_occupied[i] = True
                            else:
                                columns += 1
                                is_occupied = [False] * self.img_width
                                for j in range(top, bottom + 1):
                                    is_occupied[j] = True
                                break
                        # symetric gates have angle labels
                        if op['name'] in ['cu1']:
                            columns += 1
                            is_occupied = [False] * self.img_width
                            is_occupied[top] = True

                elif len(qarglist) == 3:
                    pos_1 = self.img_regs[(qarglist[0][0], qarglist[0][1])]
                    pos_2 = self.img_regs[(qarglist[1][0], qarglist[1][1])]
                    pos_3 = self.img_regs[(qarglist[2][0], qarglist[2][1])]

                    if 'condition' in op and op['condition']:
                        mask = self._get_mask(op['condition'][0])
                        cl_reg = self.clbit_list[self._ffs(mask)]
                        if_reg = cl_reg[0]
                        pos_4 = self.img_regs[(if_reg, 0)]

                        temp = [pos_1, pos_2, pos_3, pos_4]
                        temp.sort(key=int)
                        top = temp[0]
                        bottom = temp[2]

                        for i in range(top, pos_4 + 1):
                            if is_occupied[i] is False:
                                is_occupied[i] = True
                            else:
                                columns += 1
                                is_occupied = [False] * self.img_width
                                for j in range(top, pos_4 + 1):
                                    is_occupied[j] = True
                                break
                    else:
                        temp = [pos_1, pos_2, pos_3]
                        temp.sort(key=int)
                        top = temp[0]
                        bottom = temp[2]

                        for i in range(top, bottom + 1):
                            if is_occupied[i] is False:
                                is_occupied[i] = True
                            else:
                                columns += 1
                                is_occupied = [False] * self.img_width
                                for j in range(top, bottom + 1):
                                    is_occupied[j] = True
                                break

                # update current column width
                arg_str_len = 0
                for arg in op['op'].param:
                    arg_str = re.sub(r'[-+]?\d*\.\d{2,}|\d{2,}',
                                     _truncate_float, str(arg))
                    arg_str_len += len(arg_str)
                if columns not in max_column_width:
                    max_column_width[columns] = 0
                max_column_width[columns] = max(arg_str_len,
                                                max_column_width[columns])
            elif op['name'] == "measure":
                if len(op['cargs']) != 1 or len(op['qargs']) != 1:
                    raise _error.VisualizationError("bad operation record")
                if 'condition' in op and op['condition']:
                    raise _error.VisualizationError(
                        'conditional measures currently not supported.')
                qname, qindex = op['qargs'][0]
                cname, cindex = op['cargs'][0]
                if aliases:
                    newq = aliases[(qname, qindex)]
                    qname = newq[0]
                    qindex = newq[1]
                pos_1 = self.img_regs[(qname, qindex)]
                pos_2 = self.img_regs[(cname, cindex)]
                temp = [pos_1, pos_2]
                temp.sort(key=int)
                [pos_1, pos_2] = temp
                for i in range(pos_1, pos_2 + 1):
                    if is_occupied[i] is False:
                        is_occupied[i] = True
                    else:
                        columns += 1
                        is_occupied = [False] * self.img_width
                        for j in range(pos_1, pos_2 + 1):
                            is_occupied[j] = True
                        break
                # update current column width
                if columns not in max_column_width:
                    max_column_width[columns] = 0
            elif op['name'] == "reset":
                if 'conditional' in op and op['condition']:
                    raise _error.VisualizationError(
                        'conditional reset currently not supported.')
                qname, qindex = op['qargs'][0]
                if aliases:
                    newq = aliases[(qname, qindex)]
                    qname = newq[0]
                    qindex = newq[1]
                pos_1 = self.img_regs[(qname, qindex)]
                if is_occupied[pos_1] is False:
                    is_occupied[pos_1] = True
                else:
                    columns += 1
                    is_occupied = [False] * self.img_width
                    is_occupied[pos_1] = True
            elif op['name'] in ["barrier", 'snapshot', 'load', 'save',
                                'noise']:
                if self.plot_barriers:
                    qarglist = op['qargs']
                    indexes = [self._get_qubit_index(x) for x in qarglist]
                    start_bit = self.qubit_list[min(indexes)]
                    if aliases is not None:
                        qarglist = map(lambda x: aliases[x], qarglist)
                    start = self.img_regs[start_bit]
                    span = len(op['qargs']) - 1
                    for i in range(start, span + 1):
                        if is_occupied[i] is False:
                            is_occupied[i] = True
                        else:
                            columns += 1
                            is_occupied = [False] * self.img_width
                            for j in range(start, span + 1):
                                is_occupied[j] = True
                            break
                    # update current column width
                    if columns not in max_column_width:
                        max_column_width[columns] = 0
            else:
                raise _error.VisualizationError("bad node data")
        # every 3 characters is roughly one extra 'unit' of width in the cell
        # the gate name is 1 extra 'unit'
        # the qubit/cbit labels plus initial states is 2 more
        # the wires poking out at the ends is 2 more
        sum_column_widths = sum(1 + v / 3 for v in max_column_width.values())
        return columns + 1, math.ceil(sum_column_widths) + 4

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
            if creg_name == cbit[0]:
                mask |= (1 << index)
        return mask

    def _build_latex_array(self, aliases=None):
        """Returns an array of strings containing \\LaTeX for this circuit.

        If aliases is not None, aliases contains a dict mapping
        the current qubits in the circuit to new qubit names.
        We will deduce the register names and sizes from aliases.
        """
        columns = 1
        is_occupied = [False] * self.img_width

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

        for _, op in enumerate(self.ops):
            if 'condition' in op and op['condition']:
                mask = self._get_mask(op['condition'][0])
                cl_reg = self.clbit_list[self._ffs(mask)]
                if_reg = cl_reg[0]
                pos_2 = self.img_regs[cl_reg]
                if_value = format(op['condition'][1],
                                  'b').zfill(self.cregs[if_reg])[::-1]
            if op['name'] not in ['measure', 'barrier', 'snapshot', 'load',
                                  'save', 'noise']:
                nm = op['name']
                qarglist = op['qargs']
                if aliases is not None:
                    qarglist = map(lambda x: aliases[x], qarglist)
                if len(qarglist) == 1:
                    pos_1 = self.img_regs[(qarglist[0][0],
                                           qarglist[0][1])]
                    if 'condition' in op and op['condition']:
                        mask = self._get_mask(op['condition'][0])
                        cl_reg = self.clbit_list[self._ffs(mask)]
                        if_reg = cl_reg[0]
                        pos_2 = self.img_regs[cl_reg]
                        for i in range(pos_1, pos_2 + self.cregs[if_reg]):
                            if is_occupied[i] is False:
                                is_occupied[i] = True
                            else:
                                columns += 1
                                is_occupied = [False] * self.img_width
                                for j in range(pos_1, pos_2 + 1):
                                    is_occupied[j] = True
                                break

                        if nm == "x":
                            self._latex[pos_1][columns] = "\\gate{X}"
                        elif nm == "y":
                            self._latex[pos_1][columns] = "\\gate{Y}"
                        elif nm == "z":
                            self._latex[pos_1][columns] = "\\gate{Z}"
                        elif nm == "h":
                            self._latex[pos_1][columns] = "\\gate{H}"
                        elif nm == "s":
                            self._latex[pos_1][columns] = "\\gate{S}"
                        elif nm == "sdg":
                            self._latex[pos_1][columns] = "\\gate{S^\\dag}"
                        elif nm == "t":
                            self._latex[pos_1][columns] = "\\gate{T}"
                        elif nm == "tdg":
                            self._latex[pos_1][columns] = "\\gate{T^\\dag}"
                        elif nm == "u0":
                            self._latex[pos_1][columns] = "\\gate{U_0(%s)}" % (
                                op["op"].param[0])
                        elif nm == "u1":
                            self._latex[pos_1][columns] = "\\gate{U_1(%s)}" % (
                                op["op"].param[0])
                        elif nm == "u2":
                            self._latex[pos_1][columns] = \
                                "\\gate{U_2\\left(%s,%s\\right)}" % (
                                    op["op"].param[0], op["op"].param[1])
                        elif nm == "u3":
                            self._latex[pos_1][columns] = ("\\gate{U_3(%s,%s,%s)}" % (
                                op["op"].param[0],
                                op["op"].param[1],
                                op["op"].param[2]))
                        elif nm == "rx":
                            self._latex[pos_1][columns] = "\\gate{R_x(%s)}" % (
                                op["op"].param[0])
                        elif nm == "ry":
                            self._latex[pos_1][columns] = "\\gate{R_y(%s)}" % (
                                op["op"].param[0])
                        elif nm == "rz":
                            self._latex[pos_1][columns] = "\\gate{R_z(%s)}" % (
                                op["op"].param[0])

                        gap = pos_2 - pos_1
                        for i in range(self.cregs[if_reg]):
                            if if_value[i] == '1':
                                self._latex[pos_2 + i][columns] = \
                                    "\\control \\cw \\cwx[-" + str(gap) + "]"
                                gap = 1
                            else:
                                self._latex[pos_2 + i][columns] = \
                                    "\\controlo \\cw \\cwx[-" + str(gap) + "]"
                                gap = 1

                    else:
                        if not is_occupied[pos_1]:
                            is_occupied[pos_1] = True
                        else:
                            columns += 1
                            is_occupied = [False] * self.img_width
                            is_occupied[pos_1] = True

                        if nm == "x":
                            self._latex[pos_1][columns] = "\\gate{X}"
                        elif nm == "y":
                            self._latex[pos_1][columns] = "\\gate{Y}"
                        elif nm == "z":
                            self._latex[pos_1][columns] = "\\gate{Z}"
                        elif nm == "h":
                            self._latex[pos_1][columns] = "\\gate{H}"
                        elif nm == "s":
                            self._latex[pos_1][columns] = "\\gate{S}"
                        elif nm == "sdg":
                            self._latex[pos_1][columns] = "\\gate{S^\\dag}"
                        elif nm == "t":
                            self._latex[pos_1][columns] = "\\gate{T}"
                        elif nm == "tdg":
                            self._latex[pos_1][columns] = "\\gate{T^\\dag}"
                        elif nm == "u0":
                            self._latex[pos_1][columns] = "\\gate{U_0(%s)}" % (
                                op["op"].param[0])
                        elif nm == "u1":
                            self._latex[pos_1][columns] = "\\gate{U_1(%s)}" % (
                                op["op"].param[0])
                        elif nm == "u2":
                            self._latex[pos_1][columns] = \
                                "\\gate{U_2\\left(%s,%s\\right)}" % (
                                    op["op"].param[0], op["op"].param[1])
                        elif nm == "u3":
                            self._latex[pos_1][columns] = ("\\gate{U_3(%s,%s,%s)}" % (
                                op["op"].param[0],
                                op["op"].param[1],
                                op["op"].param[2]))
                        elif nm == "rx":
                            self._latex[pos_1][columns] = "\\gate{R_x(%s)}" % (
                                op["op"].param[0])
                        elif nm == "ry":
                            self._latex[pos_1][columns] = "\\gate{R_y(%s)}" % (
                                op["op"].param[0])
                        elif nm == "rz":
                            self._latex[pos_1][columns] = "\\gate{R_z(%s)}" % (
                                op["op"].param[0])
                        elif nm == "reset":
                            self._latex[pos_1][columns] = (
                                "\\push{\\rule{.6em}{0em}\\ket{0}\\"
                                "rule{.2em}{0em}} \\qw")

                elif len(qarglist) == 2:
                    pos_1 = self.img_regs[(qarglist[0][0], qarglist[0][1])]
                    pos_2 = self.img_regs[(qarglist[1][0], qarglist[1][1])]

                    if 'condition' in op and op['condition']:
                        pos_3 = self.img_regs[(if_reg, 0)]
                        temp = [pos_1, pos_2, pos_3]
                        temp.sort(key=int)
                        top = temp[0]
                        bottom = temp[1]

                        for i in range(top, pos_3 + 1):
                            if is_occupied[i] is False:
                                is_occupied[i] = True
                            else:
                                columns += 1
                                is_occupied = [False] * self.img_width
                                for j in range(top, pos_3 + 1):
                                    is_occupied[j] = True
                                break
                        # symetric gates have angle labels
                        if op['name'] in ['cu1']:
                            columns += 1
                            is_occupied = [False] * self.img_width
                            is_occupied[top] = True

                        gap = pos_3 - bottom
                        for i in range(self.cregs[if_reg]):
                            if if_value[i] == '1':
                                self._latex[pos_3 + i][columns] = \
                                    "\\control \\cw \\cwx[-" + str(gap) + "]"
                                gap = 1
                            else:
                                self._latex[pos_3 + i][columns] = \
                                    "\\controlo \\cw \\cwx[-" + str(gap) + "]"
                                gap = 1

                        if nm == "cx":
                            self._latex[pos_1][columns] = \
                                "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\targ"
                        elif nm == "cz":
                            self._latex[pos_1][columns] = \
                                "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\control\\qw"
                        elif nm == "cy":
                            self._latex[pos_1][columns] = \
                                "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\gate{Y}"
                        elif nm == "ch":
                            self._latex[pos_1][columns] = \
                                "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\gate{H}"
                        elif nm == "swap":
                            self._latex[pos_1][columns] = "\\qswap"
                            self._latex[pos_2][columns] = \
                                "\\qswap \\qwx[" + str(pos_1 - pos_2) + "]"
                        elif nm == "crz":
                            self._latex[pos_1][columns] = \
                                "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = \
                                "\\gate{R_z(%s)}" % (op["op"].param[0])
                        elif nm == "cu1":
                            self._latex[pos_1][columns - 1] = "\\ctrl{" + str(
                                pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns - 1] = "\\control\\qw"
                            self._latex[min(pos_1, pos_2)][columns] = \
                                "\\dstick{%s}\\qw" % (op["op"].param[0])
                            self._latex[max(pos_1, pos_2)][columns] = "\\qw"
                        elif nm == "cu3":
                            self._latex[pos_1][columns] = \
                                "\\ctrl{" + str(pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = \
                                "\\gate{U_3(%s,%s,%s)}" % (op["op"].param[0],
                                                           op["op"].param[1],
                                                           op["op"].param[2])
                    else:
                        temp = [pos_1, pos_2]
                        temp.sort(key=int)
                        top = temp[0]
                        bottom = temp[1]

                        for i in range(top, bottom + 1):
                            if is_occupied[i] is False:
                                is_occupied[i] = True
                            else:
                                columns += 1
                                is_occupied = [False] * self.img_width
                                for j in range(top, bottom + 1):
                                    is_occupied[j] = True
                                break
                        # symetric gates have angle labels
                        if op['name'] in ['cu1']:
                            columns += 1
                            is_occupied = [False] * self.img_width
                            is_occupied[top] = True

                        if nm == "cx":
                            self._latex[pos_1][columns] = "\\ctrl{" + str(
                                pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\targ"
                        elif nm == "cz":
                            self._latex[pos_1][columns] = "\\ctrl{" + str(
                                pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\control\\qw"
                        elif nm == "cy":
                            self._latex[pos_1][columns] = "\\ctrl{" + str(
                                pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\gate{Y}"
                        elif nm == "ch":
                            self._latex[pos_1][columns] = "\\ctrl{" + str(
                                pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\gate{H}"
                        elif nm == "swap":
                            self._latex[pos_1][columns] = "\\qswap"
                            self._latex[pos_2][columns] = \
                                "\\qswap \\qwx[" + str(pos_1 - pos_2) + "]"
                        elif nm == "crz":
                            self._latex[pos_1][columns] = "\\ctrl{" + str(
                                pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = \
                                "\\gate{R_z(%s)}" % (op["op"].param[0])
                        elif nm == "cu1":
                            self._latex[pos_1][columns - 1] = "\\ctrl{" + str(
                                pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns - 1] = "\\control\\qw"
                            self._latex[min(pos_1, pos_2)][columns] = \
                                "\\dstick{%s}\\qw" % (op["op"].param[0])
                            self._latex[max(pos_1, pos_2)][columns] = "\\qw"
                        elif nm == "cu3":
                            self._latex[pos_1][columns] = "\\ctrl{" + str(
                                pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = ("\\gate{U_3(%s,%s,%s)}" % (
                                op["op"].param[0],
                                op["op"].param[1],
                                op["op"].param[2]))

                elif len(qarglist) == 3:
                    pos_1 = self.img_regs[(qarglist[0][0], qarglist[0][1])]
                    pos_2 = self.img_regs[(qarglist[1][0], qarglist[1][1])]
                    pos_3 = self.img_regs[(qarglist[2][0], qarglist[2][1])]

                    if 'condition' in op and op['condition']:
                        pos_4 = self.img_regs[(if_reg, 0)]

                        temp = [pos_1, pos_2, pos_3, pos_4]
                        temp.sort(key=int)
                        top = temp[0]
                        bottom = temp[2]

                        for i in range(top, pos_4 + 1):
                            if is_occupied[i] is False:
                                is_occupied[i] = True
                            else:
                                columns += 1
                                is_occupied = [False] * self.img_width
                                for j in range(top, pos_4 + 1):
                                    is_occupied[j] = True
                                break

                        prev_column = [x[columns - 1] for x in self._latex]
                        for item, prev_entry in enumerate(prev_column):
                            if 'barrier' in prev_entry:
                                span = re.search('barrier{(.*)}', prev_entry)
                                if span and any(i in temp for i in range(
                                        item, int(span.group(1)))):
                                    self._latex[item][columns - 1] = \
                                        prev_entry.replace(
                                            '\\barrier{',
                                            '\\barrier[-0.65em]{')

                        gap = pos_4 - bottom
                        for i in range(self.cregs[if_reg]):
                            if if_value[i] == '1':
                                self._latex[pos_4 + i][columns] = \
                                    "\\control \\cw \\cwx[-" + str(gap) + "]"
                                gap = 1
                            else:
                                self._latex[pos_4 + i][columns] = \
                                    "\\controlo \\cw \\cwx[-" + str(gap) + "]"
                                gap = 1

                        if nm == "ccx":
                            self._latex[pos_1][columns] = "\\ctrl{" + str(
                                pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\ctrl{" + str(
                                pos_3 - pos_2) + "}"
                            self._latex[pos_3][columns] = "\\targ"

                        if nm == "cswap":
                            self._latex[pos_1][columns] = "\\ctrl{" + str(
                                pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\qswap"
                            self._latex[pos_3][columns] = \
                                "\\qswap \\qwx[" + str(pos_2 - pos_3) + "]"
                    else:
                        temp = [pos_1, pos_2, pos_3]
                        temp.sort(key=int)
                        top = temp[0]
                        bottom = temp[2]

                        for i in range(top, bottom + 1):
                            if is_occupied[i] is False:
                                is_occupied[i] = True
                            else:
                                columns += 1
                                is_occupied = [False] * self.img_width
                                for j in range(top, bottom + 1):
                                    is_occupied[j] = True
                                break

                        prev_column = [x[columns - 1] for x in self._latex]
                        for item, prev_entry in enumerate(prev_column):
                            if 'barrier' in prev_entry:
                                span = re.search('barrier{(.*)}', prev_entry)
                                if span and any(i in temp for i in range(
                                        item, int(span.group(1)))):
                                    self._latex[item][columns - 1] = \
                                        prev_entry.replace(
                                            '\\barrier{',
                                            '\\barrier[-0.65em]{')

                        if nm == "ccx":
                            self._latex[pos_1][columns] = "\\ctrl{" + str(
                                pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\ctrl{" + str(
                                pos_3 - pos_2) + "}"
                            self._latex[pos_3][columns] = "\\targ"

                        if nm == "cswap":
                            self._latex[pos_1][columns] = "\\ctrl{" + str(
                                pos_2 - pos_1) + "}"
                            self._latex[pos_2][columns] = "\\qswap"
                            self._latex[pos_3][columns] = \
                                "\\qswap \\qwx[" + str(pos_2 - pos_3) + "]"

            elif op["name"] == "measure":
                if (len(op['cargs']) != 1
                        or len(op['qargs']) != 1
                        or op['op'].param):
                    raise _error.VisualizationError("bad operation record")
                if 'condition' in op and op['condition']:
                    raise _error.VisualizationError(
                        "If controlled measures currently not supported.")

                qname, qindex = op['qargs'][0]
                cname, cindex = op['cargs'][0]
                if aliases:
                    newq = aliases[(qname, qindex)]
                    qname = newq[0]
                    qindex = newq[1]

                pos_1 = self.img_regs[(qname, qindex)]
                pos_2 = self.img_regs[(cname, cindex)]

                for i in range(pos_1, pos_2 + 1):
                    if is_occupied[i] is False:
                        is_occupied[i] = True
                    else:
                        columns += 1
                        is_occupied = [False] * self.img_width
                        for j in range(pos_1, pos_2 + 1):
                            is_occupied[j] = True
                        break

                try:
                    self._latex[pos_1][columns] = "\\meter"
                    prev_column = [x[columns - 1] for x in self._latex]
                    for item, prev_entry in enumerate(prev_column):
                        if 'barrier' in prev_entry:
                            span = re.search('barrier{(.*)}', prev_entry)
                            if span and (
                                    item + int(span.group(1))) - pos_1 >= 0:
                                self._latex[item][columns - 1] = \
                                    prev_entry.replace(
                                        '\\barrier{',
                                        '\\barrier[-1.15em]{')

                    self._latex[pos_2][columns] = \
                        "\\cw \\cwx[-" + str(pos_2 - pos_1) + "]"
                except Exception as e:
                    raise _error.VisualizationError(
                        'Error during Latex building: %s' % str(e))
            elif op['name'] in ["barrier", 'snapshot', 'load', 'save',
                                'noise']:
                if self.plot_barriers:
                    qarglist = op['qargs']
                    indexes = [self._get_qubit_index(x) for x in qarglist]
                    start_bit = self.qubit_list[min(indexes)]
                    if aliases is not None:
                        qarglist = map(lambda x: aliases[x], qarglist)
                    start = self.img_regs[start_bit]
                    span = len(op['qargs']) - 1
                    for i in range(start, span + 1):
                        if is_occupied[i] is False:
                            is_occupied[i] = True
                        else:
                            columns += 1
                            is_occupied = [False] * self.img_width
                            for j in range(start, span + 1):
                                is_occupied[j] = True
                            break
                    self._latex[start][columns] = "\\qw \\barrier{" + str(
                        span) + "}"
            else:
                raise _error.VisualizationError("bad node data")

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
            raise _error.VisualizationError(
                "unable to find bit for operation")
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


def _get_register_specs(bit_labels):
    """Get the number and size of unique registers from bit_labels list.

    Args:
        bit_labels (list): this list is of the form::

            [['reg1', 0], ['reg1', 1], ['reg2', 0]]

            which indicates a register named "reg1" of size 2
            and a register named "reg2" of size 1. This is the
            format of classic and quantum bit labels in qobj
            header.

    Yields:
        tuple: iterator of register_name:size pairs.
    """
    it = itertools.groupby(bit_labels, operator.itemgetter(0))
    for register_name, sub_it in it:
        yield register_name, max(ind[1] for ind in sub_it) + 1


def _truncate_float(matchobj, format_str='0.2g'):
    """Truncate long floats

    Args:
        matchobj (re.Match): contains original float
        format_str (str): format specifier
    Returns:
       str: returns truncated float
    """
    if matchobj.group(0):
        return format(float(matchobj.group(0)), format_str)
    return ''
