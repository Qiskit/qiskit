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

# pylint: disable=invalid-name,missing-docstring,inconsistent-return-statements

"""mpl circuit visualization backend."""

import collections
import itertools
import json
import logging
import re
from warnings import warn

import numpy as np

try:
    from matplotlib import get_backend
    from matplotlib import patches
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from pylatexenc.latex2text import LatexNodes2Text

    HAS_PYLATEX = True
except ImportError:
    HAS_PYLATEX = False

from qiskit.circuit import ControlledGate
from qiskit.visualization.qcstyle import DefaultStyle, BWStyle
from qiskit import user_config
from qiskit.circuit.tools.pi_check import pi_check

logger = logging.getLogger(__name__)

# Default gate width and height
WID = 0.65
HIG = 0.65

BASE_SIZE = 3.01
PORDER_GATE = 5
PORDER_LINE = 3
PORDER_REGLINE = 2
PORDER_GRAY = 3
PORDER_TEXT = 6
PORDER_SUBP = 4


class Anchor:
    def __init__(self, reg_num, yind, fold):
        self.__yind = yind
        self.__fold = fold
        self.__reg_num = reg_num
        self.__gate_placed = []
        self.gate_anchor = 0

    def plot_coord(self, index, gate_width, x_offset):
        h_pos = index % self.__fold + 1
        # check folding
        if self.__fold > 0:
            if h_pos + (gate_width - 1) > self.__fold:
                index += self.__fold - (h_pos - 1)
            x_pos = index % self.__fold + 1 + 0.5 * (gate_width - 1)
            y_pos = self.__yind - (index // self.__fold) * (self.__reg_num + 1)
        else:
            x_pos = index + 1 + 0.5 * (gate_width - 1)
            y_pos = self.__yind

        # could have been updated, so need to store
        self.gate_anchor = index
        return x_pos + x_offset, y_pos

    def is_locatable(self, index, gate_width):
        hold = [index + i for i in range(gate_width)]
        for p in hold:
            if p in self.__gate_placed:
                return False
        return True

    def set_index(self, index, gate_width):
        if self.__fold < 2:
            _index = index
        else:
            h_pos = index % self.__fold + 1
            if h_pos + (gate_width - 1) > self.__fold:
                _index = index + self.__fold - (h_pos - 1) + 1
            else:
                _index = index
        for ii in range(gate_width):
            if _index + ii not in self.__gate_placed:
                self.__gate_placed.append(_index + ii)
        self.__gate_placed.sort()

    def get_index(self):
        if self.__gate_placed:
            return self.__gate_placed[-1] + 1
        return 0


class MatplotlibDrawer:
    def __init__(self, qregs, cregs, ops,
                 scale=None, style=None, plot_barriers=True,
                 reverse_bits=False, layout=None, fold=25, ax=None,
                 initial_state=False, cregbundle=True, global_phase=None):

        if not HAS_MATPLOTLIB:
            raise ImportError('The class MatplotlibDrawer needs matplotlib. '
                              'To install, run "pip install matplotlib".')

        if not HAS_PYLATEX:
            raise ImportError('The class MatplotlibDrawer needs pylatexenc. '
                              'to install, run "pip install pylatexenc".')

        self._ast = None
        self._scale = 1.0 if scale is None else scale
        self._creg = []
        self._qreg = []
        self._registers(cregs, qregs)
        self._ops = ops
        self.global_phase = global_phase

        self._qreg_dict = collections.OrderedDict()
        self._creg_dict = collections.OrderedDict()
        self._cond = {
            'n_lines': 0,
            'xmax': 0,
            'ymax': 0,
        }
        config = user_config.get_config()
        if config and (style is None):
            config_style = config.get('circuit_mpl_style', 'default')
            if config_style == 'default':
                self._style = DefaultStyle()
            elif config_style == 'bw':
                self._style = BWStyle()
        elif style is False:
            self._style = BWStyle()
        else:
            self._style = DefaultStyle()

        self.plot_barriers = plot_barriers
        self.reverse_bits = reverse_bits
        self.layout = layout
        self.initial_state = initial_state
        if style and 'cregbundle' in style.keys():
            self.cregbundle = style['cregbundle']
            del style['cregbundle']
            warn("The style dictionary key 'cregbundle' has been deprecated and will be removed"
                 " in a future release. cregbundle is now a parameter to draw()."
                 " Example: circuit.draw(output='mpl', cregbundle=False)", DeprecationWarning, 2)
        else:
            self.cregbundle = cregbundle

        if style:
            if isinstance(style, dict):
                self._style.set_style(style)
            elif isinstance(style, str):
                with open(style, 'r') as infile:
                    dic = json.load(infile)
                self._style.set_style(dic)

        if ax is None:
            self.return_fig = True
            self.figure = plt.figure()
            self.figure.patch.set_facecolor(color=self._style.bg)
            self.ax = self.figure.add_subplot(111)
        else:
            self.return_fig = False
            self.ax = ax
            self.figure = ax.get_figure()

        self.x_offset = 0
        self._reg_long_text = 0

        # default is to use character table for text width,
        # but get_renderer will work with some mpl backends
        """fig = plt.figure()
        if hasattr(fig.canvas, 'get_renderer'):
            self.renderer = fig.canvas.get_renderer()
        else:
            self.renderer = None"""
        self.renderer = None

        self.fold = fold
        if self.fold < 2:
            self.fold = -1

        self.ax.axis('off')
        self.ax.set_aspect('equal')
        self.ax.tick_params(labelbottom=False, labeltop=False,
                            labelleft=False, labelright=False)

        self._style.fs *= self._scale
        self._style.sfs *= self._scale
        self._lwidth15 = 1.5 * self._scale
        self._lwidth2 = 2.0 * self._scale

        # these char arrays are for finding text_width when not
        # using get_renderer method for the matplotlib backend
        self._char_list = {' ': (0.0958, 0.0583), '!': (0.1208, 0.0729), '"': (0.1396, 0.0875),
                           '#': (0.2521, 0.1562), '$': (0.1917, 0.1167), '%': (0.2854, 0.1771),
                           '&': (0.2333, 0.1458), "'": (0.0833, 0.0521), '(': (0.1167, 0.0729),
                           ')': (0.1167, 0.0729), '*': (0.15, 0.0938), '+': (0.25, 0.1562),
                           ',': (0.0958, 0.0583), '-': (0.1083, 0.0667), '.': (0.0958, 0.0604),
                           '/': (0.1021, 0.0625), '0': (0.1875, 0.1167), '1': (0.1896, 0.1167),
                           '2': (0.1917, 0.1188), '3': (0.1917, 0.1167), '4': (0.1917, 0.1188),
                           '5': (0.1917, 0.1167), '6': (0.1896, 0.1167), '7': (0.1917, 0.1188),
                           '8': (0.1896, 0.1188), '9': (0.1917, 0.1188), ':': (0.1021, 0.0604),
                           ';': (0.1021, 0.0604), '<': (0.25, 0.1542), '=': (0.25, 0.1562),
                           '>': (0.25, 0.1542), '?': (0.1583, 0.0979), '@': (0.2979, 0.1854),
                           'A': (0.2062, 0.1271), 'B': (0.2042, 0.1271), 'C': (0.2083, 0.1292),
                           'D': (0.2312, 0.1417), 'E': (0.1875, 0.1167), 'F': (0.1708, 0.1062),
                           'G': (0.2312, 0.1438), 'H': (0.225, 0.1396), 'I': (0.0875, 0.0542),
                           'J': (0.0875, 0.0542), 'K': (0.1958, 0.1208), 'L': (0.1667, 0.1042),
                           'M': (0.2583, 0.1604), 'N': (0.225, 0.1396), 'O': (0.2354, 0.1458),
                           'P': (0.1812, 0.1125), 'Q': (0.2354, 0.1458), 'R': (0.2083, 0.1292),
                           'S': (0.1896, 0.1188), 'T': (0.1854, 0.1125), 'U': (0.2208, 0.1354),
                           'V': (0.2062, 0.1271), 'W': (0.2958, 0.1833), 'X': (0.2062, 0.1271),
                           'Y': (0.1833, 0.1125), 'Z': (0.2042, 0.1271), '[': (0.1167, 0.075),
                           '\\': (0.1021, 0.0625), ']': (0.1167, 0.0729), '^': (0.2521, 0.1562),
                           '_': (0.1521, 0.0938), '`': (0.15, 0.0938), 'a': (0.1854, 0.1146),
                           'b': (0.1917, 0.1167), 'c': (0.1646, 0.1021), 'd': (0.1896, 0.1188),
                           'e': (0.1854, 0.1146), 'f': (0.1042, 0.0667), 'g': (0.1896, 0.1188),
                           'h': (0.1896, 0.1188), 'i': (0.0854, 0.0521), 'j': (0.0854, 0.0521),
                           'k': (0.1729, 0.1083), 'l': (0.0854, 0.0521), 'm': (0.2917, 0.1812),
                           'n': (0.1896, 0.1188), 'o': (0.1833, 0.1125), 'p': (0.1917, 0.1167),
                           'q': (0.1896, 0.1188), 'r': (0.125, 0.0771), 's': (0.1562, 0.0958),
                           't': (0.1167, 0.0729), 'u': (0.1896, 0.1188), 'v': (0.1771, 0.1104),
                           'w': (0.2458, 0.1521), 'x': (0.1771, 0.1104), 'y': (0.1771, 0.1104),
                           'z': (0.1562, 0.0979), '{': (0.1917, 0.1188), '|': (0.1, 0.0604),
                           '}': (0.1896, 0.1188)}

        self._mathmode_regex = re.compile(r"(?<!\\)\$(.*)(?<!\\)\$")

    def _registers(self, creg, qreg):
        self._creg = []
        for r in creg:
            self._creg.append(r)
        self._qreg = []
        for r in qreg:
            self._qreg.append(r)

    @property
    def ast(self):
        return self._ast

    # This computes the width of a string in the default font
    def _get_text_width(self, text, fontsize):
        if not text:
            return 0.0

        if self.renderer:
            t = plt.text(0.5, 0.5, text, fontsize=fontsize)
            return t.get_window_extent(renderer=self.renderer).width / 60.0
        else:
            math_mode_match = self._mathmode_regex.search(text)
            num_underscores = 0
            num_carets = 0
            if math_mode_match:
                math_mode_text = math_mode_match.group(1)
                num_underscores = math_mode_text.count('_')
                num_carets = math_mode_text.count('^')
            text = LatexNodes2Text().latex_to_text(text)
            # If there are subscripts or superscripts in mathtext string
            # we need to account for that spacing by manually removing
            # from text string for text length
            if num_underscores:
                text = text.replace('_', '', num_underscores)
            if num_carets:
                text = text.replace('^', '', num_carets)

            f = 0 if fontsize == self._style.fs else 1
            sum_text = 0.0
            for c in text:
                try:
                    sum_text += self._char_list[c][f]
                except KeyError:
                    # if non-ASCII char, use width of 'r', an average size
                    sum_text += self._char_list['r'][f]
            return sum_text

    def param_parse(self, v):
        # create an empty list to store the parameters in
        param_parts = [None] * len(v)
        for i, e in enumerate(v):
            try:
                param_parts[i] = pi_check(e, output='mpl', ndigits=3)
            except TypeError:
                param_parts[i] = str(e)

            if param_parts[i].startswith('-'):
                param_parts[i] = '$-$' + param_parts[i][1:]

        param_parts = ', '.join(param_parts)
        return param_parts

    def _get_gate_ctrl_text(self, op):
        op_label = getattr(op.op, 'label', None)
        base_name = None if not hasattr(op.op, 'base_gate') else op.op.base_gate.name
        base_label = None if not hasattr(op.op, 'base_gate') else op.op.base_gate.label
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

        if gate_text in self._style.disptex:
            gate_text = "{}".format(self._style.disptex[gate_text])
        else:
            gate_text = "{}".format(gate_text[0].upper() + gate_text[1:])

        if ctrl_text:
            ctrl_text = "{}".format(ctrl_text[0].upper() + ctrl_text[1:])
        return gate_text, ctrl_text

    def _get_colors(self, op):
        if op.name in self._style.dispcol:
            fc = self._style.dispcol[op.name]
        else:
            fc = self._style.gc
        if self._style.name != 'bw':
            ec = fc
            lc = fc
        else:
            ec = self._style.edge_color
            lc = self._style.lc
        if op.name == 'reset':
            gt = self._style.not_gate_lc
        else:
            gt = self._style.gt
        return fc, ec, gt, self._style.tc, self._style.sc, lc

    def _multiqubit_gate(self, xy, fc=None, ec=None, gt=None, sc=None, text='', subtext=''):
        xpos = min([x[0] for x in xy])
        ypos = min([y[1] for y in xy])
        ypos_max = max([y[1] for y in xy])

        # added .21 is for qubit numbers on the left side
        text_width = self._get_text_width(text, self._style.fs) + .21
        sub_width = self._get_text_width(subtext, self._style.sfs) + .21
        wid = max((text_width, sub_width, WID))

        qubit_span = abs(ypos) - abs(ypos_max) + 1
        height = HIG + (qubit_span - 1)
        box = patches.Rectangle(
            xy=(xpos - 0.5 * wid, ypos - .5 * HIG), width=wid, height=height,
            fc=fc, ec=ec, linewidth=self._lwidth15, zorder=PORDER_GATE)
        self.ax.add_patch(box)

        # annotate inputs
        for bit, y in enumerate([x[1] for x in xy]):
            self.ax.text(xpos + .07 - 0.5 * wid, y, str(bit), ha='left', va='center',
                         fontsize=self._style.fs, color=gt,
                         clip_on=True, zorder=PORDER_TEXT)
        if text:
            if subtext:
                self.ax.text(xpos + .11, ypos + 0.4 * height, text, ha='center',
                             va='center', fontsize=self._style.fs,
                             color=gt, clip_on=True,
                             zorder=PORDER_TEXT)
                self.ax.text(xpos + .11, ypos + 0.2 * height, subtext, ha='center',
                             va='center', fontsize=self._style.sfs,
                             color=sc, clip_on=True,
                             zorder=PORDER_TEXT)
            else:
                self.ax.text(xpos + .11, ypos + .5 * (qubit_span - 1), text,
                             ha='center', va='center', fontsize=self._style.fs,
                             color=gt, clip_on=True,
                             zorder=PORDER_TEXT, wrap=True)

    def _gate(self, xy, fc=None, ec=None, gt=None, sc=None, text='', subtext=''):
        xpos, ypos = xy

        text_width = self._get_text_width(text, self._style.fs)
        sub_width = self._get_text_width(subtext, self._style.sfs)
        wid = max((text_width, sub_width, WID))

        box = patches.Rectangle(xy=(xpos - 0.5 * wid, ypos - 0.5 * HIG),
                                width=wid, height=HIG, fc=fc, ec=ec,
                                linewidth=self._lwidth15, zorder=PORDER_GATE)
        self.ax.add_patch(box)

        if text:
            if subtext:
                self.ax.text(xpos, ypos + 0.15 * HIG, text, ha='center',
                             va='center', fontsize=self._style.fs, color=gt,
                             clip_on=True, zorder=PORDER_TEXT)
                self.ax.text(xpos, ypos - 0.3 * HIG, subtext, ha='center',
                             va='center', fontsize=self._style.sfs, color=sc,
                             clip_on=True, zorder=PORDER_TEXT)
            else:
                self.ax.text(xpos, ypos, text, ha='center', va='center',
                             fontsize=self._style.fs, color=gt,
                             clip_on=True, zorder=PORDER_TEXT)

    def _sidetext(self, xy, tc=None, text=''):
        xpos, ypos = xy

        # 0.08 = the initial gap, add 1/2 text width to place on the right
        text_width = self._get_text_width(text, self._style.sfs)
        xp = xpos + 0.08 + text_width / 2
        self.ax.text(xp, ypos + HIG, text, ha='center', va='top',
                     fontsize=self._style.sfs, color=tc,
                     clip_on=True, zorder=PORDER_TEXT)

    def _line(self, xy0, xy1, lc=None, ls=None, zorder=PORDER_LINE):
        x0, y0 = xy0
        x1, y1 = xy1
        linecolor = self._style.lc if lc is None else lc
        linestyle = 'solid' if ls is None else ls

        if linestyle == 'doublet':
            theta = np.arctan2(np.abs(x1 - x0), np.abs(y1 - y0))
            dx = 0.05 * WID * np.cos(theta)
            dy = 0.05 * WID * np.sin(theta)
            self.ax.plot([x0 + dx, x1 + dx], [y0 + dy, y1 + dy],
                         color=linecolor, linewidth=self._lwidth2,
                         linestyle='solid', zorder=zorder)
            self.ax.plot([x0 - dx, x1 - dx], [y0 - dy, y1 - dy],
                         color=linecolor, linewidth=self._lwidth2,
                         linestyle='solid', zorder=zorder)
        else:
            self.ax.plot([x0, x1], [y0, y1],
                         color=linecolor, linewidth=self._lwidth2,
                         linestyle=linestyle, zorder=zorder)

    def _measure(self, qxy, cxy, cid, fc=None, ec=None, gt=None, sc=None):
        qx, qy = qxy
        cx, cy = cxy

        # draw gate box
        self._gate(qxy, fc=fc, ec=ec, gt=gt, sc=sc)

        # add measure symbol
        arc = patches.Arc(xy=(qx, qy - 0.15 * HIG), width=WID * 0.7,
                          height=HIG * 0.7, theta1=0, theta2=180, fill=False,
                          ec=self._style.not_gate_lc, linewidth=self._lwidth2, zorder=PORDER_GATE)
        self.ax.add_patch(arc)
        self.ax.plot([qx, qx + 0.35 * WID], [qy - 0.15 * HIG, qy + 0.20 * HIG],
                     color=self._style.not_gate_lc, linewidth=self._lwidth2, zorder=PORDER_GATE)
        # arrow
        self._line(qxy, [cx, cy + 0.35 * WID], lc=self._style.cc, ls=self._style.cline)
        arrowhead = patches.Polygon(((cx - 0.20 * WID, cy + 0.35 * WID),
                                     (cx + 0.20 * WID, cy + 0.35 * WID),
                                     (cx, cy)), fc=self._style.cc, ec=None)
        self.ax.add_artist(arrowhead)
        # target
        if self.cregbundle:
            self.ax.text(cx + .25, cy + .1, str(cid), ha='left', va='bottom',
                         fontsize=0.8 * self._style.fs, color=self._style.tc,
                         clip_on=True, zorder=PORDER_TEXT)

    def _conds(self, xy, istrue=False):
        xpos, ypos = xy

        fc = self._style.lc if istrue else self._style.bg
        box = patches.Circle(xy=(xpos, ypos), radius=WID * 0.15, fc=fc,
                             ec=self._style.lc, linewidth=self._lwidth15, zorder=PORDER_GATE)
        self.ax.add_patch(box)

    def _ctrl_qubit(self, xy, fc=None, ec=None, tc=None, text='', text_top=None):
        xpos, ypos = xy
        box = patches.Circle(xy=(xpos, ypos), radius=WID * 0.15,
                             fc=fc, ec=ec, linewidth=self._lwidth15, zorder=PORDER_GATE)
        self.ax.add_patch(box)
        # display the control label at the top or bottom if there is one
        if text_top is True:
            self.ax.text(xpos, ypos + 0.7 * HIG, text, ha='center', va='top',
                         fontsize=self._style.sfs, color=self._style.tc,
                         clip_on=True, zorder=PORDER_TEXT)
        elif text_top is False:
            self.ax.text(xpos, ypos - 0.3 * HIG, text, ha='center', va='top',
                         fontsize=self._style.sfs, color=tc,
                         clip_on=True, zorder=PORDER_TEXT)

    def _set_ctrl_bits(self, ctrl_state, num_ctrl_qubits, qbit, ec=None, tc=None,
                       text='', qargs=None):
        # place the control label at the top or bottom of controls
        if text:
            qlist = [qubit.index for qubit in qargs]
            ctbits = qlist[:num_ctrl_qubits]
            qubits = qlist[num_ctrl_qubits:]
            max_ctbit = max(ctbits)
            min_ctbit = min(ctbits)
            top = min(qubits) > min_ctbit

        # display the control qubits as open or closed based on ctrl_state
        cstate = "{0:b}".format(ctrl_state).rjust(num_ctrl_qubits, '0')[::-1]
        for i in range(num_ctrl_qubits):
            fc_open_close = ec if cstate[i] == '1' else self._style.bg
            text_top = None
            if text:
                if top and qlist[i] == min_ctbit:
                    text_top = True
                elif not top and qlist[i] == max_ctbit:
                    text_top = False
            self._ctrl_qubit(qbit[i], fc=fc_open_close, ec=ec, tc=tc,
                             text=text, text_top=text_top)

    def _x_tgt_qubit(self, xy, ec=None, ac=None):
        linewidth = self._lwidth2
        xpos, ypos = xy
        box = patches.Circle(xy=(xpos, ypos), radius=HIG * 0.35,
                             fc=ec, ec=ec, linewidth=linewidth,
                             zorder=PORDER_GATE)
        self.ax.add_patch(box)

        # add '+' symbol
        self.ax.plot([xpos, xpos], [ypos - 0.2 * HIG, ypos + 0.2 * HIG],
                     color=ac, linewidth=linewidth, zorder=PORDER_GATE + 1)
        self.ax.plot([xpos - 0.2 * HIG, xpos + 0.2 * HIG], [ypos, ypos],
                     color=ac, linewidth=linewidth, zorder=PORDER_GATE + 1)

    def _swap(self, xy, color=None):
        xpos, ypos = xy

        self.ax.plot([xpos - 0.20 * WID, xpos + 0.20 * WID],
                     [ypos - 0.20 * WID, ypos + 0.20 * WID],
                     color=color, linewidth=self._lwidth2, zorder=PORDER_LINE + 1)
        self.ax.plot([xpos - 0.20 * WID, xpos + 0.20 * WID],
                     [ypos + 0.20 * WID, ypos - 0.20 * WID],
                     color=color, linewidth=self._lwidth2, zorder=PORDER_LINE + 1)

    def _barrier(self, config):
        xys = config['coord']
        for xy in xys:
            xpos, ypos = xy
            self.ax.plot([xpos, xpos], [ypos + 0.5, ypos - 0.5],
                         linewidth=self._scale, linestyle="dashed",
                         color=self._style.lc, zorder=PORDER_TEXT)
            box = patches.Rectangle(xy=(xpos - (0.3 * WID), ypos - 0.5),
                                    width=0.6 * WID, height=1,
                                    fc=self._style.bc, ec=None, alpha=0.6,
                                    linewidth=self._lwidth15, zorder=PORDER_GRAY)
            self.ax.add_patch(box)

    def _linefeed_mark(self, xy):
        xpos, ypos = xy

        self.ax.plot([xpos - .1, xpos - .1],
                     [ypos, ypos - self._cond['n_lines'] + 1],
                     color=self._style.lc, linewidth=self._lwidth15, zorder=PORDER_LINE)
        self.ax.plot([xpos + .1, xpos + .1],
                     [ypos, ypos - self._cond['n_lines'] + 1],
                     color=self._style.lc, linewidth=self._lwidth15, zorder=PORDER_LINE)

    def draw(self, filename=None, verbose=False):
        self._draw_regs()
        self._draw_ops(verbose)
        _xl = - self._style.margin[0]
        _xr = self._cond['xmax'] + self._style.margin[1]
        _yb = - self._cond['ymax'] - self._style.margin[2] + 1 - 0.5
        _yt = self._style.margin[3] + 0.5
        self.ax.set_xlim(_xl, _xr)
        self.ax.set_ylim(_yb, _yt)

        # update figure size
        fig_w = _xr - _xl
        fig_h = _yt - _yb
        if self._style.figwidth < 0.0:
            self._style.figwidth = fig_w * BASE_SIZE * self._style.fs / 72 / WID
        self.figure.set_size_inches(self._style.figwidth, self._style.figwidth * fig_h / fig_w)
        if self.global_phase:
            plt.text(_xl, _yt, 'Global Phase: %s' % pi_check(self.global_phase,
                                                             output='mpl'))

        if filename:
            self.figure.savefig(filename, dpi=self._style.dpi,
                                bbox_inches='tight', facecolor=self.figure.get_facecolor())
        if self.return_fig:
            if get_backend() in ['module://ipykernel.pylab.backend_inline',
                                 'nbAgg']:
                plt.close(self.figure)
            return self.figure

    def _draw_regs(self):
        longest_label_width = 0
        if self.initial_state:
            initial_qbit = ' |0>'
            initial_cbit = ' 0'
        else:
            initial_qbit = ''
            initial_cbit = ''

        def _fix_double_script(label):
            words = label.split(' ')
            words = [word.replace('_', r'\_') if word.count('_') > 1 else word
                     for word in words]
            words = [word.replace('^', r'\^{\ }') if word.count('^') > 1 else word
                     for word in words]
            label = ' '.join(words).replace(' ', '\\;')
            return label

        # quantum register
        for ii, reg in enumerate(self._qreg):
            if len(self._qreg) > 1:
                if self.layout is None:
                    label = '${{{name}}}_{{{index}}}$'.format(name=reg.register.name,
                                                              index=reg.index)
                    label = _fix_double_script(label) + initial_qbit
                    text_width = self._get_text_width(label, self._style.fs)
                else:
                    if self.layout[reg.index]:
                        label = '${{{name}}}_{{{index}}} \\mapsto {{{physical}}}$'.format(
                            name=self.layout[reg.index].register.name,
                            index=self.layout[reg.index].index, physical=reg.index)
                    else:
                        label = '${{{physical}}}$'.format(physical=reg.index)
                    label = _fix_double_script(label) + initial_qbit
                    text_width = self._get_text_width(label, self._style.fs)
            else:
                label = '{name}'.format(name=reg.register.name)
                label = _fix_double_script(label) + initial_qbit
                text_width = self._get_text_width(label, self._style.fs)

            text_width = text_width * 1.15  # to account for larger font used
            if text_width > longest_label_width:
                longest_label_width = text_width

            pos = -ii
            self._qreg_dict[ii] = {
                'y': pos, 'label': label, 'index': reg.index, 'group': reg.register}
            self._cond['n_lines'] += 1

        # classical register
        if self._creg:
            n_creg = self._creg.copy()
            n_creg.pop(0)
            idx = 0
            y_off = -len(self._qreg)
            for ii, (reg, nreg) in enumerate(itertools.zip_longest(self._creg, n_creg)):
                pos = y_off - idx
                if self.cregbundle:
                    label = '{}'.format(reg.register.name)
                    label = _fix_double_script(label) + initial_cbit
                    text_width = self._get_text_width(reg.register.name, self._style.fs) * 1.15
                    if text_width > longest_label_width:
                        longest_label_width = text_width
                    self._creg_dict[ii] = {'y': pos, 'label': label, 'index': reg.index,
                                           'group': reg.register}
                    if not (not nreg or reg.register != nreg.register):
                        continue
                else:
                    label = '${}_{{{}}}$'.format(reg.register.name, reg.index)
                    label = _fix_double_script(label) + initial_cbit
                    text_width = self._get_text_width(reg.register.name, self._style.fs) * 1.15
                    if text_width > longest_label_width:
                        longest_label_width = text_width
                    self._creg_dict[ii] = {'y': pos, 'label': label, 'index': reg.index,
                                           'group': reg.register}
                self._cond['n_lines'] += 1
                idx += 1

        self._reg_long_text = longest_label_width
        self.x_offset = -1.2 + self._reg_long_text

    def _draw_regs_sub(self, n_fold, feedline_l=False, feedline_r=False):
        # quantum register
        for qreg in self._qreg_dict.values():
            label = qreg['label']
            y = qreg['y'] - n_fold * (self._cond['n_lines'] + 1)
            self.ax.text(self.x_offset - 0.2, y, label, ha='right', va='center',
                         fontsize=1.25 * self._style.fs, color=self._style.tc,
                         clip_on=True, zorder=PORDER_TEXT)
            self._line([self.x_offset + 0.2, y], [self._cond['xmax'], y],
                       zorder=PORDER_REGLINE)

        # classical register
        this_creg_dict = {}
        for creg in self._creg_dict.values():
            label = creg['label']
            y = creg['y'] - n_fold * (self._cond['n_lines'] + 1)
            if y not in this_creg_dict.keys():
                this_creg_dict[y] = {'val': 1, 'label': label}
            else:
                this_creg_dict[y]['val'] += 1
        for y, this_creg in this_creg_dict.items():
            # cregbundle
            if this_creg['val'] > 1:
                self.ax.plot([self.x_offset + 0.64, self.x_offset + 0.74], [y - .1, y + .1],
                             color=self._style.cc, zorder=PORDER_LINE)
                self.ax.text(self.x_offset + 0.54, y + .1, str(this_creg['val']), ha='left',
                             va='bottom', fontsize=0.8 * self._style.fs,
                             color=self._style.tc, clip_on=True, zorder=PORDER_TEXT)
            self.ax.text(self.x_offset - 0.2, y, this_creg['label'], ha='right', va='center',
                         fontsize=1.25 * self._style.fs, color=self._style.tc,
                         clip_on=True, zorder=PORDER_TEXT)
            self._line([self.x_offset + 0.2, y], [self._cond['xmax'], y], lc=self._style.cc,
                       ls=self._style.cline, zorder=PORDER_REGLINE)

        # lf line
        if feedline_r:
            self._linefeed_mark((self.fold + self.x_offset + 1 - 0.1,
                                 - n_fold * (self._cond['n_lines'] + 1)))
        if feedline_l:
            self._linefeed_mark((self.x_offset + 0.3,
                                 - n_fold * (self._cond['n_lines'] + 1)))

    def _draw_ops(self, verbose=False):
        _standard_gates = ['x', 'y', 'z', 'id', 'h', 'r', 's', 'sdg', 't', 'tdg', 'rx', 'ry', 'rz',
                           'rxx', 'ryy', 'rzx', 'u1', 'u2', 'u3', 'swap', 'reset', 'sx', 'sxdg']
        _barrier_gates = ['barrier', 'snapshot', 'load', 'save', 'noise']
        _barriers = {'coord': [], 'group': []}

        #
        # generate coordinate manager
        #
        q_anchors = {}
        for key, qreg in self._qreg_dict.items():
            q_anchors[key] = Anchor(reg_num=self._cond['n_lines'],
                                    yind=qreg['y'], fold=self.fold)
        c_anchors = {}
        for key, creg in self._creg_dict.items():
            c_anchors[key] = Anchor(reg_num=self._cond['n_lines'],
                                    yind=creg['y'], fold=self.fold)
        #
        # draw the ops
        #
        prev_anc = -1
        for layer in self._ops:
            widest_box = 0.0
            #
            # compute the layer_width for this layer
            #
            for op in layer:
                if op.name in [*_barrier_gates, 'measure']:
                    continue

                base_name = None if not hasattr(op.op, 'base_gate') else op.op.base_gate.name
                gate_text, ctrl_text = self._get_gate_ctrl_text(op)

                # if a standard_gate, no params, and no labels, layer_width is 1
                if (not hasattr(op.op, 'params') and
                        ((op.name in _standard_gates or base_name in _standard_gates)
                         and gate_text in (op.name, base_name) and ctrl_text is None)):
                    continue

                # small increments at end of the 3 _get_text_width calls are for small
                # spacing adjustments between gates
                ctrl_width = self._get_text_width(ctrl_text, fontsize=self._style.sfs) - 0.05

                # get param_width, but 0 for gates with array params
                if (hasattr(op.op, 'params')
                        and not any([isinstance(param, np.ndarray) for param in op.op.params])
                        and len(op.op.params) > 0):
                    param = self.param_parse(op.op.params)
                    if op.name == 'initialize':
                        param = '[%s]' % param
                    param = "${}$".format(param)
                    param_width = self._get_text_width(param, fontsize=self._style.sfs) + 0.08
                else:
                    param_width = 0.0

                if op.name == 'cu1' or op.name == 'rzz' or base_name == 'rzz':
                    tname = 'U1' if op.name == 'cu1' else 'zz'
                    gate_width = (self._get_text_width(tname + ' ()',
                                                       fontsize=self._style.sfs)
                                  + param_width) * 1.5
                else:
                    gate_width = self._get_text_width(gate_text, fontsize=self._style.fs) + 0.10
                    # add .21 for the qubit numbers on the left of the multibit gates
                    if (op.name not in _standard_gates and base_name not in _standard_gates):
                        gate_width += 0.21

                box_width = max((gate_width, ctrl_width, param_width, WID))
                if box_width > widest_box:
                    widest_box = box_width

            layer_width = int(widest_box) + 1
            this_anc = prev_anc + 1
            #
            # draw the gates in this layer
            #
            for op in layer:
                base_name = None if not hasattr(op.op, 'base_gate') else op.op.base_gate.name
                gate_text, ctrl_text = self._get_gate_ctrl_text(op)
                fc, ec, gt, tc, sc, lc = self._get_colors(op)

                # get qreg index
                q_idxs = []
                for qarg in op.qargs:
                    for index, reg in self._qreg_dict.items():
                        if (reg['group'] == qarg.register and
                                reg['index'] == qarg.index):
                            q_idxs.append(index)
                            break

                # get creg index
                c_idxs = []
                for carg in op.cargs:
                    for index, reg in self._creg_dict.items():
                        if (reg['group'] == carg.register and
                                reg['index'] == carg.index):
                            c_idxs.append(index)
                            break

                # only add the gate to the anchors if it is going to be plotted.
                # this prevents additional blank wires at the end of the line if
                # the last instruction is a barrier type
                if self.plot_barriers or op.name not in _barrier_gates:
                    for ii in q_idxs:
                        q_anchors[ii].set_index(this_anc, layer_width)

                # qreg coordinate
                q_xy = [q_anchors[ii].plot_coord(this_anc, layer_width, self.x_offset)
                        for ii in q_idxs]
                # creg coordinate
                c_xy = [c_anchors[ii].plot_coord(this_anc, layer_width, self.x_offset)
                        for ii in c_idxs]
                # bottom and top point of qreg
                qreg_b = min(q_xy, key=lambda xy: xy[1])
                qreg_t = max(q_xy, key=lambda xy: xy[1])

                # update index based on the value from plotting
                this_anc = q_anchors[q_idxs[0]].gate_anchor

                if verbose:
                    print(op)

                # load param
                if (op.type == 'op' and hasattr(op.op, 'params') and len(op.op.params) > 0
                        and not any([isinstance(param, np.ndarray) for param in op.op.params])):
                    param = "{}".format(self.param_parse(op.op.params))
                else:
                    param = ''

                # conditional gate
                if op.condition:
                    c_xy = [c_anchors[ii].plot_coord(this_anc, layer_width, self.x_offset) for
                            ii in self._creg_dict]
                    mask = 0
                    for index, cbit in enumerate(self._creg):
                        if cbit.register == op.condition[0]:
                            mask |= (1 << index)
                    val = op.condition[1]
                    # cbit list to consider
                    fmt_c = '{{:0{}b}}'.format(len(c_xy))
                    cmask = list(fmt_c.format(mask))[::-1]
                    # value
                    fmt_v = '{{:0{}b}}'.format(cmask.count('1'))
                    vlist = list(fmt_v.format(val))[::-1]
                    # plot conditionals
                    v_ind = 0
                    xy_plot = []
                    for xy, m in zip(c_xy, cmask):
                        if m == '1':
                            if xy not in xy_plot:
                                if vlist[v_ind] == '1' or self.cregbundle:
                                    self._conds(xy, istrue=True)
                                else:
                                    self._conds(xy, istrue=False)
                                xy_plot.append(xy)
                            v_ind += 1
                    creg_b = sorted(xy_plot, key=lambda xy: xy[1])[0]
                    xpos, ypos = creg_b
                    self.ax.text(xpos, ypos - 0.3 * HIG, hex(val), ha='center', va='top',
                                 fontsize=self._style.sfs, color=self._style.tc,
                                 clip_on=True, zorder=PORDER_TEXT)
                    self._line(qreg_t, creg_b, lc=self._style.cc,
                               ls=self._style.cline)
                #
                # draw special gates
                #
                if op.name == 'measure':
                    vv = self._creg_dict[c_idxs[0]]['index']
                    self._measure(q_xy[0], c_xy[0], vv, fc=fc, ec=ec, gt=gt, sc=sc)

                elif op.name in _barrier_gates:
                    _barriers = {'coord': [], 'group': []}
                    for index, qbit in enumerate(q_idxs):
                        q_group = self._qreg_dict[qbit]['group']
                        if q_group not in _barriers['group']:
                            _barriers['group'].append(q_group)
                        _barriers['coord'].append(q_xy[index])
                    if self.plot_barriers:
                        self._barrier(_barriers)

                elif op.name == 'initialize':
                    vec = "$[{}]$".format(param.replace('$', ''))
                    if len(q_xy) == 1:
                        self._gate(q_xy[0], fc=fc, ec=ec, gt=gt, sc=sc,
                                   text=gate_text, subtext=vec)
                    else:
                        self._multiqubit_gate(q_xy, fc=fc, ec=ec, gt=gt, sc=sc,
                                              text=gate_text, subtext=vec)
                #
                # draw single qubit gates
                #
                elif len(q_xy) == 1:
                    self._gate(q_xy[0], fc=fc, ec=ec, gt=gt, sc=sc,
                               text=gate_text, subtext=str(param))
                #
                # draw controlled and special gates
                #
                # cx gates
                elif isinstance(op.op, ControlledGate) and base_name == 'x':
                    num_ctrl_qubits = op.op.num_ctrl_qubits
                    self._set_ctrl_bits(op.op.ctrl_state, num_ctrl_qubits,
                                        q_xy, ec=ec, tc=tc, text=ctrl_text, qargs=op.qargs)
                    self._x_tgt_qubit(q_xy[num_ctrl_qubits], ec=ec,
                                      ac=self._style.dispcol['target'])
                    self._line(qreg_b, qreg_t, lc=lc)

                # cz gate
                elif op.name == 'cz':
                    num_ctrl_qubits = op.op.num_ctrl_qubits
                    self._set_ctrl_bits(op.op.ctrl_state, num_ctrl_qubits,
                                        q_xy, ec=ec, tc=tc, text=ctrl_text, qargs=op.qargs)
                    self._ctrl_qubit(q_xy[1], fc=ec, ec=ec, tc=tc)
                    self._line(qreg_b, qreg_t, lc=lc, zorder=PORDER_LINE + 1)

                # cu1, rzz, and controlled rzz gates (sidetext gates)
                elif (op.name == 'cu1' or op.name == 'rzz' or base_name == 'rzz'):
                    num_ctrl_qubits = 0 if op.name == 'rzz' else op.op.num_ctrl_qubits
                    if op.name != 'rzz':
                        self._set_ctrl_bits(op.op.ctrl_state, num_ctrl_qubits,
                                            q_xy, ec=ec, tc=tc, text=ctrl_text, qargs=op.qargs)
                    self._ctrl_qubit(q_xy[num_ctrl_qubits], fc=ec, ec=ec, tc=tc)
                    if op.name != 'cu1':
                        self._ctrl_qubit(q_xy[num_ctrl_qubits + 1], fc=ec, ec=ec, tc=tc)
                    stext = self._style.disptex['u1'] if op.name == 'cu1' else 'zz'
                    self._sidetext(qreg_b, tc=tc,
                                   text='{}'.format(stext) + ' ' + '({})'.format(param))
                    self._line(qreg_b, qreg_t, lc=lc)

                # swap gate
                elif op.name == 'swap':
                    self._swap(q_xy[0], color=lc)
                    self._swap(q_xy[1], color=lc)
                    self._line(qreg_b, qreg_t, lc=lc)

                # cswap gate
                elif op.name != 'swap' and base_name == 'swap':
                    num_ctrl_qubits = op.op.num_ctrl_qubits
                    self._set_ctrl_bits(op.op.ctrl_state, num_ctrl_qubits,
                                        q_xy, ec=ec, tc=tc, text=ctrl_text, qargs=op.qargs)
                    self._swap(q_xy[num_ctrl_qubits], color=lc)
                    self._swap(q_xy[num_ctrl_qubits + 1], color=lc)
                    self._line(qreg_b, qreg_t, lc=lc)

                # all other controlled gates
                elif isinstance(op.op, ControlledGate):
                    num_ctrl_qubits = op.op.num_ctrl_qubits
                    num_qargs = len(q_xy) - num_ctrl_qubits
                    self._set_ctrl_bits(op.op.ctrl_state, num_ctrl_qubits,
                                        q_xy, ec=ec, tc=tc, text=ctrl_text, qargs=op.qargs)
                    self._line(qreg_b, qreg_t, lc=lc)
                    if num_qargs == 1:
                        self._gate(q_xy[num_ctrl_qubits], fc=fc, ec=ec, gt=gt, sc=sc,
                                   text=gate_text, subtext='{}'.format(param))
                    else:
                        self._multiqubit_gate(q_xy[num_ctrl_qubits:], fc=fc, ec=ec, gt=gt,
                                              sc=sc, text=gate_text, subtext='{}'.format(param))

                # draw multi-qubit gate as final default
                else:
                    self._multiqubit_gate(q_xy, fc=fc, ec=ec, gt=gt, sc=sc,
                                          text=gate_text, subtext='{}'.format(param))

            # adjust the column if there have been barriers encountered, but not plotted
            barrier_offset = 0
            if not self.plot_barriers:
                # only adjust if everything in the layer wasn't plotted
                barrier_offset = -1 if all([op.name in _barrier_gates for op in layer]) else 0

            prev_anc = this_anc + layer_width + barrier_offset - 1
        #
        # adjust window size and draw horizontal lines
        #
        anchors = [q_anchors[ii].get_index() for ii in self._qreg_dict]
        max_anc = max(anchors) if anchors else 0
        n_fold = max(0, max_anc - 1) // self.fold if self.fold > 0 else 0

        # window size
        if max_anc > self.fold > 0:
            self._cond['xmax'] = self.fold + 1 + self.x_offset
            self._cond['ymax'] = (n_fold + 1) * (self._cond['n_lines'] + 1) - 1
        else:
            self._cond['xmax'] = max_anc + 1 + self.x_offset
            self._cond['ymax'] = self._cond['n_lines']

        # add horizontal lines
        for ii in range(n_fold + 1):
            feedline_r = (n_fold > 0 and n_fold > ii)
            feedline_l = (ii > 0)
            self._draw_regs_sub(ii, feedline_l, feedline_r)

        # draw anchor index number
        if self._style.index:
            for ii in range(max_anc):
                if self.fold > 0:
                    x_coord = ii % self.fold + self._reg_long_text - 0.2
                    y_coord = - (ii // self.fold) * (self._cond['n_lines'] + 1) + 0.7
                else:
                    x_coord = ii + self._reg_long_text - 0.2
                    y_coord = 0.7
                self.ax.text(x_coord, y_coord, str(ii + 1), ha='center',
                             va='center', fontsize=self._style.sfs,
                             color=self._style.tc, clip_on=True, zorder=PORDER_TEXT)
