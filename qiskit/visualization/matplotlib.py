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
import fractions
import itertools
import json
import logging
import math

import numpy as np

try:
    from matplotlib import get_backend
    from matplotlib import patches
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from qiskit.circuit import ControlledGate
from qiskit.visualization import exceptions
from qiskit.visualization.qcstyle import DefaultStyle, BWStyle
from qiskit import user_config
from .tools.pi_check import pi_check

logger = logging.getLogger(__name__)

WID = 0.65
HIG = 0.65
DEFAULT_SCALE = 4.3
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
        h_pos = index % self.__fold + 1
        if h_pos + (gate_width - 1) > self.__fold:
            _index = index + self.__fold - (h_pos - 1)
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
                 scale=1.0, style=None, plot_barriers=True,
                 reverse_bits=False, layout=None, fold=25, ax=None):

        if not HAS_MATPLOTLIB:
            raise ImportError('The class MatplotlibDrawer needs matplotlib. '
                              'Run "pip install matplotlib" before.')

        self._ast = None
        self._scale = DEFAULT_SCALE * scale
        self._creg = []
        self._qreg = []
        self._registers(cregs, qregs)
        self._ops = ops

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

        # TODO: self._style.fold should be removed after deprecation
        self.fold = self._style.fold or fold
        if self.fold < 2:
            self.fold = -1

        self.ax.axis('off')
        self.ax.set_aspect('equal')
        self.ax.tick_params(labelbottom=False, labeltop=False,
                            labelleft=False, labelright=False)

        self.x_offset = 0

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

    def _custom_multiqubit_gate(self, xy, cxy=None, fc=None, wide=True, text=None,
                                subtext=None):
        xpos = min([x[0] for x in xy])
        ypos = min([y[1] for y in xy])
        ypos_max = max([y[1] for y in xy])

        if cxy:
            ypos = min([y[1] for y in cxy])
        if wide:
            if subtext:
                boxes_length = round(max([len(text), len(subtext)]) / 7) or 1
            else:
                boxes_length = math.ceil(len(text) / 7) or 1
            wid = WID * 2.5 * boxes_length
        else:
            wid = WID

        if fc:
            _fc = fc
        else:
            if self._style.name != 'bw':
                if self._style.gc != DefaultStyle().gc:
                    _fc = self._style.gc
                else:
                    _fc = self._style.dispcol['multi']
                _ec = self._style.dispcol['multi']
            else:
                _fc = self._style.gc

        qubit_span = abs(ypos) - abs(ypos_max) + 1
        height = HIG + (qubit_span - 1)
        box = patches.Rectangle(
            xy=(xpos - 0.5 * wid, ypos - .5 * HIG),
            width=wid, height=height,
            fc=_fc,
            ec=self._style.dispcol['multi'],
            linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)
        # Annotate inputs
        for bit, y in enumerate([x[1] for x in xy]):
            self.ax.text(xpos - 0.45 * wid, y, str(bit), ha='left', va='center',
                         fontsize=self._style.fs, color=self._style.gt,
                         clip_on=True, zorder=PORDER_TEXT)

        if text:

            disp_text = text
            if subtext:
                self.ax.text(xpos, ypos + 0.5 * height, disp_text, ha='center',
                             va='center', fontsize=self._style.fs,
                             color=self._style.gt, clip_on=True,
                             zorder=PORDER_TEXT)
                self.ax.text(xpos, ypos + 0.3 * height, subtext, ha='center',
                             va='center', fontsize=self._style.sfs,
                             color=self._style.sc, clip_on=True,
                             zorder=PORDER_TEXT)
            else:
                self.ax.text(xpos, ypos + .5 * (qubit_span - 1), disp_text,
                             ha='center',
                             va='center',
                             fontsize=self._style.fs,
                             color=self._style.gt,
                             clip_on=True,
                             zorder=PORDER_TEXT,
                             wrap=True)

    def _gate(self, xy, fc=None, wide=False, text=None, subtext=None):
        xpos, ypos = xy

        if wide:
            if subtext:
                subtext_len = len(subtext)
                if '$\\pi$' in subtext:
                    pi_count = subtext.count('pi')
                    subtext_len = subtext_len - (4 * pi_count)

                boxes_wide = round(max(subtext_len, len(text)) / 10, 1) or 1
                wid = WID * 1.5 * boxes_wide
            else:
                boxes_wide = round(len(text) / 10) or 1
                wid = WID * 2.2 * boxes_wide
            if wid < WID:
                wid = WID
        else:
            wid = WID
        if fc:
            _fc = fc
        elif self._style.gc != DefaultStyle().gc:
            _fc = self._style.gc
        elif text and text in self._style.dispcol:
            _fc = self._style.dispcol[text]
        else:
            _fc = self._style.gc

        box = patches.Rectangle(
            xy=(xpos - 0.5 * wid, ypos - 0.5 * HIG), width=wid, height=HIG,
            fc=_fc, ec=self._style.edge_color, linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)

        if text:
            font_size = self._style.fs
            sub_font_size = self._style.sfs
            # check if gate is not unitary
            if text in ['reset']:
                disp_color = self._style.not_gate_lc
                sub_color = self._style.not_gate_lc
                font_size = self._style.math_fs

            else:
                disp_color = self._style.gt
                sub_color = self._style.sc

            if text in self._style.dispcol:
                disp_text = "${}$".format(self._style.disptex[text])
            else:
                disp_text = text

            if subtext:
                self.ax.text(xpos, ypos + 0.15 * HIG, disp_text, ha='center',
                             va='center', fontsize=font_size,
                             color=disp_color, clip_on=True,
                             zorder=PORDER_TEXT)
                self.ax.text(xpos, ypos - 0.3 * HIG, subtext, ha='center',
                             va='center', fontsize=sub_font_size,
                             color=sub_color, clip_on=True,
                             zorder=PORDER_TEXT)
            else:
                self.ax.text(xpos, ypos, disp_text, ha='center', va='center',
                             fontsize=font_size,
                             color=disp_color,
                             clip_on=True,
                             zorder=PORDER_TEXT)

    def _subtext(self, xy, text):
        xpos, ypos = xy

        self.ax.text(xpos, ypos - 0.3 * HIG, text, ha='center', va='top',
                     fontsize=self._style.sfs,
                     color=self._style.tc,
                     clip_on=True,
                     zorder=PORDER_TEXT)

    def _sidetext(self, xy, text):
        xpos, ypos = xy

        # 0.15 = the initial gap, each char means it needs to move
        # another 0.0375 over
        xp = xpos + 0.15 + (0.0375 * len(text))
        self.ax.text(xp, ypos + HIG, text, ha='center', va='top',
                     fontsize=self._style.sfs,
                     color=self._style.tc,
                     clip_on=True,
                     zorder=PORDER_TEXT)

    def _line(self, xy0, xy1, lc=None, ls=None, zorder=PORDER_LINE):
        x0, y0 = xy0
        x1, y1 = xy1
        if lc is None:
            linecolor = self._style.lc
        else:
            linecolor = lc
        if ls is None:
            linestyle = 'solid'
        else:
            linestyle = ls

        if linestyle == 'doublet':
            theta = np.arctan2(np.abs(x1 - x0), np.abs(y1 - y0))
            dx = 0.05 * WID * np.cos(theta)
            dy = 0.05 * WID * np.sin(theta)
            self.ax.plot([x0 + dx, x1 + dx], [y0 + dy, y1 + dy],
                         color=linecolor,
                         linewidth=2,
                         linestyle='solid',
                         zorder=zorder)
            self.ax.plot([x0 - dx, x1 - dx], [y0 - dy, y1 - dy],
                         color=linecolor,
                         linewidth=2,
                         linestyle='solid',
                         zorder=zorder)
        else:
            self.ax.plot([x0, x1], [y0, y1],
                         color=linecolor,
                         linewidth=2,
                         linestyle=linestyle,
                         zorder=zorder)

    def _measure(self, qxy, cxy, cid):
        qx, qy = qxy
        cx, cy = cxy

        self._gate(qxy, fc=self._style.dispcol['meas'])

        # add measure symbol
        arc = patches.Arc(xy=(qx, qy - 0.15 * HIG), width=WID * 0.7,
                          height=HIG * 0.7, theta1=0, theta2=180, fill=False,
                          ec=self._style.not_gate_lc, linewidth=2,
                          zorder=PORDER_GATE)
        self.ax.add_patch(arc)
        self.ax.plot([qx, qx + 0.35 * WID],
                     [qy - 0.15 * HIG, qy + 0.20 * HIG],
                     color=self._style.not_gate_lc, linewidth=2, zorder=PORDER_GATE)
        # arrow
        self._line(qxy, [cx, cy + 0.35 * WID], lc=self._style.cc,
                   ls=self._style.cline)
        arrowhead = patches.Polygon(((cx - 0.20 * WID, cy + 0.35 * WID),
                                     (cx + 0.20 * WID, cy + 0.35 * WID),
                                     (cx, cy)),
                                    fc=self._style.cc,
                                    ec=None)
        self.ax.add_artist(arrowhead)
        # target
        if self._style.bundle:
            self.ax.text(cx + .25, cy + .1, str(cid), ha='left', va='bottom',
                         fontsize=0.8 * self._style.fs,
                         color=self._style.tc,
                         clip_on=True,
                         zorder=PORDER_TEXT)

    def _conds(self, xy, istrue=False):
        xpos, ypos = xy

        if istrue:
            _fc = self._style.lc
        else:
            _fc = self._style.gc

        box = patches.Circle(xy=(xpos, ypos), radius=WID * 0.15,
                             fc=_fc, ec=self._style.lc,
                             linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)

    def _ctrl_qubit(self, xy, fc=None, ec=None):
        if self._style.gc != DefaultStyle().gc:
            fc = self._style.gc
            ec = self._style.gc
        if fc is None:
            fc = self._style.lc
        if ec is None:
            ec = self._style.lc
        xpos, ypos = xy
        box = patches.Circle(xy=(xpos, ypos), radius=WID * 0.15,
                             fc=fc, ec=ec,
                             linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)

    def _tgt_qubit(self, xy, fc=None, ec=None, ac=None,
                   add_width=None):
        if self._style.gc != DefaultStyle().gc:
            fc = self._style.gc
            ec = self._style.gc
        if fc is None:
            fc = self._style.dispcol['target']
        if ec is None:
            ec = self._style.lc
        if ac is None:
            ac = self._style.lc
        if add_width is None:
            add_width = 0.35

        linewidth = 2

        if self._style.dispcol['target'] == '#ffffff':
            add_width = self._style.colored_add_width

        xpos, ypos = xy

        box = patches.Circle(xy=(xpos, ypos), radius=HIG * 0.35,
                             fc=fc, ec=ec, linewidth=linewidth,
                             zorder=PORDER_GATE)
        self.ax.add_patch(box)
        # add '+' symbol
        self.ax.plot([xpos, xpos], [ypos - add_width * HIG,
                                    ypos + add_width * HIG],
                     color=ac, linewidth=linewidth, zorder=PORDER_GATE + 1)

        self.ax.plot([xpos - add_width * HIG, xpos + add_width * HIG],
                     [ypos, ypos], color=ac, linewidth=linewidth,
                     zorder=PORDER_GATE + 1)

    def _swap(self, xy):
        xpos, ypos = xy
        color = self._style.dispcol['swap']
        self.ax.plot([xpos - 0.20 * WID, xpos + 0.20 * WID],
                     [ypos - 0.20 * WID, ypos + 0.20 * WID],
                     color=color, linewidth=2, zorder=PORDER_LINE + 1)
        self.ax.plot([xpos - 0.20 * WID, xpos + 0.20 * WID],
                     [ypos + 0.20 * WID, ypos - 0.20 * WID],
                     color=color, linewidth=2, zorder=PORDER_LINE + 1)

    def _barrier(self, config, anc):
        xys = config['coord']
        group = config['group']
        y_reg = []
        for qreg in self._qreg_dict.values():
            if qreg['group'] in group:
                y_reg.append(qreg['y'])
        x0 = xys[0][0]

        box_y0 = min(y_reg) - int(anc / self.fold) * (self._cond['n_lines'] + 1) - 0.5
        box_y1 = max(y_reg) - int(anc / self.fold) * (self._cond['n_lines'] + 1) + 0.5
        box = patches.Rectangle(xy=(x0 - 0.3 * WID, box_y0),
                                width=0.6 * WID, height=box_y1 - box_y0,
                                fc=self._style.bc, ec=None, alpha=0.6,
                                linewidth=1.5, zorder=PORDER_GRAY)
        self.ax.add_patch(box)
        for xy in xys:
            xpos, ypos = xy
            self.ax.plot([xpos, xpos], [ypos + 0.5, ypos - 0.5],
                         linewidth=1, linestyle="dashed",
                         color=self._style.lc,
                         zorder=PORDER_TEXT)

    def _linefeed_mark(self, xy):
        xpos, ypos = xy

        self.ax.plot([xpos - .1, xpos - .1],
                     [ypos, ypos - self._cond['n_lines'] + 1],
                     color=self._style.lc, zorder=PORDER_LINE)
        self.ax.plot([xpos + .1, xpos + .1],
                     [ypos, ypos - self._cond['n_lines'] + 1],
                     color=self._style.lc, zorder=PORDER_LINE)

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
            self._style.figwidth = fig_w * self._scale * self._style.fs / 72 / WID
        self.figure.set_size_inches(self._style.figwidth, self._style.figwidth * fig_h / fig_w)
        if filename:
            self.figure.savefig(filename, dpi=self._style.dpi,
                                bbox_inches='tight')
        if self.return_fig:
            if get_backend() in ['module://ipykernel.pylab.backend_inline',
                                 'nbAgg']:
                plt.close(self.figure)
            return self.figure

    def _draw_regs(self):

        len_longest_label = 0
        # quantum register
        for ii, reg in enumerate(self._qreg):
            if len(self._qreg) > 1:
                if self.layout is None:
                    label = '${{{name}}}_{{{index}}}$'.format(name=reg.register.name,
                                                              index=reg.index)
                else:
                    label = '${{{name}}}_{{{index}}} \\mapsto {{{physical}}}$'.format(
                        name=self.layout[reg.index].register.name,
                        index=self.layout[reg.index].index,
                        physical=reg.index)
            else:
                label = '${name}$'.format(name=reg.register.name)

            if len(label) > len_longest_label:
                len_longest_label = len(label)

            pos = -ii
            self._qreg_dict[ii] = {
                'y': pos,
                'label': label,
                'index': reg.index,
                'group': reg.register
            }
            self._cond['n_lines'] += 1
        # classical register
        if self._creg:
            n_creg = self._creg.copy()
            n_creg.pop(0)
            idx = 0
            y_off = -len(self._qreg)
            for ii, (reg, nreg) in enumerate(itertools.zip_longest(
                    self._creg, n_creg)):
                pos = y_off - idx
                if self._style.bundle:
                    label = '${}$'.format(reg.register.name)
                    self._creg_dict[ii] = {
                        'y': pos,
                        'label': label,
                        'index': reg.index,
                        'group': reg.register
                    }
                    if not (not nreg or reg.register != nreg.register):
                        continue
                else:
                    label = '${}_{{{}}}$'.format(reg.register.name, reg.index)
                    self._creg_dict[ii] = {
                        'y': pos,
                        'label': label,
                        'index': reg.index,
                        'group': reg.register
                    }
                if len(label) > len_longest_label:
                    len_longest_label = len(label)

                self._cond['n_lines'] += 1
                idx += 1

        # 7 is the length of the smallest possible label
        self.x_offset = -.5 + 0.18 * (len_longest_label - 7)

    def _draw_regs_sub(self, n_fold, feedline_l=False, feedline_r=False):
        # quantum register
        for qreg in self._qreg_dict.values():
            if n_fold == 0:
                label = qreg['label']
            else:
                label = qreg['label']
            y = qreg['y'] - n_fold * (self._cond['n_lines'] + 1)
            self.ax.text(self.x_offset, y, label, ha='right', va='center',
                         fontsize=1.25 * self._style.fs,
                         color=self._style.tc,
                         clip_on=True,
                         zorder=PORDER_TEXT)
            self._line([self.x_offset + 0.5, y], [self._cond['xmax'], y],
                       zorder=PORDER_REGLINE)
        # classical register
        this_creg_dict = {}
        for creg in self._creg_dict.values():
            if n_fold == 0:
                label = creg['label']
            else:
                label = creg['label']
            y = creg['y'] - n_fold * (self._cond['n_lines'] + 1)
            if y not in this_creg_dict.keys():
                this_creg_dict[y] = {'val': 1, 'label': label}
            else:
                this_creg_dict[y]['val'] += 1
        for y, this_creg in this_creg_dict.items():
            # bundle
            if this_creg['val'] > 1:
                self.ax.plot([self.x_offset + 1.1, self.x_offset + 1.2], [y - .1, y + .1],
                             color=self._style.cc,
                             zorder=PORDER_LINE)
                self.ax.text(self.x_offset + 1.0, y + .1, str(this_creg['val']), ha='left',
                             va='bottom',
                             fontsize=0.8 * self._style.fs,
                             color=self._style.tc,
                             clip_on=True,
                             zorder=PORDER_TEXT)
            self.ax.text(self.x_offset, y, this_creg['label'], ha='right', va='center',
                         fontsize=1.5 * self._style.fs,
                         color=self._style.tc,
                         clip_on=True,
                         zorder=PORDER_TEXT)
            self._line([self.x_offset + 0.5, y], [self._cond['xmax'], y], lc=self._style.cc,
                       ls=self._style.cline, zorder=PORDER_REGLINE)

        # lf line
        if feedline_r:
            self._linefeed_mark((self.fold + 1 - 0.1,
                                 - n_fold * (self._cond['n_lines'] + 1)))
        if feedline_l:
            self._linefeed_mark((0.1,
                                 - n_fold * (self._cond['n_lines'] + 1)))

    def _draw_ops(self, verbose=False):
        _wide_gate = ['u2', 'u3', 'cu2', 'cu3', 'unitary', 'r']
        _barriers = {'coord': [], 'group': []}

        #
        # generate coordinate manager
        #
        q_anchors = {}
        for key, qreg in self._qreg_dict.items():
            q_anchors[key] = Anchor(reg_num=self._cond['n_lines'],
                                    yind=qreg['y'],
                                    fold=self.fold)
        c_anchors = {}
        for key, creg in self._creg_dict.items():
            c_anchors[key] = Anchor(reg_num=self._cond['n_lines'],
                                    yind=creg['y'],
                                    fold=self.fold)
        #
        # draw gates
        #
        prev_anc = -1
        for layer in self._ops:
            layer_width = 1

            for op in layer:

                if op.name in _wide_gate:
                    if layer_width < 2:
                        layer_width = 2
                    if op.type == 'op' and hasattr(op.op, 'params'):
                        param = self.param_parse(op.op.params)
                        if '$\\pi$' in param:
                            pi_count = param.count('pi')
                            len_param = len(param) - (4 * pi_count)
                        else:
                            len_param = len(param)
                        if len_param > len(op.name):
                            box_width = math.floor(len(param) / 10)
                            if op.name == 'unitary':
                                box_width = 2
                            # If more than 4 characters min width is 2
                            if box_width <= 1:
                                box_width = 2
                            if layer_width < box_width:
                                if box_width > 2:
                                    layer_width = box_width
                                else:
                                    layer_width = 2
                            continue

                # if custom gate with a longer than standard name determine
                # width
                elif op.name not in ['barrier', 'snapshot', 'load', 'save',
                                     'noise', 'cswap', 'swap', 'measure'] and len(op.name) >= 4:
                    box_width = math.ceil(len(op.name) / 6)

                    # handle params/subtext longer than op names
                    if op.type == 'op' and hasattr(op.op, 'params'):
                        param = self.param_parse(op.op.params)
                        if '$\\pi$' in param:
                            pi_count = param.count('pi')
                            len_param = len(param) - (4 * pi_count)
                        else:
                            len_param = len(param)
                        if len_param > len(op.name):
                            box_width = math.floor(len(param) / 8)
                            # If more than 4 characters min width is 2
                            if box_width <= 1:
                                box_width = 2
                            if layer_width < box_width:
                                if box_width > 2:
                                    layer_width = box_width * 2
                                else:
                                    layer_width = 2
                            continue
                    # If more than 4 characters min width is 2
                    layer_width = math.ceil(box_width * WID * 2.5)

            this_anc = prev_anc + 1

            for op in layer:

                _iswide = op.name in _wide_gate
                if op.name not in ['barrier', 'snapshot', 'load', 'save',
                                   'noise', 'cswap', 'swap', 'measure',
                                   'reset'] and len(op.name) >= 4:
                    _iswide = True

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

                # Only add the gate to the anchors if it is going to be plotted.
                # This prevents additional blank wires at the end of the line if
                # the last instruction is a barrier type
                if self.plot_barriers or \
                        op.name not in ['barrier', 'snapshot', 'load', 'save',
                                        'noise']:

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

                if op.type == 'op' and hasattr(op.op, 'params'):
                    param = self.param_parse(op.op.params)
                else:
                    param = None
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
                                if vlist[v_ind] == '1' or self._style.bundle:
                                    self._conds(xy, istrue=True)
                                else:
                                    self._conds(xy, istrue=False)
                                xy_plot.append(xy)
                            v_ind += 1
                    creg_b = sorted(xy_plot, key=lambda xy: xy[1])[0]
                    self._subtext(creg_b, hex(val))
                    self._line(qreg_t, creg_b, lc=self._style.cc,
                               ls=self._style.cline)
                #
                # draw special gates
                #
                if op.name == 'measure':
                    vv = self._creg_dict[c_idxs[0]]['index']
                    self._measure(q_xy[0], c_xy[0], vv)
                elif op.name in ['barrier', 'snapshot', 'load', 'save',
                                 'noise']:
                    _barriers = {'coord': [], 'group': []}
                    for index, qbit in enumerate(q_idxs):
                        q_group = self._qreg_dict[qbit]['group']

                        if q_group not in _barriers['group']:
                            _barriers['group'].append(q_group)
                        _barriers['coord'].append(q_xy[index])
                    if self.plot_barriers:
                        self._barrier(_barriers, this_anc)
                elif op.name == 'initialize':
                    vec = '[%s]' % param
                    self._custom_multiqubit_gate(q_xy, wide=_iswide,
                                                 text="|psi>",
                                                 subtext=vec)
                elif op.name == 'unitary':
                    # TODO(mtreinish): Look into adding the unitary to the
                    # subtext
                    self._custom_multiqubit_gate(q_xy, wide=_iswide,
                                                 text="Unitary")
                elif isinstance(op.op, ControlledGate) and op.name not in [
                        'ccx', 'cx', 'cz', 'cu1', 'ccz', 'cu3', 'crz',
                        'cswap']:
                    disp = op.op.base_gate.name
                    num_ctrl_qubits = op.op.num_ctrl_qubits
                    num_qargs = len(q_xy) - num_ctrl_qubits

                    for i in range(num_ctrl_qubits):
                        self._ctrl_qubit(q_xy[i], fc=self._style.dispcol['multi'],
                                         ec=self._style.dispcol['multi'])
                    # add qubit-qubit wiring
                    self._line(qreg_b, qreg_t, lc=self._style.dispcol['multi'])
                    if num_qargs == 1:
                        self._gate(q_xy[-1], wide=_iswide, text=disp)
                    else:
                        self._custom_multiqubit_gate(
                            q_xy[num_ctrl_qubits:], wide=_iswide, text=disp)

                #
                # draw single qubit gates
                #
                elif len(q_xy) == 1:
                    disp = op.name
                    if param:
                        self._gate(q_xy[0], wide=_iswide, text=disp,
                                   subtext=str(param))
                    else:
                        self._gate(q_xy[0], wide=_iswide, text=disp)
                #
                # draw multi-qubit gates (n=2)
                #
                elif len(q_xy) == 2:
                    # cx
                    if op.name == 'cx':
                        if self._style.dispcol['cx'] != '#ffffff':
                            add_width = self._style.colored_add_width
                        else:
                            add_width = None
                        self._ctrl_qubit(q_xy[0], fc=self._style.dispcol['cx'],
                                         ec=self._style.dispcol['cx'])
                        if self._style.name != 'bw':
                            self._tgt_qubit(q_xy[1], fc=self._style.dispcol['cx'],
                                            ec=self._style.dispcol['cx'],
                                            ac=self._style.dispcol['target'],
                                            add_width=add_width)
                        else:
                            self._tgt_qubit(q_xy[1], fc=self._style.dispcol['target'],
                                            ec=self._style.dispcol['cx'],
                                            ac=self._style.dispcol['cx'],
                                            add_width=add_width)
                        # add qubit-qubit wiring
                        self._line(qreg_b, qreg_t, lc=self._style.dispcol['cx'])
                    # cz for latexmode
                    elif op.name == 'cz':
                        disp = op.name.replace('c', '')
                        if self._style.name != 'bw':
                            color = self._style.dispcol['multi']
                            self._ctrl_qubit(q_xy[0],
                                             fc=color,
                                             ec=color)
                        else:
                            self._ctrl_qubit(q_xy[0])
                        self._gate(q_xy[1], wide=_iswide, text=disp, fc=color)
                        # add qubit-qubit wiring
                        if self._style.name != 'bw':
                            self._line(qreg_b, qreg_t,
                                       lc=self._style.dispcol['multi'])
                        else:
                            self._line(qreg_b, qreg_t, zorder=PORDER_LINE + 1)
                    # control gate
                    elif op.name in ['cy', 'ch', 'cu3', 'cu1', 'crz']:
                        disp = op.name.replace('c', '')

                        color = None
                        if self._style.name != 'bw':
                            color = self._style.dispcol['multi']

                        self._ctrl_qubit(q_xy[0], fc=color, ec=color)
                        if param:
                            self._gate(q_xy[1], wide=_iswide,
                                       text=disp,
                                       fc=color,
                                       subtext='{}'.format(param))
                        else:
                            self._gate(q_xy[1], wide=_iswide, text=disp,
                                       fc=color)
                        # add qubit-qubit wiring
                        self._line(qreg_b, qreg_t, lc=color)

                    # rzz gate
                    elif op.name == 'rzz':
                        self._ctrl_qubit(q_xy[0])
                        self._ctrl_qubit(q_xy[1])
                        self._sidetext(qreg_b, text='zz({})'.format(param))

                        # add qubit-qubit wiring
                        self._line(qreg_b, qreg_t)
                    # swap gate
                    elif op.name == 'swap':
                        self._swap(q_xy[0])
                        self._swap(q_xy[1])
                        # add qubit-qubit wiring
                        self._line(qreg_b, qreg_t, lc=self._style.dispcol['swap'])
                    # Custom gate
                    else:
                        self._custom_multiqubit_gate(q_xy, c_xy, wide=_iswide,
                                                     text=op.name)
                #
                # draw multi-qubit gates (n=3)
                #
                elif len(q_xy) == 3:
                    # cswap gate
                    if op.name == 'cswap':
                        self._ctrl_qubit(q_xy[0],
                                         fc=self._style.dispcol['multi'],
                                         ec=self._style.dispcol['multi'])
                        self._swap(q_xy[1])
                        self._swap(q_xy[2])
                        # add qubit-qubit wiring
                        self._line(qreg_b, qreg_t, lc=self._style.dispcol['multi'])
                    # ccx gate
                    elif op.name == 'ccx':
                        self._ctrl_qubit(q_xy[0], fc=self._style.dispcol['multi'],
                                         ec=self._style.dispcol['multi'])
                        self._ctrl_qubit(q_xy[1], fc=self._style.dispcol['multi'],
                                         ec=self._style.dispcol['multi'])
                        if self._style.name != 'bw':
                            self._tgt_qubit(q_xy[2], fc=self._style.dispcol['multi'],
                                            ec=self._style.dispcol['multi'],
                                            ac=self._style.dispcol['target'])
                        else:
                            self._tgt_qubit(q_xy[2], fc=self._style.dispcol['target'],
                                            ec=self._style.dispcol['multi'],
                                            ac=self._style.dispcol['multi'])
                        # add qubit-qubit wiring
                        self._line(qreg_b, qreg_t, lc=self._style.dispcol['multi'])
                    # custom gate
                    else:
                        self._custom_multiqubit_gate(q_xy, c_xy, wide=_iswide,
                                                     text=op.name)

                # draw custom multi-qubit gate
                elif len(q_xy) > 3:
                    self._custom_multiqubit_gate(q_xy, c_xy, wide=_iswide,
                                                 text=op.name)
                else:
                    logger.critical('Invalid gate %s', op)
                    raise exceptions.VisualizationError('invalid gate {}'.format(op))

            # adjust the column if there have been barriers encountered, but not plotted
            barrier_offset = 0
            if not self.plot_barriers:
                # only adjust if everything in the layer wasn't plotted
                barrier_offset = -1 if all([op.name in
                                            ['barrier', 'snapshot', 'load', 'save', 'noise']
                                            for op in layer]) else 0
            prev_anc = this_anc + layer_width + barrier_offset - 1
        #
        # adjust window size and draw horizontal lines
        #
        anchors = [q_anchors[ii].get_index() for ii in self._qreg_dict]
        if anchors:
            max_anc = max(anchors)
        else:
            max_anc = 0
        n_fold = max(0, max_anc - 1) // self.fold
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
        # draw gate number
        if self._style.index:
            for ii in range(max_anc):
                if self.fold > 0:
                    x_coord = ii % self.fold + 1
                    y_coord = - (ii // self.fold) * (self._cond['n_lines'] + 1) + 0.7
                else:
                    x_coord = ii + 1
                    y_coord = 0.7
                self.ax.text(x_coord, y_coord, str(ii + 1), ha='center',
                             va='center', fontsize=self._style.sfs,
                             color=self._style.tc, clip_on=True,
                             zorder=PORDER_TEXT)

    @staticmethod
    def param_parse(v):
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

    @staticmethod
    def format_numeric(val, tol=1e-5):
        if isinstance(val, complex):
            return str(val)
        elif complex(val).imag != 0:
            val = complex(val)
        abs_val = abs(val)
        if math.isclose(abs_val, 0.0, abs_tol=1e-100):
            return '0'
        if math.isclose(math.fmod(abs_val, 1.0),
                        0.0, abs_tol=tol) and 0.5 < abs_val < 9999.5:
            return str(int(val))
        if 0.1 <= abs_val < 100.0:
            return '{:.2f}'.format(val)
        return '{:.1e}'.format(val)

    @staticmethod
    def fraction(val, base=np.pi, n=100, tol=1e-5):
        abs_val = abs(val)
        for i in range(1, n):
            for j in range(1, n):
                if math.isclose(abs_val, i / j * base, rel_tol=tol):
                    if val < 0:
                        i *= -1
                    return fractions.Fraction(i, j)
        return None
