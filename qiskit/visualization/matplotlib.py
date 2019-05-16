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

# pylint: disable=invalid-name,missing-docstring

"""mpl circuit visualization backend."""

import collections
import fractions
import itertools
import json
import logging
import math

import numpy as np

try:
    from matplotlib import patches
    from matplotlib import pyplot as plt
    from matplotlib import pyplot as plt, gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from qiskit.visualization import exceptions
from qiskit.visualization import interpolation
from qiskit.visualization.qcstyle import (OPStylePulse, OPStyleSched,
                                          DefaultStyle, BWStyle)
from qiskit.pulse.channels import (DriveChannel, ControlChannel, MeasureChannel,
                                   AcquireChannel, SnapshotChannel)
from qiskit.pulse import (SamplePulse, FrameChange, PersistentValue, Snapshot, Acquire,
                          PulseError)
from qiskit import user_config

logger = logging.getLogger(__name__)

Register = collections.namedtuple('Register', 'reg index')

WID = 0.65
HIG = 0.65
DEFAULT_SCALE = 4.3
PORDER_GATE = 5
PORDER_LINE = 2
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

    def plot_coord(self, index, gate_width):
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
        return x_pos, y_pos

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
                 reverse_bits=False):

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
        if config:
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
        if style:
            if isinstance(style, dict):
                self._style.set_style(style)
            elif isinstance(style, str):
                with open(style, 'r') as infile:
                    dic = json.load(infile)
                self._style.set_style(dic)

        self.figure = plt.figure()
        self.figure.patch.set_facecolor(color=self._style.bg)
        self.ax = self.figure.add_subplot(111)
        self.ax.axis('off')
        self.ax.set_aspect('equal')
        self.ax.tick_params(labelbottom=False, labeltop=False,
                            labelleft=False, labelright=False)

    def _registers(self, creg, qreg):
        self._creg = []
        for r in creg:
            self._creg.append(Register(reg=r[0], index=r[1]))
        self._qreg = []
        for r in qreg:
            self._qreg.append(Register(reg=r[0], index=r[1]))

    @property
    def ast(self):
        return self._ast

    def _custom_multiqubit_gate(self, xy, fc=None, wide=True, text=None,
                                subtext=None):
        xpos = min([x[0] for x in xy])
        ypos = min([y[1] for y in xy])
        ypos_max = max([y[1] for y in xy])
        if wide:
            if subtext:
                boxes_length = round(max([len(text), len(subtext)]) / 8) or 1
            else:
                boxes_length = round(len(text) / 8) or 1
            wid = WID * 2.8 * boxes_length
        else:
            wid = WID
        if fc:
            _fc = fc
        else:
            _fc = self._style.gc
        qubit_span = abs(ypos) - abs(ypos_max) + 1
        height = HIG + (qubit_span - 1)
        box = patches.Rectangle(
            xy=(xpos - 0.5 * wid, ypos - .5 * HIG),
            width=wid, height=height, fc=_fc, ec=self._style.lc,
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
                self.ax.text(xpos, ypos + 0.15 * height, disp_text, ha='center',
                             va='center', fontsize=self._style.fs,
                             color=self._style.gt, clip_on=True,
                             zorder=PORDER_TEXT)
                self.ax.text(xpos, ypos - 0.3 * height, subtext, ha='center',
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
                             zorder=PORDER_TEXT)

    def _gate(self, xy, fc=None, wide=False, text=None, subtext=None):
        xpos, ypos = xy

        if wide:
            if subtext:
                wid = WID * 2.8
            else:
                boxes_wide = round(len(text) / 10) or 1
                wid = WID * 2.8 * boxes_wide
        else:
            wid = WID
        if fc:
            _fc = fc
        elif text and text in self._style.dispcol:
            _fc = self._style.dispcol[text]
        else:
            _fc = self._style.gc

        box = patches.Rectangle(
            xy=(xpos - 0.5 * wid, ypos - 0.5 * HIG), width=wid, height=HIG,
            fc=_fc, ec=self._style.lc, linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)

        if text:
            if text in self._style.dispcol:
                disp_text = "${}$".format(self._style.disptex[text])
            else:
                disp_text = text
            if subtext:
                self.ax.text(xpos, ypos + 0.15 * HIG, disp_text, ha='center',
                             va='center', fontsize=self._style.fs,
                             color=self._style.gt, clip_on=True,
                             zorder=PORDER_TEXT)
                self.ax.text(xpos, ypos - 0.3 * HIG, subtext, ha='center',
                             va='center', fontsize=self._style.sfs,
                             color=self._style.sc, clip_on=True,
                             zorder=PORDER_TEXT)
            else:
                self.ax.text(xpos, ypos, disp_text, ha='center', va='center',
                             fontsize=self._style.fs,
                             color=self._style.gt,
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
        self.ax.text(xp, ypos+HIG, text, ha='center', va='top',
                     fontsize=self._style.sfs,
                     color=self._style.tc,
                     clip_on=True,
                     zorder=PORDER_TEXT)

    def _line(self, xy0, xy1, lc=None, ls=None):
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
                         linewidth=1.0,
                         linestyle='solid',
                         zorder=PORDER_LINE)
            self.ax.plot([x0 - dx, x1 - dx], [y0 - dy, y1 - dy],
                         color=linecolor,
                         linewidth=1.0,
                         linestyle='solid',
                         zorder=PORDER_LINE)
        else:
            self.ax.plot([x0, x1], [y0, y1],
                         color=linecolor,
                         linewidth=1.0,
                         linestyle=linestyle,
                         zorder=PORDER_LINE)

    def _measure(self, qxy, cxy, cid):
        qx, qy = qxy
        cx, cy = cxy

        self._gate(qxy, fc=self._style.dispcol['meas'])
        # add measure symbol
        arc = patches.Arc(xy=(qx, qy - 0.15 * HIG), width=WID * 0.7,
                          height=HIG * 0.7, theta1=0, theta2=180, fill=False,
                          ec=self._style.lc, linewidth=1.5,
                          zorder=PORDER_GATE)
        self.ax.add_patch(arc)
        self.ax.plot([qx, qx + 0.35 * WID],
                     [qy - 0.15 * HIG, qy + 0.20 * HIG],
                     color=self._style.lc, linewidth=1.5, zorder=PORDER_GATE)
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

    def _ctrl_qubit(self, xy):
        xpos, ypos = xy

        box = patches.Circle(xy=(xpos, ypos), radius=WID * 0.15,
                             fc=self._style.lc, ec=self._style.lc,
                             linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)

    def _tgt_qubit(self, xy):
        xpos, ypos = xy

        box = patches.Circle(xy=(xpos, ypos), radius=HIG * 0.35,
                             fc=self._style.dispcol['target'],
                             ec=self._style.lc, linewidth=1.5,
                             zorder=PORDER_GATE)
        self.ax.add_patch(box)
        # add '+' symbol
        self.ax.plot([xpos, xpos], [ypos - 0.35 * HIG, ypos + 0.35 * HIG],
                     color=self._style.lc, linewidth=1.0, zorder=PORDER_GATE)
        self.ax.plot([xpos - 0.35 * HIG, xpos + 0.35 * HIG], [ypos, ypos],
                     color=self._style.lc, linewidth=1.0, zorder=PORDER_GATE)

    def _swap(self, xy):
        xpos, ypos = xy

        self.ax.plot([xpos - 0.20 * WID, xpos + 0.20 * WID],
                     [ypos - 0.20 * WID, ypos + 0.20 * WID],
                     color=self._style.lc, linewidth=1.5, zorder=PORDER_LINE)
        self.ax.plot([xpos - 0.20 * WID, xpos + 0.20 * WID],
                     [ypos + 0.20 * WID, ypos - 0.20 * WID],
                     color=self._style.lc, linewidth=1.5, zorder=PORDER_LINE)

    def _barrier(self, config, anc):
        xys = config['coord']
        group = config['group']
        y_reg = []
        for qreg in self._qreg_dict.values():
            if qreg['group'] in group:
                y_reg.append(qreg['y'])
        x0 = xys[0][0]

        box_y0 = min(y_reg) - int(anc / self._style.fold) * (self._cond['n_lines'] + 1) - 0.5
        box_y1 = max(y_reg) - int(anc / self._style.fold) * (self._cond['n_lines'] + 1) + 0.5
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
        plt.close(self.figure)
        return self.figure

    def _draw_regs(self):
        # quantum register
        for ii, reg in enumerate(self._qreg):
            if len(self._qreg) > 1:
                label = '${}_{{{}}}$'.format(reg.reg.name, reg.index)
            else:
                label = '${}$'.format(reg.reg.name)

            pos = -ii
            self._qreg_dict[ii] = {
                'y': pos,
                'label': label,
                'index': reg.index,
                'group': reg.reg
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
                    label = '${}$'.format(reg.reg.name)
                    self._creg_dict[ii] = {
                        'y': pos,
                        'label': label,
                        'index': reg.index,
                        'group': reg.reg
                    }
                    if not (not nreg or reg.reg != nreg.reg):
                        continue
                else:
                    label = '${}_{{{}}}$'.format(reg.reg.name, reg.index)
                    self._creg_dict[ii] = {
                        'y': pos,
                        'label': label,
                        'index': reg.index,
                        'group': reg.reg
                    }

                self._cond['n_lines'] += 1
                idx += 1

    def _draw_regs_sub(self, n_fold, feedline_l=False, feedline_r=False):
        # quantum register
        for qreg in self._qreg_dict.values():
            if n_fold == 0:
                label = qreg['label'] + ' : $\\left|0\\right\\rangle$'
            else:
                label = qreg['label']
            y = qreg['y'] - n_fold * (self._cond['n_lines'] + 1)
            self.ax.text(-0.5, y, label, ha='right', va='center',
                         fontsize=self._style.fs,
                         color=self._style.tc,
                         clip_on=True,
                         zorder=PORDER_TEXT)
            self._line([0, y], [self._cond['xmax'], y])
        # classical register
        this_creg_dict = {}
        for creg in self._creg_dict.values():
            if n_fold == 0:
                label = creg['label'] + ' :  0 '
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
                self.ax.plot([.6, .7], [y - .1, y + .1],
                             color=self._style.cc,
                             zorder=PORDER_LINE)
                self.ax.text(0.5, y + .1, str(this_creg['val']), ha='left',
                             va='bottom',
                             fontsize=0.8 * self._style.fs,
                             color=self._style.tc,
                             clip_on=True,
                             zorder=PORDER_TEXT)
            self.ax.text(-0.5, y, this_creg['label'], ha='right', va='center',
                         fontsize=self._style.fs,
                         color=self._style.tc,
                         clip_on=True,
                         zorder=PORDER_TEXT)
            self._line([0, y], [self._cond['xmax'], y], lc=self._style.cc,
                       ls=self._style.cline)

        # lf line
        if feedline_r:
            self._linefeed_mark((self._style.fold + 1 - 0.1,
                                 - n_fold * (self._cond['n_lines'] + 1)))
        if feedline_l:
            self._linefeed_mark((0.1,
                                 - n_fold * (self._cond['n_lines'] + 1)))

    def _draw_ops(self, verbose=False):
        _wide_gate = ['u2', 'u3', 'cu2', 'cu3']
        _barriers = {'coord': [], 'group': []}

        #
        # generate coordinate manager
        #
        q_anchors = {}
        for key, qreg in self._qreg_dict.items():
            q_anchors[key] = Anchor(reg_num=self._cond['n_lines'],
                                    yind=qreg['y'],
                                    fold=self._style.fold)
        c_anchors = {}
        for key, creg in self._creg_dict.items():
            c_anchors[key] = Anchor(reg_num=self._cond['n_lines'],
                                    yind=creg['y'],
                                    fold=self._style.fold)
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
                # if custom gate with a longer than standard name determine
                # width
                elif op.name not in ['barrier', 'snapshot', 'load', 'save',
                                     'noise', 'cswap', 'swap'] and len(
                                         op.name) >= 4:
                    box_width = round(len(op.name) / 8)
                    # If more than 4 characters min width is 2
                    if box_width <= 1:
                        box_width = 2
                    if layer_width < box_width:
                        if box_width > 2:
                            layer_width = box_width * 2
                        else:
                            layer_width = 2

            this_anc = prev_anc + 1

            for op in layer:

                _iswide = op.name in _wide_gate
                if op.name not in ['barrier', 'snapshot', 'load', 'save',
                                   'noise', 'cswap', 'swap'] and len(
                                       op.name) >= 4:
                    _iswide = True

                # get qreg index
                q_idxs = []
                for qarg in op.qargs:
                    for index, reg in self._qreg_dict.items():
                        if (reg['group'] == qarg[0] and
                                reg['index'] == qarg[1]):
                            q_idxs.append(index)
                            break

                # get creg index
                c_idxs = []
                for carg in op.cargs:
                    for index, reg in self._creg_dict.items():
                        if (reg['group'] == carg[0] and
                                reg['index'] == carg[1]):
                            c_idxs.append(index)
                            break
                for ii in q_idxs:
                    q_anchors[ii].set_index(this_anc, layer_width)

                # qreg coordinate
                q_xy = [q_anchors[ii].plot_coord(this_anc, layer_width) for ii in q_idxs]
                # creg coordinate
                c_xy = [c_anchors[ii].plot_coord(this_anc, layer_width) for ii in c_idxs]
                # bottom and top point of qreg
                qreg_b = min(q_xy, key=lambda xy: xy[1])
                qreg_t = max(q_xy, key=lambda xy: xy[1])

                # update index based on the value from plotting
                this_anc = q_anchors[q_idxs[0]].gate_anchor

                if verbose:
                    print(op)

                if op.type == 'op' and hasattr(op.op, 'params'):
                    param = self.param_parse(op.op.params, self._style.pimode)
                else:
                    param = None
                # conditional gate
                if op.condition:
                    c_xy = [c_anchors[ii].plot_coord(this_anc, layer_width) for
                            ii in self._creg_dict]
                    mask = 0
                    for index, cbit in enumerate(self._creg):
                        if cbit.reg == op.condition[0]:
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
                #
                # draw single qubit gates
                #
                elif len(q_xy) == 1:
                    disp = op.name
                    if param:
                        prm = '{}'.format(param)
                        if len(prm) < 20:
                            self._gate(q_xy[0], wide=_iswide, text=disp,
                                       subtext=prm)
                        else:
                            self._gate(q_xy[0], wide=_iswide, text=disp)
                    else:
                        self._gate(q_xy[0], wide=_iswide, text=disp)
                #
                # draw multi-qubit gates (n=2)
                #
                elif len(q_xy) == 2:
                    # cx
                    if op.name == 'cx':
                        self._ctrl_qubit(q_xy[0])
                        self._tgt_qubit(q_xy[1])
                        # add qubit-qubit wiring
                        self._line(qreg_b, qreg_t)
                    # cz for latexmode
                    elif op.name == 'cz':
                        if self._style.latexmode:
                            self._ctrl_qubit(q_xy[0])
                            self._ctrl_qubit(q_xy[1])
                        else:
                            disp = op.name.replace('c', '')
                            self._ctrl_qubit(q_xy[0])
                            self._gate(q_xy[1], wide=_iswide, text=disp)
                        # add qubit-qubit wiring
                        self._line(qreg_b, qreg_t)
                    # control gate
                    elif op.name in ['cy', 'ch', 'cu3', 'crz']:
                        disp = op.name.replace('c', '')
                        self._ctrl_qubit(q_xy[0])
                        if param:
                            self._gate(q_xy[1], wide=_iswide, text=disp,
                                       subtext='{}'.format(param))
                        else:
                            self._gate(q_xy[1], wide=_iswide, text=disp)
                        # add qubit-qubit wiring
                        self._line(qreg_b, qreg_t)
                    # cu1
                    elif op.name == 'cu1':
                        self._ctrl_qubit(q_xy[0])
                        self._ctrl_qubit(q_xy[1])
                        self._sidetext(qreg_b, param)

                        # add qubit-qubit wiring
                        self._line(qreg_b, qreg_t)
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
                        self._line(qreg_b, qreg_t)
                    # Custom gate
                    else:
                        self._custom_multiqubit_gate(q_xy, wide=_iswide,
                                                     text=op.name)
                #
                # draw multi-qubit gates (n=3)
                #
                elif len(q_xy) == 3:
                    # cswap gate
                    if op.name == 'cswap':
                        self._ctrl_qubit(q_xy[0])
                        self._swap(q_xy[1])
                        self._swap(q_xy[2])
                        # add qubit-qubit wiring
                        self._line(qreg_b, qreg_t)
                    # ccx gate
                    elif op.name == 'ccx':
                        self._ctrl_qubit(q_xy[0])
                        self._ctrl_qubit(q_xy[1])
                        self._tgt_qubit(q_xy[2])
                        # add qubit-qubit wiring
                        self._line(qreg_b, qreg_t)
                    # custom gate
                    else:
                        self._custom_multiqubit_gate(q_xy, wide=_iswide,
                                                     text=op.name)

                # draw custom multi-qubit gate
                elif len(q_xy) > 3:
                    self._custom_multiqubit_gate(q_xy, wide=_iswide,
                                                 text=op.name)
                else:
                    logger.critical('Invalid gate %s', op)
                    raise exceptions.VisualizationError('invalid gate {}'.format(op))

            prev_anc = this_anc + layer_width - 1
        #
        # adjust window size and draw horizontal lines
        #
        anchors = [q_anchors[ii].get_index() for ii in self._qreg_dict]
        if anchors:
            max_anc = max(anchors)
        else:
            max_anc = 0
        n_fold = max(0, max_anc - 1) // self._style.fold
        # window size
        if max_anc > self._style.fold > 0:
            self._cond['xmax'] = self._style.fold + 1
            self._cond['ymax'] = (n_fold + 1) * (self._cond['n_lines'] + 1) - 1
        else:
            self._cond['xmax'] = max_anc + 1
            self._cond['ymax'] = self._cond['n_lines']
        # add horizontal lines
        for ii in range(n_fold + 1):
            feedline_r = (n_fold > 0 and n_fold > ii)
            feedline_l = (ii > 0)
            self._draw_regs_sub(ii, feedline_l, feedline_r)
        # draw gate number
        if self._style.index:
            for ii in range(max_anc):
                if self._style.fold > 0:
                    x_coord = ii % self._style.fold + 1
                    y_coord = - (ii // self._style.fold) * (self._cond['n_lines'] + 1) + 0.7
                else:
                    x_coord = ii + 1
                    y_coord = 0.7
                self.ax.text(x_coord, y_coord, str(ii + 1), ha='center',
                             va='center', fontsize=self._style.sfs,
                             color=self._style.tc, clip_on=True,
                             zorder=PORDER_TEXT)

    @staticmethod
    def param_parse(v, pimode=False):

        # create an empty list to store the parameters in
        param_parts = [None] * len(v)
        for i, e in enumerate(v):

            if pimode:
                try:
                    param_parts[i] = MatplotlibDrawer.format_pi(e)
                except TypeError:
                    param_parts[i] = str(e)
            else:
                try:
                    param_parts[i] = MatplotlibDrawer.format_numeric(e)
                except TypeError:
                    param_parts[i] = str(e)
            if param_parts[i].startswith('-'):
                param_parts[i] = '$-$' + param_parts[i][1:]

        param_parts = ', '.join(param_parts)
        return param_parts

    @staticmethod
    def format_pi(val):
        fracvals = MatplotlibDrawer.fraction(val)
        buf = ''
        if fracvals:
            nmr, dnm = fracvals.numerator, fracvals.denominator
            if nmr == 1:
                buf += '$\\pi$'
            elif nmr == -1:
                buf += '-$\\pi$'
            else:
                buf += '{}$\\pi$'.format(nmr)
            if dnm > 1:
                buf += '/{}'.format(dnm)
            return buf
        else:
            coef = MatplotlibDrawer.format_numeric(val / np.pi)
            if coef == '0':
                return '0'
            return '{}$\\pi$'.format(coef)

    @staticmethod
    def format_numeric(val, tol=1e-5):
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


class EventsOutputChannels:
    """Pulse dataset for channel."""

    def __init__(self, t0, tf):
        """Create new channel dataset.

        Args:
            t0 (int): starting time of plot
            tf (int): ending time of plot
        """
        self.pulses = {}
        self.t0 = t0
        self.tf = tf

        self._waveform = None
        self._framechanges = None
        self._conditionals = None
        self._snapshots = None
        self._labels = None
        self.enable = False

    def add_instruction(self, start_time, pulse):
        """Add new pulse instruction to channel.

        Args:
            start_time (int): Starting time of instruction
            pulse (Instruction): Instruction object to be added
        """

        if start_time in self.pulses.keys():
            self.pulses[start_time].append(pulse.command)
        else:
            self.pulses[start_time] = [pulse.command]

    @property
    def waveform(self):
        """Get waveform."""
        if self._waveform is None:
            self._build_waveform()

        return self._waveform[self.t0:self.tf]

    @property
    def framechanges(self):
        """Get frame changes."""
        if self._framechanges is None:
            self._build_waveform()

        return self._trim(self._framechanges)

    @property
    def conditionals(self):
        """Get conditionals."""
        if self._conditionals is None:
            self._build_waveform()

        return self._trim(self._conditionals)

    @property
    def snapshots(self):
        """Get snapshots."""
        if self._snapshots is None:
            self._build_waveform()

        return self._trim(self._snapshots)

    @property
    def labels(self):
        """Get labels."""
        if self._labels is None:
            self._build_waveform()

        return self._trim(self._labels)

    def is_empty(self):
        """Return if pulse is empty.

        Returns:
            bool: if the channel has nothing to plot
        """
        if any(self.waveform) or self.framechanges or self.conditionals or self.snapshots:
            return False

        return True

    def to_table(self, name):
        """Get table contains.

        Args:
            name (str): name of channel

        Returns:
            dict: dictionary of events in the channel
        """
        time_event = []

        framechanges = self.framechanges
        conditionals = self.conditionals
        snapshots = self.snapshots

        for key, val in framechanges.items():
            data_str = 'framechange: %.2f' % val
            time_event.append((key, name, data_str))
        for key, val in conditionals.items():
            data_str = 'conditional, %s' % val
            time_event.append((key, name, data_str))
        for key, val in snapshots.items():
            data_str = 'snapshot: %s' % val
            time_event.append((key, name, data_str))

        return time_event

    def _build_waveform(self):
        """Create waveform from stored pulses.
        """
        self._framechanges = {}
        self._conditionals = {}
        self._snapshots = {}
        self._labels = {}
        fc = 0
        pv = np.zeros(self.tf + 1, dtype=np.complex128)
        wf = np.zeros(self.tf + 1, dtype=np.complex128)
        last_pv = None
        for time, commands in sorted(self.pulses.items()):
            if time > self.tf:
                break
            tmp_fc = 0
            for command in commands:
                if isinstance(command, FrameChange):
                    tmp_fc += command.phase
                    pv[time:] = 0
                elif isinstance(command, Snapshot):
                    self._snapshots[time] = command.name
            if tmp_fc != 0:
                self._framechanges[time] = tmp_fc
                fc += tmp_fc
            for command in commands:
                if isinstance(command, PersistentValue):
                    pv[time:] = np.exp(1j*fc) * command.value
                    last_pv = (time, command)
                    break

            for command in commands:
                duration = command.duration
                tf = min(time + duration, self.tf)
                if isinstance(command, SamplePulse):
                    wf[time:tf] = np.exp(1j*fc) * command.samples[:tf-time]
                    pv[time:] = 0
                    self._labels[time] = (tf, command)
                    if last_pv is not None:
                        pv_cmd = last_pv[1]
                        self._labels[last_pv[0]] = (time, pv_cmd)
                        last_pv = None

                elif isinstance(command, Acquire):
                    wf[time:tf] = np.ones(command.duration)
                    self._labels[time] = (tf, command)
        self._waveform = wf + pv

    def _trim(self, events):
        """Return events during given `time_range`.

        Args:
            events (dict): time and operation of events

        Returns:
            dict: dictionary of events within the time
        """
        events_in_time_range = {}

        for k, v in events.items():
            if self.t0 <= k <= self.tf:
                events_in_time_range[k] = v

        return events_in_time_range


class SamplePulseDrawer:
    """A class to create figure for sample pulse."""

    def __init__(self, style):
        """Create new figure.

        Args:
            style (OPStylePulse): style sheet
        """
        self.style = style or OPStylePulse()

    def draw(self, pulse, dt, interp_method, scaling=1):
        """Draw figure.
        Args:
            pulse (SamplePulse): SamplePulse to draw
            dt (float): time interval
            interp_method (Callable): interpolation function
                See `qiskit.visualization.interpolation` for more information
            scaling (float): Relative visual scaling of waveform amplitudes

        Returns:
            matplotlib.figure: A matplotlib figure object of the pulse envelope
        """
        figure = plt.figure()

        interp_method = interp_method or interpolation.step_wise

        figure.set_size_inches(self.style.figsize[0], self.style.figsize[1])
        ax = figure.add_subplot(111)
        ax.set_facecolor(self.style.bg_color)

        samples = pulse.samples
        time = np.arange(0, len(samples) + 1, dtype=float) * dt

        time, re, im = interp_method(time, samples, self.style.num_points)

        # plot
        ax.fill_between(x=time, y1=re, y2=np.zeros_like(time),
                        facecolor=self.style.wave_color[0], alpha=0.3,
                        edgecolor=self.style.wave_color[0], linewidth=1.5,
                        label='real part')
        ax.fill_between(x=time, y1=im, y2=np.zeros_like(time),
                        facecolor=self.style.wave_color[1], alpha=0.3,
                        edgecolor=self.style.wave_color[1], linewidth=1.5,
                        label='imaginary part')

        ax.set_xlim(0, pulse.duration * dt)
        if scaling:
            ax.set_ylim(-scaling, scaling)
        else:
            v_max = max(max(np.abs(re)), max(np.abs(im)))
            ax.set_ylim(-1.2 * v_max, 1.2 * v_max)

        return figure


class ScheduleDrawer:
    """A class to create figure for schedule and channel."""

    def __init__(self, style):
        """Create new figure.

        Args:
            style (OPStyleSched): style sheet
        """
        self.style = style or OPStyleSched()

    def _build_channels(self, schedule, t0, tf):
        # prepare waveform channels
        drive_channels = collections.OrderedDict()
        measure_channels = collections.OrderedDict()
        control_channels = collections.OrderedDict()
        acquire_channels = collections.OrderedDict()
        snapshot_channels = collections.OrderedDict()

        for chan in schedule.channels:
            if isinstance(chan, DriveChannel):
                try:
                    drive_channels[chan] = EventsOutputChannels(t0, tf)
                except PulseError:
                    pass
            elif isinstance(chan, MeasureChannel):
                try:
                    measure_channels[chan] = EventsOutputChannels(t0, tf)
                except PulseError:
                    pass
            elif isinstance(chan, ControlChannel):
                try:
                    control_channels[chan] = EventsOutputChannels(t0, tf)
                except PulseError:
                    pass
            elif isinstance(chan, AcquireChannel):
                try:
                    acquire_channels[chan] = EventsOutputChannels(t0, tf)
                except PulseError:
                    pass
            elif isinstance(chan, SnapshotChannel):
                try:
                    snapshot_channels[chan] = EventsOutputChannels(t0, tf)
                except PulseError:
                    pass

        output_channels = {**drive_channels, **measure_channels,
                           **control_channels, **acquire_channels}
        channels = {**output_channels, **acquire_channels, **snapshot_channels}
        # sort by index then name to group qubits together.
        output_channels = collections.OrderedDict(sorted(output_channels.items(),
                                                         key=lambda x: (x[0].index, x[0].name)))
        channels = collections.OrderedDict(sorted(channels.items(),
                                                  key=lambda x: (x[0].index, x[0].name)))

        for start_time, instruction in schedule.instructions:
            for channel in instruction.channels:
                if channel in output_channels:
                    output_channels[channel].add_instruction(start_time, instruction)
                elif channel in snapshot_channels:
                    snapshot_channels[channel].add_instruction(start_time, instruction)
        return channels, output_channels, snapshot_channels

    def _count_valid_waveforms(self, channels, scaling=1, channels_to_plot=None,
                               plot_all=False):
        # count numbers of valid waveform
        n_valid_waveform = 0
        v_max = 0
        for channel, events in channels.items():
            if channels_to_plot:
                if channel in channels_to_plot:
                    waveform = events.waveform
                    v_max = max(v_max,
                                max(np.abs(np.real(waveform))),
                                max(np.abs(np.imag(waveform))))
                    n_valid_waveform += 1
                    events.enable = True
            else:
                if not events.is_empty() or plot_all:
                    waveform = events.waveform
                    v_max = max(v_max,
                                max(np.abs(np.real(waveform))),
                                max(np.abs(np.imag(waveform))))
                    n_valid_waveform += 1
                    events.enable = True
        if scaling:
            v_max = 0.5 * scaling
        else:
            v_max = 0.5 / (1.2 * v_max)

        return n_valid_waveform, v_max

    # pylint: disable=unused-argument
    def _draw_table(self, figure, channels, dt, n_valid_waveform):
        # create table
        table_data = []
        if self.style.use_table:
            for channel, events in channels.items():
                if events.enable:
                    table_data.extend(events.to_table(channel.name))
            table_data = sorted(table_data, key=lambda x: x[0])

        # plot table
        if table_data:
            # table area size
            ncols = self.style.table_columns
            nrows = int(np.ceil(len(table_data)/ncols))

            # fig size
            h_table = nrows * self.style.fig_unit_h_table
            h_waves = (self.style.figsize[1] - h_table)

            # create subplots
            gs = gridspec.GridSpec(2, 1, height_ratios=[h_table, h_waves], hspace=0)
            tb = plt.subplot(gs[0])
            ax = plt.subplot(gs[1])

            # configure each cell
            tb.axis('off')
            cell_value = [['' for _kk in range(ncols * 3)] for _jj in range(nrows)]
            cell_color = [self.style.table_color * ncols for _jj in range(nrows)]
            cell_width = [*([0.2, 0.2, 0.5] * ncols)]
            for ii, data in enumerate(table_data):
                # pylint: disable=unbalanced-tuple-unpacking
                r, c = np.unravel_index(ii, (nrows, ncols), order='f')
                # pylint: enable=unbalanced-tuple-unpacking
                time, ch_name, data_str = data
                # item
                cell_value[r][3 * c + 0] = 't = %s' % time * dt
                cell_value[r][3 * c + 1] = 'ch %s' % ch_name
                cell_value[r][3 * c + 2] = data_str
            table = tb.table(cellText=cell_value,
                             cellLoc='left',
                             rowLoc='center',
                             colWidths=cell_width,
                             bbox=[0, 0, 1, 1],
                             cellColours=cell_color)
            table.auto_set_font_size(False)
            table.set_fontsize = self.style.table_font_size
        else:
            ax = figure.add_subplot(111)

        figure.set_size_inches(self.style.figsize[0], self.style.figsize[1])

        return ax

    def _draw_snapshots(self, ax, snapshot_channels, dt, y0):
        for events in snapshot_channels.values():
            snapshots = events.snapshots
            if snapshots:
                for time in snapshots:
                    ax.annotate(s="\u25D8", xy=(time*dt, y0), xytext=(time*dt, y0+0.08),
                                arrowprops={'arrowstyle': 'wedge'}, ha='center')

    def _draw_framechanges(self, ax, fcs, dt, y0):
        framechanges_present = True
        for time in fcs.keys():
            ax.text(x=time*dt, y=y0, s=r'$\circlearrowleft$',
                    fontsize=self.style.icon_font_size,
                    ha='center', va='center')
        return framechanges_present

    def _get_channel_color(self, channel):
        # choose color
        if isinstance(channel, DriveChannel):
            color = self.style.d_ch_color
        elif isinstance(channel, ControlChannel):
            color = self.style.u_ch_color
        elif isinstance(channel, MeasureChannel):
            color = self.style.m_ch_color
        elif isinstance(channel, AcquireChannel):
            color = self.style.a_ch_color
        else:
            color = 'black'
        return color

    def _prev_label_at_time(self, prev_labels, time):
        for _, labels in enumerate(prev_labels):
            for t0, (tf, _) in labels.items():
                if time in (t0, tf):
                    return True
        return False

    def _draw_labels(self, ax, labels, prev_labels, dt, y0):
        for t0, (tf, cmd) in labels.items():
            if isinstance(cmd, PersistentValue):
                name = cmd.name if cmd.name else 'pv'
            elif isinstance(cmd, Acquire):
                name = cmd.name if cmd.name else 'acquire'
            else:
                name = cmd.name

            ax.annotate(r'%s' % name,
                        xy=((t0+tf)//2*dt, y0),
                        xytext=((t0+tf)//2*dt, y0-0.07),
                        fontsize=self.style.label_font_size,
                        ha='center', va='center')

            linestyle = self.style.label_ch_linestyle
            alpha = self.style.label_ch_alpha
            color = self.style.label_ch_color

            if not self._prev_label_at_time(prev_labels, t0):
                ax.axvline(t0*dt, -1, 1, color=color,
                           linestyle=linestyle, alpha=alpha)
            if not (self._prev_label_at_time(prev_labels, tf) or tf in labels):
                ax.axvline(tf*dt, -1, 1, color=color,
                           linestyle=linestyle, alpha=alpha)

    def _draw_channels(self, ax, output_channels, interp_method, t0, tf, dt, v_max,
                       label=False, framechange=True):
        y0 = 0
        prev_labels = []
        for channel, events in output_channels.items():
            if events.enable:
                # plot waveform
                waveform = events.waveform
                time = np.arange(t0, tf + 1, dtype=float) * dt
                time, re, im = interp_method(time, waveform, self.style.num_points)
                color = self._get_channel_color(channel)
                # scaling and offset
                re = v_max * re + y0
                im = v_max * im + y0
                offset = np.zeros_like(time) + y0
                # plot
                ax.fill_between(x=time, y1=re, y2=offset,
                                facecolor=color[0], alpha=0.3,
                                edgecolor=color[0], linewidth=1.5,
                                label='real part')
                ax.fill_between(x=time, y1=im, y2=offset,
                                facecolor=color[1], alpha=0.3,
                                edgecolor=color[1], linewidth=1.5,
                                label='imaginary part')
                ax.plot((t0, tf), (y0, y0), color='#000000', linewidth=1.0)

                # plot frame changes
                fcs = events.framechanges
                if fcs and framechange:
                    self._draw_framechanges(ax, fcs, dt, y0)
                # plot labels
                labels = events.labels
                if labels and label:
                    self._draw_labels(ax, labels, prev_labels, dt, y0)
                prev_labels.append(labels)

            else:
                continue
            # plot label
            ax.text(x=0, y=y0, s=channel.name,
                    fontsize=self.style.axis_font_size,
                    ha='right', va='center')

            y0 -= 1
        return y0

    def draw(self, schedule, dt, interp_method, plot_range,
             scaling=1, channels_to_plot=None, plot_all=True,
             table=True, label=False, framechange=True):
        """Draw figure.
        Args:
            schedule (ScheduleComponent): Schedule to draw
            dt (float): time interval
            interp_method (Callable): interpolation function
                See `qiskit.visualization.interpolation` for more information
            plot_range (tuple[float]): plot range
            scaling (float): Relative visual scaling of waveform amplitudes
            channels_to_plot (list[OutputChannel]): channels to draw
            plot_all (bool): if plot all channels even it is empty
            table (bool): Draw event table
            label (bool): Label individual instructions
            framechange (bool): Add framechange indicators

        Returns:
            matplotlib.figure: A matplotlib figure object for the pulse schedule

        Raises:
            VisualizationError: when schedule cannot be drawn
        """
        figure = plt.figure()

        if not channels_to_plot:
            channels_to_plot = []
        interp_method = interp_method or interpolation.step_wise

        # setup plot range
        if plot_range:
            t0 = int(np.floor(plot_range[0]/dt))
            tf = int(np.floor(plot_range[1]/dt))
        else:
            t0 = 0
            tf = schedule.stop_time
        # prepare waveform channels
        (channels, output_channels,
         snapshot_channels) = self._build_channels(schedule, t0, tf)

        # count numbers of valid waveform
        n_valid_waveform, v_max = self._count_valid_waveforms(output_channels, scaling=scaling,
                                                              channels_to_plot=channels_to_plot,
                                                              plot_all=plot_all)

        if table:
            ax = self._draw_table(figure, channels, dt, n_valid_waveform)

        else:
            ax = figure.add_subplot(111)
            figure.set_size_inches(self.style.figsize[0], self.style.figsize[1])

        ax.set_facecolor(self.style.bg_color)

        y0 = self._draw_channels(ax, output_channels, interp_method,
                                 t0, tf, dt, v_max, label=label,
                                 framechange=framechange)

        self._draw_snapshots(ax, snapshot_channels, dt, y0)

        ax.set_xlim(t0 * dt, tf * dt)
        ax.set_ylim(y0, 1)
        ax.set_yticklabels([])

        return figure
