# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,anomalous-backslash-in-string,missing-docstring

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
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from qiskit.tools.visualization import _error
from qiskit.tools.visualization import _qcstyle
from qiskit.tools.visualization import _utils


logger = logging.getLogger(__name__)

Register = collections.namedtuple('Register', 'name index')

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
    def __init__(self,
                 scale=1.0, style=None, plot_barriers=True,
                 reverse_bits=False):

        if not HAS_MATPLOTLIB:
            raise ImportError('The class MatplotlibDrawer needs matplotlib. '
                              'Run "pip install matplotlib" before.')

        self._ast = None
        self._scale = DEFAULT_SCALE * scale
        self._creg = []
        self._qreg = []
        self._ops = []
        self._qreg_dict = collections.OrderedDict()
        self._creg_dict = collections.OrderedDict()
        self._cond = {
            'n_lines': 0,
            'xmax': 0,
            'ymax': 0,
        }

        self._style = _qcstyle.QCStyle()
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

    def parse_circuit(self, circuit):
        qregs, cregs, ops = _utils._get_instructions(
            circuit, reversebits=self.reverse_bits)
        self._registers(cregs, qregs)
        self._ops = ops

    def _registers(self, creg, qreg):
        # NOTE: formats of clbit and qubit are different!
        self._creg = []
        for e in creg:
            self._creg.append(Register(name=e[0], index=e[1]))
        self._qreg = []
        for e in qreg:
            self._qreg.append(Register(name=e[0], index=e[1]))

    @property
    def ast(self):
        return self._ast

    def _gate(self, xy, fc=None, wide=False, text=None, subtext=None):
        xpos, ypos = xy

        if wide:
            wid = WID * 2.8
        else:
            wid = WID
        if fc:
            _fc = fc
        elif text:
            _fc = self._style.dispcol[text]
        else:
            _fc = self._style.gc

        box = patches.Rectangle(
            xy=(xpos - 0.5 * wid, ypos - 0.5 * HIG), width=wid, height=HIG,
            fc=_fc, ec=self._style.lc, linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)

        if text:
            disp_text = "${}$".format(self._style.disptex[text])
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

        box_y0 = min(y_reg) - int(anc / self._style.fold) * (
            self._cond['n_lines'] + 1) - 0.5
        box_y1 = max(y_reg) - int(anc / self._style.fold) * (
            self._cond['n_lines'] + 1) + 0.5
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
                label = '${}_{{{}}}$'.format(reg.name.name, reg.index)
            else:
                label = '${}$'.format(reg.name.name)
            pos = -ii
            self._qreg_dict[ii] = {
                'y': pos,
                'label': label,
                'index': reg.index,
                'group': reg.name
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
                    label = '${}$'.format(reg.name.name)
                    self._creg_dict[ii] = {
                        'y': pos,
                        'label': label,
                        'index': reg.index,
                        'group': reg.name
                    }
                    if not (not nreg or reg.name != nreg.name):
                        continue
                else:
                    label = '${}_{{{}}}$'.format(reg.name.name, reg.index)
                    self._creg_dict[ii] = {
                        'y': pos,
                        'label': label,
                        'index': reg.index,
                        'group': reg.name
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
        _force_next = 'measure barrier'.split()
        _wide_gate = 'u2 u3 cu2 cu3'.split()
        _barriers = {'coord': [], 'group': []}
        next_ops = self._ops.copy()
        next_ops.pop(0)
        this_anc = 0

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
        for i, (op, op_next) in enumerate(
                itertools.zip_longest(self._ops, next_ops)):
            # wide gate
            if op['name'] in _wide_gate:
                _iswide = True
                gw = 2
            else:
                _iswide = False
                gw = 1
            # get qreg index
            if 'qargs' in op.keys():
                q_idxs = []
                for qarg in op['qargs']:
                    for index, reg in self._qreg_dict.items():
                        if (reg['group'] == qarg[0] and
                                reg['index'] == qarg[1]):
                            q_idxs.append(index)
                            break
            else:
                q_idxs = []
            # get creg index
            if 'cargs' in op.keys():
                c_idxs = []
                for carg in op['cargs']:
                    for index, reg in self._creg_dict.items():
                        if (reg['group'] == carg[0] and
                                reg['index'] == carg[1]):
                            c_idxs.append(index)
                            break
            else:
                c_idxs = []
            # find empty space to place gate
            if not _barriers['group']:
                this_anc = max([q_anchors[ii].get_index() for ii in q_idxs])
                while True:
                    if op['name'] in _force_next or (
                            'condition' in op.keys() and op['condition']) or \
                            not self._style.compress:
                        occupied = self._qreg_dict.keys()
                    else:
                        occupied = q_idxs
                    q_list = [ii for ii in range(min(occupied),
                                                 max(occupied) + 1)]
                    locs = [q_anchors[jj].is_locatable(
                        this_anc, gw) for jj in q_list]
                    if all(locs):
                        for ii in q_list:
                            if op['name'] in [
                                    'barrier', 'snapshot', 'load', 'save',
                                    'noise'] and not self.plot_barriers:
                                q_anchors[ii].set_index(this_anc - 1, gw)
                            else:
                                q_anchors[ii].set_index(this_anc, gw)
                        break
                    else:
                        this_anc += 1
            # qreg coordinate
            q_xy = [q_anchors[ii].plot_coord(this_anc, gw) for ii in q_idxs]
            # creg coordinate
            c_xy = [c_anchors[ii].plot_coord(this_anc, gw) for ii in c_idxs]
            # bottom and top point of qreg
            qreg_b = min(q_xy, key=lambda xy: xy[1])
            qreg_t = max(q_xy, key=lambda xy: xy[1])

            if verbose:
                print(i, op)

            # rotation parameter
            if 'op' in op.keys() and hasattr(op['op'], 'param'):
                param = self.param_parse(op['op'].param, self._style.pimode)
            else:
                param = None
            # conditional gate
            if 'condition' in op.keys() and op['condition']:
                c_xy = [c_anchors[ii].plot_coord(this_anc, gw) for
                        ii in self._creg_dict]
                mask = 0
                for index, cbit in enumerate(self._creg):
                    if cbit.name == op['condition'][0]:
                        mask |= (1 << index)
                val = op['condition'][1]
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
            if op['name'] == 'measure':
                vv = self._creg_dict[c_idxs[0]]['index']
                self._measure(q_xy[0], c_xy[0], vv)
            elif op['name'] in ['barrier', 'snapshot', 'load', 'save',
                                'noise']:
                q_group = self._qreg_dict[q_idxs[0]]['group']
                if q_group not in _barriers['group']:
                    _barriers['group'].append(q_group)
                _barriers['coord'].append(q_xy[0])
                if op_next and op_next['name'] == 'barrier':
                    continue
                else:
                    if self.plot_barriers:
                        self._barrier(_barriers, this_anc)
                    _barriers['group'].clear()
                    _barriers['coord'].clear()
            #
            # draw single qubit gates
            #
            elif len(q_xy) == 1:
                disp = op['name']
                if param:
                    self._gate(q_xy[0], wide=_iswide, text=disp,
                               subtext='{}'.format(param))
                else:
                    self._gate(q_xy[0], wide=_iswide, text=disp)
            #
            # draw multi-qubit gates (n=2)
            #
            elif len(q_xy) == 2:
                # cx
                if op['name'] in ['cx']:
                    self._ctrl_qubit(q_xy[0])
                    self._tgt_qubit(q_xy[1])
                # cz for latexmode
                elif op['name'] == 'cz':
                    if self._style.latexmode:
                        self._ctrl_qubit(q_xy[0])
                        self._ctrl_qubit(q_xy[1])
                    else:
                        disp = op['name'].replace('c', '')
                        self._ctrl_qubit(q_xy[0])
                        self._gate(q_xy[1], wide=_iswide, text=disp)
                # control gate
                elif op['name'] in ['cy', 'ch', 'cu3', 'crz']:
                    disp = op['name'].replace('c', '')
                    self._ctrl_qubit(q_xy[0])
                    if param:
                        self._gate(q_xy[1], wide=_iswide, text=disp,
                                   subtext='{}'.format(param))
                    else:
                        self._gate(q_xy[1], wide=_iswide, text=disp)
                # cu1 for latexmode
                elif op['name'] in ['cu1']:
                    disp = op['name'].replace('c', '')
                    self._ctrl_qubit(q_xy[0])
                    if self._style.latexmode:
                        self._ctrl_qubit(q_xy[1])
                        self._subtext(qreg_b, param)
                    else:
                        self._gate(q_xy[1], wide=_iswide, text=disp,
                                   subtext='{}'.format(param))
                # swap gate
                elif op['name'] == 'swap':
                    self._swap(q_xy[0])
                    self._swap(q_xy[1])
                # add qubit-qubit wiring
                self._line(qreg_b, qreg_t)
            #
            # draw multi-qubit gates (n=3)
            #
            elif len(q_xy) == 3:
                # cswap gate
                if op['name'] == 'cswap':
                    self._ctrl_qubit(q_xy[0])
                    self._swap(q_xy[1])
                    self._swap(q_xy[2])
                # ccx gate
                elif op['name'] == 'ccx':
                    self._ctrl_qubit(q_xy[0])
                    self._ctrl_qubit(q_xy[1])
                    self._tgt_qubit(q_xy[2])
                # add qubit-qubit wiring
                self._line(qreg_b, qreg_t)
            else:
                logger.critical('Invalid gate %s', op)
                raise _error.VisualizationError('invalid gate {}'.format(op))
        #
        # adjust window size and draw horizontal lines
        #
        max_anc = max([q_anchors[ii].get_index() for ii in self._qreg_dict])
        n_fold = (max_anc - 1) // self._style.fold
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
                    y_coord = - (ii // self._style.fold) * (
                        self._cond['n_lines'] + 1) + 0.7
                else:
                    x_coord = ii + 1
                    y_coord = 0.7
                self.ax.text(x_coord, y_coord, str(ii + 1), ha='center',
                             va='center', fontsize=self._style.sfs,
                             color=self._style.tc, clip_on=True,
                             zorder=PORDER_TEXT)

    @staticmethod
    def param_parse(v, pimode=False):
        for i, e in enumerate(v):
            if pimode:
                v[i] = MatplotlibDrawer.format_pi(e)
            else:
                v[i] = MatplotlibDrawer.format_numeric(e)
            if v[i].startswith('-'):
                v[i] = '$-$' + v[i][1:]
        param = ', '.join(v)
        return param

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
