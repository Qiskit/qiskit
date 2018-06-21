#!/usr/bin/env python
# coding: utf-8
#
# Copyright 2018 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------
# pylint: disable=invalid-name
# -----------------------------------------------------------------------------
#
# Quantum circuit drawer based on matplotlib:
# A drop-in replacement of latex_drawer.
# This module visualizes quantum circuit of qiskit.QuantumCircuit
# as well as a qasm file
#
# Authors: Takashi Imamichi, Naoki Kanazawa
# -----------------------------------------------------------------------------

import json
import logging
import re
from collections import namedtuple
from fractions import Fraction
from itertools import zip_longest
from math import fmod, isclose

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from qiskit import QuantumCircuit, QISKitError
from qiskit.qasm import Qasm
from qiskit.unroll import Unroller, JsonBackend

try:
    # for qiskit 0.5
    from qiskit import load_qasm_file

    use_qp = False
except ImportError:
    # for qiskit 0.4
    from qiskit import QuantumProgram

    use_qp = True

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# definitions
# -----------------------------------------------------------------------------
WID = 0.65
HIG = 0.65
DEFAULT_SCALE = 3
PORDER_GATE = 5
PORDER_LINE = 2
PORDER_GRAY = 3
PORDER_TEXT = 6
PORDER_SUBP = 4


def matplotlib_drawer(circuit,
                      basis='id,u0,u1,u2,u3,x,y,z,h,s,sdg,t,tdg,rx,ry,rz,'
                            'cx,cy,cz,ch,crz,cu1,cu3,swap,ccx,cswap',
                      scale=1.0, style=None, filename=None):
    """Draw a quantum circuit based on matplotlib

    Args:
        circuit (QuantumCircuit): a quantum circuit
        basis (str): comma separated list of gates
        scale (float): scaling factor
        style (dict or str): dictionary of style or file name of style file
        filename (str): output filename of circuit drawing if filename is not None
    """
    qcd = MatplotlibDrawer(basis=basis, scale=scale, style=style)
    qcd.parse_circuit(circuit)
    qcd.draw(filename)


Register = namedtuple('Register', 'name index')


class QCStyle:
    def __init__(self):
        self.tc = '#000000'
        self.sc = '#000000'
        self.lc = '#000000'
        self.cc = '#778899'
        self.gc = '#ffffff'
        self.gt = '#000000'
        self.bc = '#bdbdbd'
        self.bg = '#ffffff'
        self.fs = 10
        self.sfs = 6
        self.disptex = {
            'id': 'id',
            'u0': 'U_0',
            'u1': 'U_1',
            'u2': 'U_2',
            'u3': 'U_3',
            'x': 'X',
            'y': 'Y',
            'z': 'Z',
            'h': 'H',
            's': 'S',
            'sdg': 'S^\\dagger',
            't': 'T',
            'tdg': 'T^\\dagger',
            'rx': 'R_x',
            'ry': 'R_y',
            'rz': 'R_z',
            'reset': '\\left|0\\right\\rangle'
        }
        self.dispcol = {
            'id': '#ffffff',
            'u0': '#ffffff',
            'u1': '#ffffff',
            'u2': '#ffffff',
            'u3': '#ffffff',
            'x': '#ffffff',
            'y': '#ffffff',
            'z': '#ffffff',
            'h': '#ffffff',
            's': '#ffffff',
            'sdg': '#ffffff',
            't': '#ffffff',
            'tdg': '#ffffff',
            'rx': '#ffffff',
            'ry': '#ffffff',
            'rz': '#ffffff',
            'reset': '#ffffff',
            'target': '#ffffff',
            'meas': '#ffffff'
        }
        self.latexmode = True
        self.pimode = False
        self.fold = 20
        self.bundle = False
        self.barrier = False
        self.index = False
        self.compress = False
        self.figwidth = -1
        self.dpi = 150

    def set_style(self, dic):
        self.tc = dic.get('textcolor', self.tc)
        self.sc = dic.get('subtextcolor', self.sc)
        self.lc = dic.get('linecolor', self.lc)
        self.cc = dic.get('creglinecolor', self.cc)
        self.gt = dic.get('gatetextcolor', self.tc)
        self.gc = dic.get('gatefacecolor', self.gc)
        self.bc = dic.get('barrierfacecolor', self.bc)
        self.bg = dic.get('backgroundcolor', self.bg)
        self.fs = dic.get('fontsize', self.fs)
        self.sfs = dic.get('subfontsize', self.sfs)
        self.disptex = dic.get('displaytext', self.disptex)
        for key in self.dispcol.keys():
            self.dispcol[key] = self.gc
        self.dispcol = dic.get('displaycolor', self.dispcol)
        self.latexmode = dic.get('latexdrawerstyle', self.latexmode)
        self.pimode = dic.get('usepiformat', self.pimode)
        self.fold = dic.get('fold', self.fold)
        if self.fold < 2:
            self.fold = -1
        self.bundle = dic.get('cregbundle', self.bundle)
        self.barrier = dic.get('plotbarrier', self.barrier)
        self.index = dic.get('showindex', self.index)
        self.compress = dic.get('compress', self.compress)
        self.figwidth = dic.get('figwidth', self.figwidth)
        self.dpi = dic.get('dpi', self.dpi)


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
            self.__gate_placed.append(_index + ii)
        self.__gate_placed.sort()

    def get_index(self):
        if len(self.__gate_placed) > 0:
            return self.__gate_placed[-1] + 1
        else:
            return 0


class MatplotlibDrawer:
    def __init__(self,
                 basis='id,u0,u1,u2,u3,x,y,z,h,s,sdg,t,tdg,rx,ry,rz,'
                       'cx,cy,cz,ch,crz,cu1,cu3,swap,ccx,cswap',
                 scale=1.0, style=None):

        self._ast = None
        if ',' not in basis:
            logger.warning('Warning: basis is not comma separated: "%s". '
                           'Perhaps you set `filename` to `basis`.', basis)
        self._basis = basis.split(',')
        self._scale = DEFAULT_SCALE * scale
        self._creg = []
        self._qreg = []
        self._ops = []
        self._qreg_dict = {}
        self._creg_dict = {}
        self._cond = {
            'n_lines': 0,
            'xmax': 0,
            'ymax': 0,
        }

        self._style = QCStyle()
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
        self.ax.set_aspect('equal', 'datalim')

    def load_qasm_file(self, filename):
        if use_qp:
            qp = QuantumProgram()
            qp.load_qasm_file(filename, name='draw', basis_gates=','.join(self._basis))
            circuit = qp.get_circuit(qp.get_circuit_names()[0])
        else:
            circuit = load_qasm_file(filename, name='draw', basis_gates=','.join(self._basis))
        self.parse_circuit(circuit)

    def parse_circuit(self, circuit: QuantumCircuit):
        ast = Qasm(data=circuit.qasm()).parse()
        u = Unroller(ast, JsonBackend(self._basis))
        u.execute()
        self._ast = u.backend.circuit
        self._registers()
        self._ops = self._ast['operations']

    def _registers(self):
        # NOTE: formats of clbit and qubit are different!
        header = self._ast['header']
        self._creg = []
        for e in header['clbit_labels']:
            for i in range(e[1]):
                self._creg.append(Register(name=e[0], index=i))
        assert len(self._creg) == header['number_of_clbits']
        self._qreg = []
        for e in header['qubit_labels']:
            self._qreg.append(Register(name=e[0], index=e[1]))
        assert len(self._qreg) == header['number_of_qubits']

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

        box = patches.Rectangle(xy=(xpos - 0.5 * wid, ypos - 0.5 * HIG), width=wid, height=HIG,
                                fc=_fc, ec=self._style.lc, linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)

        if text:
            disp_text = "${}$".format(self._style.disptex[text])
            if subtext:
                self.ax.text(xpos, ypos + 0.15 * HIG, disp_text, ha='center', va='center',
                             fontsize=self._style.fs,
                             color=self._style.gt,
                             zorder=PORDER_TEXT)
                self.ax.text(xpos, ypos - 0.3 * HIG, subtext, ha='center', va='center',
                             fontsize=self._style.sfs,
                             color=self._style.sc,
                             zorder=PORDER_TEXT)
            else:
                self.ax.text(xpos, ypos, disp_text, ha='center', va='center',
                             fontsize=self._style.fs,
                             color=self._style.gt,
                             zorder=PORDER_TEXT)

    def _subtext(self, xy, text):
        xpos, ypos = xy

        self.ax.text(xpos, ypos - 0.3 * HIG, text, ha='center', va='top',
                     fontsize=self._style.sfs,
                     color=self._style.tc,
                     zorder=PORDER_TEXT)

    def _line(self, xy0, xy1):
        x0, y0 = xy0
        x1, y1 = xy1
        self.ax.plot([x0, x1], [y0, y1],
                     color=self._style.lc,
                     linewidth=1.0,
                     zorder=PORDER_LINE)

    def _measure(self, qxy, cxy, cid):
        qx, qy = qxy
        cx, cy = cxy

        self._gate(qxy, fc=self._style.dispcol['meas'])
        # add measure symbol
        arc = patches.Arc(xy=(qx, qy - 0.15 * HIG), width=WID * 0.7, height=HIG * 0.7,
                          theta1=0, theta2=180, fill=False,
                          ec=self._style.lc, linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(arc)
        self.ax.plot([qx, qx + 0.35 * WID], [qy - 0.15 * HIG, qy + 0.20 * HIG],
                     color=self._style.lc, linewidth=1.5, zorder=PORDER_GATE)
        # arrow
        self.ax.arrow(x=qx, y=qy, dx=0, dy=cy - qy, width=0.01, head_width=0.2, head_length=0.2,
                      length_includes_head=True, color=self._style.cc, zorder=PORDER_LINE)
        # target
        if self._style.bundle:
            self.ax.text(cx + .25, cy + .1, str(cid), ha='left', va='bottom',
                         fontsize=0.8 * self._style.fs,
                         color=self._style.tc,
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
                             fc=self._style.dispcol['target'], ec=self._style.lc,
                             linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)
        # add '+' symbol
        self.ax.plot([xpos, xpos], [ypos - 0.35 * HIG, ypos + 0.35 * HIG],
                     color=self._style.lc, linewidth=1.0, zorder=PORDER_GATE)
        self.ax.plot([xpos - 0.35 * HIG, xpos + 0.35 * HIG], [ypos, ypos],
                     color=self._style.lc, linewidth=1.0, zorder=PORDER_GATE)

    def _swap(self, xy):
        xpos, ypos = xy

        self.ax.plot([xpos - 0.20 * WID, xpos + 0.20 * WID], [ypos - 0.20 * WID, ypos + 0.20 * WID],
                     color=self._style.lc, linewidth=1.5, zorder=PORDER_LINE)
        self.ax.plot([xpos - 0.20 * WID, xpos + 0.20 * WID], [ypos + 0.20 * WID, ypos - 0.20 * WID],
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
        self.ax.set_xlim(-1.5, self._cond['xmax'] + 1.5)
        self.ax.set_ylim(self._cond['ymax'] - 1.5, 1.5)
        # update figure size
        fig_w = abs(self._cond['xmax']) + 2
        fig_h = abs(self._cond['ymax']) + 2
        if self._style.figwidth < 0.0:
            self._style.figwidth = fig_w * self._scale * self._style.fs / 72 / WID
        self.figure.set_size_inches(self._style.figwidth, self._style.figwidth * fig_h / fig_w)
        if filename:
            self.figure.savefig(filename, dpi=self._style.dpi, bbox_inches='tight')
        else:
            plt.show()

    def _draw_regs(self):
        self._pl_qxl = [0] * len(self._qreg)
        self._pl_cxl = [0] * len(self._creg)
        # quantum register
        for ii, reg in enumerate(self._qreg):
            if len(self._qreg) > 1:
                label = '${}_{{{}}}$'.format(reg.name, reg.index)
            else:
                label = '${}$'.format(reg.name)
            pos = -ii
            self._qreg_dict[ii] = {'y': pos, 'label': label, 'index': reg.index, 'group': reg.name}
            self._cond['n_lines'] += 1
        # classical register
        if self._creg:
            n_creg = self._creg.copy()
            n_creg.pop(0)
            idx = 0
            y_off = -len(self._qreg)
            for ii, (reg, nreg) in enumerate(zip_longest(self._creg, n_creg)):
                pos = y_off - idx
                if self._style.bundle:
                    label = '${}$'.format(reg.name)
                    self._creg_dict[ii] = {'y': pos, 'label': label, 'index': reg.index, 'group': reg.name}
                    if not (not nreg or reg.name != nreg.name):
                        continue
                else:
                    label = '${}_{{{}}}$'.format(reg.name, reg.index)
                    self._creg_dict[ii] = {'y': pos, 'label': label, 'index': reg.index, 'group': reg.name}
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
                         zorder=PORDER_TEXT)
            self.ax.plot([0, self._cond['xmax']], [y, y], color=self._style.lc, zorder=PORDER_LINE)
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
        for this_creg in this_creg_dict.values():
            # bundle
            if this_creg['val'] > 1:
                self.ax.plot([.6, .7], [y - .1, y + .1],
                             color=self._style.cc,
                             zorder=PORDER_LINE)
                self.ax.text(0.5, y + .1, str(this_creg['val']), ha='left', va='bottom',
                             fontsize=0.8 * self._style.fs,
                             color=self._style.tc,
                             zorder=PORDER_TEXT)
            self.ax.text(-0.5, y, this_creg['label'], ha='right', va='center',
                         fontsize=self._style.fs,
                         color=self._style.tc,
                         zorder=PORDER_TEXT)
            self.ax.plot([0, self._cond['xmax']], [y, y], color=self._style.cc, zorder=PORDER_LINE)

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
        for i, (op, op_next) in enumerate(zip_longest(self._ops, next_ops)):
            # no-barrier
            if op['name'] == 'barrier' and not self._style.barrier:
                continue
            # wide gate
            if op['name'] in _wide_gate:
                _iswide = True
                gw = 2
            else:
                _iswide = False
                gw = 1
            # get qreg index
            q_idxs = op['qubits']
            # get creg index
            if 'clbits' in op.keys():
                c_idxs = op['clbits']
            else:
                c_idxs = []
            # find empty space to place gate
            if len(_barriers['group']) == 0:
                this_anc = max([q_anchors[ii].get_index() for ii in q_idxs])
                while True:
                    if op['name'] in _force_next or 'conditional' in op.keys() or not self._style.compress:
                        occupied = self._qreg_dict.keys()
                    else:
                        occupied = q_idxs
                    q_list = [ii for ii in range(min(occupied), max(occupied) + 1)]
                    locs = [q_anchors[jj].is_locatable(this_anc, gw) for jj in q_list]
                    if all(locs):
                        for ii in q_list:
                            q_anchors[ii].set_index(this_anc, gw)
                        break
                    else:
                        this_anc += 1
            # qreg coordinate
            q_xy = [q_anchors[ii].plot_coord(this_anc, gw) for ii in q_idxs]
            # creg corrdinate
            c_xy = [c_anchors[ii].plot_coord(this_anc, gw) for ii in c_idxs]
            # bottom and top point of qreg
            qreg_b = sorted(q_xy, key=lambda xy: xy[1])[0]
            qreg_t = sorted(q_xy, key=lambda xy: xy[1])[-1]

            if verbose:
                print(i, op)

            # rotation parameter
            if 'texparams' in op.keys():
                param = self.param_parse(op['texparams'], self._style.pimode)
            else:
                param = None
            # conditional gate
            if 'conditional' in op.keys():
                c_xy = [c_anchors[ii].plot_coord(this_anc, gw) for ii in self._creg_dict.keys()]
                if self._style.bundle:
                    c_xy = list(set(c_xy))
                    for xy in c_xy:
                        self._conds(xy, istrue=True)
                else:
                    fmt = '{{:0{}b}}'.format(len(c_xy))
                    vlist = list(fmt.format(int(op['conditional']['val'], 16)))[::-1]
                    for xy, v in zip(c_xy, vlist):
                        if v == '0':
                            bv = False
                        else:
                            bv = True
                        self._conds(xy, istrue=bv)
                creg_b = sorted(c_xy, key=lambda xy: xy[1])[0]
                self._subtext(creg_b, op['conditional']['val'])
                self._line(qreg_t, creg_b)
            #
            # draw special gates
            #
            if op['name'] == 'measure':
                vv = self._creg_dict[c_idxs[0]]['index']
                self._measure(q_xy[0], c_xy[0], vv)
            elif op['name'] == 'barrier':
                if self._style.barrier:
                    q_group = self._qreg_dict[q_idxs[0]]['group']
                    if q_group not in _barriers['group']:
                        _barriers['group'].append(q_group)
                    _barriers['coord'].append(q_xy[0])
                    if op_next and op_next['name'] == 'barrier':
                        continue
                    else:
                        self._barrier(_barriers, this_anc)
                        _barriers['group'].clear()
                        _barriers['coord'].clear()
                else:
                    continue
            #
            # draw single qubit gates
            #
            elif len(q_xy) == 1:
                disp = op['name']
                if param:
                    self._gate(q_xy[0], wide=_iswide, text=disp, subtext='{}'.format(param))
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
                        self._gate(q_xy[1], wide=_iswide, text=disp, subtext='{}'.format(param))
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
                        self._gate(q_xy[1], wide=_iswide, text=disp, subtext='{}'.format(param))
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
                raise QISKitError('invalid gate {}'.format(op))
        #
        # adjust window size and draw horizontal lines
        #
        max_anc = max([q_anchors[ii].get_index() for ii in self._qreg_dict.keys()])
        n_fold = (max_anc - 1) // self._style.fold
        # window size
        if max_anc > self._style.fold > 0:
            self._cond['xmax'] = self._style.fold + 1
            self._cond['ymax'] = - (n_fold + 1) * (self._cond['n_lines'] + 1) + 1
        else:
            self._cond['xmax'] = max_anc + 1
            self._cond['ymax'] = - self._cond['n_lines']
        # add horizontal lines
        for ii in range(n_fold + 1):
            if n_fold > 0 and n_fold > ii:
                feedline_r = True
            else:
                feedline_r = False
            if ii > 0:
                feedline_l = True
            else:
                feedline_l = False
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
                self.ax.text(x_coord, y_coord, str(ii + 1), ha='center', va='center',
                             fontsize=self._style.sfs,
                             color=self._style.tc,
                             zorder=PORDER_TEXT)

    @staticmethod
    def param_parse(v, pimode=False):
        for i, e in enumerate(v):
            z = e.split(',')
            buf = ''
            for k in z:
                val = MatplotlibDrawer.parse_numeric(k, pimode)
                if isinstance(val, (int, float)):
                    if pimode:
                        buf += MatplotlibDrawer.format_pi(val)
                    else:
                        buf += MatplotlibDrawer.format_numeric(val)
                elif '\\pi' in val:
                    t = val.split()  # `val` is in a form '{coef} \\pi' or '- {coef} \\pi'
                    buf += MatplotlibDrawer.format_numeric(''.join(t[:-1])) + t[-1]
                else:
                    buf += val
            v[i] = buf
        param = re.sub(r'\\pi', '$\\pi$', ','.join(v))
        param = re.sub(r' ', '', param)
        param = re.sub(r',', ', ', param)
        param = re.sub(r'-', '−', param)
        return param

    @staticmethod
    def parse_numeric(k, pimode=False):
        # parse a string and return number or string
        f = 0.0
        is_numeric = False
        try:
            f = float(k)
            is_numeric = True
        except ValueError:
            if pimode and '\\pi' in k:
                if k == '\\pi':
                    f = np.pi
                    is_numeric = True
                elif k == '- \\pi':
                    f = -np.pi
                    is_numeric = True
                else:
                    _k = re.sub(r' ', '', k)
                    _k = re.sub(r'\\pi', '', _k)
                    try:
                        f = float(_k) * np.pi
                        is_numeric = True
                    except ValueError:
                        pass
        return f if is_numeric else k

    @staticmethod
    def format_pi(val):
        fracvals = MatplotlibDrawer.fraction(val)
        buf = ''
        if fracvals:
            nmr, dnm = fracvals.numerator, fracvals.denominator
            if nmr == 1:
                buf += '\\pi'
            elif nmr == -1:
                buf += '-\\pi'
            else:
                buf += '{}\\pi'.format(nmr)
            if dnm > 1:
                buf += '/{}'.format(dnm)
            return buf
        else:
            return MatplotlibDrawer.format_numeric(val)

    @staticmethod
    def format_numeric(val, tol=1e-5):
        try:
            val = float(val)
        except ValueError:
            return val
        abs_val = abs(val)
        if isclose(fmod(abs_val, 1.0), 0.0, abs_tol=tol) and 0.0 <= abs_val < 10000.0:
            return str(int(val))
        elif 0.1 <= abs_val < 100.0:
            return '{:.2f}'.format(val)
        return '{:.1e}'.format(val)

    @staticmethod
    def fraction(val, base=np.pi, n=100, tol=1e-5):
        abs_val = abs(val)
        for i in range(1, n):
            for j in range(1, n):
                if isclose(abs_val, i / j * base, rel_tol=tol):
                    if val < 0:
                        i *= -1
                    return Fraction(i, j)
        return None
