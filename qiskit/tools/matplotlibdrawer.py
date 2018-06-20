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
            'n_linefeeds': 0,
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

        self._pl_qxl = []
        self._pl_cxl = []

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

    def _gate(self, xpos, ypos, fc=None, wide=False, text=None, subtext=None):
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

    def _subtext(self, xpos, ypos, text):
        self.ax.text(xpos, ypos - 0.3 * HIG, text, ha='center', va='top',
                     fontsize=self._style.sfs,
                     color=self._style.tc,
                     zorder=PORDER_TEXT)

    def _measure(self, xpos, yqpos, ycpos, cid):
        self._gate(xpos, yqpos, fc=self._style.dispcol['meas'])
        arc = patches.Arc(xy=(xpos, yqpos - 0.15 * HIG), width=WID * 0.7, height=HIG * 0.7,
                          theta1=0, theta2=180, fill=False,
                          ec=self._style.lc, linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(arc)
        self.ax.plot([xpos, xpos + 0.35 * WID], [yqpos - 0.15 * HIG, yqpos + 0.20 * HIG],
                     color=self._style.lc, linewidth=1.5, zorder=PORDER_GATE)
        # arrow
        self.ax.arrow(x=xpos, y=yqpos, dx=0, dy=ycpos - yqpos, width=0.01, head_width=0.2,
                      head_length=0.2, length_includes_head=True, color=self._style.cc,
                      zorder=PORDER_LINE)
        # target
        if self._style.bundle:
            self.ax.text(xpos + .25, ycpos + .1, str(cid), ha='left', va='bottom',
                         fontsize=0.8 * self._style.fs,
                         color=self._style.tc,
                         zorder=PORDER_TEXT)

    def _conds(self, xpos, ypos, istrue=False):
        if istrue:
            _fc = self._style.lc
        else:
            _fc = self._style.gc

        box = patches.Circle(xy=(xpos, ypos), radius=WID * 0.15,
                             fc=_fc, ec=self._style.lc,
                             linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)

    def _ctrl_qubit(self, xpos, ypos):
        box = patches.Circle(xy=(xpos, ypos), radius=WID * 0.15,
                             fc=self._style.lc, ec=self._style.lc,
                             linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)

    def _tgt_qubit(self, xpos, ypos):
        box = patches.Circle(xy=(xpos, ypos), radius=WID * 0.35,
                             fc=self._style.dispcol['target'], ec=self._style.lc,
                             linewidth=1.5, zorder=PORDER_LINE)
        self.ax.add_patch(box)

    def _swap(self, xpos, ypos):
        self.ax.plot([xpos - 0.20 * WID, xpos + 0.20 * WID], [ypos - 0.20 * WID, ypos + 0.20 * WID],
                     color=self._style.lc, linewidth=1.5, zorder=PORDER_LINE)
        self.ax.plot([xpos - 0.20 * WID, xpos + 0.20 * WID], [ypos + 0.20 * WID, ypos - 0.20 * WID],
                     color=self._style.lc, linewidth=1.5, zorder=PORDER_LINE)

    def _barrier(self, xpos, ypos):
        box_y0 = self._linefeed_y(self._qreg_dict[max(self._qreg_dict.keys())]['y']) - 0.5
        box_y1 = self._linefeed_y(self._qreg_dict[min(self._qreg_dict.keys())]['y']) + 0.5
        box = patches.Rectangle(xy=(xpos - 0.3 * WID, box_y0),
                                width=0.6 * WID, height=box_y1 - box_y0,
                                fc=self._style.bc, ec=None, alpha=0.6,
                                linewidth=1.5, zorder=PORDER_GRAY)
        self.ax.add_patch(box)
        for y in ypos:
            self.ax.plot([xpos, xpos], [y + 0.5, y - 0.5],
                         linewidth=1, linestyle="dashed",
                         color=self._style.lc,
                         zorder=PORDER_TEXT)

    def _linefeed_mark(self, xpos):
        self.ax.plot([xpos - .1, xpos - .1],
                     [self._linefeed_y(0), self._linefeed_y(-self._cond['n_lines'] + 1)],
                     color=self._style.lc, zorder=PORDER_LINE)
        self.ax.plot([xpos + .1, xpos + .1],
                     [self._linefeed_y(0), self._linefeed_y(-self._cond['n_lines'] + 1)],
                     color=self._style.lc, zorder=PORDER_LINE)

    def draw(self, filename=None, verbose=False):
        self._draw_regs()
        self._draw_ops(verbose)
        self.ax.set_xlim(-1, self._cond['xmax'] + 1)
        self.ax.set_ylim(self._cond['ymax'] - 1, 1)
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
            self._qreg_dict[ii] = {'y': pos, 'label': label, 'index': reg.index}
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
                    self._creg_dict[ii] = {'y': pos, 'label': label, 'index': reg.index}
                    if not (not nreg or reg.name != nreg.name):
                        continue
                else:
                    label = '${}_{{{}}}$'.format(reg.name, reg.index)
                    self._creg_dict[ii] = {'y': pos, 'label': label, 'index': reg.index}
                idx += 1
        self._pl_qxl, self._pl_cxl = self._draw_regs_sub()
        self._cond['n_lines'] = len(self._pl_qxl) + len(self._pl_cxl)

    def _draw_regs_sub(self):
        pl_qxl = [0] * len(self._qreg_dict.keys())
        pl_cxl = [0] * len(self._creg_dict.keys())
        # quantum register
        for ii, idx in enumerate(self._qreg_dict.keys()):
            if self._cond['n_linefeeds'] == 0:
                label = self._qreg_dict[idx]['label'] + ' : $\\left|0\\right\\rangle$'
            else:
                label = self._qreg_dict[idx]['label']
            y = self._linefeed_y(self._qreg_dict[idx]['y'])
            self.ax.text(-0.5, y, label, ha='right', va='center',
                         fontsize=self._style.fs,
                         color=self._style.tc,
                         zorder=PORDER_TEXT)
            pl_qxl[ii], = self.ax.plot([0, 0], [y, y],
                                       color=self._style.lc,
                                       zorder=PORDER_LINE)
        # classical register
        this_creg_dict = {}
        for v in self._creg_dict.values():
            if self._cond['n_linefeeds'] == 0:
                label = v['label'] + ' :  0 '
            else:
                label = v['label']
            y = self._linefeed_y(v['y'])
            if y not in this_creg_dict.keys():
                this_creg_dict[y] = {'val': 1, 'label': label}
            else:
                this_creg_dict[y]['val'] += 1
        for ii, y in enumerate(this_creg_dict.keys()):
            if this_creg_dict[y]['val'] > 1:
                self.ax.plot([.6, .7], [y - .1, y + .1],
                             color=self._style.cc,
                             zorder=PORDER_LINE)
                self.ax.text(0.5, y + .1, str(this_creg_dict[y]['val']), ha='left', va='bottom',
                             fontsize=0.8 * self._style.fs,
                             color=self._style.tc,
                             zorder=PORDER_TEXT)
            self.ax.text(-0.5, y, this_creg_dict[y]['label'], ha='right', va='center',
                         fontsize=self._style.fs,
                         color=self._style.tc,
                         zorder=PORDER_TEXT)
            pl_cxl[ii], = self.ax.plot([0, 0], [y, y],
                                       color=self._style.cc,
                                       zorder=PORDER_LINE)
        # lf line
        if self._cond['n_linefeeds'] > 0:
            self._linefeed_mark(0.1)

        del pl_cxl[len(this_creg_dict):]

        return pl_qxl, pl_cxl

    def _draw_ops(self, verbose=False):
        _force_next = 'measure barrier'.split()
        _wide_gate = 'u2 u3 cu2 cu3'.split()
        _barriers = []

        anchors = np.ones(len(self._qreg_dict))
        gate_occupied = {key: [] for key in self._qreg_dict}
        this_anc = 0
        yqs = []
        ycs = []
        next_ops = self._ops.copy()
        next_ops.pop(0)

        for i, (op, op_next) in enumerate(zip_longest(self._ops, next_ops)):
            # gate position before folding
            if 'qubits' in op.keys():
                anc = [anchors[v] for v in op['qubits']]
            else:
                continue
            if _barriers:
                this_anc = max(this_anc, max(anc))
            else:
                this_anc = max(anc)
            # gate width
            if op['name'] in _wide_gate:
                dx = 2
                x_shift = 0.5
                _iswide = True
            else:
                dx = 1
                x_shift = 0
                _iswide = False
            # gate alignment
            is_locatable = False
            while not is_locatable:
                if op['name'] in _force_next:
                    gidxs = self._qreg_dict.keys()
                else:
                    gidxs = op['qubits']
                for gidx in gidxs:
                    footprint = {this_anc + i for i in range(dx)}
                    if any(n in footprint for n in gate_occupied[gidx]):
                        is_locatable = False
                        this_anc += 1
                        break
                    else:
                        is_locatable = True
            # folding
            if 0 < self._style.fold + 1 < self._linefeed_x(this_anc) + dx:
                deltapos = self._style.fold + 1 - self._linefeed_x(this_anc)
                self._linefeed_mark(self._style.fold + 1 - 0.1)
                self._cond['n_linefeeds'] += 1
                pl_qlx, pl_clx = self._draw_regs_sub()
                self._pl_qxl += pl_qlx
                self._pl_cxl += pl_clx
                this_anc += deltapos
            # gate position after folding
            if 'qubits' in op.keys():
                yqs = [self._linefeed_y(self._qreg_dict[v]['y'], this_anc) for v in op['qubits']]
            if 'clbits' in op.keys():
                ycs = [self._linefeed_y(self._creg_dict[v]['y'], this_anc) for v in op['clbits']]
            # new horizontal position
            xpos = self._linefeed_x(this_anc) + x_shift

            if verbose:
                print(i, op)

            if 'texparams' in op.keys():
                param = self.param_parse(op['texparams'], self._style.pimode)
            else:
                param = None

            if 'conditional' in op.keys():
                yc_cnds = [self._linefeed_y(self._creg_dict[v]['y'], this_anc)
                           for v in self._creg_dict]
                if self._style.bundle:
                    yc_cnds = list(set(yc_cnds))
                    for y in yc_cnds:
                        self._conds(xpos, y, istrue=True)
                else:
                    fmt = '{:0%sb}' % len(yc_cnds)
                    vlist = list(fmt.format(int(op['conditional']['val'], 16)))[::-1]
                    for y, v in zip(yc_cnds, vlist):
                        if v == '0':
                            bv = False
                        else:
                            bv = True
                        self._conds(xpos, y, istrue=bv)
                self._subtext(xpos, min(yc_cnds), op['conditional']['val'])
                self.ax.plot([xpos, xpos], [min(yqs), min(yc_cnds)],
                             color=self._style.lc,
                             linewidth=1.0,
                             zorder=PORDER_LINE)

            if op['name'] == 'measure':
                vv = self._creg_dict[op['clbits'][0]]['index']
                self._measure(xpos, yqs[0], ycs[0], vv)
            elif op['name'] == 'barrier':
                if self._style.barrier:
                    if op_next and op_next['name'] == 'barrier':
                        _barriers.append(yqs[0])
                        continue
                    else:
                        _barriers.append(yqs[0])
                        self._barrier(xpos, _barriers)
                        _barriers = []
                else:
                    continue
            elif len(yqs) == 1:
                disp = op['name']
                if param:
                    self._gate(xpos, yqs[0], wide=_iswide, text=disp, subtext='{}'.format(param))
                else:
                    self._gate(xpos, yqs[0], wide=_iswide, text=disp)
            elif len(yqs) == 2:
                if op['name'] in ['cx']:
                    self._ctrl_qubit(xpos, yqs[0])
                    self._tgt_qubit(xpos, yqs[1])
                    if yqs[0] > yqs[1]:
                        self.ax.plot([xpos, xpos], [yqs[0], yqs[1] - 0.3 * WID],
                                     color=self._style.lc,
                                     linewidth=1.0,
                                     zorder=PORDER_LINE)
                    else:
                        self.ax.plot([xpos, xpos], [yqs[0], yqs[1] + 0.3 * WID],
                                     color=self._style.lc,
                                     linewidth=1.0,
                                     zorder=PORDER_LINE)
                elif op['name'] == 'cz':
                    if self._style.latexmode:
                        self._ctrl_qubit(xpos, yqs[0])
                        self._ctrl_qubit(xpos, yqs[1])
                    else:
                        disp = op['name'].replace('c', '')
                        self._ctrl_qubit(xpos, yqs[0])
                        self._gate(xpos, yqs[1], wide=_iswide, text=disp)
                    self.ax.plot([xpos, xpos], [yqs[0], yqs[1]],
                                 color=self._style.lc,
                                 linewidth=1.0,
                                 zorder=PORDER_LINE)
                elif op['name'] in ['cy', 'ch', 'cu3', 'crz']:
                    disp = op['name'].replace('c', '')
                    self._ctrl_qubit(xpos, yqs[0])
                    if param:
                        self._gate(xpos, yqs[1], wide=_iswide, text=disp,
                                   subtext='{}'.format(param))
                    else:
                        self._gate(xpos, yqs[1], wide=_iswide, text=disp)
                    self.ax.plot([xpos, xpos], [yqs[0], yqs[1]],
                                 color=self._style.lc,
                                 linewidth=1.0,
                                 zorder=PORDER_LINE)
                elif op['name'] in ['cu1']:
                    disp = op['name'].replace('c', '')
                    self._ctrl_qubit(xpos, yqs[0])
                    if self._style.latexmode:
                        self._ctrl_qubit(xpos, yqs[1])
                        self._subtext(xpos, min(yqs), param)
                    else:
                        self._gate(xpos, yqs[1], wide=_iswide, text=disp,
                                   subtext='{}'.format(param))
                    self.ax.plot([xpos, xpos], [yqs[0], yqs[1]],
                                 color=self._style.lc,
                                 linewidth=1.0,
                                 zorder=PORDER_LINE)
                elif op['name'] == 'swap':
                    self._swap(xpos, yqs[0])
                    self._swap(xpos, yqs[1])
                    self.ax.plot([xpos, xpos], [yqs[0], yqs[1]],
                                 color=self._style.lc,
                                 linewidth=1.0,
                                 zorder=PORDER_LINE)
            elif len(yqs) == 3:
                if op['name'] == 'cswap':
                    self._ctrl_qubit(xpos, yqs[0])
                    self._swap(xpos, yqs[1])
                    self._swap(xpos, yqs[2])
                    self.ax.plot([xpos, xpos], [min(yqs), max(yqs)],
                                 color=self._style.lc,
                                 linewidth=1.0,
                                 zorder=PORDER_LINE)
                elif op['name'] == 'ccx':
                    self._ctrl_qubit(xpos, yqs[0])
                    self._ctrl_qubit(xpos, yqs[1])
                    self._tgt_qubit(xpos, yqs[2])
                    if yqs.index(max(yqs)) == 2:
                        self.ax.plot([xpos, xpos], [min(yqs), yqs[2] + 0.3 * WID],
                                     color=self._style.lc,
                                     linewidth=1.0,
                                     zorder=PORDER_LINE)
                    elif yqs.index(min(yqs)) == 2:
                        self.ax.plot([xpos, xpos], [max(yqs), yqs[2] - 0.3 * WID],
                                     color=self._style.lc,
                                     linewidth=1.0,
                                     zorder=PORDER_LINE)
                    else:
                        self.ax.plot([xpos, xpos], [min(yqs), max(yqs)],
                                     color=self._style.lc,
                                     linewidth=1.0,
                                     zorder=PORDER_LINE)
            else:
                logger.critical('Invalid gate %s', op)
                raise QISKitError('invalid gate {}'.format(op))

            # gate alignment
            if self._style.compress:
                for v in op['qubits']:
                    anchors[v] = this_anc + dx
                if op['name'] in _force_next or 'conditional' in op.keys():
                    for key in self._qreg_dict:
                        gate_occupied[key].extend([this_anc + ii for ii in range(dx)])
                else:
                    for key, qdc in self._qreg_dict.items():
                        if min(yqs) <= self._linefeed_y(qdc['y'], this_anc) <= max(yqs):
                            gate_occupied[key].extend([this_anc + ii for ii in range(dx)])
            else:
                for v in range(anchors.shape[0]):
                    anchors[v] = this_anc + dx

        if self._cond['n_linefeeds'] > 0:
            self._cond['xmax'] = self._style.fold + 1
        else:
            self._cond['xmax'] = max(anchors)
        self._cond['ymax'] = self._linefeed_y(-self._cond['n_lines'] + 1)

        for ii in range(len(self._pl_qxl)):
            self._pl_qxl[ii].set_xdata([0, self._cond['xmax']])
        for ii in range(len(self._pl_cxl)):
            self._pl_cxl[ii].set_xdata([0, self._cond['xmax']])

        # gate number
        if self._style.index:
            for ii in range(int(max(anchors)) - 1):
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

    def _linefeed_x(self, x):
        if self._style.fold > 0:
            _lf = min(x // self._style.fold, self._cond['n_linefeeds'])
        else:
            _lf = 0
        return x - _lf * self._style.fold

    def _linefeed_y(self, y, indx=None):
        if indx:
            if self._style.fold > 0:
                _lf = min(indx // self._style.fold, self._cond['n_linefeeds'])
            else:
                _lf = 0
        else:
            _lf = self._cond['n_linefeeds']
        return y - _lf * (self._cond['n_lines'] + 1.0)

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
