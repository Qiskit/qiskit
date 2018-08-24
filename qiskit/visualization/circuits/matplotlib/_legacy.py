# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

from collections import namedtuple
from fractions import Fraction
from itertools import zip_longest
from math import fmod, isclose

import os
import logging
import json
import tempfile

from matplotlib import get_backend as get_matplotlib_backend
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageChops

from qiskit import QuantumCircuit, QISKitError, load_qasm_file

from ._styleconfig import default_circuit_scheme
from ._anchor import Anchor

logger = logging.getLogger(__name__)


WID = 0.65
HIG = 0.65
DEFAULT_SCALE = 4.3
PORDER_GATE = 5
PORDER_LINE = 2
PORDER_GRAY = 3
PORDER_TEXT = 6
PORDER_SUBP = 4


Register = namedtuple('Register', 'name index')


class CircuitPlot:
    """Represents the output of plotting a circtuit."""

    def __init__(self, figure, dpi):
        self._figure = figure
        self._dpi = dpi

    def save(self, filepath, format='png'):
        self._figure.savefig(filepath, format=format, dpi=self._dpi,
                             bbox_inches='tight')

    def show(self):
        pass


class MatplotlibRenderer:
    """A circuit plotter renderer that uses matplotlib to draw a diagram of
    a quantum circuit in JSON QASM representation."""

    def __init__(self, style=None):
        self._ast = None
        self._scale = DEFAULT_SCALE * style.scale
        self._cregisters = []
        self._qregisters = []
        self._operations = []
        self._qreg_dict = {}
        self._creg_dict = {}
        self._cond = {
            'n_lines': 0,
            'xmax': 0,
            'ymax': 0,
        }
        self._style = default_circuit_scheme()
        self.figure = plt.figure()
        self._axes = self.figure.add_subplot(1, 1, 1)
        self.figure.patch.set_facecolor(color=self._style.background_color)
        self._axes.axis('off')
        self._axes.set_aspect('equal', 'datalim')

    def render(self, json_circuit, filename=None, verbose=False):
        """Render a JSON QASM circuit into an in-memory representation."""

        self._ast = json_circuit
        self._build_registers()
        self._draw_registers(verbose)
        self._build_operations()
        self._draw_operations(verbose)
        self._center_diagram()
        self._resize()

        return CircuitPlot(self.figure, self._style.dpi)

    def _build_registers(self):
        self._cregisters = []
        self._qregisters = []

        header = self._ast['header']
        for register_name, register_width in header['clbit_labels']:
            for index in range(register_width):
                self._cregisters.append(
                    Register(name=register_name, index=index))
        assert len(self._cregisters) == header['number_of_clbits']

        # NOTE: formats of classical and quantic register are different!
        for register_name, register_qubit_index in header['qubit_labels']:
            self._qregisters.append(
                Register(name=register_name, index=register_qubit_index))
        assert len(self._qregisters) == header['number_of_qubits']

    def _build_operations(self):
        self._operations = self._ast['operations']

    def _gate(self, xy, fc=None, wide=False, text=None, subtext=None):
        xpos, ypos = xy

        if wide:
            wid = WID * 2.8
        else:
            wid = WID
        if fc:
            _fc = fc
        elif text:
            _fc = self._style.gate_colors[text]
        else:
            _fc = self._style.gate_face_color

        box = patches.Rectangle(xy=(xpos - 0.5 * wid, ypos - 0.5 * HIG), width=wid, height=HIG,
                                fc=_fc, ec=self._style.line_color, linewidth=1.5, zorder=PORDER_GATE)
        self._axes.add_patch(box)

        if text:
            disp_text = "${}$".format(self._style.gate_labels[text])
            if subtext:
                self._axes.text(xpos, ypos + 0.15 * HIG, disp_text, ha='center', va='center',
                             fontsize=self._style.font_size,
                             color=self._style.gate_text_color,
                             zorder=PORDER_TEXT)
                self._axes.text(xpos, ypos - 0.3 * HIG, subtext, ha='center', va='center',
                             fontsize=self._style.subfont_size,
                             color=self._style.subtext_color,
                             zorder=PORDER_TEXT)
            else:
                self._axes.text(xpos, ypos, disp_text, ha='center', va='center',
                             fontsize=self._style.font_size,
                             color=self._style.gate_text_color,
                             zorder=PORDER_TEXT)

    def _subtext(self, xy, text):
        xpos, ypos = xy

        self._axes.text(xpos, ypos - 0.3 * HIG, text, ha='center', va='top',
                     fontsize=self._style.subfont_size,
                     color=self._style.text_color,
                     zorder=PORDER_TEXT)

    def _line(self, xy0, xy1):
        x0, y0 = xy0
        x1, y1 = xy1
        self._axes.plot([x0, x1], [y0, y1],
                     color=self._style.line_color,
                     linewidth=1.0,
                     zorder=PORDER_LINE)

    def _measure(self, qxy, cxy, cid):
        qx, qy = qxy
        cx, cy = cxy

        self._gate(qxy, fc=self._style.gate_colors['meas'])
        # add measure symbol
        arc = patches.Arc(xy=(qx, qy - 0.15 * HIG), width=WID * 0.7, height=HIG * 0.7,
                          theta1=0, theta2=180, fill=False,
                          ec=self._style.line_color, linewidth=1.5, zorder=PORDER_GATE)
        self._axes.add_patch(arc)
        self._axes.plot([qx, qx + 0.35 * WID], [qy - 0.15 * HIG, qy + 0.20 * HIG],
                     color=self._style.line_color, linewidth=1.5, zorder=PORDER_GATE)
        # arrow
        self._axes.arrow(x=qx, y=qy, dx=0, dy=cy - qy, width=0.01, head_width=0.2, head_length=0.2,
                      length_includes_head=True, color=self._style.creg_line_color, zorder=PORDER_LINE)
        # target
        if self._style.bundle:
            self._axes.text(cx + .25, cy + .1, str(cid), ha='left', va='bottom',
                         fontsize=0.8 * self._style.font_size,
                         color=self._style.text_color,
                         zorder=PORDER_TEXT)

    def _conds(self, xy, istrue=False):
        xpos, ypos = xy

        if istrue:
            _fc = self._style.line_color
        else:
            _fc = self._style.gate_face_color

        box = patches.Circle(xy=(xpos, ypos), radius=WID * 0.15,
                             fc=_fc, ec=self._style.line_color,
                             linewidth=1.5, zorder=PORDER_GATE)
        self._axes.add_patch(box)

    def _ctrl_qubit(self, xy):
        xpos, ypos = xy

        box = patches.Circle(xy=(xpos, ypos), radius=WID * 0.15,
                             fc=self._style.line_color, ec=self._style.line_color,
                             linewidth=1.5, zorder=PORDER_GATE)
        self._axes.add_patch(box)

    def _tgt_qubit(self, xy):
        xpos, ypos = xy

        box = patches.Circle(xy=(xpos, ypos), radius=HIG * 0.35,
                             fc=self._style.gate_colors['target'], ec=self._style.line_color,
                             linewidth=1.5, zorder=PORDER_GATE)
        self._axes.add_patch(box)
        # add '+' symbol
        self._axes.plot([xpos, xpos], [ypos - 0.35 * HIG, ypos + 0.35 * HIG],
                     color=self._style.line_color, linewidth=1.0, zorder=PORDER_GATE)
        self._axes.plot([xpos - 0.35 * HIG, xpos + 0.35 * HIG], [ypos, ypos],
                     color=self._style.line_color, linewidth=1.0, zorder=PORDER_GATE)

    def _swap(self, xy):
        xpos, ypos = xy

        self._axes.plot([xpos - 0.20 * WID, xpos + 0.20 * WID], [ypos - 0.20 * WID, ypos + 0.20 * WID],
                     color=self._style.line_color, linewidth=1.5, zorder=PORDER_LINE)
        self._axes.plot([xpos - 0.20 * WID, xpos + 0.20 * WID], [ypos + 0.20 * WID, ypos - 0.20 * WID],
                     color=self._style.line_color, linewidth=1.5, zorder=PORDER_LINE)

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
                                fc=self._style.barrier_face_color, ec=None, alpha=0.6,
                                linewidth=1.5, zorder=PORDER_GRAY)
        self._axes.add_patch(box)
        for xy in xys:
            xpos, ypos = xy
            self._axes.plot([xpos, xpos], [ypos + 0.5, ypos - 0.5],
                         linewidth=1, linestyle="dashed",
                         color=self._style.line_color,
                         zorder=PORDER_TEXT)

    def _linefeed_mark(self, xy):
        xpos, ypos = xy

        self._axes.plot([xpos - .1, xpos - .1],
                     [ypos, ypos - self._cond['n_lines'] + 1],
                     color=self._style.line_color, zorder=PORDER_LINE)
        self._axes.plot([xpos + .1, xpos + .1],
                     [ypos, ypos - self._cond['n_lines'] + 1],
                     color=self._style.line_color, zorder=PORDER_LINE)

    def _center_diagram(self):
        self._axes.set_xlim(-1.5, self._cond['xmax'] + 1.5)
        self._axes.set_ylim(self._cond['ymax'] - 1.5, 1.5)

    def _resize(self):
        width = abs(self._cond['xmax']) + 2
        height = abs(self._cond['ymax']) + 2
        aspect_ratio = height / width
        final_width = width * self._scale * self._style.font_size / 72 / WID
        final_height = final_width * aspect_ratio
        self.figure.set_size_inches(final_width, final_height)


    def _draw_registers(self, verbose=False):
        # quantum register
        for ii, reg in enumerate(self._qregisters):
            if len(self._qregisters) > 1:
                label = '${}_{{{}}}$'.format(reg.name, reg.index)
            else:
                label = '${}$'.format(reg.name)
            pos = -ii
            self._qreg_dict[ii] = {'y': pos, 'label': label, 'index': reg.index, 'group': reg.name}
            self._cond['n_lines'] += 1
        # classical register
        if self._cregisters:
            n_creg = self._cregisters.copy()
            n_creg.pop(0)
            idx = 0
            y_off = -len(self._qregisters)
            for ii, (reg, nreg) in enumerate(zip_longest(self._cregisters, n_creg)):
                pos = y_off - idx
                if self._style.bundle:
                    label = '${}$'.format(reg.name)
                    self._creg_dict[ii] = {'y': pos, 'label': label, 'index': reg.index,
                                           'group': reg.name}
                    if not (not nreg or reg.name != nreg.name):
                        continue
                else:
                    label = '${}_{{{}}}$'.format(reg.name, reg.index)
                    self._creg_dict[ii] = {'y': pos, 'label': label, 'index': reg.index,
                                           'group': reg.name}
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
            self._axes.text(-0.5, y, label, ha='right', va='center',
                         fontsize=self._style.font_size,
                         color=self._style.text_color,
                         zorder=PORDER_TEXT)
            self._axes.plot([0, self._cond['xmax']], [y, y], color=self._style.line_color, zorder=PORDER_LINE)
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
                self._axes.plot([.6, .7], [y - .1, y + .1],
                             color=self._style.creg_line_color,
                             zorder=PORDER_LINE)
                self._axes.text(0.5, y + .1, str(this_creg['val']), ha='left', va='bottom',
                             fontsize=0.8 * self._style.font_size,
                             color=self._style.text_color,
                             zorder=PORDER_TEXT)
            self._axes.text(-0.5, y, this_creg['label'], ha='right', va='center',
                         fontsize=self._style.font_size,
                         color=self._style.text_color,
                         zorder=PORDER_TEXT)
            self._axes.plot([0, self._cond['xmax']], [y, y], color=self._style.creg_line_color, zorder=PORDER_LINE)

        # lf line
        if feedline_r:
            self._linefeed_mark((self._style.fold + 1 - 0.1,
                                 - n_fold * (self._cond['n_lines'] + 1)))
        if feedline_l:
            self._linefeed_mark((0.1,
                                 - n_fold * (self._cond['n_lines'] + 1)))

    def _draw_operations(self, verbose=False):
        _force_next = 'measure barrier'.split()
        _wide_gate = 'u2 u3 cu2 cu3'.split()
        _barriers = {'coord': [], 'group': []}
        next_ops = self._operations.copy()
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
        for i, (op, op_next) in enumerate(zip_longest(self._operations, next_ops)):
            # wide gate
            if op['name'] in _wide_gate:
                _iswide = True
                gw = 2
            else:
                _iswide = False
                gw = 1
            # get qreg index
            if 'qubits' in op.keys():
                q_idxs = op['qubits']
            else:
                q_idxs = []
            # get creg index
            if 'clbits' in op.keys():
                c_idxs = op['clbits']
            else:
                c_idxs = []
            # find empty space to place gate
            if not _barriers['group']:
                this_anc = max([q_anchors[ii].get_index() for ii in q_idxs])
                while True:
                    if op['name'] in _force_next or 'conditional' in op.keys() or \
                            not self._style.compress:
                        occupied = self._qreg_dict.keys()
                    else:
                        occupied = q_idxs
                    q_list = [ii for ii in range(min(occupied), max(occupied) + 1)]
                    locs = [q_anchors[jj].is_locatable(this_anc, gw) for jj in q_list]
                    if all(locs):
                        for ii in q_list:
                            if op['name'] == 'barrier' and not self._style.barrier:
                                q_anchors[ii].set_index(this_anc - 1, gw)
                            else:
                                q_anchors[ii].set_index(this_anc, gw)
                        break
                    else:
                        this_anc += 1
            # qreg coordinate
            q_xy = [q_anchors[ii].plot_coord(this_anc, gw) for ii in q_idxs]
            # creg corrdinate
            c_xy = [c_anchors[ii].plot_coord(this_anc, gw) for ii in c_idxs]
            # bottom and top point of qreg
            qreg_b = min(q_xy, key=lambda xy: xy[1])
            qreg_t = max(q_xy, key=lambda xy: xy[1])

            if verbose:
                print(i, op)

            # rotation parameter
            if 'params' in op.keys():
                param = self.param_parse(op['params'], self._style.pimode)
            else:
                param = None
            # conditional gate
            if 'conditional' in op.keys():
                c_xy = [c_anchors[ii].plot_coord(this_anc, gw) for ii in self._creg_dict]
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
                q_group = self._qreg_dict[q_idxs[0]]['group']
                if q_group not in _barriers['group']:
                    _barriers['group'].append(q_group)
                _barriers['coord'].append(q_xy[0])
                if op_next and op_next['name'] == 'barrier':
                    continue
                else:
                    if self._style.barrier:
                        self._barrier(_barriers, this_anc)
                    _barriers['group'].clear()
                    _barriers['coord'].clear()
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
        max_anc = max([q_anchors[ii].get_index() for ii in self._qreg_dict])
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
                self._axes.text(x_coord, y_coord, str(ii + 1), ha='center', va='center',
                             fontsize=self._style.subfont_size,
                             color=self._style.text_color,
                             zorder=PORDER_TEXT)

    @staticmethod
    def param_parse(v, pimode=False):
        for i, e in enumerate(v):
            if pimode:
                v[i] = MatplotlibRenderer.format_pi(e)
            else:
                v[i] = MatplotlibRenderer.format_numeric(e)
            if v[i].startswith('-'):
                v[i] = '$-$' + v[i][1:]
        param = ', '.join(v)
        return param

    @staticmethod
    def format_pi(val):
        fracvals = MatplotlibRenderer.fraction(val)
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
            coef = MatplotlibRenderer.format_numeric(val / np.pi)
            if coef == '0':
                return '0'
            return '{}$\\pi$'.format(coef)

    @staticmethod
    def format_numeric(val, tol=1e-5):
        abs_val = abs(val)
        if isclose(abs_val, 0.0, abs_tol=1e-100):
            return '0'
        if isclose(fmod(abs_val, 1.0), 0.0, abs_tol=tol) and 0.5 < abs_val < 9999.5:
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

