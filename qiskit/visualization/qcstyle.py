# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name,missing-docstring


class DefaultStyle:
    """IBM Design Style colors
    """
    def __init__(self):
        # Set colors
        basis_color = '#FA74A6'
        clifford_color = '#6FA4FF'
        non_gate_color = '#000000'
        other_color = '#BB8BFF'
        pauli_color = '#05BAB6'
        iden_color = '#05BAB6'

        self.name = 'iqx'
        self.tc = '#000000'
        self.sc = '#000000'
        self.lc = '#000000'
        self.not_gate_lc = '#ffffff'
        self.cc = '#778899'
        self.gc = '#ffffff'
        self.gt = '#000000'
        self.bc = '#bdbdbd'
        self.bg = '#ffffff'
        self.edge_color = None
        self.math_fs = 15
        self.fs = 13
        self.sfs = 8
        self.colored_add_width = 0.2
        self.disptex = {
            'id': 'Id',
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
            'u0': basis_color,
            'u1': basis_color,
            'u2': basis_color,
            'u3': basis_color,
            'id': iden_color,
            'x': pauli_color,
            'y': pauli_color,
            'z': pauli_color,
            'h': clifford_color,
            'cx': clifford_color,
            's': clifford_color,
            'sdg': clifford_color,
            't': other_color,
            'tdg': other_color,
            'rx': other_color,
            'ry': other_color,
            'rz': other_color,
            'reset': non_gate_color,
            'target': '#ffffff',
            'swap': other_color,
            'multi': other_color,
            'meas': non_gate_color
        }
        self.latexmode = False
        self.fold = 25
        self.bundle = True
        self.barrier = True
        self.index = False
        self.figwidth = -1
        self.dpi = 150
        self.margin = [2.0, 0.1, 0.1, 0.3]
        self.cline = 'doublet'

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
        self.dispcol = dic.get('displaycolor', self.dispcol)
        self.latexmode = dic.get('latexdrawerstyle', self.latexmode)
        self.fold = dic.get('fold', self.fold)
        if self.fold < 2:
            self.fold = -1
        self.bundle = dic.get('cregbundle', self.bundle)
        self.barrier = dic.get('plotbarrier', self.barrier)
        self.index = dic.get('showindex', self.index)
        self.figwidth = dic.get('figwidth', self.figwidth)
        self.dpi = dic.get('dpi', self.dpi)
        self.margin = dic.get('margin', self.margin)
        self.cline = dic.get('creglinestyle', self.cline)


class BWStyle:
    def __init__(self):
        self.name = 'bw'
        self.tc = '#000000'
        self.sc = '#000000'
        self.lc = '#000000'
        self.not_gate_lc = '#000000'
        self.cc = '#778899'
        self.gc = '#ffffff'
        self.gt = '#000000'
        self.bc = '#bdbdbd'
        self.bg = '#ffffff'
        self.edge_color = '#000000'
        self.fs = 13
        self.math_fs = 15
        self.colored_add_width = 0.2
        self.sfs = 8
        self.disptex = {
            'id': 'Id',
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
            'cx': '#000000',
            's': '#ffffff',
            'sdg': '#ffffff',
            't': '#ffffff',
            'tdg': '#ffffff',
            'rx': '#ffffff',
            'ry': '#ffffff',
            'rz': '#ffffff',
            'reset': '#ffffff',
            'target': '#ffffff',
            'meas': '#ffffff',
            'swap': '#000000',
            'multi': '#000000'
        }
        self.latexmode = False
        self.fold = 25
        self.bundle = True
        self.barrier = True
        self.index = False
        self.figwidth = -1
        self.dpi = 150
        self.margin = [2.0, 0.0, 0.0, 0.3]
        self.cline = 'doublet'

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
        self.fold = dic.get('fold', self.fold)
        if self.fold < 2:
            self.fold = -1
        self.bundle = dic.get('cregbundle', self.bundle)
        self.barrier = dic.get('plotbarrier', self.barrier)
        self.index = dic.get('showindex', self.index)
        self.figwidth = dic.get('figwidth', self.figwidth)
        self.dpi = dic.get('dpi', self.dpi)
        self.margin = dic.get('margin', self.margin)
        self.cline = dic.get('creglinestyle', self.cline)
