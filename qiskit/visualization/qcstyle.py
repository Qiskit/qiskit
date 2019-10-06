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

from copy import copy
from warnings import warn


class DefaultStyle:
    """IBM Design Style colors
    """

    def __init__(self):
        # Set colors
        basis_color = '#EE538B'
        clifford_color = '#30b0ff'
        non_gate_color = '#202529'
        other_color = '#8A3FFC'
        pauli_color = '#20D5D2'
        iden_color = '#20D5D2'
        rot_color = '#006161'
        dark_font = '#13171A'
        light_font = '#F2F4F8'

        self.name = 'iqx'
        self.tc = light_font
        self.sc = light_font
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
            'reset': '|0\\rangle'
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
            'rx': rot_color,
            'ry': rot_color,
            'rz': rot_color,
            'reset': non_gate_color,
            'cx_target': dark_font,
            'other_target': light_font,
            'swap': clifford_color,
            'multi': other_color,
            'meas': non_gate_color
        }
        self.fontcol = {
            'u0': light_font,
            'u1': light_font,
            'u2': light_font,
            'u3': light_font,
            'id': dark_font,
            'x': dark_font,
            'y': dark_font,
            'z': dark_font,
            'h': dark_font,
            'cx': dark_font,
            's': dark_font,
            'sdg': dark_font,
            't': light_font,
            'tdg': light_font,
            'rx': light_font,
            'ry': light_font,
            'rz': light_font,
            'reset': light_font,
            'multi': light_font,
            'meas': light_font,
            'other': light_font
        }
        self.latexmode = False
        self.fold = None  # To be removed after 0.10 is released
        self.bundle = True
        self.index = False
        self.figwidth = -1
        self.dpi = 150
        self.margin = [2.0, 0.1, 0.1, 0.3]
        self.cline = 'doublet'

    def set_style(self, style_dic):
        dic = copy(style_dic)
        self.tc = dic.pop('textcolor', self.tc)
        self.sc = dic.pop('subtextcolor', self.sc)
        self.lc = dic.pop('linecolor', self.lc)
        self.cc = dic.pop('creglinecolor', self.cc)
        self.gt = dic.pop('gatetextcolor', self.tc)
        self.gc = dic.pop('gatefacecolor', self.gc)
        self.bc = dic.pop('barrierfacecolor', self.bc)
        self.bg = dic.pop('backgroundcolor', self.bg)
        self.fs = dic.pop('fontsize', self.fs)
        self.sfs = dic.pop('subfontsize', self.sfs)
        self.disptex = dic.pop('displaytext', self.disptex)
        self.dispcol = dic.pop('displaycolor', self.dispcol)
        self.latexmode = dic.pop('latexdrawerstyle', self.latexmode)
        self.bundle = dic.pop('cregbundle', self.bundle)
        self.index = dic.pop('showindex', self.index)
        self.figwidth = dic.pop('figwidth', self.figwidth)
        self.dpi = dic.pop('dpi', self.dpi)
        self.margin = dic.pop('margin', self.margin)
        self.cline = dic.pop('creglinestyle', self.cline)
        if 'fold' in dic:
            warn('The key "fold" in the argument "style" is being replaced by the argument "fold"',
                 DeprecationWarning, 5)
            self.fold = dic.pop('fold', self.fold)
            if self.fold < 2:
                self.fold = -1

        if dic:
            warn('style option/s ({}) is/are not supported'.format(', '.join(dic.keys())),
                 DeprecationWarning, 2)


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
            'cx_target': '#ffffff',
            'other_target':'#ffffff',
            'meas': '#ffffff',
            'swap': '#000000',
            'multi': '#000000'
        }
        self.fontcol = {
            'u0': '#000000',
            'u1': '#000000',
            'u2': '#000000',
            'u3': '#000000',
            'id': '#000000',
            'x': '#000000',
            'y': '#000000',
            'z': '#000000',
            'h': '#000000',
            'cx': '#000000',
            's': '#000000',
            'sdg': '#000000',
            't': '#000000',
            'tdg': '#000000',
            'rx': '#000000',
            'ry': '#000000',
            'rz': '#000000',
            'reset': '#000000',
            'multi': '#000000',
            'meas': '#000000',
            'other': '#000000',
        }
        self.latexmode = False
        self.fold = 25
        self.bundle = True
        self.index = False
        self.figwidth = -1
        self.dpi = 150
        self.margin = [2.0, 0.0, 0.0, 0.3]
        self.cline = 'doublet'

    def set_style(self, style_dic):
        dic = copy(style_dic)
        self.tc = dic.pop('textcolor', self.tc)
        self.sc = dic.pop('subtextcolor', self.sc)
        self.lc = dic.pop('linecolor', self.lc)
        self.cc = dic.pop('creglinecolor', self.cc)
        self.gt = dic.pop('gatetextcolor', self.tc)
        self.gc = dic.pop('gatefacecolor', self.gc)
        self.bc = dic.pop('barrierfacecolor', self.bc)
        self.bg = dic.pop('backgroundcolor', self.bg)
        self.fs = dic.pop('fontsize', self.fs)
        self.sfs = dic.pop('subfontsize', self.sfs)
        self.disptex = dic.pop('displaytext', self.disptex)
        for key in self.dispcol.keys():
            self.dispcol[key] = self.gc
        self.dispcol = dic.pop('displaycolor', self.dispcol)
        self.latexmode = dic.pop('latexdrawerstyle', self.latexmode)
        self.bundle = dic.pop('cregbundle', self.bundle)
        self.index = dic.pop('showindex', self.index)
        self.figwidth = dic.pop('figwidth', self.figwidth)
        self.dpi = dic.pop('dpi', self.dpi)
        self.margin = dic.pop('margin', self.margin)
        self.cline = dic.pop('creglinestyle', self.cline)
        if 'fold' in dic:
            warn('The key "fold" in the argument "style" is being replaced by the argument "fold"',
                 DeprecationWarning, 5)
            self.fold = dic.pop('fold', self.fold)
            if self.fold < 2:
                self.fold = -1

        if dic:
            warn('style option/s ({}) is/are not supported'.format(', '.join(dic.keys())),
                 DeprecationWarning, 2)
