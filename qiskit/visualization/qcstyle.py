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
        other_color = '#9D1C54'
        hadamard_color = '#f84f5a'
        classical_color = '#06329A'
        phase_color = '#3CB3FC'
        nonunit_color = '#A8A8A8'
        quantum_color = '#9D1C54'

        white_text = '#ffffff'
        other_text = '#000000'

        self.name = 'iqx'
        self.tc = '#000000'
        self.gt = {
            'h': other_text,

            'x': white_text,
            'cx': white_text,
            'ccx': white_text,
            'swap': white_text,
            'cswap': white_text,
            'iswap': white_text,

            't': other_text,
            'tdg': other_text,
            's': other_text,
            'sdg': other_text,
            'z': other_text,
            'u1': other_text,

            'meas': other_text,
            'measure': other_text,
            'reset': other_text,


            'u0': white_text,
            'u2': white_text,
            'u3': white_text,
            'id': white_text,
            'y': white_text,
            'cy': white_text,
            'cz': white_text,


            'r': white_text,
            'rx': white_text,
            'ry': white_text,
            'rz': white_text,
            'rxx': white_text,
            'ryy': white_text,
            'rzx': white_text,
            'rzz': white_text,
            'ch': white_text,
            'crx': white_text,
            'cry': white_text,
            'crz': white_text,

            'target': other_text,
            'multi': other_text,

        }
        self.sc = white_text
        self.lc = '#000000'
        self.not_gate_lc = '#ffffff'
        self.cc = '#778899'         # Medium Gray
        self.gc = other_color
        self.bc = '#bdbdbd'         # Dark Gray
        self.bg = '#ffffff'
        self.edge_color = None
        self.math_fs = 15
        self.fs = 13
        self.sfs = 8
        self.disptex = {
            'id': 'I',
            'u0': '$\\mathrm{U}_0$',
            'u1': '$\\mathrm{U}_1$',
            'u2': '$\\mathrm{U}_2$',
            'u3': '$\\mathrm{U}_3$',
            'x': 'X',
            'y': 'Y',
            'z': 'Z',
            'h': 'H',
            's': 'S',
            'sdg': '$\\mathrm{S}^\\dagger$',
            'sx': '$\\sqrt{\\mathrm{X}}$',
            'sxdg': '$\\sqrt{\\mathrm{X}}^\\dagger$',
            't': 'T',
            'tdg': '$\\mathrm{T}^\\dagger$',
            'iswap': 'Iswap',
            'dcx': 'Dcx',
            'ms': 'MS',
            'diagonal': 'Diagonal',
            'unitary': 'Unitary',
            'r': 'R',
            'rx': '$\\mathrm{R}_\\mathrm{X}$',
            'ry': '$\\mathrm{R}_\\mathrm{Y}$',
            'rz': '$\\mathrm{R}_\\mathrm{Z}$',
            'rxx': '$\\mathrm{R}_{\\mathrm{XX}}$',
            'ryy': '$\\mathrm{R}_{\\mathrm{YY}}$',
            'rzx': '$\\mathrm{R}_{\\mathrm{ZX}}$',
            'rzz': '$\\mathrm{R}_{\\mathrm{ZZ}}$',
            'reset': '$\\left|0\\right\\rangle$',
            'initialize': '$|\\psi\\rangle$'
        }
        self.dispcol = {
            'h': hadamard_color,

            'x': classical_color,
            'cx': classical_color,
            'ccx': classical_color,
            'swap': classical_color,
            'cswap': classical_color,
            'iswap': classical_color,

            't': phase_color,
            'tdg': phase_color,
            's': phase_color,
            'sdg': phase_color,
            'z': phase_color,
            'u1': phase_color,

            'meas': nonunit_color,
            'measure': nonunit_color,
            'reset': nonunit_color,


            'u0': quantum_color,
            'u2': quantum_color,
            'u3': quantum_color,
            'id': quantum_color,
            'y': quantum_color,
            'cy': quantum_color,
            'cz': quantum_color,


            'r': quantum_color,
            'rx': quantum_color,
            'ry': quantum_color,
            'rz': quantum_color,
            'rxx': quantum_color,
            'ryy': quantum_color,
            'rzx': quantum_color,
            'rzz': quantum_color,
            'ch': quantum_color,
            'crx': quantum_color,
            'cry': quantum_color,
            'crz': quantum_color,
            'cu3': quantum_color,

            'target': '#ffffff',
            'multi': other_color,

        }
        self.latexmode = False
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
        self.index = dic.pop('showindex', self.index)
        self.figwidth = dic.pop('figwidth', self.figwidth)
        self.dpi = dic.pop('dpi', self.dpi)
        self.margin = dic.pop('margin', self.margin)
        self.cline = dic.pop('creglinestyle', self.cline)

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
        self.sfs = 8
        self.disptex = {
            'id': 'I',
            'u0': '$\\mathrm{U}_0$',
            'u1': '$\\mathrm{U}_1$',
            'u2': '$\\mathrm{U}_2$',
            'u3': '$\\mathrm{U}_3$',
            'x': 'X',
            'y': 'Y',
            'z': 'Z',
            'h': 'H',
            's': 'S',
            'sdg': '$\\mathrm{S}^\\dagger$',
            't': 'T',
            'tdg': '$\\mathrm{T}^\\dagger$',
            'iswap': 'Iswap',
            'dcx': 'Dcx',
            'ms': 'MS',
            'diagonal': 'Diagonal',
            'unitary': 'Unitary',
            'r': 'R',
            'rx': '$\\mathrm{R}_\\mathrm{X}$',
            'ry': '$\\mathrm{R}_\\mathrm{Y}$',
            'rz': '$\\mathrm{R}_\\mathrm{Z}$',
            'rxx': '$\\mathrm{R}_{\\mathrm{XX}}$',
            'ryy': '$\\mathrm{R}_{\\mathrm{YY}}$',
            'rzx': '$\\mathrm{R}_{\\mathrm{ZX}}$',
            'rzz': '$\\mathrm{R}_{\\mathrm{ZZ}}$',
            'reset': '$\\left|0\\right\\rangle$',
            'initialize': '$|\\psi\\rangle$'
        }
        self.dispcol = {
            'u0': '#ffffff',
            'u1': '#ffffff',
            'u2': '#ffffff',
            'u3': '#ffffff',
            'id': '#ffffff',
            'x': '#ffffff',
            'y': '#ffffff',
            'z': '#ffffff',
            'h': '#ffffff',
            'cx': '#000000',
            'cy': '#ffffff',
            'cz': '#000000',
            'swap': '#000000',
            's': '#ffffff',
            'sdg': '#ffffff',
            'dcx': '#ffffff',
            'iswap': '#ffffff',
            't': '#ffffff',
            'tdg': '#ffffff',
            'r': '#ffffff',
            'rx': '#ffffff',
            'ry': '#ffffff',
            'rz': '#ffffff',
            'rxx': '#ffffff',
            'ryy': '#ffffff',
            'rzx': '#ffffff',
            'reset': '#ffffff',
            'target': '#ffffff',
            'multi': '#ffffff',
            'meas': '#ffffff',
            'measure': '#ffffff'
        }
        self.latexmode = False
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
        for key in self.dispcol.keys():
            self.dispcol[key] = self.gc
        self.dispcol = dic.pop('displaycolor', self.dispcol)
        self.latexmode = dic.pop('latexdrawerstyle', self.latexmode)
        self.index = dic.pop('showindex', self.index)
        self.figwidth = dic.pop('figwidth', self.figwidth)
        self.dpi = dic.pop('dpi', self.dpi)
        self.margin = dic.pop('margin', self.margin)
        self.cline = dic.pop('creglinestyle', self.cline)

        if dic:
            warn('style option/s ({}) is/are not supported'.format(', '.join(dic.keys())),
                 DeprecationWarning, 2)
