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
        basis_color = '#FA74A6'               # Red
        clifford_color = '#6FA4FF'            # Blue
        non_gate_color = '#000000'            # Black
        other_color = '#BB8BFF'               # Purple
        pauli_color = '#05BAB6'               # Green
        iden_color = '#05BAB6'                # Green

        black_font = '#000000'                # Black font color
        white_font = '#ffffff'                # White font color

        self.name = 'default'
        self.tc = '#000000'
        self.sc = '#000000'
        self.lc = '#000000'
        self.not_gate_lc = '#ffffff'
        self.cc = '#778899'         # Medium Gray
        self.gc = other_color
        self.gt = '#000000'
        self.bc = '#bdbdbd'         # Dark Gray
        self.bg = '#ffffff'
        self.edge_color = None
        self.math_fs = 15
        self.fs = 13
        self.sfs = 8
        self.disptex = {
            'id': 'I',
            'p': 'P',
            'u': 'U',
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
            'u1': (basis_color, black_font),
            'u2': (basis_color, black_font),
            'u3': (basis_color, black_font),
            'id': (iden_color, black_font),
            'x': (pauli_color, black_font),
            'y': (pauli_color, black_font),
            'z': (pauli_color, black_font),
            'h': (clifford_color, black_font),
            'cx': (clifford_color, black_font),
            'cy': (clifford_color, black_font),
            'cz': (clifford_color, black_font),
            'swap': (clifford_color, black_font),
            's': (clifford_color, black_font),
            'sdg': (clifford_color, black_font),
            'dcx': (clifford_color, black_font),
            'iswap': (clifford_color, black_font),
            't': (other_color, black_font),
            'tdg': (other_color, black_font),
            'r': (other_color, black_font),
            'rx': (other_color, black_font),
            'ry': (other_color, black_font),
            'rz': (other_color, black_font),
            'rxx': (other_color, black_font),
            'ryy': (other_color, black_font),
            'rzx': (other_color, black_font),
            'reset': (non_gate_color, white_font),
            'target': ('#ffffff', white_font),
            'measure': (non_gate_color, white_font),
            'ccx': (other_color, black_font),
            'cdcx': (other_color, black_font),
            'ccdcx': (other_color, black_font),
            'cswap': (other_color, black_font),
            'ccswap': (other_color, black_font),
            'mcx': (other_color, black_font),
            'mcx_gray': (other_color, black_font),
            'u': (other_color, black_font),
            'p': (other_color, black_font),
            'sx': (other_color, black_font),
            'sxdg': (other_color, black_font)
        }
        self.latexmode = False
        self.index = False
        self.figwidth = -1
        self.dpi = 150
        self.margin = [2.0, 0.1, 0.1, 0.3]
        self.cline = 'doublet'

    def set_style(self, style_dic):
        dic = copy(style_dic)
        self.name = dic.pop('name', self.name)
        self.tc = dic.pop('textcolor', self.tc)
        self.sc = dic.pop('subtextcolor', self.sc)
        self.lc = dic.pop('linecolor', self.lc)
        self.cc = dic.pop('creglinecolor', self.cc)
        self.gt = dic.pop('gatetextcolor', self.gt)
        self.gc = dic.pop('gatefacecolor', self.gc)
        self.bc = dic.pop('barrierfacecolor', self.bc)
        self.bg = dic.pop('backgroundcolor', self.bg)
        self.fs = dic.pop('fontsize', self.fs)
        self.sfs = dic.pop('subfontsize', self.sfs)
        self.disptex = dic.pop('displaytext', self.disptex)
        dcol = dic.pop('displaycolor', self.dispcol)
        for col in dcol.keys():
            if col in self.dispcol.keys():
                self.dispcol[col] = dcol[col]
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
        face_gate_color = '#ffffff'             # White face color

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
            'p': 'P',
            'u': 'U',
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
            'u1': (face_gate_color, '#000000'),
            'u2': (face_gate_color, '#000000'),
            'u3': (face_gate_color, '#000000'),
            'id': (face_gate_color, '#000000'),
            'x': (face_gate_color, '#000000'),
            'y': (face_gate_color, '#000000'),
            'z': (face_gate_color, '#000000'),
            'h': (face_gate_color, '#000000'),
            'cx': (face_gate_color, '#000000'),
            'cy': (face_gate_color, '#000000'),
            'cz': (face_gate_color, '#000000'),
            'swap': (face_gate_color, '#000000'),
            's': (face_gate_color, '#000000'),
            'sdg': (face_gate_color, '#000000'),
            'dcx': (face_gate_color, '#000000'),
            'iswap': (face_gate_color, '#000000'),
            't': (face_gate_color, '#000000'),
            'tdg': (face_gate_color, '#000000'),
            'r': (face_gate_color, '#000000'),
            'rx': (face_gate_color, '#000000'),
            'ry': (face_gate_color, '#000000'),
            'rz': (face_gate_color, '#000000'),
            'rxx': (face_gate_color, '#000000'),
            'ryy': (face_gate_color, '#000000'),
            'rzx': (face_gate_color, '#000000'),
            'reset': (face_gate_color, '#000000'),
            'target': (face_gate_color, '#000000'),
            'measure': (face_gate_color, '#000000'),
            'ccx': (face_gate_color, '#000000'),
            'cdcx': (face_gate_color, '#000000'),
            'ccdcx': (face_gate_color, '#000000'),
            'cswap': (face_gate_color, '#000000'),
            'ccswap': (face_gate_color, '#000000'),
            'mcx': (face_gate_color, '#000000'),
            'mcx_gray': (face_gate_color, '#000000'),
            'u': (face_gate_color, '#000000'),
            'p': (face_gate_color, '#000000'),
            'sx': (face_gate_color, '#000000'),
            'sxdg': (face_gate_color, '#000000')
        }
        self.latexmode = False
        self.index = False
        self.figwidth = -1
        self.dpi = 150
        self.margin = [2.0, 0.1, 0.1, 0.3]
        self.cline = 'doublet'

    def set_style(self, style_dic):
        dic = copy(style_dic)
        self.name = dic.pop('name', self.name)
        self.tc = dic.pop('textcolor', self.tc)
        self.sc = dic.pop('subtextcolor', self.sc)
        self.lc = dic.pop('linecolor', self.lc)
        self.cc = dic.pop('creglinecolor', self.cc)
        self.gt = dic.pop('gatetextcolor', self.gt)
        self.gc = dic.pop('gatefacecolor', self.gc)
        self.bc = dic.pop('barrierfacecolor', self.bc)
        self.bg = dic.pop('backgroundcolor', self.bg)
        self.fs = dic.pop('fontsize', self.fs)
        self.sfs = dic.pop('subfontsize', self.sfs)
        self.disptex = dic.pop('displaytext', self.disptex)
        dcol = dic.pop('displaycolor', self.dispcol)
        for col in dcol.keys():
            if col in self.dispcol.keys():
                self.dispcol[col] = dcol[col]
        self.latexmode = dic.pop('latexdrawerstyle', self.latexmode)
        self.index = dic.pop('showindex', self.index)
        self.figwidth = dic.pop('figwidth', self.figwidth)
        self.dpi = dic.pop('dpi', self.dpi)
        self.margin = dic.pop('margin', self.margin)
        self.cline = dic.pop('creglinestyle', self.cline)

        if dic:
            warn('style option/s ({}) is/are not supported'.format(', '.join(dic.keys())),
                 DeprecationWarning, 2)


class IQXStyle:
    def __init__(self):
        # Set colors
        classical_gate_color = '#002D9C'        # Dark Blue
        phase_gate_color = '#33B1FF'            # Cyan
        hadamard_color = '#FA4D56'              # Red
        other_quantum_gate = '#9F1853'          # Dark Red
        non_unitary_gate = '#A8A8A8'            # Grey

        black_font = '#000000'                  # Black font color
        white_font = '#ffffff'                  # White font color

        self.name = 'iqx'
        self.tc = '#000000'
        self.sc = '#ffffff'
        self.lc = '#000000'
        self.not_gate_lc = '#ffffff'
        self.cc = '#778899'                     # Medium Gray
        self.gc = other_quantum_gate
        self.gt = '#ffffff'
        self.bc = non_unitary_gate              # Dark Gray
        self.bg = '#ffffff'
        self.edge_color = None
        self.math_fs = 15
        self.fs = 13
        self.sfs = 8
        self.disptex = {
            'id': 'I',
            'p': 'P',
            'u': 'U',
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
            'u1': (phase_gate_color, black_font),
            'u2': (other_quantum_gate, white_font),
            'u3': (other_quantum_gate, white_font),
            'id': (classical_gate_color, white_font),
            'x': (classical_gate_color, white_font),
            'y': (other_quantum_gate, white_font),
            'z': (phase_gate_color, black_font),
            'h': (hadamard_color, black_font),
            'cx': (classical_gate_color, white_font),
            'cy': (other_quantum_gate, white_font),
            'cz': (other_quantum_gate, white_font),
            'swap': (classical_gate_color, white_font),
            's': (phase_gate_color, black_font),
            'sdg': (phase_gate_color, black_font),
            'dcx': (classical_gate_color, white_font),
            'iswap': (phase_gate_color, black_font),
            't': (phase_gate_color, black_font),
            'tdg': (phase_gate_color, black_font),
            'r': (other_quantum_gate, white_font),
            'rx': (other_quantum_gate, white_font),
            'ry': (other_quantum_gate, white_font),
            'rz': (other_quantum_gate, white_font),
            'rxx': (other_quantum_gate, white_font),
            'ryy': (other_quantum_gate, white_font),
            'rzx': (other_quantum_gate, white_font),
            'reset': (non_unitary_gate, black_font),
            'target': ('#ffffff', '#ffffff'),
            'measure': (non_unitary_gate, black_font),
            'ccx': (classical_gate_color, white_font),
            'cdcx': (classical_gate_color, white_font),
            'ccdcx': (classical_gate_color, white_font),
            'cswap': (classical_gate_color, white_font),
            'ccswap': (classical_gate_color, white_font),
            'mcx': (classical_gate_color, white_font),
            'mcx_gray': (classical_gate_color, white_font),
            'u': (other_quantum_gate, white_font),
            'p': (phase_gate_color, black_font),
            'sx': (other_quantum_gate, white_font),
            'sxdg': (other_quantum_gate, white_font),
        }
        self.latexmode = False
        self.index = False
        self.figwidth = -1
        self.dpi = 150
        self.margin = [2.0, 0.1, 0.1, 0.3]
        self.cline = 'doublet'

    def set_style(self, style_dic):
        dic = copy(style_dic)
        self.name = dic.pop('name', self.name)
        self.tc = dic.pop('textcolor', self.tc)
        self.sc = dic.pop('subtextcolor', self.sc)
        self.lc = dic.pop('linecolor', self.lc)
        self.cc = dic.pop('creglinecolor', self.cc)
        self.gt = dic.pop('gatetextcolor', self.gt)
        self.gc = dic.pop('gatefacecolor', self.gc)
        self.bc = dic.pop('barrierfacecolor', self.bc)
        self.bg = dic.pop('backgroundcolor', self.bg)
        self.fs = dic.pop('fontsize', self.fs)
        self.sfs = dic.pop('subfontsize', self.sfs)
        self.disptex = dic.pop('displaytext', self.disptex)
        dcol = dic.pop('displaycolor', self.dispcol)
        for col in dcol.keys():
            if col in self.dispcol.keys():
                self.dispcol[col] = dcol[col]
        self.latexmode = dic.pop('latexdrawerstyle', self.latexmode)
        self.index = dic.pop('showindex', self.index)
        self.figwidth = dic.pop('figwidth', self.figwidth)
        self.dpi = dic.pop('dpi', self.dpi)
        self.margin = dic.pop('margin', self.margin)
        self.cline = dic.pop('creglinestyle', self.cline)

        if dic:
            warn('style option/s ({}) is/are not supported'.format(', '.join(dic.keys())),
                 DeprecationWarning, 2)
