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
        colors = {
            # Default Colors
            'basis': '#FA74A6',         # Red
            'clifford': '#6FA4FF',      # Light Blue
            'pauli': '#05BAB6',         # Green
            'def_other': '#BB8BFF',     # Purple
            # IQX Colors
            'classical': '#002D9C',     # Dark Blue
            'phase': '#33B1FF',         # Cyan
            'hadamard': '#FA4D56',      # Light Red
            'non_unitary': '#A8A8A8',   # Medium Gray
            'iqx_other': '#9F1853',     # Dark Red
            # B/W
            'black': '#000000',
            'white': '#FFFFFF',
            'light_gray': '#778899',
            'dark_gray': '#BDBDBD'
        }
        self.style = {

            'name': 'default',
            'tc': colors['black'],          # Non-gate Text Color
            'gt': colors['black'],          # Gate Text Color
            'sc': colors['black'],          # Gate Subtext Color
            'lc': colors['black'],          # Line Color
            'cc': colors['light_gray'],     # creg Line Color
            'gc': colors['def_other'],      # Default Gate Color
            'bc': colors['dark_gray'],      # Barrier Color
            'bg': colors['white'],          # Background Color
            'ec': None,                     # Edge Color (B/W only)
            'fs': 13,                       # Gate Font Size
            'sfs': 8,                       # Subtext Font Size
            'latexmode': False,
            'index': False,
            'figwidth': -1,
            'dpi': 150,
            'margin': [2.0, 0.1, 0.1, 0.3],
            'cline': 'doublet',

            'disptex': {
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
            },
            'dispcol': {
                'u1': (colors['basis'], colors['black']),
                'u2': (colors['basis'], colors['black']),
                'u3': (colors['basis'], colors['black']),
                'id': (colors['pauli'], colors['black']),
                'x': (colors['pauli'], colors['black']),
                'y': (colors['pauli'], colors['black']),
                'z': (colors['pauli'], colors['black']),
                'h': (colors['clifford'], colors['black']),
                'cx': (colors['clifford'], colors['black']),
                'cy': (colors['clifford'], colors['black']),
                'cz': (colors['clifford'], colors['black']),
                'swap': (colors['clifford'], colors['black']),
                's': (colors['clifford'], colors['black']),
                'sdg': (colors['clifford'], colors['black']),
                'dcx': (colors['clifford'], colors['black']),
                'iswap': (colors['clifford'], colors['black']),
                't': (colors['def_other'], colors['black']),
                'tdg': (colors['def_other'], colors['black']),
                'r': (colors['def_other'], colors['black']),
                'rx': (colors['def_other'], colors['black']),
                'ry': (colors['def_other'], colors['black']),
                'rz': (colors['def_other'], colors['black']),
                'rxx': (colors['def_other'], colors['black']),
                'ryy': (colors['def_other'], colors['black']),
                'rzx': (colors['def_other'], colors['black']),
                'reset': (colors['black'], colors['white']),
                'target': (colors['white'], colors['white']),
                'measure': (colors['black'], colors['white']),
                'ccx': (colors['def_other'], colors['black']),
                'cdcx': (colors['def_other'], colors['black']),
                'ccdcx': (colors['def_other'], colors['black']),
                'cswap': (colors['def_other'], colors['black']),
                'ccswap': (colors['def_other'], colors['black']),
                'mcx': (colors['def_other'], colors['black']),
                'mcx_gray': (colors['def_other'], colors['black']),
                'u': (colors['def_other'], colors['black']),
                'p': (colors['def_other'], colors['black']),
                'sx': (colors['def_other'], colors['black']),
                'sxdg': (colors['def_other'], colors['black'])
            }
        }

def set_style(base_style, update_style):
    upd_dict = copy(update_style)
    base_style['name'] = upd_dict.pop('name', base_style['name'])
    base_style['tc'] = upd_dict.pop('textcolor', base_style['tc'])
    base_style['sc'] = upd_dict.pop('subtextcolor', base_style['sc'])
    base_style['lc'] = upd_dict.pop('linecolor', base_style['lc'])
    base_style['cc'] = upd_dict.pop('creglinecolor', base_style['cc'])
    base_style['gt'] = upd_dict.pop('gatetextcolor', base_style['gt'])
    base_style['gc'] = upd_dict.pop('gatefacecolor', base_style['gc'])
    base_style['bc'] = upd_dict.pop('barrierfacecolor', base_style['bc'])
    base_style['bg'] = upd_dict.pop('backgroundcolor', base_style['bg'])
    base_style['fs'] = upd_dict.pop('fontsize', base_style['fs'])
    base_style['sfs'] = upd_dict.pop('subfontsize', base_style['sfs'])
    base_style['disptex'] = upd_dict.pop('displaytext', base_style['disptex'])
    dcol = upd_dict.pop('displaycolor', base_style['dispcol'])
    for col in dcol.keys():
        if col in base_style['dispcol'].keys():
            base_style['dispcol'][col] = dcol[col]
    base_style['latexmode'] = upd_dict.pop('latexdrawerstyle', base_style['latexmode'])
    base_style['index'] = upd_dict.pop('showindex', base_style['index'])
    base_style['figwidth'] = upd_dict.pop('figwidth', base_style['figwidth'])
    base_style['dpi'] = upd_dict.pop('dpi', base_style['dpi'])
    base_style['margin'] = upd_dict.pop('margin', base_style['margin'])
    base_style['cline'] = upd_dict.pop('creglinestyle', base_style['cline'])

    if upd_dict:
        warn('style option/s ({}) is/are not supported'.format(', '.join(upd_dict.keys())),
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
