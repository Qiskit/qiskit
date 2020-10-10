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

"""mpl circuit visualization style."""

from warnings import warn


class DefaultStyle:
    """IBM Design Style colors
    """
    def __init__(self):
        """Creates a Default Style dictionary
        """
        colors = {
            '### Default Colors': 'Default Colors',
            'basis': '#FA74A6',         # Red
            'clifford': '#6FA4FF',      # Light Blue
            'pauli': '#05BAB6',         # Green
            'def_other': '#BB8BFF',     # Purple
            '### IQX Colors': 'IQX Colors',
            'classical': '#002D9C',     # Dark Blue
            'phase': '#33B1FF',         # Cyan
            'hadamard': '#FA4D56',      # Light Red
            'non_unitary': '#A8A8A8',   # Medium Gray
            'iqx_other': '#9F1853',     # Dark Red
            '### B/W': 'B/W',
            'black': '#000000',
            'white': '#FFFFFF',
            'dark_gray': '#778899',
            'light_gray': '#BDBDBD'
        }
        self.style = {
            'name': 'default',
            'tc': colors['black'],          # Non-gate Text Color
            'gt': colors['black'],          # Gate Text Color
            'sc': colors['black'],          # Gate Subtext Color
            'lc': colors['black'],          # Line Color
            'cc': colors['dark_gray'],      # creg Line Color
            'gc': colors['def_other'],      # Default Gate Color
            'bc': colors['light_gray'],     # Barrier Color
            'bg': colors['white'],          # Background Color
            'ec': None,                     # Edge Color (B/W only)
            'fs': 13,                       # Gate Font Size
            'sfs': 8,                       # Subtext Font Size
            'index': False,
            'figwidth': -1,
            'dpi': 150,
            'margin': [2.0, 0.1, 0.1, 0.3],
            'cline': 'doublet',

            'disptex': {
                'u1': '$\\mathrm{U}_1$',
                'u2': '$\\mathrm{U}_2$',
                'u3': '$\\mathrm{U}_3$',
                'u': 'U',
                'p': 'P',
                'id': 'I',
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
                'dcx': 'Dcx',
                'iswap': 'Iswap',
                'ms': 'MS',
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
                'u': (colors['def_other'], colors['black']),
                'p': (colors['def_other'], colors['black']),
                'id': (colors['pauli'], colors['black']),
                'x': (colors['pauli'], colors['black']),
                'y': (colors['pauli'], colors['black']),
                'z': (colors['pauli'], colors['black']),
                'h': (colors['clifford'], colors['black']),
                'cx': (colors['clifford'], colors['black']),
                'ccx': (colors['def_other'], colors['black']),
                'mcx': (colors['def_other'], colors['black']),
                'mcx_gray': (colors['def_other'], colors['black']),
                'cy': (colors['clifford'], colors['black']),
                'cz': (colors['clifford'], colors['black']),
                'swap': (colors['clifford'], colors['black']),
                'cswap': (colors['def_other'], colors['black']),
                'ccswap': (colors['def_other'], colors['black']),
                'dcx': (colors['clifford'], colors['black']),
                'cdcx': (colors['def_other'], colors['black']),
                'ccdcx': (colors['def_other'], colors['black']),
                'iswap': (colors['clifford'], colors['black']),
                's': (colors['clifford'], colors['black']),
                'sdg': (colors['clifford'], colors['black']),
                't': (colors['def_other'], colors['black']),
                'tdg': (colors['def_other'], colors['black']),
                'sx': (colors['def_other'], colors['black']),
                'sxdg': (colors['def_other'], colors['black']),
                'r': (colors['def_other'], colors['black']),
                'rx': (colors['def_other'], colors['black']),
                'ry': (colors['def_other'], colors['black']),
                'rz': (colors['def_other'], colors['black']),
                'rxx': (colors['def_other'], colors['black']),
                'ryy': (colors['def_other'], colors['black']),
                'rzx': (colors['def_other'], colors['black']),
                'reset': (colors['black'], colors['white']),
                'target': (colors['white'], colors['white']),
                'measure': (colors['black'], colors['white'])
            }
        }


def set_style(def_style, json_style):
    """Utility function to take elements in json_style and
    write them into def_style.
    """
    def_style['name'] = json_style.pop('name', def_style['name'])
    def_style['tc'] = json_style.pop('textcolor', def_style['tc'])
    def_style['gt'] = json_style.pop('gatetextcolor', def_style['gt'])
    def_style['sc'] = json_style.pop('subtextcolor', def_style['sc'])
    def_style['lc'] = json_style.pop('linecolor', def_style['lc'])
    def_style['cc'] = json_style.pop('creglinecolor', def_style['cc'])
    def_style['gc'] = json_style.pop('gatefacecolor', def_style['gc'])
    def_style['bc'] = json_style.pop('barrierfacecolor', def_style['bc'])
    def_style['bg'] = json_style.pop('backgroundcolor', def_style['bg'])
    def_style['ec'] = json_style.pop('edgecolor', def_style['ec'])
    def_style['fs'] = json_style.pop('fontsize', def_style['fs'])
    def_style['sfs'] = json_style.pop('subfontsize', def_style['sfs'])
    def_style['index'] = json_style.pop('showindex', def_style['index'])
    def_style['figwidth'] = json_style.pop('figwidth', def_style['figwidth'])
    def_style['dpi'] = json_style.pop('dpi', def_style['dpi'])
    def_style['margin'] = json_style.pop('margin', def_style['margin'])
    def_style['cline'] = json_style.pop('creglinestyle', def_style['cline'])
    dtex = json_style.pop('displaytext', def_style['disptex'])
    for tex in dtex.keys():
        if tex in def_style['disptex'].keys():
            def_style['disptex'][tex] = dtex[tex]
    dcol = json_style.pop('displaycolor', def_style['dispcol'])
    for col in dcol.keys():
        if col in def_style['dispcol'].keys():
            def_style['dispcol'][col] = dcol[col]

    if json_style:
        warn('style option/s ({}) is/are not supported'.format(', '.join(json_style.keys())),
             DeprecationWarning, 2)

    return def_style
