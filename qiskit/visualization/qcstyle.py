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


def set_style(current_style, new_style):
    """Utility function to take elements in new_style and
    write them into current_style.
    """
    current_style['name'] = new_style.pop('name', current_style['name'])
    current_style['tc'] = new_style.pop('textcolor', current_style['tc'])
    current_style['gt'] = new_style.pop('gatetextcolor', current_style['gt'])
    current_style['sc'] = new_style.pop('subtextcolor', current_style['sc'])
    current_style['lc'] = new_style.pop('linecolor', current_style['lc'])
    current_style['cc'] = new_style.pop('creglinecolor', current_style['cc'])
    current_style['gc'] = new_style.pop('gatefacecolor', current_style['gc'])
    current_style['bc'] = new_style.pop('barrierfacecolor', current_style['bc'])
    current_style['bg'] = new_style.pop('backgroundcolor', current_style['bg'])
    current_style['ec'] = new_style.pop('edgecolor', current_style['ec'])
    current_style['fs'] = new_style.pop('fontsize', current_style['fs'])
    current_style['sfs'] = new_style.pop('subfontsize', current_style['sfs'])
    current_style['index'] = new_style.pop('showindex', current_style['index'])
    current_style['figwidth'] = new_style.pop('figwidth', current_style['figwidth'])
    current_style['dpi'] = new_style.pop('dpi', current_style['dpi'])
    current_style['margin'] = new_style.pop('margin', current_style['margin'])
    current_style['cline'] = new_style.pop('creglinestyle', current_style['cline'])
    dtex = new_style.pop('displaytext', current_style['disptex'])
    for tex in dtex.keys():
        if tex in current_style['disptex'].keys():
            current_style['disptex'][tex] = dtex[tex]
    dcol = new_style.pop('displaycolor', current_style['dispcol'])
    for col in dcol.keys():
        if col in current_style['dispcol'].keys():
            current_style['dispcol'][col] = dcol[col]

    if new_style:
        warn('style option/s ({}) is/are not supported'.format(', '.join(new_style.keys())),
             DeprecationWarning, 2)

    return current_style
