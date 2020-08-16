# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
r"""
A collection of functions that decide layout of figure.

Those functions are assigned to the `layout` key of the stylesheet.
User can change the layout of the output image by writing own function.
"""


def default_color_table(gate_name: str) -> str:
    """Color table that returns color code associated with the gate name."""
    _color_code_name_table = {
        'u0': '#FA74A6',
        'u1': '#FA74A6',
        'u2': '#FA74A6',
        'u3': '#FA74A6',
        'id': '#05BAB6',
        'x': '#05BAB6',
        'y': '#05BAB6',
        'z': '#05BAB6',
        'h': '#6FA4FF',
        'cx': '#6FA4FF',
        'cy': '#6FA4FF',
        'cz': '#6FA4FF',
        'swap': '#6FA4FF',
        's': '#6FA4FF',
        'sdg': '#6FA4FF',
        'dcx': '#6FA4FF',
        'iswap': '#6FA4FF',
        't': '#BB8BFF',
        'tdg': '#BB8BFF',
        'r': '#BB8BFF',
        'rx': '#BB8BFF',
        'ry': '#BB8BFF',
        'rz': '#BB8BFF',
        'reset': '#000000',
        'measure': '#000000'
    }
    return _color_code_name_table.get(gate_name, '#BB8BFF')


def default_latex_gate_name(gate_name: str) -> str:
    """Conversion table that returns latex syntax gate name."""
    _latex_name_table = {
        'u0': r'{\rm U}_0',
        'u1': r'{\rm U}_1',
        'u2': r'{\rm U}_2',
        'u3': r'{\rm U}_3',
        'id': r'{\rm Id}',
        'x': r'{\rm X}',
        'y': r'{\rm Y}',
        'z': r'{\rm Z}',
        'h': r'{\rm H}',
        'cx': r'{\rm Cx}',
        'cy': r'{\rm Cy}',
        'cz': r'{\rm Cz}',
        'swap': r'{\rm SWAP}',
        's': r'{\rm S}',
        'sdg': r'{\rm S}^\dagger',
        'dcx': r'{\rm DCX}',
        'iswap': r'{\rm iSWAP}',
        't': r'{\rm T}',
        'tdg': r'{\rm T}^\dagger',
        'r': r'{\rm R}',
        'rx': r'{\rm R}_x',
        'ry': r'{\rm R}_y',
        'rz': r'{\rm R}_z',
        'reset': r'|0\rangle',
        'measure': r'{\rm Measure}'
    }
    return _latex_name_table.get(gate_name, r'{{\rm {name}}}'.format(name=gate_name))
