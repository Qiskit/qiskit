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

"""
Default stylesheet of the pulse drawer.
"""


default_style = {
    'formatter': {
        'general': {
            'fig_size': [8, 6],
            'dpi': 160},
        'color': {
            'fill_waveform_d': ['#648fff', '#002999'],
            'fill_waveform_u': ['#ffb000', '#994A00'],
            'fill_waveform_m': ['#dc267f', '#760019'],
            'fill_waveform_a': ['#dc267f', '#760019'],
            'baseline': '#000000',
            'barrier': '#222222',
            'background': 'f2f3f4',
            'annotate': '#222222',
            'frame_change': '#000000',
            'snapshot': '#000000',
            'axis_label': '#000000'},
        'alpha': {
            'fill_waveform': 1.0,
            'baseline': 1.0,
            'barrier': 0.7},
        'layer': {
            'fill_waveform': 2,
            'baseline': 1,
            'barrier': 1,
            'annotate': 4,
            'axis_label': 4,
            'frame_change': 3,
            'snapshot': 3},
        'margin': {
            'top': 0.2,
            'bottom': 0.2,
            'left': 0.05,
            'right': 0.05,
            'between_channel': 0.1},
        'label_offset': {
            'pulse_name': -0.1,
            'scale_factor': -0.1,
            'frame_change': 0.1,
            'snapshot': 0.1},
        'text_size': {
            'axis_label': 15,
            'annotate': 12,
            'frame_change': 20,
            'snapshot': 20,
            'fig_title': 15},
        'line_width': {
            'fill_waveform': 0,
            'baseline': 1,
            'barrier': 1},
        'line_style': {
            'fill_waveform': '-',
            'baseline': '-',
            'barrier': ':'},
        'control': {
            'apply_phase_modulation': True,
            'show_snapshot_channel': True,
            'show_acquire_channel': True,
            'show_empty_channel': True},
        'unicode_symbol': {
            'frame_change': u'\u21BA',
            'snapshot': u'\u21AF'},
        'latex_symbol': {
            'frame_change': r'\circlearrowleft',
            'snapshot': None}},
    'generator': {
        'waveform': [],
        'frame': [],
        'channel': [],
        'snapshot': [],
        'barrier': []}}
