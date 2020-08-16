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

# pylint: disable=invalid-name

"""
Scheduled circuit visualization module.
"""
import warnings
from typing import Dict, Any, Mapping


class QiskitTimelineStyle(dict):
    """Stylesheet for timeline drawer."""
    _deprecated_keys = {}

    def __init__(self):
        super().__init__()
        self.current_stylesheet = None
        self.update(default_style())

    def update(self, __m: Mapping[str, Any], **kwargs) -> None:
        super().update(__m, **kwargs)
        for key, value in __m.items():
            if key in self._deprecated_keys:
                warnings.warn('%s is deprecated. Use %s instead.'
                              % (key, self._deprecated_keys[key]),
                              DeprecationWarning)
                self.__setitem__(self._deprecated_keys[key], value)
            else:
                self.__setitem__(key, value)


def default_style() -> Dict[str, Any]:
    """Define default values of the timeline stylesheet."""
    return {
        'formatter.general.fig_unit_height': 0.7,
        'formatter.general.fig_width': 6,
        'formatter.margin.top': 0.5,
        'formatter.margin.bottom': 0.5,
        'formatter.margin.left_percent': 0.05,
        'formatter.margin.right_percent': 0.05,
        'formatter.margin.interval': 0.3,
        'formatter.margin.link_interval_dt': 100,
        'formatter.time_bucket.edge_dt': 10,
        'formatter.color.background': '#FFFFFF',
        'formatter.color.timeslot': '#DDDDDD',
        'formatter.color.gate_name': '#000000',
        'formatter.color.bit_name': '#000000',
        'formatter.color.barrier': '#222222',
        'formatter.box_height.gate': 0.7,
        'formatter.box_height.timeslot': 0.8,
        'formatter.layer.gate': 3,
        'formatter.layer.timeslot': 0,
        'formatter.layer.gate_name': 5,
        'formatter.layer.bit_name': 5,
        'formatter.layer.frame_change': 4,
        'formatter.layer.barrier': 1,
        'formatter.layer.bit_link': 2,
        'formatter.alpha.gate': 0.5,
        'formatter.alpha.timeslot': 0.7,
        'formatter.alpha.barrier': 0.5,
        'formatter.alpha.bit_link': 0.5,
        'formatter.line_width.gate': 0,
        'formatter.line_width.timeslot': 0,
        'formatter.line_width.barrier': 3,
        'formatter.line_width.bit_link': 3,
        'formatter.line_style.barrier': '-',
        'formatter.line_style.bit_link': '-',
        'formatter.font_size.gate_name': 12,
        'formatter.font_size.bit_name': 15,
        'formatter.font_size.frame_change': 20,
        'formatter.label_offset.frame_change': 0.25,
        'formatter.unicode_symbol.frame_change': u'\u21BA',
        'formatter.latex_symbol.frame_change': r'\circlearrowleft',
        'formatter.control.show_idle': True,
        'formatter.control.show_clbits': True,
        'layout.gate_color': None,
        'layout.latex_gate_name': None,
        'layout.bit_arange': None,
        'generator.gates': [],
        'generator.bits': [],
        'generator.barriers': [],
        'generator.bit_links': []}


drawer_style = QiskitTimelineStyle()
