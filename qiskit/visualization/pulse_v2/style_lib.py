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
Stylesheet for pulse drawer.
"""

from typing import Callable, List, Tuple
from collections import namedtuple

ComplexColors = namedtuple('ComplexColors', 'real imaginary')


class QiskitPulseStyle:
    """Stylesheet for pulse drawer.
    """
    def __init__(self,
                 # canvas general
                 fig_size: Tuple[float, float],
                 dpi: int,
                 # colors
                 color_ch_d: ComplexColors,
                 color_ch_u: ComplexColors,
                 color_ch_m: ComplexColors,
                 color_baseline: str,
                 color_background: str,
                 color_annotate: str,
                 color_label_axis: str,
                 # layer indices
                 z_order_waveform: int,
                 z_order_baseline: int,
                 z_order_annotate: int,
                 # margins
                 margin_top: float,
                 margin_bottom: float,
                 margin_left: float,
                 margin_right: float,
                 margin_between_channels: float,
                 # text sizes
                 text_size_axis: float,
                 text_size_annotate: float,
                 text_size_symbol: float,
                 # generators
                 gen_waveform: List[Callable],
                 gen_baseline: List[Callable],
                 gen_pulse_label: List[Callable],
                 gen_channel_info: List[Callable],
                 gen_frame_info: List[Callable]):
        """Create new stylesheet.
        """
        # general setting
        self.fig_size = fig_size
        self.dpi = dpi

        # colors
        self.color_ch_d = color_ch_d
        self.color_ch_u = color_ch_u
        self.color_ch_m = color_ch_m

        self.color_baseline = color_baseline
        self.color_background = color_background
        self.color_annotate = color_annotate
        self.color_label_axis = color_label_axis

        # layer
        self.waveform_z_order = z_order_waveform
        self.baseline_z_order = z_order_baseline
        self.annotate_z_order = z_order_annotate

        # margin
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        self.margin_left = margin_left
        self.margin_right = margin_right
        self.margin_between_channel = margin_between_channels

        # text size
        self.text_size_axis = text_size_axis
        self.text_size_annotate = text_size_annotate
        self.text_size_symbol = text_size_symbol

        # object generators
        self.gen_waveform = gen_waveform
        self.gen_baseline = gen_baseline
        self.gen_pulse_label = gen_pulse_label
        self.gen_channel_info = gen_channel_info
        self.gen_frame_info = gen_frame_info
