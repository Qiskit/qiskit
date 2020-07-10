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


class QiskitPulseStyle(dict):
    """Stylesheet for pulse drawer.
    """
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass








class _QiskitPulseStyle:
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
                 color_ch_a: ComplexColors,
                 color_baseline: str,
                 color_background: str,
                 color_annotate: str,
                 color_axis_label: str,
                 # alpha
                 alpha_waveform: float,
                 alpha_baseline: float,
                 # layer indices
                 z_order_waveform: int,
                 z_order_baseline: int,
                 z_order_annotate: int,
                 z_order_axis_label: int,
                 # margins
                 margin_top: float,
                 margin_bottom: float,
                 margin_left: float,
                 margin_right: float,
                 margin_between_channels: float,
                 # text sizes
                 text_size_axis_label: float,
                 text_size_annotate: float,
                 text_size_symbol: float,
                 # line width
                 line_width_waveform: float,
                 line_width_baseline: float,
                 # options
                 option_phase_modulation: bool,
                 # generators
                 gen_waveform: List[Callable],
                 gen_baseline: List[Callable],
                 gen_pulse_label: List[Callable],
                 gen_channel_label: List[Callable],
                 gen_frame_label: List[Callable]):
        """Create new stylesheet.
        """
        self.formatter = dict(
            general={
                'fig_size': fig_size,
                'dpi': dpi
            },
            color={
                'ch_d': color_ch_d,
                'ch_u': color_ch_u,
                'ch_m': color_ch_m,
                'ch_a': color_ch_a,
                'baseline': color_baseline,
                'background': color_background,
                'annotate': color_annotate,
                'axis_label': color_axis_label
            },
            alpha={
                'waveform': alpha_waveform,
                'baseline': alpha_baseline
            },
            layer={
                'waveform': z_order_waveform,
                'baseline': z_order_baseline,
                'annotate': z_order_annotate,
                'axis_label': z_order_axis_label,
            },
            margin={
                'top': margin_top,
                'bottom': margin_bottom,
                'left': margin_left,
                'right': margin_right,
                'between_channel': margin_between_channels
            },
            text_size={
                'axis_label': text_size_axis_label,
                'annotate': text_size_annotate,
                'symbol': text_size_symbol
            },
            line_width={
                'waveform': line_width_waveform,
                'baseline': line_width_baseline
            },
            option={
                'phase_modulation': option_phase_modulation
            }
        )
        self.generator = dict(
            waveform=gen_waveform,
            baseline=gen_baseline,
            pulse_label=gen_pulse_label,
            channel_label=gen_channel_label,
            frame_label=gen_frame_label
        )
