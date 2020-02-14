# -*- coding: utf-8 -*-

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

# pylint: disable=missing-docstring

from typing import Tuple, List


class SchedStyle:
    def __init__(self, figsize: Tuple[float, float] = (10, 12),
                 fig_unit_h_table: float = 0.4,
                 use_table: bool = True,
                 table_columns: int = 2,
                 table_font_size: int = 10,
                 axis_font_size: int = 18,
                 label_font_size: int = 10,
                 icon_font_size: int = 18,
                 label_ch_linestyle: str = '--',
                 label_ch_color: str = None,
                 label_ch_alpha: float = 0.3,
                 d_ch_color: List[str] = None,
                 u_ch_color: List[str] = None,
                 m_ch_color: List[str] = None,
                 s_ch_color: List[str] = None,
                 s_ch_linestyle: str = '-',
                 table_color: str = None,
                 bg_color: str = None,
                 num_points: int = 1000,
                 dpi: int = 150,
                 remove_spacing: bool = True,
                 max_table_ratio: float = 0.5,
                 vertical_span: float = 0.2):
        """Set style sheet for Qiskit-Pulse schedule drawer.

        Args:
            figsize: Size of figure.
            fig_unit_h_table: Height of row of event table. See Example.
            use_table: When set `True` use event table.
            table_columns: Number of event table columns.
            table_font_size: Font size of event table.
            axis_font_size: Font size of channel aliases.
            label_font_size: Font size of labels in canvas.
            icon_font_size: Size of symbols.
            label_ch_linestyle: Line style for channel pulse label line.
            label_ch_color: Color code or name of color for channel pulse label line.
            label_ch_alpha: Transparency for channel pulse label line from 0 to 1.
            d_ch_color: Color code or name of colors for real and imaginary part
                of waveform at d channels.
            u_ch_color: Color code or name of colors for real and imaginary part
                of waveform at u channels.
            m_ch_color: Color code or name of colors for real and imaginary part
                of waveform at m channels.
            s_ch_color: Color code or name of color for snapshot channel line.
            s_ch_linestyle: Line style for snapshot line.
            table_color: Color code or name of color for event table columns of
                time, channel, event information.
            bg_color: Color code or name of color for canvas background.
            num_points: Number of points for interpolation of each channel.
            dpi: Resolution in the unit of dot per inch to save image.
            remove_spacing: Remove redundant spacing
                when the waveform has no negative values.
            max_table_ratio: Maximum portion of the plot the table can take up.
                Limited to range between 0 and 1.
            vertical_span: Spacing on top and bottom of pulse canvas.

        Example:
            Height of the event table is decided by multiple parameters.::

                figsize = (10, 12)
                fig_unit_h_table = 0.4
                table_columns = 2
                max_table_ratio = 0.5

            With this setup, events are shown in double-column style with
            each line height of 0.4 inch and the table cannot exceed 5 inch.
            Thus 12 lines are maximum and up to 24 events can be shown.
            If you want to show more events, increase figure height or
            reduce size of line height and table font size.
        """
        self.figsize = figsize
        self.fig_unit_h_table = fig_unit_h_table
        self.use_table = use_table
        self.table_columns = table_columns
        self.table_font_size = table_font_size
        self.axis_font_size = axis_font_size
        self.label_font_size = label_font_size
        self.icon_font_size = icon_font_size
        self.d_ch_color = d_ch_color or ['#648fff', '#002999']
        self.label_ch_linestyle = label_ch_linestyle
        self.label_ch_color = label_ch_color or '#222222'
        self.label_ch_alpha = label_ch_alpha
        self.u_ch_color = u_ch_color or ['#ffb000', '#994A00']
        self.m_ch_color = m_ch_color or ['#dc267f', '#760019']
        self.a_ch_color = m_ch_color or ['#333333', '#666666']
        self.s_ch_color = s_ch_color or '#7da781'
        self.s_ch_linestyle = s_ch_linestyle
        self.table_color = table_color or ['#e0e0e0', '#f6f6f6', '#f6f6f6']
        self.bg_color = bg_color or '#f2f3f4'
        self.num_points = num_points
        self.dpi = dpi
        self.remove_spacing = remove_spacing
        self.max_table_ratio = max(min(max_table_ratio, 0.0), 1.0)
        self.vertical_span = vertical_span


class PulseStyle:
    def __init__(self, figsize: Tuple[float, float] = (7, 5),
                 wave_color: List[str] = None,
                 bg_color: str = None,
                 num_points: int = None,
                 dpi: int = None):
        """Set style sheet for Qiskit-Pulse sample pulse drawer.

        Args:
            figsize: Size of figure.
            wave_color: Color code or name of colors for real and imaginary part
                of SamplePulse waveform.
            bg_color: Color code or name of color for pulse canvas background.
            num_points: Number of points for interpolation.
            dpi: Resolution in the unit of dot per inch to save image.
        """
        self.figsize = figsize
        self.wave_color = wave_color or ['#ff0000', '#0000ff']
        self.bg_color = bg_color or '#f2f3f4'
        self.num_points = num_points or 1000
        self.dpi = dpi or 150
