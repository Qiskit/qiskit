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


class SchedStyle:
    def __init__(self, figsize=(10, 12), fig_unit_h_table=0.4,
                 use_table=True, table_columns=2, table_font_size=10, axis_font_size=18,
                 label_font_size=10, icon_font_size=18, label_ch_linestyle='--',
                 label_ch_color=None, label_ch_alpha=0.3, d_ch_color=None, u_ch_color=None,
                 m_ch_color=None, s_ch_color=None, s_ch_linestyle='-', table_color=None,
                 bg_color=None, num_points=1000, dpi=150, remove_spacing=True,
                 max_table_ratio=0.5):
        """Set style sheet for OpenPulse schedule drawer.

        Args:
            figsize (tuple): Size of figure.
            fig_unit_h_table (float): height of table row
            use_table (bool): use table
            table_columns (int): number of table columns
            table_font_size (float): font size of table
            axis_font_size (float): font size of axis label
            label_font_size (float): font size of labels
            icon_font_size (float): font size of labels
            label_ch_linestyle (str): Linestyle for labelling output channels
            label_ch_color (str): Color for channel pulse label line
            label_ch_alpha (float): Alpha for channel labels
            d_ch_color (list[str]): colors for real and imaginary part of waveform at d channels
            u_ch_color (list[str]): colors for real and imaginary part of waveform at u channels
            m_ch_color (list[str]): colors for real and imaginary part of waveform at m channels
            s_ch_color (str): color for snapshot channel line
            s_ch_linestyle (str): Linestyle for snapshot line
            table_color(list[str]): colors for table columns
            bg_color(str): color for figure background
            num_points (int): number of points for interpolation
            dpi (int): dpi to save image
            remove_spacing(bool): Remove redundant spacing when the waveform has no negative values
            max_table_ratio (float): Maximum portion of the plot the table can take up. Limited to
                range between 0 and 1.
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


class PulseStyle:
    def __init__(self, figsize=(7, 5), wave_color=None,
                 bg_color=None, num_points=None, dpi=None):
        """Set style sheet for OpenPulse sample pulse drawer.

        Args:
            figsize (tuple): Size of figure.
            wave_color (list[str]): colors for real and imaginary part of waveform.
            bg_color(str): color for figure background.
            num_points (int): number of points for interpolation.
            dpi (int): dpi to save image.
        """
        self.figsize = figsize
        self.wave_color = wave_color or ['#ff0000', '#0000ff']
        self.bg_color = bg_color or '#f2f3f4'
        self.num_points = num_points or 1000
        self.dpi = dpi or 150
