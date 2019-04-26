# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree

# pylint: disable=invalid-name,anomalous-backslash-in-string,missing-docstring


class QCStyle:
    def __init__(self):
        self.tc = '#000000'
        self.sc = '#000000'
        self.lc = '#000000'
        self.cc = '#778899'
        self.gc = '#ffffff'
        self.gt = '#000000'
        self.bc = '#bdbdbd'
        self.bg = '#ffffff'
        self.fs = 13
        self.sfs = 8
        self.disptex = {
            'id': 'id',
            'u0': 'U_0',
            'u1': 'U_1',
            'u2': 'U_2',
            'u3': 'U_3',
            'x': 'X',
            'y': 'Y',
            'z': 'Z',
            'h': 'H',
            's': 'S',
            'sdg': 'S^\\dagger',
            't': 'T',
            'tdg': 'T^\\dagger',
            'rx': 'R_x',
            'ry': 'R_y',
            'rz': 'R_z',
            'reset': '\\left|0\\right\\rangle'
        }
        self.dispcol = {
            'id': '#ffffff',
            'u0': '#ffffff',
            'u1': '#ffffff',
            'u2': '#ffffff',
            'u3': '#ffffff',
            'x': '#ffffff',
            'y': '#ffffff',
            'z': '#ffffff',
            'h': '#ffffff',
            's': '#ffffff',
            'sdg': '#ffffff',
            't': '#ffffff',
            'tdg': '#ffffff',
            'rx': '#ffffff',
            'ry': '#ffffff',
            'rz': '#ffffff',
            'reset': '#ffffff',
            'target': '#ffffff',
            'meas': '#ffffff'
        }
        self.latexmode = True
        self.pimode = False
        self.fold = 20
        self.bundle = False
        self.barrier = True
        self.index = False
        self.figwidth = -1
        self.dpi = 150
        self.margin = [2.0, 0.0, 0.0, 0.3]
        self.cline = 'doublet'

    def set_style(self, dic):
        self.tc = dic.get('textcolor', self.tc)
        self.sc = dic.get('subtextcolor', self.sc)
        self.lc = dic.get('linecolor', self.lc)
        self.cc = dic.get('creglinecolor', self.cc)
        self.gt = dic.get('gatetextcolor', self.tc)
        self.gc = dic.get('gatefacecolor', self.gc)
        self.bc = dic.get('barrierfacecolor', self.bc)
        self.bg = dic.get('backgroundcolor', self.bg)
        self.fs = dic.get('fontsize', self.fs)
        self.sfs = dic.get('subfontsize', self.sfs)
        self.disptex = dic.get('displaytext', self.disptex)
        for key in self.dispcol.keys():
            self.dispcol[key] = self.gc
        self.dispcol = dic.get('displaycolor', self.dispcol)
        self.latexmode = dic.get('latexdrawerstyle', self.latexmode)
        self.pimode = dic.get('usepiformat', self.pimode)
        self.fold = dic.get('fold', self.fold)
        if self.fold < 2:
            self.fold = -1
        self.bundle = dic.get('cregbundle', self.bundle)
        self.barrier = dic.get('plotbarrier', self.barrier)
        self.index = dic.get('showindex', self.index)
        self.figwidth = dic.get('figwidth', self.figwidth)
        self.dpi = dic.get('dpi', self.dpi)
        self.margin = dic.get('margin', self.margin)
        self.cline = dic.get('creglinestyle', self.cline)


class OPStyleSched:
    def __init__(self, fig_w=None, fig_unit_h_waveform=None, fig_unit_h_table=None,
                 use_table=None, table_columns=None, table_font_size=None, label_font_size=None,
                 icon_font_size=None, d_ch_color=None, u_ch_color=None, m_ch_color=None,
                 s_ch_color=None, s_ch_linestyle=None,
                 table_color=None, bg_color=None, num_points=None, dpi=None):
        """Set style sheet for OpenPulse schedule drawer.

        Args:
            fig_w (float): width of figure.
            fig_unit_h_waveform (float): height of single waveform plot.
            fig_unit_h_table (float): height of table row.
            use_table (bool): use table.
            table_columns (int): number of table columns.
            table_font_size (float): font size of table.
            label_font_size (float): font size of labels.
            icon_font_size (float): font size of labels.
            d_ch_color (list[str]): colors for real and imaginary part of waveform at d channels.
            u_ch_color (list[str]): colors for real and imaginary part of waveform at u channels.
            m_ch_color (list[str]): colors for real and imaginary part of waveform at m channels.
            s_ch_color (str): color for snapshot channel line.
            s_ch_linestyle (str): Linestyle for snapshot line.
            table_color(list[str]): colors for table columns.
            bg_color(str): color for figure background.
            num_points (int): number of points for interpolation.
            dpi (int): dpi to save image.
        """
        self.fig_w = fig_w or 10
        self.fig_unit_h_waveform = fig_unit_h_waveform or 2.5
        self.fig_unit_h_table = fig_unit_h_table or 0.4
        self.use_table = use_table or True
        self.table_columns = table_columns or 2
        self.table_font_size = table_font_size or 10
        self.label_font_size = label_font_size or 18
        self.icon_font_size = icon_font_size or 18
        self.d_ch_color = d_ch_color or ['#648fff', '#002999']
        self.u_ch_color = u_ch_color or ['#ffb000', '#994A00']
        self.m_ch_color = m_ch_color or ['#dc267f', '#760019']
        self.a_ch_color = m_ch_color or ['#333333', '#666666']
        self.s_ch_color = s_ch_color or '#7da781'
        self.s_ch_linestyle = s_ch_linestyle or '--'
        self.table_color = table_color or ['#e0e0e0', '#f6f6f6', '#f6f6f6']
        self.bg_color = bg_color or '#f2f3f4'
        self.num_points = num_points or 1000
        self.dpi = dpi or 150


class OPStylePulse:
    def __init__(self, fig_w=None, fig_h=None, wave_color=None,
                 bg_color=None, num_points=None, dpi=None):
        """Set style sheet for OpenPulse sample pulse drawer.

        Args:
            fig_w (float): width of figure.
            fig_h (float): hight of figure.
            wave_color (list[str]): colors for real and imaginary part of waveform.
            bg_color(str): color for figure background.
            num_points (int): number of points for interpolation.
            dpi (int): dpi to save image.
        """
        self.fig_w = fig_w or 6
        self.fig_h = fig_h or 5
        self.wave_color = wave_color or ['#ff0000', '#0000ff']
        self.bg_color = bg_color or '#f2f3f4'
        self.num_points = num_points or 1000
        self.dpi = dpi or 150
