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

# pylint: disable=invalid-name,anomalous-backslash-in-string,missing-docstring


class DefaultStyle:
    """A colorblind friendly gate styling modelled
    after:
        B. Wang, “Points of view: Color blindness“,
        Nat. Methods 8, 441 (2011).
    """
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
            'id': '#F0E442',
            'u0': '#E7AB3B',
            'u1': '#E7AB3B',
            'u2': '#E7AB3B',
            'u3': '#E7AB3B',
            'x': '#58C698',
            'y': '#58C698',
            'z': '#58C698',
            'h': '#70B7EB',
            's': '#E0722D',
            'sdg': '#E0722D',
            't': '#E0722D',
            'tdg': '#E0722D',
            'rx': '#ffffff',
            'ry': '#ffffff',
            'rz': '#ffffff',
            'reset': '#D188B4',
            'target': '#70B7EB',
            'meas': '#D188B4'
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


class BWStyle:
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
    def __init__(self, figsize=(10, 12), fig_unit_h_table=0.4,
                 use_table=True, table_columns=2, table_font_size=10, axis_font_size=18,
                 label_font_size=10, icon_font_size=18, label_ch_linestyle='--',
                 label_ch_color=None, label_ch_alpha=0.3, d_ch_color=None, u_ch_color=None,
                 m_ch_color=None, s_ch_color=None, s_ch_linestyle='-', table_color=None,
                 bg_color=None, num_points=1000, dpi=150):
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


class OPStylePulse:
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
