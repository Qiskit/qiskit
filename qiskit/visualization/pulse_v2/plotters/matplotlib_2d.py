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

from typing import Optional

import matplotlib.pyplot as plt

from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import core, drawing_objects, types
import numpy as np


class Mpl2DPlotter:

    def __init__(self,
                 canvas: core.DrawerCanvas,
                 axis: Optional[plt.Axes] = None):

        self.canvas = canvas

        # calculate height of all charts
        canvas_height = 0
        for chart in self.canvas.charts:
            canvas_height += chart.vmax - chart.vmin

        if axis is None:
            fig_h = canvas_height * self.canvas.formatter['general.fig_chart_height']
            fig_w = self.canvas.formatter['general.fig_width']

            figure = plt.figure(figsize=(fig_w, fig_h))
            self.ax = figure.add_subplot(1, 1, 1)
        else:
            self.ax = axis

        self.initialize_canvas()

    def initialize_canvas(self):
        self.ax.set_facecolor(self.canvas.formatter['color.background'])

        # axis labels
        self.ax.set_yticklabels([])

    def draw(self):
        # axis configuration
        axis_config = self.canvas.layout['time_axis_map'](
            time_window=self.canvas.time_range,
            axis_breaks=self.canvas.time_breaks,
            dt=self.canvas.device.dt
        )

        current_y = 0
        margin_y = self.canvas.formatter['margin.between_channel']
        for chart in self.canvas.charts:
            current_y -= chart.vmax
            for _, data in chart.collections:
                # calculate scaling factor
                if not data.ignore_scaling:
                    # product of channel-wise scaling and chart level scaling
                    scale = max(self.canvas.chan_scales.get(chan, 1.0) for chan in data.channels)
                    scale *= chart.scale
                else:
                    scale = 1.0

                x = data.xvals
                y = scale * data.yvals + current_y

                if isinstance(data, drawing_objects.LineData):
                    # line object
                    if data.fill:
                        self.ax.fill_between(x, y1=y,  y2=current_y * np.ones_like(y),
                                             **data.styles)
                    else:
                        self.ax.plot(x, y, **data.styles)
                elif isinstance(data, drawing_objects.TextData):
                    # text object
                    text = r'${s}$'.format(s=data.latex) if data.latex else data.text
                    # replace dynamic text
                    text = text.replace(types.DynamicString.SCALE,
                                        '{val:.1f}'.format(val=chart.scale))
                    self.ax.text(x=x[0], y=y[0], s=text, **data.styles)
                else:
                    VisualizationError('Data {name} is not supported '
                                       'by {plotter}'.format(name=data,
                                                             plotter=self.__class__.__name__))
            # axis break
            for pos in axis_config.axis_break_pos:
                self.ax.text(x=pos, y=current_y,
                             s='//',
                             ha='center',
                             va='center',
                             zorder=self.canvas.formatter['layer.axis_label'],
                             fontsize=self.canvas.formatter['text_size.axis_break_symbol'],
                             rotation=180)

            # shift chart position
            current_y += chart.vmin - margin_y

        # remove the last margin
        current_y += margin_y

        y_max = self.canvas.formatter['margin.top']
        y_min = current_y - self.canvas.formatter['margin.bottom']

        # plot axis break line
        for pos in axis_config.axis_break_pos:
            self.ax.plot([pos, pos], [y_min, y_max],
                         zorder=self.canvas.formatter['layer.fill_waveform'] + 1,
                         linewidth=self.canvas.formatter['line_width.axis_break'],
                         color=self.canvas.formatter['color.background'])

        # label
        self.ax.set_xticks(list(axis_config.axis_map.keys()))
        self.ax.set_xticklabels(list(axis_config.axis_map.values()),
                                fontsize=self.canvas.formatter['text_size.axis_label'])
        self.ax.set_xlabel(axis_config.label,
                           fontsize=self.canvas.formatter['text_size.axis_label'])

        # boundary
        self.ax.set_xlim(*axis_config.window)
        self.ax.set_ylim(y_min, y_max)

        # misc
        self.ax.yaxis.set_tick_params(left=False)
