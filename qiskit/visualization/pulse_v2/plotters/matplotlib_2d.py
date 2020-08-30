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
from qiskit.visualization.pulse_v2 import core, drawing_objects


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

        # boundary
        self.ax.set_xlim(*self.canvas.time_range)

    def draw(self):
        current_y = 0
        for chart in self.canvas.charts:
            current_y -= chart.vmax
            for data in chart.collections:
                # calculate scaling factor
                if not data.ignore_scaling:
                    # product of channel-wise scaling and chart level scaling
                    scale = max(self.canvas.chan_scales.get(chan, 1.0) for chan in data.channels)
                    scale *= chart.scale
                else:
                    scale = 1.0

                if isinstance(data, drawing_objects.FilledAreaData):
                    # filled area object
                    x = chart.bind_coordinate(data.x)
                    y1 = scale * chart.bind_coordinate(data.y1) + current_y
                    y2 = scale * chart.bind_coordinate(data.y2) + current_y
                    self.ax.fill_between(x=x, y1=y1, y2=y2, **data.styles)
                elif isinstance(data, drawing_objects.LineData):
                    # line object
                    x = chart.bind_coordinate(data.x)
                    y = scale * chart.bind_coordinate(data.y) + current_y
                    self.ax.plot(x, y, **data.styles)
                elif isinstance(data, drawing_objects.TextData):
                    # text object
                    x = chart.bind_coordinate([data.x])[0]
                    y = scale * chart.bind_coordinate([data.y])[0] + current_y
                    text = r'${s}$'.format(s=data.latex) if data.latex else data.text
                    self.ax.text(x=x, y=y, s=text, **data.styles)
                else:
                    VisualizationError('Data {name} is not supported '
                                       'by {plotter}'.format(name=data,
                                                             plotter=self.__class__.__name__))
            current_y -= chart.vmin

        self.ax.set_ylim(current_y, 0)
