# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=import-outside-toplevel
"""A module of Plotly class wrappers.
"""

import plotly.graph_objects as go


class PlotlyFigure():
    """A simple wrapper around Plotly Figure class.

    Allows the figures to be more or less drop in replacements
    for Matplotlib Figures, e.g. savefig is a method here.
    """
    def __init__(self, fig):
        self._fig = fig

    def __repr__(self):
        return self._fig.__repr__()

    def _ipython_display_(self):
        import plotly.io as pio

        if pio.renderers.render_on_display and pio.renderers.default:
            pio.show(self._fig, config={'displayModeBar': False,
                                        'editable': False})
        else:
            print(repr(self))

    def show(self, *args, **kwargs):
        """Display the figure.
        """
        import plotly.io as pio

        config = {}
        if 'config' not in kwargs.keys():
            config = {'displayModeBar': False,
                      'editable': False}

        return pio.show(self._fig, *args, config=config, **kwargs)

    def savefig(self, filename, figsize=(None, None), scale=1, transparent=False):
        """Save the figure.

        Parameters:
            filename (str): Filename to save to.
            figsize (tuple): Figure size (W x H) in pixels.
            scale (float): Scales the output figure.
            transparent (bool): Transparent background.
        """
        if transparent:
            plot_color = self._fig.layout['plot_bgcolor']
            paper_color = self._fig.layout['paper_bgcolor']
            self._fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)')
        self._fig.write_image(filename, width=figsize[0], height=figsize[1], scale=scale)
        if transparent:
            self._fig.update_layout(plot_bgcolor=plot_color,
                                    paper_bgcolor=paper_color)


class PlotlyWidget(go.FigureWidget):
    """A wrapper around the Plotly widget class.
    """
    def show(self, *args, **kwargs):
        """Display the figure.
        """
        import plotly.io as pio

        config = {}
        if 'config' not in kwargs.keys():
            config = {'displayModeBar': False,
                      'editable': False}

        return pio.show(self, *args, config=config, **kwargs)

    def savefig(self, filename, figsize=(None, None), scale=1, transparent=False):
        """Safe the figure as a static image.

        Parameters:
            filename (str): Name of the file to which the image is saved.
            figsize (tuple): Size of figure in pixels.
            scale (float): Scale factor for non-vectorized image formats.
            transparent (bool): Set the background to transparent.
        """
        if transparent:
            plot_color = self.layout['plot_bgcolor']
            paper_color = self.layout['paper_bgcolor']
            self.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)')

        self.write_image(filename, width=figsize[0], height=figsize[1], scale=scale)
        if transparent:
            self.update_layout(plot_bgcolor=plot_color,
                               paper_bgcolor=paper_color)
