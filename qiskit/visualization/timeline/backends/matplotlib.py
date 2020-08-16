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

"""

from typing import Optional, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.timeline import drawer_style, core, types, drawing_objects


class MplPlotter:

    def __init__(self,
                 draw_data: core.DrawDataContainer,
                 axis: Optional[plt.Axes] = None):

        self.draw_data = draw_data

        if axis is None:
            fig_height = self.draw_data.bbox_top - self.draw_data.bbox_bottom
            fig_h = drawer_style['formatter.general.fig_unit_height'] * fig_height
            fig_w = drawer_style['formatter.general.fig_width']

            figure = plt.figure(figsize=(fig_w, fig_h))
            self.ax = figure.add_subplot(1, 1, 1)
        else:
            self.ax = axis

        self.initialize_canvas()

    def initialize_canvas(self):
        self.ax.set_facecolor(drawer_style['formatter.color.background'])

        # axis labels
        self.ax.set_yticklabels([])

        # boundary
        self.ax.set_xlim(self.draw_data.bbox_left, self.draw_data.bbox_right)
        self.ax.set_ylim(self.draw_data.bbox_bottom, self.draw_data.bbox_top)

    def draw(self):

        for drawing in self.draw_data.drawings:
            if not drawing.visible:
                continue

            # plot drawing element to canvas
            if isinstance(drawing, drawing_objects.BoxData):
                self._draw_box(drawing)
            elif isinstance(drawing, drawing_objects.LineData):
                self._draw_line(drawing)
            elif isinstance(drawing, drawing_objects.TextData):
                self._draw_text(drawing)
            elif isinstance(drawing, drawing_objects.BitLinkData):
                self._draw_bit_link(drawing)
            else:
                raise VisualizationError('Data type %s is not supported in matplotlib.' %
                                         drawing.__class__.__name__)

        # format
        self.ax.set_xlim(self.draw_data.bbox_left, self.draw_data.bbox_right)
        self.ax.set_ylim(self.draw_data.bbox_bottom, self.draw_data.bbox_top)

        return self.ax

    def _draw_box(self,
                  draw_obj: drawing_objects.BoxData):
        """Draw box data.

        Args:
            draw_obj: drawing object.
        """
        y_offset = self.draw_data.bit_offsets[draw_obj.bit]

        if draw_obj.data_type == types.DrawingBox.SCHED_GATE:
            # draw time bucket
            x, y1, y2 = _time_bucket_outline(x0=draw_obj.x0,
                                             y0=draw_obj.y0,
                                             x1=draw_obj.x1,
                                             y1=draw_obj.y1)

            self.ax.fill_between(x=x, y1=y1+y_offset, y2=y2+y_offset, **draw_obj.styles)
        else:
            # draw general rectangle
            x0 = self._get_coordinate(draw_obj.x0)
            y0 = self._get_coordinate(draw_obj.y0) + y_offset
            x1 = self._get_coordinate(draw_obj.x1)
            y1 = self._get_coordinate(draw_obj.y1) + y_offset

            rect = Rectangle(xy=(x0, y0), width=x1 - x0, height=y1 - y0)
            pc = PatchCollection([rect], **draw_obj.styles)
            self.ax.add_collection(pc)

    def _draw_line(self,
                   draw_obj: drawing_objects.LineData):
        """Draw line data.

        Args:
            draw_obj: drawing object.
        """
        y_offset = self.draw_data.bit_offsets[draw_obj.bit]

        xs = list(map(self._get_coordinate, draw_obj.x))
        ys = np.array(list(map(self._get_coordinate, draw_obj.y))) + y_offset

        self.ax.plot(xs, ys, **draw_obj.styles)

    def _draw_text(self,
                   draw_obj: drawing_objects.TextData):
        """Draw text data.

        Args:
            draw_obj: drawing object.
        """
        y_offset = self.draw_data.bit_offsets[draw_obj.bit]

        x = self._get_coordinate(draw_obj.x)
        y = self._get_coordinate(draw_obj.y) + y_offset

        if draw_obj.latex is not None:
            s = r'${latex}$'.format(latex=draw_obj.latex)
        else:
            s = draw_obj.text

        self.ax.text(x=x, y=y + y_offset, s=s, **draw_obj.styles)

    def _draw_bit_link(self,
                       draw_obj: drawing_objects.BitLinkData):
        """Draw bit link data.

        Args:
            draw_obj: drawing object.
        """
        ys = np.array([self.draw_data.bit_offsets.get(bit, None) for bit in draw_obj.bits])

        x_pos = draw_obj.x + draw_obj.offset

        self.ax.plot([x_pos, x_pos], [np.nanmin(ys), np.nanmax(ys)], **draw_obj.styles)

    def _get_coordinate(self,
                        value: types.Coordinate) -> Union[int, float]:

        if not isinstance(value, types.AbstractCoordinate):
            return value
        elif value == types.AbstractCoordinate.RIGHT:
            return self.draw_data.bbox_right
        elif value == types.AbstractCoordinate.LEFT:
            return self.draw_data.bbox_left
        elif value == types.AbstractCoordinate.TOP:
            return self.draw_data.bbox_top
        elif value == types.AbstractCoordinate.BOTTOM:
            return self.draw_data.bbox_bottom
        else:
            raise VisualizationError('Invalid coordinate %s is specified.' % value)


def _time_bucket_outline(x0: int,
                         y0: int,
                         x1: int,
                         y1: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate outline of time bucket. Edges are smoothly faded.

    Args:
        x0: Left coordinate.
        y0: Bottom coordinate.
        x1: Right coordinate.
        y1: Top coordinate.
    """
    ew = drawer_style['formatter.time_bucket.edge_dt']
    y_mid = 0.5 * (y0 + y1)

    edge = np.sin(np.pi / 2 * np.arange(0, ew) / ew)

    xs = np.concatenate([np.arange(x0, x0+ew),
                         [x0+ew, x1-ew],
                         np.arange(x1-ew+1, x1+1)])

    l1 = (y1-y_mid) * np.concatenate([edge, [1, 1], edge[::-1]])
    l2 = (y0-y_mid) * np.concatenate([edge, [1, 1], edge[::-1]])

    return xs, l1, l2
