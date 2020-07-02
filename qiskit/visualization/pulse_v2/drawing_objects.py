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
Drawing object IRs for pulse drawer.

In this module, we support following IRs.

- ``FilledAreaData``
- ``LineData``
- ``TextData``

Those object is designed based upon `matplotlib` since it is the primary plotter of
the pulse drawer. However those object should be backend plotter agnostic rather
than designed specific to the `matplotlib`.

In interactive visualization, for example, we may use other plotter such as `bokeh`
and drawing IRs should be able to be interpreted by all plotters supported by the pulse drawer.

To satisfy this requirement, the drawing IRs should be simple and preferably represent
a primitive shape that can be universally expressed by most of plotters.

For example, a pulse envelope is complex valued and may be represented by two lines
with different colors corresponding to the real and imaginary component.
However, this object should be represented with two ``FilledAreaData`` IRs
rather than defining a dedicated IR. A complicated IR may not be handled by some plotters.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np

from qiskit import pulse


class ElementaryData(ABC):
    """Abstract class of visualization intermediate representation."""
    def __init__(self,
                 data_type: str,
                 bind: pulse.channels.Channel,
                 meta: Dict[str, Any],
                 offset: float,
                 visible: bool,
                 styles: Dict[str, Any]):
        """Create new visualization IR.
        Args:
            data_type: String representation of this drawing object.
            bind: Pulse channel object bound to this drawing.
            offset: Offset coordinate of vertical axis.
            visible: Set ``True`` to show the component on the canvas.
        """
        self.data_type = data_type
        self.bind = bind
        self.meta = meta
        self.offset = offset
        self.visible = visible
        self.styles = styles

    @property
    @abstractmethod
    def data_key(self):
        pass

    def __repr__(self):
        return "{}(data_key={})".format(self.__class__.__name__, self.data_key)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.data_key == other.data_key


class FilledAreaData(ElementaryData):
    """Drawing IR to represent object appears as a filled area.
    This is the counterpart of `matplotlib.axes.Axes.fill_between`.
    """
    def __init__(self,
                 data_type: str,
                 bind: pulse.channels.Channel,
                 x: np.ndarray,
                 y1: np.ndarray,
                 y2: np.ndarray,
                 meta: Dict[str, Any],
                 offset: float,
                 visible: bool,
                 styles: Dict[str, Any]):  # pylint: disable=invalid-name
        """Create new visualization IR.
        Args:
            data_type: String representation of this drawing object.
            bind: Pulse channel object bound to this drawing.
            x: Series of horizontal coordinate that the object is drawn.
            y1: Series of vertical coordinate of upper boundary of filling area.
            y2: Series of vertical coordinate of lower boundary of filling area.
            offset: Offset coordinate of vertical axis.
            visible: Set ``True`` to show the component on the canvas.
        """
        self.x = x
        self.y1 = y1
        self.y2 = y2

        super().__init__(
            data_type=data_type,
            bind=bind,
            meta=meta,
            offset=offset,
            visible=visible,
            styles=styles
        )

    @property
    def data_key(self):
        return str(hash((self.__class__.__name__,
                         self.data_type,
                         self.bind,
                         tuple(self.x),
                         tuple(self.y1),
                         tuple(self.y2))))


class LineData(ElementaryData):
    """Drawing IR to represent object appears as a line.
    This is the counterpart of `matplotlib.pyploy.plot`.
    """
    def __init__(self,
                 data_type: str,
                 bind: pulse.channels.Channel,
                 x: np.ndarray,
                 y: np.ndarray,
                 meta: Dict[str, Any],
                 offset: float,
                 visible: bool,
                 styles: Dict[str, Any]):  # pylint: disable=invalid-name
        """Create new visualization IR.
        Args:
            data_type: String representation of this drawing object.
            bind: Pulse channel object bound to this drawing.
            x: Series of horizontal coordinate that the object is drawn.
            y: Series of vertical coordinate that the object is drawn.
            offset: Offset coordinate of vertical axis.
            visible: Set ``True`` to show the component on the canvas.
        """
        self.x = x
        self.y = y

        super().__init__(
            data_type=data_type,
            bind=bind,
            meta=meta,
            offset=offset,
            visible=visible,
            styles=styles
        )

    @property
    def data_key(self):
        return str(hash((self.__class__.__name__,
                         self.data_type,
                         self.bind,
                         tuple(self.x),
                         tuple(self.y))))


class TextData(ElementaryData):
    """Drawing IR to represent object appears as a text.
    This is the counterpart of `matplotlib.pyploy.text`.
    """
    def __init__(self,
                 data_type: str,
                 bind: pulse.channels.Channel,
                 x: np.ndarray,
                 y: np.ndarray,
                 text: str,
                 meta: Dict[str, Any],
                 offset: float,
                 visible: bool,
                 styles: Dict[str, Any]):  # pylint: disable=invalid-name
        """Create new visualization IR.
        Args:
            data_type: String representation of this drawing object.
            bind: Pulse channel object bound to this drawing.
            x: Series of horizontal coordinate that the object is drawn.
            y: Series of vertical coordinate that the object is drawn.
            text: String to show in the canvas.
            offset: Offset coordinate of vertical axis.
            visible: Set ``True`` to show the component on the canvas.
        """
        self.x = x
        self.y = y
        self.text = text

        super().__init__(
            data_type=data_type,
            bind=bind,
            meta=meta,
            offset=offset,
            visible=visible,
            styles=styles
        )

    @property
    def data_key(self):
        return str(hash((self.__class__.__name__,
                         self.data_type,
                         self.bind,
                         tuple(self.x),
                         tuple(self.y),
                         self.text)))
