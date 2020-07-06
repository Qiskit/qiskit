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

Drawing IRs play two important roles:
- Allowing unittests of visualization module. Usually it is hard for image files to be tested.
- Removing program parser from each plotter interface. We can easily add new plotter.

IRs supported by this module is designed based on `matplotlob` since it is the primary plotter
of the pulse drawer. However IRs should be agnostic to the actual plotter.

When we think about the dynamic update of drawing objects, it will be efficient to
just update properties of drawing objects rather than regenerating everything from scratch.
Thus the core drawing function generates all possible drawings in the beginning and
then updates the visibility and the coordinate of each item according to the end-user request.
Drawing properties are designed based on this line of thinking.

In the abstract class ``ElementaryData`` common properties to represent a drawing object are
specified. In addition, it has the `data_key` property that returns an unique hash for
the drawing for comparing objects. This property should be defined in each sub-class by
considering important properties to identify that object, i.e. `visible` should not
be a part of the key, because change on this property just set visibility of
the same drawing.

To support not only `matplotlib` but also multiple plotters, those drawing IRs should be
universal and designed without strong dependency on modules in `matplotlib`.
Thus, an IR should represent a primitive geometry that is supported by many plotters.
It should be noted that there will be no unittest for an actual plotter interface, which takes
drawing IRs and output image data, we should avoid adding a complicated data structure
that has a context of the pulse program.

For example, a pulse envelope is complex valued number array and may be represented
by two lines with different colors corresponding to the real and imaginary component.
We may use two line-type IRs rather than defining a new IR that takes complex value,
because many plotter doesn't support a function that visualize complex values.
If we introduce such IR and write a custom wrapper function on top of a plotter API,
it could be difficult to prevent bugs by the CI process.
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
        return "{}(type={}, key={})".format(self.__class__.__name__,
                                            self.data_type,
                                            self.data_key)

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
                 x: float,
                 y: float,
                 text: str,
                 meta: Dict[str, Any],
                 offset: float,
                 visible: bool,
                 styles: Dict[str, Any]):  # pylint: disable=invalid-name
        """Create new visualization IR.
        Args:
            data_type: String representation of this drawing object.
            bind: Pulse channel object bound to this drawing.
            x: A horizontal coordinate that the object is drawn.
            y: A vertical coordinate that the object is drawn.
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
                         self.x,
                         self.y,
                         self.text)))
