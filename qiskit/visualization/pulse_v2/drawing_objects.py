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

# pylint: disable=invalid-name

r"""
Drawing objects for pulse drawer.

Drawing objects play two important roles:
    - Allowing unittests of visualization module. Usually it is hard for image files to be tested.
    - Removing program parser from each plotter interface. We can easily add new plotter.

This module is based on the structure of matplotlib as it is the primary plotter
of the pulse drawer. However this interface is agnostic to the actual plotter.

Design concept
~~~~~~~~~~~~~~
When we think about dynamically updating drawing objects, it will be most efficient to
update only the changed properties of drawings rather than regenerating entirely from scratch.
Thus the core drawing function generates all possible drawings in the beginning and
then updates the visibility and the offset coordinate of each item according to
the end-user request.

Data key
~~~~~~~~
In the abstract class ``ElementaryData`` common properties to represent a drawing object are
specified. In addition, drawing objects have the `data_key` property that returns an
unique hash of the object for comparison. This property should be defined in each sub-class by
considering necessary properties to identify that object, i.e. `visible` should not
be a part of the key, because any change on this property just sets the visibility of
the same drawing object.

To support not only `matplotlib` but also multiple plotters, those drawing objectss should be
universal and designed without strong dependency on modules in `matplotlib`.
This means drawing objects that represent primitive geometries are preferred.
It should be noted that there will be no unittest for a plotter interface, which takes
drawing objects and output an image data, we should avoid adding a complicated data structure
that has a context of the pulse program.

For example, a pulse envelope is complex valued number array and may be represented
by two lines with different colors associated with the real and the imaginary component.
In this case, we can use two line-type objects rather than defining a new drwaing object
that takes complex value. Because many plotters don't support an API that visualizes
complex valued data array. If we introduce such drawing object and write a custom wrapper function
on top of the existing plotter API, it could be difficult to prevent bugs with the CI tools
due to lack of the effective unittest.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union

import numpy as np

from qiskit.pulse import channels
from qiskit.visualization.exceptions import VisualizationError


class ElementaryData(ABC):
    """Base class of the pulse visualization interface."""
    def __init__(self,
                 data_type: str,
                 channel: channels.Channel,
                 meta: Optional[Dict[str, Any]],
                 offset: float,
                 scale: float,
                 visible: bool,
                 styles: Optional[Dict[str, Any]]):
        """Create new drawing object.

        Args:
            data_type: String representation of this drawing object.
            channel: Pulse channel object bound to this drawing.
            meta: Meta data dictionary of the object.
            offset: Offset coordinate of vertical axis.
            scale: Vertical scaling factor of this object.
            visible: Set ``True`` to show the component on the canvas.
            styles: Style keyword args of the object. This conforms to `matplotlib`.
        """
        self.data_type = data_type
        self.channel = channel
        self.meta = meta
        self.scale = scale
        self.offset = offset
        self.visible = visible
        self.styles = styles

    @property
    @abstractmethod
    def data_key(self):
        """Return unique hash of this object."""
        pass

    def __repr__(self):
        return "{}(type={}, key={})".format(self.__class__.__name__,
                                            self.data_type,
                                            self.data_key)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.data_key == other.data_key


class FilledAreaData(ElementaryData):
    """Drawing object to represent object appears as a filled area.

    This is the counterpart of `matplotlib.axes.Axes.fill_between`.
    """
    def __init__(self,
                 data_type: str,
                 channel: channels.Channel,
                 x: np.ndarray,
                 y1: np.ndarray,
                 y2: np.ndarray,
                 meta: Optional[Dict[str, Any]] = None,
                 offset: float = 0,
                 scale: float = 1,
                 visible: bool = True,
                 styles: Optional[Dict[str, Any]] = None):
        """Create new drawing object of filled area.

        Args:
            data_type: String representation of this drawing object.
            channel: Pulse channel object bound to this drawing.
            x: Series of horizontal coordinate that the object is drawn.
            y1: Series of vertical coordinate of upper boundary of filling area.
            y2: Series of vertical coordinate of lower boundary of filling area.
            meta: Meta data dictionary of the object.
            offset: Offset coordinate of vertical axis.
            scale: Vertical scaling factor of this object.
            visible: Set ``True`` to show the component on the canvas.
            styles: Style keyword args of the object. This conforms to `matplotlib`.
        """
        self.x = x
        self.y1 = y1
        self.y2 = y2

        super().__init__(data_type=data_type,
                         channel=channel,
                         meta=meta,
                         offset=offset,
                         scale=scale,
                         visible=visible,
                         styles=styles)

    @property
    def data_key(self):
        """Return unique hash of this object."""
        return str(hash((self.__class__.__name__,
                         self.data_type,
                         self.channel,
                         tuple(self.x),
                         tuple(self.y1),
                         tuple(self.y2))))


class LineData(ElementaryData):
    """Drawing object to represent object appears as a line.

    This is the counterpart of `matplotlib.pyploy.plot`.
    """
    def __init__(self,
                 data_type: str,
                 channel: channels.Channel,
                 x: Optional[Union[np.ndarray, float]],
                 y: Optional[Union[np.ndarray, float]],
                 meta: Optional[Dict[str, Any]] = None,
                 offset: float = 0,
                 scale: float = 1,
                 visible: bool = True,
                 styles: Optional[Dict[str, Any]] = None):
        """Create new drawing object of line data.

        Args:
            data_type: String representation of this drawing object.
            channel: Pulse channel object bound to this drawing.
            x: Series of horizontal coordinate that the object is drawn.
                If `x` is `None`, a horizontal line is drawn at `y`.
            y: Series of vertical coordinate that the object is drawn.
                If `y` is `None`, a vertical line is drawn at `x`.
            meta: Meta data dictionary of the object.
            offset: Offset coordinate of vertical axis.
            scale: Vertical scaling factor of this object.
            visible: Set ``True`` to show the component on the canvas.
            styles: Style keyword args of the object. This conforms to `matplotlib`.

        Raises:
            VisualizationError: When both `x` and `y` are None.
        """
        if x is None and y is None:
            raise VisualizationError('`x` and `y` cannot be None simultaneously.')

        self.x = x
        self.y = y

        super().__init__(data_type=data_type,
                         channel=channel,
                         meta=meta,
                         offset=offset,
                         scale=scale,
                         visible=visible,
                         styles=styles)

    @property
    def data_key(self):
        """Return unique hash of this object."""
        return str(hash((self.__class__.__name__,
                         self.data_type,
                         self.channel,
                         tuple(self.x),
                         tuple(self.y))))


class TextData(ElementaryData):
    """Drawing object to represent object appears as a text.

    This is the counterpart of `matplotlib.pyploy.text`.
    """
    def __init__(self,
                 data_type: str,
                 channel: channels.Channel,
                 x: float,
                 y: float,
                 text: str,
                 latex: Optional[str] = None,
                 meta: Optional[Dict[str, Any]] = None,
                 offset: float = 0,
                 scale: float = 1,
                 visible: bool = True,
                 styles: Optional[Dict[str, Any]] = None):
        """Create new drawing object of text data.

        Args:
            data_type: String representation of this drawing object.
            channel: Pulse channel object bound to this drawing.
            x: A horizontal coordinate that the object is drawn.
            y: A vertical coordinate that the object is drawn.
            text: String to show in the canvas.
            latex: Latex representation of the text (if backend supports latex drawing).
            meta: Meta data dictionary of the object.
            offset: Offset coordinate of vertical axis.
            scale: Vertical scaling factor of this object.
            visible: Set ``True`` to show the component on the canvas.
            styles: Style keyword args of the object. This conforms to `matplotlib`.
        """
        self.x = x
        self.y = y
        self.text = text
        self.latex = latex or ''

        super().__init__(data_type=data_type,
                         channel=channel,
                         meta=meta,
                         offset=offset,
                         scale=scale,
                         visible=visible,
                         styles=styles)

    @property
    def data_key(self):
        """Return unique hash of this object."""
        return str(hash((self.__class__.__name__,
                         self.data_type,
                         self.channel,
                         self.x,
                         self.y,
                         self.text,
                         self.latex)))
