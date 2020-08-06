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
In the abstract class ``ElementaryData`` common attributes to represent a drawing object are
specified. In addition, drawing objects have the `data_key` property that returns an
unique hash of the object for comparison.
This property should be defined in each-subclass by considering necessary data set to
identify that object while keeping sufficient flexibility for object to be updated.
Thus, this is basically the combination of data type, channel and coordinate.
See py:mod:`qiskit.visualization.pulse_v2.data_ypes` for the detail of data type.
If a data key cannot distinguish two independent objects, you need to add new data type.

The data key may be used in the plotter interface to identify the object instance
which may be dynamically updated. For example, we may use a `TextData` instance
to draw a scaling factor of certain channel. If we interactively change the scaling factor,
the plotter interface should be able to find the old object instance and
update the text field with new scaling factor. This example illustrates
the reason that the data key should be insensitive to the text value.


Drawing objects
~~~~~~~~~~~~~~~
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
from typing import Dict, Any, Optional, Union, List, NewType

import numpy as np

from qiskit.pulse import channels
from qiskit.visualization.pulse_v2 import types


Coordinate = NewType('Coordinate', Union[int, float, types.AbstractCoordinate])


class ElementaryData(ABC):
    """Base class of the pulse visualization interface."""
    __hash__ = None

    def __init__(self,
                 data_type: str,
                 channel: channels.Channel,
                 meta: Optional[Dict[str, Any]],
                 offset: float,
                 scale: float,
                 visible: bool,
                 fix_position: bool,
                 styles: Optional[Dict[str, Any]]):
        """Create new drawing object.

        Args:
            data_type: String representation of this drawing object.
            channel: Pulse channel object bound to this drawing.
            meta: Meta data dictionary of the object.
            offset: Offset coordinate of vertical axis.
            scale: Vertical scaling factor of this object.
            visible: Set ``True`` to show the component on the canvas.
            fix_position: Set ``True`` to disable scaling.
            styles: Style keyword args of the object. This conforms to `matplotlib`.
        """
        self.data_type = data_type
        self.channel = channel
        self.meta = meta or dict()
        self.scale = scale
        self.offset = offset
        self.visible = visible
        self.fix_position = fix_position
        self.styles = styles or dict()

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
                 x: Union[np.ndarray, List[Coordinate]],
                 y1: Union[np.ndarray, List[Coordinate]],
                 y2: Union[np.ndarray, List[Coordinate]],
                 meta: Optional[Dict[str, Any]] = None,
                 offset: float = 0,
                 scale: float = 1,
                 visible: bool = True,
                 fix_position: bool = False,
                 styles: Optional[Dict[str, Any]] = None):
        """Create new drawing object of filled area.

        Consecutive elements in ``y1`` and ``y2`` are compressed.

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
            fix_position: Set ``True`` to disable scaling.
            styles: Style keyword args of the object. This conforms to `matplotlib`.
        """
        # find consecutive elements in y1
        valid_inds = find_consecutive_index(y1) | find_consecutive_index(y2)

        self.x = np.array(x)[valid_inds]
        self.y1 = np.array(y1)[valid_inds]
        self.y2 = np.array(y2)[valid_inds]

        super().__init__(data_type=data_type,
                         channel=channel,
                         meta=meta,
                         offset=offset,
                         scale=scale,
                         visible=visible,
                         fix_position=fix_position,
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
                 x: Union[np.ndarray, List[Coordinate]],
                 y: Union[np.ndarray, List[Coordinate]],
                 meta: Optional[Dict[str, Any]] = None,
                 offset: float = 0,
                 scale: float = 1,
                 visible: bool = True,
                 fix_position: bool = False,
                 styles: Optional[Dict[str, Any]] = None):
        """Create new drawing object of line data.

        Consecutive elements in ``y2`` are compressed.

        Args:
            data_type: String representation of this drawing object.
            channel: Pulse channel object bound to this drawing.
            x: Series of horizontal coordinate that the object is drawn.
            y: Series of vertical coordinate that the object is drawn.
            meta: Meta data dictionary of the object.
            offset: Offset coordinate of vertical axis.
            scale: Vertical scaling factor of this object.
            visible: Set ``True`` to show the component on the canvas.
            fix_position: Set ``True`` to disable scaling.
            styles: Style keyword args of the object. This conforms to `matplotlib`.
        """
        # find consecutive elements in y
        valid_inds = find_consecutive_index(y)
        self.x = np.array(x)[valid_inds]
        self.y = np.array(y)[valid_inds]

        super().__init__(data_type=data_type,
                         channel=channel,
                         meta=meta,
                         offset=offset,
                         scale=scale,
                         visible=visible,
                         fix_position=fix_position,
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
                 x: Coordinate,
                 y: Coordinate,
                 text: str,
                 latex: Optional[str] = None,
                 meta: Optional[Dict[str, Any]] = None,
                 offset: float = 0,
                 scale: float = 1,
                 visible: bool = True,
                 fix_position: bool = False,
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
            fix_position: Set ``True`` to disable scaling.
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
                         fix_position=fix_position,
                         styles=styles)

    @property
    def data_key(self):
        """Return unique hash of this object."""
        return str(hash((self.__class__.__name__,
                         self.data_type,
                         self.channel,
                         self.x,
                         self.y)))


def find_consecutive_index(vector: Union[np.ndarray, List[Coordinate]]) -> np.ndarray:
    """A helper function to return non-consecutive index from the given list.

    This drastically reduces memory footprint to represent a drawing object,
    especially for samples of very long flat-topped Gaussian pulses.

    Args:
        vector: The array of numbers.
    """
    if all(isinstance(val, np.number) for val in vector):
        consecutive_ind_l = np.insert(np.diff(vector).astype(bool), 0, True)
        consecutive_ind_r = np.insert(np.diff(vector).astype(bool), -1, True)
        non_consecutive = consecutive_ind_l | consecutive_ind_r
    else:
        non_consecutive = np.ones_like(vector).astype(bool)

    return non_consecutive
