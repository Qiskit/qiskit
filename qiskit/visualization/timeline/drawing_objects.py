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
Drawing objects for timeline drawer.

Drawing objects play two important roles:
    - Allowing unittests of visualization module. Usually it is hard for image files to be tested.
    - Removing program parser from each plotter interface. We can easily add new plotter.

This module is based on the structure of matplotlib as it is the primary plotter
of the timeline drawer. However this interface is agnostic to the actual plotter.

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

To support not only `matplotlib` but also multiple plotters, those drawing objects should be
universal and designed without strong dependency on modules in `matplotlib`.
This means drawing objects that represent primitive geometries are preferred.
It should be noted that there will be no unittest for a plotter interface, which takes
drawing objects and output an image data, we should avoid adding a complicated data structure
that has a context of the scheduled circuit program.

Usually a drawing object is associated to the specific bit. `BitLinkData` is the
exception of this framework because it takes multiple bits to connect.
Position of the link is dynamically updated according to the user preference of bit to show.
While other objects are general format to represent specific shapes, the `BitLinkData` is
only used by the bit links and thus has no `data_type` input.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from qiskit.visualization.timeline import types


class ElementaryData(ABC):
    """Base class of the scheduled circuit visualization object.

    Note that drawing objects are mutable.
    """
    __hash__ = None

    def __init__(self,
                 data_type: str,
                 meta: Optional[Dict[str, Any]],
                 visible: bool,
                 styles: Optional[Dict[str, Any]]):
        """Create new drawing object.

        Args:
            data_type: String representation of this drawing object.
            meta: Meta data dictionary of the object.
            visible: Set ``True`` to show the component on the canvas.
            styles: Style keyword args of the object. This conforms to `matplotlib`.
        """
        self.data_type = data_type
        self.meta = meta
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


class LineData(ElementaryData):
    """Drawing object that represents line shape."""
    def __init__(self,
                 data_type: str,
                 bit: types.Bits,
                 x: List[types.Coordinate],
                 y: List[types.Coordinate],
                 meta: Dict[str, Any] = None,
                 visible: bool = True,
                 styles: Dict[str, Any] = None):
        """Create new line.

        Args:
            data_type: String representation of this drawing object.
            bit: Bit associated to this object.
            x: Horizontal coordinate sequence of this line.
            y: Vertical coordinate sequence of this line.
            meta: Meta data dictionary of the object.
            visible: Set ``True`` to show the component on the canvas.
            styles: Style keyword args of the object. This conforms to `matplotlib`.
        """
        self.bit = bit
        self.x = tuple(x)
        self.y = tuple(y)

        super().__init__(
            data_type=data_type,
            meta=meta,
            visible=visible,
            styles=styles
        )

    @property
    def data_key(self):
        """Return unique hash of this object."""
        return str(hash((self.__class__.__name__,
                         self.data_type,
                         self.bit,
                         self.x,
                         self.y)))


class BoxData(ElementaryData):
    """Drawing object that represents box shape."""
    def __init__(self,
                 data_type: str,
                 bit: types.Bits,
                 x0: types.Coordinate,
                 y0: types.Coordinate,
                 x1: types.Coordinate,
                 y1: types.Coordinate,
                 meta: Dict[str, Any] = None,
                 visible: bool = True,
                 styles: Dict[str, Any] = None):
        """Create new box.

        Args:
            data_type: String representation of this drawing object.
            bit: Bit associated to this object.
            x0: Left coordinate of this box.
            y0: Bottom coordinate of this box.
            x1: Right coordinate of this box.
            y1: Top coordinate of this box.
            meta: Meta data dictionary of the object.
            visible: Set ``True`` to show the component on the canvas.
            styles: Style keyword args of the object. This conforms to `matplotlib`.
        """
        self.bit = bit
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

        super().__init__(
            data_type=data_type,
            meta=meta,
            visible=visible,
            styles=styles
        )

    @property
    def data_key(self):
        """Return unique hash of this object."""
        return str(hash((self.__class__.__name__,
                         self.data_type,
                         self.bit,
                         self.x0,
                         self.y0,
                         self.x1,
                         self.y1)))


class TextData(ElementaryData):
    """Drawing object that represents a text on canvas."""
    def __init__(self,
                 data_type: str,
                 bit: types.Bits,
                 x: types.Coordinate,
                 y: types.Coordinate,
                 text: str,
                 latex: Optional[str] = None,
                 meta: Dict[str, Any] = None,
                 visible: bool = True,
                 styles: Dict[str, Any] = None):
        """Create new text.

        Args:
            data_type: String representation of this drawing object.
            bit: Bit associated to this object.
            x: Horizontal reference coordinate of this box.
            y: Vertical reference coordinate of this box.
            text: A string to draw on the canvas.
            latex: If set this string is used instead of `text`.
            meta: Meta data dictionary of the object.
            visible: Set ``True`` to show the component on the canvas.
            styles: Style keyword args of the object. This conforms to `matplotlib`.
        """
        self.bit = bit
        self.x = x
        self.y = y
        self.text = text
        self.latex = latex

        super().__init__(
            data_type=data_type,
            meta=meta,
            visible=visible,
            styles=styles
        )

    @property
    def data_key(self):
        """Return unique hash of this object."""
        return str(hash((self.__class__.__name__,
                         self.data_type,
                         self.bit,
                         self.x,
                         self.y,
                         self.text,
                         self.latex)))


class BitLinkData(ElementaryData):
    """A special drawing data type that represents bit link of multi-bit gates.

    Note this object takes multiple bit and be dedicated to the bit link.
    This may appear as a line on the canvas.
    """
    def __init__(self,
                 bits: List[types.Bits],
                 x: types.Coordinate,
                 offset: float = 0,
                 visible: bool = True,
                 styles: Dict[str, Any] = None):
        """Create new bit link.

        Args:
            bits:
            x: Horizontal coordinate of the link.
            offset: Horizontal offset of bit link. If multiple links are overlapped,
                the actual position of the link is automatically shifted by this argument.
            visible: Set ``True`` to show the component on the canvas.
            styles: Style keyword args of the object. This conforms to `matplotlib`.
        """
        self.bits = tuple(bits)
        self.x = x
        self.offset = offset

        super().__init__(
            data_type=types.DrawingLine.BIT_LINK,
            meta=None,
            visible=visible,
            styles=styles
        )

    @property
    def data_key(self):
        """Return unique hash of this object."""
        return str(hash((self.__class__.__name__,
                         self.data_type,
                         self.bits,
                         self.x)))
