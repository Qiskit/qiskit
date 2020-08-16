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

"""
Scheduled circuit visualization module.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any, List

from qiskit import circuit


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
                 bit: Union[circuit.Qubit, circuit.Clbit],
                 x: List[float],
                 y: List[float],
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
                 bit: Union[circuit.Qubit, circuit.Clbit],
                 x: float,
                 y: float,
                 width: float,
                 height: float,
                 meta: Dict[str, Any] = None,
                 visible: bool = True,
                 styles: Dict[str, Any] = None):
        """Create new box.

        Args:
            data_type: String representation of this drawing object.
            bit: Bit associated to this object.
            x: Left coordinate of this box.
            y: Bottom coordinate of this box.
            width: Width of this box.
            height: Height of this box.
            meta: Meta data dictionary of the object.
            visible: Set ``True`` to show the component on the canvas.
            styles: Style keyword args of the object. This conforms to `matplotlib`.
        """
        self.bit = bit
        self.x = x
        self.y = y
        self.width = width
        self.height = height

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
                         self.width,
                         self.height)))


class TextData(ElementaryData):
    """Drawing object that represents a text on canvas."""
    def __init__(self,
                 data_type: str,
                 bit: Union[circuit.Qubit, circuit.Clbit],
                 x: float,
                 y: float,
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
                 bits: List[Union[circuit.Qubit, circuit.Clbit]],
                 x: float,
                 offset: float = 0,
                 visible: bool = True,
                 styles: Dict[str, Any] = None):
        self.bits = tuple(bits)
        self.x = x
        self.offset = offset

        super().__init__(
            data_type='BitLink',
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
