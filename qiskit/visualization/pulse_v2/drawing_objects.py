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
Thus the core :py:class:`qiskit.visualization.pulse_v2.core.DrawerCanvas` generates
all possible drawings in the beginning and then the canvas instance manages
visibility of each drawing object according to the end-user request.

Data key
~~~~~~~~
In the abstract class ``ElementaryData`` common attributes to represent a drawing object are
specified. In addition, drawing objects have the `data_key` property that returns an
unique hash of the object for comparison.
This key is generated from a data type and the location of the drawing object in the canvas.
See py:mod:`qiskit.visualization.pulse_v2.types` for detail on the data type.
If a data key cannot distinguish two independent objects, you need to add a new data type.
The data key may be used in the plotter interface to identify the object.

Drawing objects
~~~~~~~~~~~~~~~
To support not only `matplotlib` but also multiple plotters, those drawing objects should be
universal and designed without strong dependency on modules in `matplotlib`.
This means drawing objects that represent primitive geometries are preferred.
It should be noted that there will be no unittest for each plotter API, which takes
drawing objects and outputs image data, we should avoid adding a complicated geometry
that has a context of the pulse program.

For example, a pulse envelope is complex valued number array and may be represented
by two lines with different colors associated with the real and the imaginary component.
We can use two line-type objects rather than defining a new drawing object that takes
complex value. As many plotters don't support an API that visualizes complex-valued
data arrays, if we introduced such a drawing object and wrote a custom wrapper function
on top of the existing API, it could be difficult to prevent bugs with the CI tools
due to lack of the effective unittest.
"""
from abc import ABC
from typing import Dict, Any, Optional, Union, List

import numpy as np
from qiskit.pulse.channels import Channel
from qiskit.visualization.pulse_v2 import types


class ElementaryData(ABC):
    """Base class of the pulse visualization interface."""
    __hash__ = None

    def __init__(self,
                 data_type: str,
                 xvals: np.ndarray,
                 yvals: np.ndarray,
                 channels: Optional[Union[Channel, List[Channel]]] = None,
                 meta: Optional[Dict[str, Any]] = None,
                 ignore_scaling: bool = False,
                 styles: Optional[Dict[str, Any]] = None):
        """Create new drawing object.

        Args:
            data_type: String representation of this drawing object.
            xvals: Series of horizontal coordinate that the object is drawn.
            yvals: Series of vertical coordinate that the object is drawn.
            channels: Pulse channel object bound to this drawing.
            meta: Meta data dictionary of the object.
            ignore_scaling: Set ``True`` to disable scaling.
            styles: Style keyword args of the object. This conforms to `matplotlib`.
        """
        if channels and isinstance(channels, Channel):
            channels = [channels]

        self.data_type = data_type
        self.xvals = np.array(xvals, dtype=object)
        self.yvals = np.array(yvals, dtype=object)
        self.channels = channels or []
        self.meta = meta or dict()
        self.ignore_scaling = ignore_scaling
        self.styles = styles or dict()

    @property
    def data_key(self):
        """Return unique hash of this object."""
        return str(hash((self.__class__.__name__,
                         self.data_type,
                         tuple(self.xvals),
                         tuple(self.yvals))))

    def __repr__(self):
        return "{}(type={}, key={})".format(self.__class__.__name__,
                                            self.data_type,
                                            self.data_key)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.data_key == other.data_key


class LineData(ElementaryData):
    """Drawing object to represent object appears as a line.

    This is the counterpart of `matplotlib.pyploy.plot`.
    """
    def __init__(self,
                 data_type: str,
                 xvals: Union[np.ndarray, List[types.Coordinate]],
                 yvals: Union[np.ndarray, List[types.Coordinate]],
                 fill: bool = False,
                 channels: Optional[Union[Channel, List[Channel]]] = None,
                 meta: Optional[Dict[str, Any]] = None,
                 ignore_scaling: bool = False,
                 styles: Optional[Dict[str, Any]] = None):
        """Create new drawing object.

        Args:
            data_type: String representation of this drawing object.
            channels: Pulse channel object bound to this drawing.
            xvals: Series of horizontal coordinate that the object is drawn.
            yvals: Series of vertical coordinate that the object is drawn.
            fill: Set ``True`` to fill the area under curve.
            meta: Meta data dictionary of the object.
            ignore_scaling: Set ``True`` to disable scaling.
            styles: Style keyword args of the object. This conforms to `matplotlib`.
        """
        self.fill = fill

        super().__init__(data_type=data_type,
                         xvals=xvals,
                         yvals=yvals,
                         channels=channels,
                         meta=meta,
                         ignore_scaling=ignore_scaling,
                         styles=styles)


class TextData(ElementaryData):
    """Drawing object to represent object appears as a text.

    This is the counterpart of `matplotlib.pyploy.text`.
    """
    def __init__(self,
                 data_type: str,
                 xvals: Union[np.ndarray, List[types.Coordinate]],
                 yvals: Union[np.ndarray, List[types.Coordinate]],
                 text: str,
                 latex: Optional[str] = None,
                 channels: Optional[Union[Channel, List[Channel]]] = None,
                 meta: Optional[Dict[str, Any]] = None,
                 ignore_scaling: bool = False,
                 styles: Optional[Dict[str, Any]] = None):
        """Create new drawing object.

        Args:
            data_type: String representation of this drawing object.
            channels: Pulse channel object bound to this drawing.
            xvals: Series of horizontal coordinate that the object is drawn.
            yvals: Series of vertical coordinate that the object is drawn.
            text: String to show in the canvas.
            latex: Latex representation of the text (if backend supports latex drawing).
            meta: Meta data dictionary of the object.
            ignore_scaling: Set ``True`` to disable scaling.
            styles: Style keyword args of the object. This conforms to `matplotlib`.
        """
        self.text = text
        self.latex = latex or ''

        super().__init__(data_type=data_type,
                         xvals=xvals,
                         yvals=yvals,
                         channels=channels,
                         meta=meta,
                         ignore_scaling=ignore_scaling,
                         styles=styles)
