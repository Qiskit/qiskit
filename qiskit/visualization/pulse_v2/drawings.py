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
Drawing objects for pulse drawer.

Drawing objects play two important roles:
    - Allowing unittests of visualization module. Usually it is hard for image files to be tested.
    - Removing program parser from each plotter interface. We can easily add new plotter.

This module is based on the structure of matplotlib as it is the primary plotter
of the pulse drawer. However this interface is agnostic to the actual plotter.

Design concept
~~~~~~~~~~~~~~
When we think about dynamically updating drawings, it will be most efficient to
update only the changed properties of drawings rather than regenerating entirely from scratch.
Thus the core :py:class:`qiskit.visualization.pulse_v2.core.DrawerCanvas` generates
all possible drawings in the beginning and then the canvas instance manages
visibility of each drawing according to the end-user request.

Data key
~~~~~~~~
In the abstract class ``ElementaryData`` common attributes to represent a drawing are
specified. In addition, drawings have the `data_key` property that returns an
unique hash of the object for comparison.
This key is generated from a data type and the location of the drawing in the canvas.
See py:mod:`qiskit.visualization.pulse_v2.types` for detail on the data type.
If a data key cannot distinguish two independent objects, you need to add a new data type.
The data key may be used in the plotter interface to identify the object.

Drawing objects
~~~~~~~~~~~~~~~
To support not only `matplotlib` but also multiple plotters, those drawings should be
universal and designed without strong dependency on modules in `matplotlib`.
This means drawings that represent primitive geometries are preferred.
It should be noted that there will be no unittest for each plotter API, which takes
drawings and outputs image data, we should avoid adding a complicated geometry
that has a context of the pulse program.

For example, a pulse envelope is complex valued number array and may be represented
by two lines with different colors associated with the real and the imaginary component.
We can use two line-type objects rather than defining a new drawing that takes
complex value. As many plotters don't support an API that visualizes complex-valued
data arrays, if we introduced such a drawing and wrote a custom wrapper function
on top of the existing API, it could be difficult to prevent bugs with the CI tools
due to lack of the effective unittest.
"""
from abc import ABC
from enum import Enum
from typing import Dict, Any, Optional, Union, List

import numpy as np

from qiskit.pulse.channels import Channel
from qiskit.visualization.pulse_v2 import types
from qiskit.visualization.exceptions import VisualizationError


class ElementaryData(ABC):
    """Base class of the pulse visualization interface."""

    __hash__ = None

    def __init__(
        self,
        data_type: Union[str, Enum],
        xvals: np.ndarray,
        yvals: np.ndarray,
        channels: Optional[Union[Channel, List[Channel]]] = None,
        meta: Optional[Dict[str, Any]] = None,
        ignore_scaling: bool = False,
        styles: Optional[Dict[str, Any]] = None,
    ):
        """Create new drawing.

        Args:
            data_type: String representation of this drawing.
            xvals: Series of horizontal coordinate that the object is drawn.
            yvals: Series of vertical coordinate that the object is drawn.
            channels: Pulse channel object bound to this drawing.
            meta: Meta data dictionary of the object.
            ignore_scaling: Set ``True`` to disable scaling.
            styles: Style keyword args of the object. This conforms to `matplotlib`.
        """
        if channels and isinstance(channels, Channel):
            channels = [channels]

        if isinstance(data_type, Enum):
            data_type = data_type.value

        self.data_type = str(data_type)
        self.xvals = np.array(xvals, dtype=object)
        self.yvals = np.array(yvals, dtype=object)
        self.channels = channels or []
        self.meta = meta or dict()
        self.ignore_scaling = ignore_scaling
        self.styles = styles or dict()

    @property
    def data_key(self):
        """Return unique hash of this object."""
        return str(
            hash((self.__class__.__name__, self.data_type, tuple(self.xvals), tuple(self.yvals)))
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(type={self.data_type}, key={self.data_key})"

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.data_key == other.data_key


class LineData(ElementaryData):
    """Drawing object to represent object appears as a line.

    This is the counterpart of `matplotlib.pyplot.plot`.
    """

    def __init__(
        self,
        data_type: Union[str, Enum],
        xvals: Union[np.ndarray, List[types.Coordinate]],
        yvals: Union[np.ndarray, List[types.Coordinate]],
        fill: bool = False,
        channels: Optional[Union[Channel, List[Channel]]] = None,
        meta: Optional[Dict[str, Any]] = None,
        ignore_scaling: bool = False,
        styles: Optional[Dict[str, Any]] = None,
    ):
        """Create new drawing.

        Args:
            data_type: String representation of this drawing.
            channels: Pulse channel object bound to this drawing.
            xvals: Series of horizontal coordinate that the object is drawn.
            yvals: Series of vertical coordinate that the object is drawn.
            fill: Set ``True`` to fill the area under curve.
            meta: Meta data dictionary of the object.
            ignore_scaling: Set ``True`` to disable scaling.
            styles: Style keyword args of the object. This conforms to `matplotlib`.
        """
        self.fill = fill

        super().__init__(
            data_type=data_type,
            xvals=xvals,
            yvals=yvals,
            channels=channels,
            meta=meta,
            ignore_scaling=ignore_scaling,
            styles=styles,
        )


class TextData(ElementaryData):
    """Drawing object to represent object appears as a text.

    This is the counterpart of `matplotlib.pyplot.text`.
    """

    def __init__(
        self,
        data_type: Union[str, Enum],
        xvals: Union[np.ndarray, List[types.Coordinate]],
        yvals: Union[np.ndarray, List[types.Coordinate]],
        text: str,
        latex: Optional[str] = None,
        channels: Optional[Union[Channel, List[Channel]]] = None,
        meta: Optional[Dict[str, Any]] = None,
        ignore_scaling: bool = False,
        styles: Optional[Dict[str, Any]] = None,
    ):
        """Create new drawing.

        Args:
            data_type: String representation of this drawing.
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
        self.latex = latex or ""

        super().__init__(
            data_type=data_type,
            xvals=xvals,
            yvals=yvals,
            channels=channels,
            meta=meta,
            ignore_scaling=ignore_scaling,
            styles=styles,
        )


class BoxData(ElementaryData):
    """Drawing object that represents box shape.

    This is the counterpart of `matplotlib.patches.Rectangle`.
    """

    def __init__(
        self,
        data_type: Union[str, Enum],
        xvals: Union[np.ndarray, List[types.Coordinate]],
        yvals: Union[np.ndarray, List[types.Coordinate]],
        channels: Optional[Union[Channel, List[Channel]]] = None,
        meta: Dict[str, Any] = None,
        ignore_scaling: bool = False,
        styles: Dict[str, Any] = None,
    ):
        """Create new box.

        Args:
            data_type: String representation of this drawing.
            xvals: Left and right coordinate that the object is drawn.
            yvals: Top and bottom coordinate that the object is drawn.
            channels: Pulse channel object bound to this drawing.
            meta: Meta data dictionary of the object.
            ignore_scaling: Set ``True`` to disable scaling.
            styles: Style keyword args of the object. This conforms to `matplotlib`.

        Raises:
            VisualizationError: When number of data points are not equals to 2.
        """
        if len(xvals) != 2 or len(yvals) != 2:
            raise VisualizationError("Length of data points are not equals to 2.")

        super().__init__(
            data_type=data_type,
            xvals=xvals,
            yvals=yvals,
            channels=channels,
            meta=meta,
            ignore_scaling=ignore_scaling,
            styles=styles,
        )
