# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import typing

import typing_extensions

from rustworkx.rustworkx import PyGraph, PyDiGraph

if typing.TYPE_CHECKING:
    from matplotlib.axes import Axes  # type: ignore
    from matplotlib.figure import Figure  # type: ignore
    from matplotlib.colors import Colormap  # type: ignore

_S = typing.TypeVar("_S")
_T = typing.TypeVar("_T")

class _DrawKwargs(typing.TypedDict, typing.Generic[_S, _T], total=False):
    arrowstyle: str
    arrow_size: int
    node_list: list[int]
    edge_list: list[int]
    node_size: int | list[int]
    node_color: (
        str
        | tuple[float, float, float]
        | tuple[float, float, float, float]
        | list[str]
        | list[tuple[float, float, float]]
        | list[tuple[float, float, float, float]]
    )
    node_shape: str
    alpha: float
    cmap: Colormap
    vmin: float
    vmax: float
    linewidths: float | list[float]
    edge_color: (
        str
        | tuple[float, float, float]
        | tuple[float, float, float, float]
        | list[str]
        | list[tuple[float, float, float]]
        | list[tuple[float, float, float, float]]
    )
    edge_cmap: Colormap
    edge_vmin: float
    edge_vmax: float
    style: str
    labels: typing.Callable[[_S], str]
    edge_labels: typing.Callable[[_T], str]
    font_size: int
    font_color: str
    font_weight: str
    font_family: str
    label: str
    connectionstyle: str

def mpl_draw(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    pos: typing.Mapping[int, tuple[float, float]] | None = ...,
    ax: Axes | None = ...,
    arrows: bool = ...,
    with_labels: bool = ...,
    **kwds: typing_extensions.Unpack[_DrawKwargs[_S, _T]],
) -> Figure | None: ...
