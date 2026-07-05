# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import typing
from rustworkx.rustworkx import PyGraph, PyDiGraph

if typing.TYPE_CHECKING:
    from PIL.Image import Image

_S = typing.TypeVar("_S")
_T = typing.TypeVar("_T")

@typing.overload
def graphviz_draw(
    graph: PyDiGraph[_S, _T] | PyGraph[_S, _T],
    node_attr_fn: typing.Callable[[_S], dict[str, str]] | None = ...,
    edge_attr_fn: typing.Callable[[_T], dict[str, str]] | None = ...,
    graph_attr: dict[str, str] | None = ...,
    filename: None = ...,
    image_type: (
        typing.Literal[
            "canon",
            "cmap",
            "cmapx",
            "cmapx_np",
            "dia",
            "dot",
            "fig",
            "gd",
            "gd2",
            "gif",
            "hpgl",
            "imap",
            "imap_np",
            "ismap",
            "jpe",
            "jpeg",
            "jpg",
            "mif",
            "mp",
            "pcl",
            "pdf",
            "pic",
            "plain",
            "plain-ext",
            "png",
            "ps",
            "ps2",
            "svg",
            "svgz",
            "vml",
            "vmlzvrml",
            "vtx",
            "wbmp",
            "xdor",
            "xlib",
        ]
        | None
    ) = ...,
    method: typing.Literal["twopi", "neato", "circo", "fdp", "sfdp", "dot"] | None = ...,
) -> Image: ...
@typing.overload
def graphviz_draw(
    graph: PyDiGraph[_S, _T] | PyGraph[_S, _T],
    node_attr_fn: typing.Callable[[_S], dict[str, str]] | None = ...,
    edge_attr_fn: typing.Callable[[_T], dict[str, str]] | None = ...,
    graph_attr: dict[str, str] | None = ...,
    filename: None = ...,
    image_type: str | None = ...,
    method: str | None = ...,
) -> Image: ...
@typing.overload
def graphviz_draw(
    graph: PyDiGraph[_S, _T] | PyGraph[_S, _T],
    node_attr_fn: typing.Callable[[_S], dict[str, str]] | None = ...,
    edge_attr_fn: typing.Callable[[_T], dict[str, str]] | None = ...,
    graph_attr: dict[str, str] | None = ...,
    filename: str = ...,
    image_type: (
        typing.Literal[
            "canon",
            "cmap",
            "cmapx",
            "cmapx_np",
            "dia",
            "dot",
            "fig",
            "gd",
            "gd2",
            "gif",
            "hpgl",
            "imap",
            "imap_np",
            "ismap",
            "jpe",
            "jpeg",
            "jpg",
            "mif",
            "mp",
            "pcl",
            "pdf",
            "pic",
            "plain",
            "plain-ext",
            "png",
            "ps",
            "ps2",
            "svg",
            "svgz",
            "vml",
            "vmlzvrml",
            "vtx",
            "wbmp",
            "xdor",
            "xlib",
        ]
        | None
    ) = ...,
    method: typing.Literal["twopi", "neato", "circo", "fdp", "sfdp", "dot"] | None = ...,
) -> None: ...
@typing.overload
def graphviz_draw(
    graph: PyDiGraph[_S, _T] | PyGraph[_S, _T],
    node_attr_fn: typing.Callable[[_S], dict[str, str]] | None = ...,
    edge_attr_fn: typing.Callable[[_T], dict[str, str]] | None = ...,
    graph_attr: dict[str, str] | None = ...,
    filename: str = ...,
    image_type: str | None = ...,
    method: str | None = ...,
) -> None: ...
