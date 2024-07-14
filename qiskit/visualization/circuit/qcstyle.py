# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Matplotlib circuit visualization style."""

from __future__ import annotations

import json
import os
from typing import Any
from pathlib import Path
from warnings import warn

from qiskit import user_config


class StyleDict(dict):
    """A dictionary for matplotlib styles.

    Defines additional abbreviations for key accesses, such as allowing
    ``"ec"`` instead of writing ``"edgecolor"``.
    """

    VALID_FIELDS = {
        "name",
        "textcolor",
        "gatetextcolor",
        "subtextcolor",
        "linecolor",
        "creglinecolor",
        "gatefacecolor",
        "barrierfacecolor",
        "backgroundcolor",
        "edgecolor",
        "fontsize",
        "subfontsize",
        "showindex",
        "figwidth",
        "dpi",
        "margin",
        "creglinestyle",
        "displaytext",
        "displaycolor",
    }

    ABBREVIATIONS = {
        "tc": "textcolor",
        "gt": "gatetextcolor",
        "sc": "subtextcolor",
        "lc": "linecolor",
        "cc": "creglinecolor",
        "gc": "gatefacecolor",
        "bc": "barrierfacecolor",
        "bg": "backgroundcolor",
        "ec": "edgecolor",
        "fs": "fontsize",
        "sfs": "subfontsize",
        "index": "showindex",
        "cline": "creglinestyle",
        "disptex": "displaytext",
        "dispcol": "displaycolor",
    }

    def __setitem__(self, key: Any, value: Any) -> None:
        # allow using field abbreviations
        if key in self.ABBREVIATIONS:
            key = self.ABBREVIATIONS[key]

        if key not in self.VALID_FIELDS:
            warn(
                f"style option ({key}) is not supported",
                UserWarning,
                2,
            )
        return super().__setitem__(key, value)

    def __getitem__(self, key: Any) -> Any:
        # allow using field abbreviations
        if key in self.ABBREVIATIONS:
            key = self.ABBREVIATIONS[key]

        return super().__getitem__(key)

    def update(self, other):
        # the attributes "displaycolor" and "displaytext" are dictionaries
        # themselves, therefore we need to propagate the update down to them
        nested_attrs = {"displaycolor", "displaytext"}
        for attr in nested_attrs.intersection(other.keys()):
            if attr in self.keys():
                self[attr].update(other[attr])
            else:
                self[attr] = other[attr]

        super().update((key, value) for key, value in other.items() if key not in nested_attrs)


class DefaultStyle:
    """Creates a Default Style dictionary

    The style dict contains numerous options that define the style of the
    output circuit visualization. The style dict is used by the `mpl` or
    `latex` output. The options available in the style dict are defined below:

    Attributes:
        name (str): The name of the style. The name can be set to ``iqp``,
            ``iqp-dark``, ``textbook``, ``bw``, ``clifford``, or the name of a
            user-created json file. This overrides the setting in the user config
            file (usually ``~/.qiskit/settings.conf``).
        textcolor (str): the color code to use for all text not inside a gate.
        subtextcolor (str): the color code to use for subtext.
        linecolor (str): the color code to use for lines.
        creglinecolor (str): The color code to use for classical register lines.
        gatetextcolor (str): The color code to use for gate text.
        gatefacecolor (str): The color code to use for a gate if no color
            specified in the 'displaycolor' dict.
        barrierfacecolor (str): The color code to use for barriers.
        backgroundcolor (str): The color code to use for the background.
        edgecolor (str): The color code to use for gate edges when using the
            `bw` style.
        fontsize (int): The font size to use for text.
        subfontsize (int): The font size to use for subtext.
        showindex (bool): If set to True, show the index numbers at the top.
        figwidth (int): The maximum width (in inches) for the output figure.
            If set to -1, the maximum displayable width will be used.
        dpi (int): The DPI to use for the output image.
        margin (list): A list of margin values to adjust spacing around output
            image. Takes a list of 4 ints: [x left, x right, y bottom, y top].
        creglinestyle (str): The style of line to use for classical registers.
            Choices are ``solid``, ``doublet``, or any valid matplotlib
            `linestyle` kwarg value.
        displaytext (dict): a dictionary of the text to use for certain element
            types in the output visualization. These items allow the use of
            LaTeX formatting for gate names. The 'displaytext' dict can contain
            any number of elements. User created names and labels may be used as
            keys, which allow these to have Latex formatting.
        displaycolor (dict): the color codes to use for each circuit element in
            the form (gate_color, text_color). Colors can also be entered without
            the text color, such as 'u1': '#FA74A6', in which case the text color
            will always be `gatetextcolor`. The `displaycolor` dict can contain
            any number of elements. User names and labels may be used as keys,
            which allows for custom colors for user-created gates.
    """

    def __init__(self):
        default_style_dict = "iqp.json"
        path = Path(__file__).parent / "styles" / default_style_dict

        with open(path, "r") as infile:
            default_style = json.load(infile)

        # set shortcuts, such as "ec" for "edgecolor"
        self.style = StyleDict(**default_style)


def load_style(style: dict | str | None) -> tuple[StyleDict, float]:
    """Utility function to load style from json files.

    Args:
        style: Depending on the type, this acts differently:

            * If a string, it can specify a supported style name (such
              as "iqp" or "clifford"). It can also specify the name of
              a custom color scheme stored as JSON file. This JSON file
              _must_ specify a complete set of colors.
            * If a dictionary, it may specify the style name via a
              ``{"name": "<desired style>"}`` entry. If this is not given,
              the default style will be used. The remaining entries in the
              dictionary can be used to override certain specs.
              E.g. ``{"name": "iqp", "ec": "#FF0000"}`` will use the ``"iqp"``
              color scheme but set the edgecolor to red.

    Returns:
        A tuple containing the style as dictionary and the default font ratio.
    """

    # if the style is not given, try to load the configured default (if set),
    # or use the default style
    config = user_config.get_config()
    if style is None:
        if config:
            style = config.get("circuit_mpl_style", "default")
        else:
            style = "default"

    # determine the style name which could also be inside a dictionary, like
    # style={"name": "clifford", <other settings...>}
    if isinstance(style, dict):
        style_name = style.get("name", "default")
    elif isinstance(style, str):
        if style.endswith(".json"):
            style_name = style[:-5]
        else:
            style_name = style
    else:
        warn(
            f'Unsupported style parameter "{style}" of type {type(style)}. '
            "Will use the default style.",
            UserWarning,
            2,
        )
        style_name = "default"

    if style_name in ["iqp", "default"]:
        current_style = DefaultStyle().style
    else:
        # Search for file in 'styles' dir, then config_path, and finally the current directory
        style_name = style_name + ".json"
        style_paths = []

        default_path = Path(__file__).parent / "styles" / style_name
        style_paths.append(default_path)

        # check configured paths, if there are any
        if config:
            config_path = config.get("circuit_mpl_style_path", "")
            if config_path:
                for path in config_path:
                    style_paths.append(Path(path) / style_name)

        # check current directory
        cwd_path = Path("") / style_name
        style_paths.append(cwd_path)

        for path in style_paths:
            # expand ~ to the user directory and check if the file exists
            exp_user = path.expanduser()
            if os.path.isfile(exp_user):
                try:
                    with open(exp_user) as infile:
                        json_style = json.load(infile)

                    current_style = StyleDict(json_style)
                    break
                except json.JSONDecodeError as err:
                    warn(
                        f"Could not decode JSON in file '{path}': {str(err)}. "
                        "Will use default style.",
                        UserWarning,
                        2,
                    )
                    break
                except (OSError, FileNotFoundError):
                    warn(
                        f"Error loading JSON file '{path}'. Will use default style.",
                        UserWarning,
                        2,
                    )
                    break
        else:
            warn(
                f"Style JSON file '{style_name}' not found in any of these locations: "
                f"{', '.join(map(str, style_paths))}. "
                "Will use default style.",
                UserWarning,
                2,
            )
            current_style = DefaultStyle().style

    # if the style is a dictionary, update the defaults with the new values
    # this _needs_ to happen after loading by name to cover cases like
    #   style = {"name": "bw", "edgecolor": "#FF0000"}
    if isinstance(style, dict):
        current_style.update(style)

    # this is the default font ratio
    # if the font- or subfont-sizes are changed, the new size is based on this ratio
    def_font_ratio = 13 / 8

    return current_style, def_font_ratio
