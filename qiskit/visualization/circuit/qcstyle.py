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
        if key in self.ABBREVIATIONS.keys():
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
        if key in self.ABBREVIATIONS.keys():
            key = self.ABBREVIATIONS[key]

        return super().__getitem__(key)

    def update(self, other):
        # the attributes "displaycolor" and "displaytext" are dictionaries
        # themselves, therefore we need to propagate the update down to them
        nested_attrs = {"displaycolor", "displaytext"}
        for attr in nested_attrs.intersection(other.keys()):
            if attr in self.keys():
                self[attr].update(other.pop(attr))
            else:
                self[attr] = other.pop(attr)

        super().update(other)


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
    """Utility function to load style from json files."""

    # figure out the type of input argument and determine the style name
    config = user_config.get_config()
    if style is None:
        if config:
            style = config.get("circuit_mpl_style", "default")
        else:
            style = "default"
    elif isinstance(style, dict):
        if "name" in style:
            style_name = style["name"]
    elif isinstance(style, str):
        style_name = style
        if style_name.endswith(".json"):
            style_name = style_name[:-5]
    else:
        warn(
            f'Unsupported style parameter "{style}" of type {type(style)}. '
            'Will use the default "iqp" style.',
            UserWarning,
            2,
        )

    # load the default style, which will be used to provide default arguments
    # if some are not provided
    current_style = DefaultStyle().style

    # if the style is a dictionary, update the defaults with the new values
    if isinstance(style, dict):
        current_style.update(style)

    # if it is the default style, we have already loaded it
    elif style in ["iqp", "default"]:
        pass

    # otherwise try to load it
    else:
        # Search for file in 'styles' dir, then config_path, and finally 'cwd'
        style_name = style_name + ".json"
        style_paths = []

        # spath = os.path.dirname(os.path.abspath(__file__))
        # style_path.append(os.path.join(spath, "styles", style_name))
        # check the default path
        default_path = Path(__file__).parent / "styles" / style_name
        style_paths.append(default_path)

        # check configured paths, if there are any
        if config:
            config_path = config.get("circuit_mpl_style_path", "")
            if config_path:
                for path in config_path:
                    path_ = Path(config_path) / style_name
                    style_paths.append(path_)

        # check current working directory
        cwd_path = Path.cwd()
        style_paths.append(cwd_path)
        # style_path.append(os.path.normpath(os.path.join("", style_name)))

        for path in style_paths:
            # expand ~ to the user directory and check if the file exists
            exp_user = path.expanduser()
            if os.path.isfile(exp_user):
                try:
                    with open(exp_user) as infile:
                        json_style = json.load(infile)

                    current_style.update(json_style)
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

    def_font_ratio = current_style["fs"] / current_style["sfs"]
    return current_style, def_font_ratio
