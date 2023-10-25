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

"""mpl circuit visualization style."""

import json
import os
from warnings import warn


from qiskit import user_config


class DefaultStyle:
    """Creates a Default Style dictionary

    **Style Dict Details**

    The style dict contains numerous options that define the style of the
    output circuit visualization. The style dict is used by the `mpl` or
    `latex` output. The options available in the style dict are defined below:

    name (str): the name of the style. The name can be set to ``iqp``,
        ``iqp-dark``, ``textbook``, ``bw``, ``clifford``, or the name of a
        user-created json file. This overrides the setting in the user config
        file (usually ``~/.qiskit/settings.conf``).

    textcolor (str): the color code to use for all text not inside a gate.
        Defaults to ``#000000``

    subtextcolor (str): the color code to use for subtext. Defaults to
        ``#000000``

    linecolor (str): the color code to use for lines. Defaults to
        ``#000000``

    creglinecolor (str): the color code to use for classical register
        lines. Defaults to ``#778899``

    gatetextcolor (str): the color code to use for gate text. Defaults to
        ``#000000``

    gatefacecolor (str): the color code to use for a gate if no color
        specified in the 'displaycolor' dict. Defaults to ``#BB8BFF``

    barrierfacecolor (str): the color code to use for barriers. Defaults to
        ``#BDBDBD``

    backgroundcolor (str): the color code to use for the background.
        Defaults to ``#FFFFFF``

    edgecolor (str): the color code to use for gate edges when using the
        `bw` style. Defaults to ``#000000``.

    fontsize (int): the font size to use for text. Defaults to 13.

    subfontsize (int): the font size to use for subtext. Defaults to 8.

    showindex (bool): if set to True, show the index numbers at the top.
        Defaults to False.

    figwidth (int): the maximum width (in inches) for the output figure.
        If set to -1, the maximum displayable width will be used.
        Defaults to -1.

    dpi (int): the DPI to use for the output image. Defaults to 150.

    margin (list): a list of margin values to adjust spacing around output
        image. Takes a list of 4 ints: [x left, x right, y bottom, y top].
        Defaults to [2.0, 0.1, 0.1, 0.3].

    creglinestyle (str): The style of line to use for classical registers.
        Choices are ``solid``, ``doublet``, or any valid matplotlib
        `linestyle` kwarg value. Defaults to ``doublet``.

    displaytext (dict): a dictionary of the text to use for certain element
        types in the output visualization. These items allow the use of
        LaTeX formatting for gate names. The 'displaytext' dict can contain
        any number of elements. User created names and labels may be used as
        keys, which allow these to have Latex formatting. The default
        values are (`default.json`)::

            {
                'u1': 'U_1',
                'u2': 'U_2',
                'u3': 'U_3',
                'sdg': 'S^\\dagger',
                'sx': '\\sqrt{X}',
                'sxdg': '\\sqrt{X}^\\dagger',
                't': 'T',
                'tdg': 'T^\\dagger',
                'dcx': 'Dcx',
                'iswap': 'Iswap',
                'ms': 'MS',
                'rx': 'R_X',
                'ry': 'R_Y',
                'rz': 'R_Z',
                'rxx': 'R_{XX}',
                'ryy': 'R_{YY}',
                'rzx': 'R_{ZX}',
                'rzz': 'ZZ',
                'reset': '\\left|0\\right\\rangle',
                'initialize': '|\\psi\\rangle'
            }

    displaycolor (dict): the color codes to use for each circuit element in
        the form (gate_color, text_color). Colors can also be entered without
        the text color, such as 'u1': '#FA74A6', in which case the text color
        will always be `gatetextcolor`. The `displaycolor` dict can contain
        any number of elements. User names and labels may be used as keys,
        which allows for custom colors for user-created gates. The default
        values are (`default.json`)::

            {
                'u1': ('#FA74A6', '#000000'),
                'u2': ('#FA74A6', '#000000'),
                'u3': ('#FA74A6', '#000000'),
                'id': ('#05BAB6', '#000000'),
                'u': ('#BB8BFF', '#000000'),
                'p': ('#BB8BFF', '#000000'),
                'x': ('#05BAB6', '#000000'),
                'y': ('#05BAB6', '#000000'),
                'z': ('#05BAB6', '#000000'),
                'h': ('#6FA4FF', '#000000'),
                'cx': ('#6FA4FF', '#000000'),
                'ccx': ('#BB8BFF', '#000000'),
                'mcx': ('#BB8BFF', '#000000'),
                'mcx_gray': ('#BB8BFF', '#000000'),
                'cy': ('#6FA4FF', '#000000'),
                'cz': ('#6FA4FF', '#000000'),
                'swap': ('#6FA4FF', '#000000'),
                'cswap': ('#BB8BFF', '#000000'),
                'ccswap': ('#BB8BFF', '#000000'),
                'dcx': ('#6FA4FF', '#000000'),
                'cdcx': ('#BB8BFF', '#000000'),
                'ccdcx': ('#BB8BFF', '#000000'),
                'iswap': ('#6FA4FF', '#000000'),
                's': ('#6FA4FF', '#000000'),
                'sdg': ('#6FA4FF', '#000000'),
                't': ('#BB8BFF', '#000000'),
                'tdg': ('#BB8BFF', '#000000'),
                'sx': ('#6FA4FF', '#000000'),
                'sxdg': ('#6FA4FF', '#000000')
                'r': ('#BB8BFF', '#000000'),
                'rx': ('#BB8BFF', '#000000'),
                'ry': ('#BB8BFF', '#000000'),
                'rz': ('#BB8BFF', '#000000'),
                'rxx': ('#BB8BFF', '#000000'),
                'ryy': ('#BB8BFF', '#000000'),
                'rzx': ('#BB8BFF', '#000000'),
                'reset': ('#000000', '#FFFFFF'),
                'target': ('#FFFFFF', '#FFFFFF'),
                'measure': ('#000000', '#FFFFFF'),
            }

    """

    def __init__(self):
        colors = {
            "### Default Colors": "Default Colors",
            "basis": "#FA74A6",  # Red
            "clifford": "#6FA4FF",  # Light Blue
            "pauli": "#05BAB6",  # Green
            "def_other": "#BB8BFF",  # Purple
            "### IQP Colors": "IQP Colors",
            "classical": "#002D9C",  # Dark Blue
            "phase": "#33B1FF",  # Cyan
            "hadamard": "#FA4D56",  # Light Red
            "non_unitary": "#A8A8A8",  # Medium Gray
            "iqp_other": "#9F1853",  # Dark Red
            "### B/W": "B/W",
            "black": "#000000",
            "white": "#FFFFFF",
            "dark_gray": "#778899",
            "light_gray": "#BDBDBD",
        }
        self.style = {
            "name": "default",
            "tc": colors["black"],  # Non-gate Text Color
            "gt": colors["black"],  # Gate Text Color
            "sc": colors["black"],  # Gate Subtext Color
            "lc": colors["black"],  # Line Color
            "cc": colors["dark_gray"],  # creg Line Color
            "gc": colors["def_other"],  # Default Gate Color
            "bc": colors["light_gray"],  # Barrier Color
            "bg": colors["white"],  # Background Color
            "ec": None,  # Edge Color (B/W only)
            "fs": 13,  # Gate Font Size
            "sfs": 8,  # Subtext Font Size
            "index": False,
            "figwidth": -1,
            "dpi": 150,
            "margin": [2.0, 0.1, 0.1, 0.3],
            "cline": "doublet",
            "disptex": {
                "u1": "U_1",
                "u2": "U_2",
                "u3": "U_3",
                "id": "I",
                "sdg": "S^\\dagger",
                "sx": "\\sqrt{X}",
                "sxdg": "\\sqrt{X}^\\dagger",
                "tdg": "T^\\dagger",
                "ms": "MS",
                "rx": "R_X",
                "ry": "R_Y",
                "rz": "R_Z",
                "rxx": "R_{XX}",
                "ryy": "R_{YY}",
                "rzx": "R_{ZX}",
                "rzz": "ZZ",
                "reset": "\\left|0\\right\\rangle",
                "initialize": "$|\\psi\\rangle$",
            },
            "dispcol": {
                "u1": (colors["basis"], colors["black"]),
                "u2": (colors["basis"], colors["black"]),
                "u3": (colors["basis"], colors["black"]),
                "u": (colors["def_other"], colors["black"]),
                "p": (colors["def_other"], colors["black"]),
                "id": (colors["pauli"], colors["black"]),
                "x": (colors["pauli"], colors["black"]),
                "y": (colors["pauli"], colors["black"]),
                "z": (colors["pauli"], colors["black"]),
                "h": (colors["clifford"], colors["black"]),
                "cx": (colors["clifford"], colors["black"]),
                "ccx": (colors["def_other"], colors["black"]),
                "mcx": (colors["def_other"], colors["black"]),
                "mcx_gray": (colors["def_other"], colors["black"]),
                "cy": (colors["clifford"], colors["black"]),
                "cz": (colors["clifford"], colors["black"]),
                "swap": (colors["clifford"], colors["black"]),
                "cswap": (colors["def_other"], colors["black"]),
                "ccswap": (colors["def_other"], colors["black"]),
                "dcx": (colors["clifford"], colors["black"]),
                "cdcx": (colors["def_other"], colors["black"]),
                "ccdcx": (colors["def_other"], colors["black"]),
                "iswap": (colors["clifford"], colors["black"]),
                "s": (colors["clifford"], colors["black"]),
                "sdg": (colors["clifford"], colors["black"]),
                "t": (colors["def_other"], colors["black"]),
                "tdg": (colors["def_other"], colors["black"]),
                "sx": (colors["clifford"], colors["black"]),
                "sxdg": (colors["clifford"], colors["black"]),
                "r": (colors["def_other"], colors["black"]),
                "rx": (colors["def_other"], colors["black"]),
                "ry": (colors["def_other"], colors["black"]),
                "rz": (colors["def_other"], colors["black"]),
                "rxx": (colors["def_other"], colors["black"]),
                "ryy": (colors["def_other"], colors["black"]),
                "rzx": (colors["def_other"], colors["black"]),
                "reset": (colors["black"], colors["white"]),
                "target": (colors["white"], colors["white"]),
                "measure": (colors["black"], colors["white"]),
            },
        }


def load_style(style):
    """Utility function to load style from json files and call set_style."""
    current_style = DefaultStyle().style
    style_name = "default"
    def_font_ratio = current_style["fs"] / current_style["sfs"]

    config = user_config.get_config()
    if style is None:
        if config:
            style = config.get("circuit_mpl_style", "default")
        else:
            style = "default"

    if style == "default":
        warn(
            'The default matplotlib drawer scheme will be changed to "iqp" in a following release. '
            'To silence this warning, specify the current default explicitly as style="clifford", '
            'or the new default as style="iqp".',
            category=FutureWarning,
            stacklevel=2,
        )

    if style is False:
        style_name = "bw"
    elif isinstance(style, dict) and "name" in style:
        style_name = style["name"]
    elif isinstance(style, str):
        style_name = style
    elif not isinstance(style, (str, dict)):
        warn(
            f"style parameter '{style}' must be a str or a dictionary. Will use default style.",
            UserWarning,
            2,
        )
    if style_name.endswith(".json"):
        style_name = style_name[:-5]

    # Alias IQP<->IQX, where we hardcode both "iqx" and "iqx-dark" instead of only replacing the
    # first three letters, in case users have a custom name starting with "iqx" too.
    # Also replace "default" with the new name "clifford".
    replacements = {"iqx": "iqp", "iqx-dark": "iqp-dark", "default": "clifford"}
    if style_name in replacements.keys():
        if style_name[:3] == "iqx":
            warn(
                'The "iqx" and "iqx-dark" matplotlib drawer schemes are pending deprecation and will '
                'be deprecated in a future release. Instead, use "iqp" and "iqp-dark".',
                category=PendingDeprecationWarning,
                stacklevel=2,
            )

        style_name = replacements[style_name]

    # Search for file in 'styles' dir, then config_path, and finally 'cwd'
    style_path = []
    if style_name != "clifford":
        style_name = style_name + ".json"
        spath = os.path.dirname(os.path.abspath(__file__))
        style_path.append(os.path.join(spath, "styles", style_name))
        if config:
            config_path = config.get("circuit_mpl_style_path", "")
            if config_path:
                for path in config_path:
                    style_path.append(os.path.normpath(os.path.join(path, style_name)))
        style_path.append(os.path.normpath(os.path.join("", style_name)))

        for path in style_path:
            exp_user = os.path.expanduser(path)
            if os.path.isfile(exp_user):
                try:
                    with open(exp_user) as infile:
                        json_style = json.load(infile)
                    set_style(current_style, json_style)
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
                f"{', '.join(style_path)}. "
                "Will use default style.",
                UserWarning,
                2,
            )

    if isinstance(style, dict):
        set_style(current_style, style)

    return current_style, def_font_ratio


def set_style(current_style, new_style):
    """Utility function to take elements in new_style and
    write them into current_style.
    """
    valid_fields = {
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

    current_style.update(new_style)
    current_style["tc"] = current_style.get("textcolor", current_style["tc"])
    current_style["gt"] = current_style.get("gatetextcolor", current_style["gt"])
    current_style["sc"] = current_style.get("subtextcolor", current_style["sc"])
    current_style["lc"] = current_style.get("linecolor", current_style["lc"])
    current_style["cc"] = current_style.get("creglinecolor", current_style["cc"])
    current_style["gc"] = current_style.get("gatefacecolor", current_style["gc"])
    current_style["bc"] = current_style.get("barrierfacecolor", current_style["bc"])
    current_style["bg"] = current_style.get("backgroundcolor", current_style["bg"])
    current_style["ec"] = current_style.get("edgecolor", current_style["ec"])
    current_style["fs"] = current_style.get("fontsize", current_style["fs"])
    current_style["sfs"] = current_style.get("subfontsize", current_style["sfs"])
    current_style["index"] = current_style.get("showindex", current_style["index"])
    current_style["cline"] = current_style.get("creglinestyle", current_style["cline"])
    current_style["disptex"] = {**current_style["disptex"], **new_style.get("displaytext", {})}
    current_style["dispcol"] = {**current_style["dispcol"], **new_style.get("displaycolor", {})}

    unsupported_keys = set(new_style) - valid_fields
    if unsupported_keys:
        warn(
            f"style option/s ({', '.join(unsupported_keys)}) is/are not supported",
            UserWarning,
            2,
        )
