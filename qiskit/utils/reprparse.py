# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
A parser Class for working with the recursively built representations
"""
from typing import Optional
from .reprbuild import is_valid_repr, split_repr, ReprBuildError

REBUILDER = "rebuild"


class ReprParser:
    """Class to parse, print and manipulate a recursive object representation

    Args:
        obj_repr (str): string representation of the dictionary produced
                        by calls to myClass.__repr__()

    Raises:
        ReprBuildError: if the representation is not a valid dictionary

    Additional Information:
    """

    def __init__(self, obj_repr, rebuilders: [Optional] = None):
        if isinstance(obj_repr, str):
            self._repr_str = obj_repr
            try:
                obj_repr = eval(obj_repr)  # pylint: disable=eval-used
            except Exception as error:
                raise ReprBuildError("ReprParser argument is invalid representation ") from error
        else:
            obj_repr = repr(obj_repr)
            self._repr_str = obj_repr

        if not is_valid_repr(obj_repr):
            raise ReprBuildError("ReprParser argument is invalid representation ")

        self._repr_str = obj_repr
        (self._summary, self._obj_defn) = split_repr(self._repr_str)
        self._name = self._summary.get("name", "")
        self._class_name = self._summary.get("class", "")
        self._is_builtin = self._summary.get("is_builtin", False)
        self._rebuilder_map = {}
        if rebuilders is not None:
            self.append_rebuilder(rebuilders)

    def __repr__(self):
        return f"ReprParse for {self._class_name}  {self._name}"

    def get(self, name):
        """Get item in the dictionary of type unknown
        Args:
            name (str): Item to be parsed
        Returns:
            str: name of class of item
        Raises:

        Additional Information:
        """
        return split_repr(self._obj_defn.get(name))

    def get_complex(self, name, default: [Optional] = None):
        """Get item in the dictionary of type complex
        Args:
            name (str): Item to be parsed
            default (class): return value if name is not found

        Returns:
            complex: Item value
        Raises:

        Additional Information:
        """
        complex_defn = split_repr(self._obj_defn.get(name))[1]
        if _builtin_repr(complex_defn) and complex_defn[1] == "complex":
            return complex(complex_defn[0])
        else:
            return default

    def get_dict(self, name, default: [Optional] = None):
        """Get item in the dictionary and return as a dictionary
        Args:
            name (str): Item to be parsed
            default (dict): Default value to return if item is not present in the dictionary
        Returns:
            dict: Item value
        Raises:

        Additional Information:
        """
        summary, item_defn = split_repr(self._obj_defn.get(name))
        if summary is not None and summary.get("class", "") in ("dict", "defaultdict"):
            return item_defn
        else:
            return default

    def get_float(self, name, default: [Optional] = None):
        """Get item in the dictionary of type complex
        Args:
            name (str): Item to be parsed
            default (class): return value if name is not found
        Returns:
            float: Item value
        Raises:

        Additional Information:
        """
        item_defn = split_repr(self._obj_defn.get(name))[1]
        if item_defn is not None and _builtin_repr(item_defn) and item_defn[1] == "complex":
            return float(item_defn[0])
        else:
            return default

    def get_int(self, name, default: [Optional] = None):
        """Get item in the dictionary of type complex
        Args:
            name (str): Item to be parsed
            default (class): return value if name is not found
        Returns:
            int: Item value
        Raises:

        Additional Information:
        """
        item_dict = split_repr(self._obj_defn.get(name, None))[1]
        if item_dict is not None and _builtin_repr(item_dict) and item_dict[1] == "int":
            return int(item_dict[0])
        return default

    def get_list(self, name, default: [Optional] = None):
        """Get item in the dictionary and return as a list
        Args:
            name (str): Item to be parsed
            default (list): Default value to return if item is not present in the dictionary
        Returns:
            list: Item value
        Raises:

        Additional Information:
        """
        item_list = self._obj_defn.get(name, None)
        if item_list is None or not isinstance(item_list, (list, set, tuple)):
            item_list = default

        return item_list

    def get_repr(self, name, default: [Optional] = None):
        """Return an representation stored in the parser
        Args:
            name (str): Item to be parsed
            default (class): return value if name is not found
        Returns:
            list:  a valid representation for name (if found)
        """
        obj_repr = self.get_list(name, None)
        if not is_valid_repr(obj_repr):
            obj_repr = default
        return obj_repr

    def get_set(self, name, default: [Optional] = None):
        """Get item in the dictionary and return as a list
        Args:
            name (str): Item to be parsed
            default (list): Default value to return if item is not present in the dictionary
        Returns:
            list: Item value
        Raises:

        Additional Information:
        """
        summary, item_defn = split_repr(self._obj_defn.get(name))
        if summary is not None and summary.get("class", "") == "set":
            return set(item_defn)
        else:
            return default

    def get_string(self, name, default: [Optional] = None):
        """Return a string element from the parsed representation
        Args:
            name (str): Item to be parsed
            default (class): return value if name is not found
        Returns:
            str:  the string value for name
        """
        new_string = self._obj_defn.get(name)
        if not isinstance(new_string, str) and is_valid_repr(new_string):
            new_string = split_repr(new_string)[1]
        if not isinstance(new_string, str):
            new_string = default
        return new_string

    def get_tuple(self, name, default: [Optional] = None):
        """Get item in the dictionary and return as a list
        Args:
            name (str): Item to be parsed
            default (list): Default value to return if item is not present in the dictionary
        Returns:
            list: Item value
        Raises:

        Additional Information:
        """
        summary, item_defn = split_repr(self._obj_defn.get(name))
        if summary is not None and summary.get("class", "") == "tuple":
            return tuple(item_defn)
        else:
            return default

    @property
    def obj_defn(self):
        """Return a string element from the parsed representation
        Args:

        Returns:
            dict:  the representation as a dictionary
        """
        return self._obj_defn

    @property
    def class_name(self):
        """Return the class of the object that generated the representation
        Args:

        Returns:
            str:  the class of the object
        """
        return self._class_name

    @property
    def name(self):
        """Return the name of the object that generated the representation
        Args:

        Returns:
            str:  the object name
        """
        return self._name

    @property
    def summary(self):
        """Return just the summary string from the representation
        Args:

        Returns:
            str:  the representation summary string
        """
        return self._summary.get("summary", "")

    def get_mapper(self, name):
        """Return the method registered by the add_class_mapper calls for the named class
        Args:
            name (str): Class name of the mapping method
        Returns:
            method:  The method to rebuild an instance of the object from a representation
        """
        return self._rebuilder_map.get(name, None)

    def get_parser(self, obj_repr):
        """Retrieve a parser for the object specified
        Args:
            obj_repr (Union[str,list]): Either the string name of the attribute or the list
                        representation of the attribute to build the ReprParser object for
        Returns:
            ReprParser: parser loaded with the requested attributes definition and current rebuild map
        Raises:
            ReprError: if the mapping is invalid
        """
        new_parser = None
        if not is_valid_repr(obj_repr):
            obj_repr = self.get_repr(obj_repr, None)
        if is_valid_repr(obj_repr):
            new_parser = ReprParser(obj_repr)
            if isinstance(new_parser, ReprParser):
                new_parser.append_rebuilder(self._rebuilder_map)
        return new_parser

    def append_rebuilder(self, rebuilder):
        """Register the method(s) used to rebuild from the recursive representation
        Args:
            rebuilder (Union[list, dict, class]):
                    A list of classes to be registered,
                    a dictionary of name,method pairs to be registered, or
                    a single class to be registered
        Returns:
        Raises:
            ReprBuildError: if the method(s) can not be found
        """
        if isinstance(rebuilder, list):
            for obj_class in rebuilder:
                if hasattr(obj_class, REBUILDER):
                    self._rebuilder_map[obj_class.__class_name__] = getattr(obj_class, REBUILDER)
                else:
                    raise ReprBuildError(
                        f"Rebuild method, {REBUILDER} not found in {obj_class.__class_name__}"
                    )
        elif isinstance(rebuilder, dict):
            for class_name, mapper in rebuilder.items():
                self._rebuilder_map[class_name] = mapper
        else:
            if hasattr(rebuilder, REBUILDER):
                self._rebuilder_map[rebuilder.__class_name__] = getattr(rebuilder, REBUILDER)
            else:
                raise ReprBuildError(
                    f"No rebuilder {REBUILDER} method in {rebuilder.__class_name__}"
                )

    def rebuild(self, name: [Optional] = None, obj_repr: [Optional] = None):
        """Build an instance of the specified object according to the representation
        Args:
            name (str): Optional class name as string. If none name from obj_repr will be used
            obj_repr(str): The representation of the object to be instantiated
        Returns:
            object:  The newly instantiated instance defined in the representation
        Raises:
            ReprBuildError: if the mapping is invalid
        """
        new_obj = None
        if obj_repr is None and name is not None:
            obj_repr = self.get_repr(name)
        else:
            obj_repr = self._repr_str

        if obj_repr is not None:
            if name is None:
                summary = split_repr(obj_repr)[0]
                if summary is not None:
                    name = summary.get("class", "")

            new_obj = self._rebuild_builtin(obj_repr)
            if new_obj is None:
                mapper = self._rebuilder_map.get(name, None)
                if mapper is None:
                    raise ReprBuildError(f"No {REBUILDER} method found for {name}")
                if not is_valid_repr(obj_repr):
                    raise ReprBuildError(f"Invalid representation for {name}")
                new_obj = mapper(obj_repr)
        return new_obj

    def _rebuild_builtin(self, obj_repr):
        summary, obj_dict = split_repr(obj_repr)
        if summary is None:
            new_attr = None
        elif isinstance(obj_dict, str):
            try:
                new_attr = eval(obj_dict)  # pylint: disable=eval-used
            except:  # pylint: disable=bare-except
                new_attr = obj_dict
        elif isinstance(obj_dict, (tuple, list, dict, set)):
            if summary.get("class", "") == "str":
                new_attr = obj_dict[0]
            elif summary.get("class", "") == "int":
                new_attr = int(obj_dict[0])
            elif summary.get("class", "") == "float":
                new_attr = float(obj_dict[0])
            elif summary.get("class", "") == "complex":
                new_attr = complex(obj_dict[0])
            elif summary.get("class", "") in ("list", "dict"):
                new_attr = obj_dict
            elif summary.get("class", "") == "tuple":
                new_attr = tuple(obj_dict)
            elif summary.get("class", "") == "set":
                new_attr = set(obj_dict)
            else:
                # TODO: Support numpy and sympy types as builtins
                new_attr = None
        else:
            new_attr = None
        return new_attr

    def format_repr(self, indent=""):
        """Return a user friendly version of the representation
        Args:
            indent (str):    Indentation level for the output
        Returns:
            str: Nicely formatted representation
        Raises:

        Additional Information:
        """
        if self._class_name in ("str", "int", "float", "complex"):
            return_str = f"{indent}{self._obj_defn}\n"
        elif self._class_name in ("tuple", "set", "list"):
            return_str = ""
            if self._name != "":
                return_str += f"{indent}{self._name} : {self._class_name}\n"
                indent += "    "
            return_str += _format_repr_list(self._obj_defn, indent=indent)
        elif self._class_name in ("defaultdict", "dict"):
            return_str = f"{indent}{self._name} : {self._class_name}\n"
            return_str += _format_repr_dict(self._obj_defn, indent=indent + "    ")
        else:
            return_str = f"{indent}{self._name} : {self._class_name}\n"
            return_str += _format_repr_dict(self._obj_defn, indent + "    ")
        return return_str

    def print(self, indent=""):
        """Print a user friendly version of the representation
        Args:
            indent (str):    Indentation level for the output
        Returns:
        Raises:

        Additional Information:
        """
        print(self.format_repr(indent=indent))


def print_repr(obj_repr, indent="    "):
    """Print the current object registration
    Args:
        obj_repr (Union[str,list]):  Recursive object representation
        indent (str): The starting indentation for this representation
    Returns:
    Raises:
        ReprBuildError: If argument is not a valid representation
    """
    print(format_repr(obj_repr, indent=indent))


def format_repr(obj_repr, indent="    "):
    """Print the current object registration
    Args:
        obj_repr (Union[str,list]):  Recursive object representation
        indent (str): The starting indentation for this representation
    Returns:
        str: Nice formatted representation
    Raises:
        ReprBuildError: If argument is not a valid representation
    """
    return ReprParser(obj_repr).format_repr(indent=indent)


def _format_repr_dict(obj_dict, indent="", name: Optional = ""):
    """print an element that is of type dict"""
    if is_valid_repr(obj_dict):
        return_str = format_repr(obj_dict, indent)
    elif isinstance(obj_dict, str):
        return_str = f"{indent}{obj_dict}\n"
    elif _builtin_repr(obj_dict):
        return_str = f"{indent}{name} : {obj_dict[0]} : {obj_dict[1]}\n"
    elif isinstance(obj_dict, dict):
        return_str = ""
        for cur_name, cur_obj in obj_dict.items():
            return_str += _format_repr_element(
                cur_obj,
                indent,
                name=cur_name,
                header=f"{indent}{cur_name}: {cur_obj.__class__.__name__}\n",
            )
    elif isinstance(obj_dict, (set, list, tuple)):
        return_str = ""
        for item in obj_dict:
            return_str += _format_repr_element(item, indent, name=name)
    elif obj_dict is not None:
        return_str = f"Type mismatch, expecting 'dict' got {type(obj_dict)}\n"
    return return_str


def _format_repr_element(cur_defn, indent="", header: Optional = None, name: Optional = ""):
    """print the details of a single element of the representation s"""
    return_str = ""
    if isinstance(cur_defn, (str, int, float, complex)):
        return_str = f"{indent}{name} : {cur_defn}\n"
    elif _builtin_repr(cur_defn):
        item_defn = split_repr(cur_defn)[1]
        return_str = f"{indent}{name} : {item_defn[1]}: {item_defn[0]}\n"
    elif is_valid_repr(cur_defn):
        if header is not None and not _builtin_repr(cur_defn):
            return_str += header
            indent += "    "
        return_str += format_repr(cur_defn, indent)
    elif isinstance(cur_defn, (tuple, set, list)):
        if header is not None:
            return_str += header
            indent += "    "
        return_str += _format_repr_list(cur_defn, indent)
    elif isinstance(cur_defn, dict):
        if header is not None:
            return_str += header
            indent += "    "
        return_str += _format_repr_dict(cur_defn, indent, name=name)
    else:
        return_str = format_repr(cur_defn, indent)
    return return_str


def _format_repr_list(obj_list, indent):
    """print an element that is of type list, set or tuple"""
    return_str = ""
    if is_valid_repr(obj_list):
        (summary, list_dict) = split_repr(obj_list)
        return_str += f"{indent}{summary.get('name','')} : {summary.get('class','Unknown')}\n"
        return_str += format_repr(list_dict, indent + "    ")
    elif isinstance(obj_list, (tuple, set, list)):
        for cur_obj in obj_list:
            return_str += _format_repr_element(cur_obj, indent)
    else:
        return_str = f"Type mismatch expecting 'list' go {type(cur_obj)}\n"
    return return_str


def _builtin_repr(list_dict):
    if is_valid_repr(list_dict):
        list_dict = split_repr(list_dict)[1]
    is_builtin = (
        isinstance(list_dict, tuple)
        and (len(list_dict) == 2)
        and isinstance(list_dict[0], str)
        and isinstance(list_dict[1], str)
    )
    return is_builtin
