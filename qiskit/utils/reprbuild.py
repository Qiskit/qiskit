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
Build a representation such that eval(repr(A)) is a dictionary which can be
unambiguous enough that we can build a class method such that cls(A).build_repr(eval(A))
is equivalent to A for most reasonable definitions of equivalence.
"""
from typing import Optional
import numpy as np

REPRATTRIBUTES = "_repr_attrs"
MAXRECURSION = 200


class ReprBuildError(Exception):
    """Base class for errors raised while processing a representation."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(" ".join(message))
        self.message = " ".join(message)

    def __str__(self):
        """Return the message."""
        if not isinstance(self.message, str):
            return repr(self.message)
        else:
            return self.message


def _get_summary(source):
    """Create the summary information about the input object for the recursive representation."""
    _summary = f"class: {source.__class__.__name__}"
    _src_name = getattr(source, "name", None)
    if _src_name is not None:
        _summary += f",name: {_src_name}"

    return _summary


def split_repr(obj_repr):
    """Parse off and return the summary and embedded definition of the input representation
    Args:
        obj_repr Union[str,list]: Object representation to be parsed
    Returns:
        dict:  Summary with the class and name elements mapped
        Union[dict,tuple,list]:  if non-iterable built-in type return tuple(type,repr)
                                 if recursively built representation returns attribute definition
                                 else return iterable type of representation
    Raises:

    Additional Information:
        The summary is a dictionary with:
            class: The name of the original class
            name:   The name, if it was assigned, of the original object
            is_builtin: A boolean indicating if the class is to be treated at a builtin
    """
    (summary, repr_defn) = (None, None)
    if isinstance(obj_repr, str):
        try:
            obj_repr = eval(obj_repr)  # pylint: disable=eval-used
        except:  # pylint: disable=bare-except
            obj_repr = None

    if isinstance(obj_repr, list) and len(obj_repr) == 2:
        if isinstance(obj_repr[0], str):
            summary_list = obj_repr[0].split(",")
            classname = summary_list[0]
            if classname.startswith("<class '"):
                summary = {}
                summary["class"] = classname
            elif classname.startswith("class: "):
                summary = {}
                summary["class"] = classname.split("class: ")[1]

            if summary is not None:
                repr_defn = obj_repr[1]
                summary["name"] = ""
                if len(summary_list) > 1:
                    name_list = summary_list[1].split("name: ")
                    if len(name_list) > 1 and name_list[1] is not None:
                        summary["name"] = name_list[1]
                summary["is_builtin"] = (
                    isinstance(repr_defn, tuple)
                    and (len(repr_defn) == 2)
                    and isinstance(repr_defn[0], str)
                    and isinstance(repr_defn[1], str)
                )
            if len(summary_list) < 3:
                summary["summary"] = f"<{summary['class']} '{summary['name']}'>"
            else:
                custom_summary = ",".join(summary_list[2:])
                summary["summary"] = f"<{summary['class']} '{summary['name']}' {custom_summary}>"

    return summary, repr_defn


def is_valid_repr(obj_repr):
    """Determine if the input is a valid representation.
    Args:
        obj_repr (Union[list,str]): Object to be validated as representation
    Returns:
        Boolean: True if input can be converted to a valid representation
                 False if not a valid representation
    Raises:

    Additional Information:
    """
    return split_repr(obj_repr)[0] is not None


def build_attribute_defn(
    source,
    attribute,
    depth=-1,
    deepdive=False,
    recursion=0,
):
    """
    Args:
        source (Unknown)    : Source object containing the attribute to be defined
        attribute (Unknown) : Name of the source's attribute to define
        depth (int)         : if == 0 : return object summary only as string
                              if != 1 : return a recursively build representation for the attribute
                                       a starting depth -1 will fully expand all attributes
        deepdive (boolean)  : if True append attributes returned from dir() to list
        recursion (int)     : prevent unlimited recursion in case of circular references
    Raises:
        ValueError: invalid value
    Returns:
        list: [summary, dictionary] representation of attribute
    Raises:
    Additional Information:
            attr_defn[0]: Summary of the attribute
            attr_defn[1]: Representation of a dictionary for members of the attribute
                                listed in _repr_attrs for its class
    """
    if attribute is None:
        attr = source
    else:
        attr = getattr(source, attribute, None)
    attr_defn = [_get_summary(attr), None]
    if attr is None:
        attr_defn = None
    elif isinstance(attr, str):
        attr_defn = attr
    elif isinstance(attr, (int, float, complex)):
        attr_defn[1] = (repr(attr), attr.__class__.__name__)
    elif isinstance(attr, (np.integer, np.floating, np.complexfloating, np.ndarray)):
        attr_defn[1] = (repr(attr), attr.__class__.__name__)
    elif depth != 0:
        if hasattr(attr, REPRATTRIBUTES):
            attr_defn = build_object_defn(
                attr,
                getattr(attr, REPRATTRIBUTES),
                depth=depth - 1,
                deepdive=deepdive,
                recursion=recursion + 1,
            )
        elif isinstance(attr, (list, tuple, set)):
            repr_list = []
            if len(attr) > 0:
                for cur_attr in attr:
                    if cur_attr is None:
                        pass
                    elif hasattr(cur_attr, REPRATTRIBUTES):
                        cur_repr = build_object_defn(
                            cur_attr,
                            getattr(cur_attr, REPRATTRIBUTES),
                            depth=depth - 1,
                            deepdive=deepdive,
                            recursion=recursion + 1,
                        )
                    elif isinstance(cur_attr, (list, tuple, set, dict)):
                        cur_repr = build_object_defn(
                            cur_attr,
                            None,
                            depth=depth - 1,
                            deepdive=deepdive,
                            recursion=recursion + 1,
                        )
                    else:
                        cur_repr = repr(cur_attr)
                    repr_list.append(cur_repr)
            if isinstance(attr, tuple):
                repr_list = tuple(repr_list)
            elif isinstance(attr, set):
                repr_list = set(repr_list)
            attr_defn[1] = repr_list
        elif isinstance(attr, dict):
            repr_list = {}
            if len(attr) > 0:
                repr_list = {}
                for cur_key, cur_attr in attr.items():
                    if cur_attr is None:
                        pass
                    elif hasattr(cur_attr, REPRATTRIBUTES):
                        cur_repr = build_object_defn(
                            cur_attr,
                            getattr(cur_attr, REPRATTRIBUTES),
                            depth=depth - 1,
                            deepdive=deepdive,
                            recursion=recursion + 1,
                        )
                    elif isinstance(cur_attr, (list, tuple, set, dict)):
                        list_dict = build_attribute_defn(
                            cur_attr,
                            None,
                            depth=depth - 1,
                            deepdive=deepdive,
                            recursion=recursion + 1,
                        )
                        cur_repr = list_dict
                    else:
                        cur_repr = repr(cur_attr)
                    repr_list[cur_key] = cur_repr
            attr_defn[1] = repr_list
        else:
            attr_defn[1] = repr(attr)

    return attr_defn


def build_object_defn(
    source,
    attr_list=None,
    depth=-1,
    deepdive=False,
    recursion=0,
):
    """Create a list with the summary string and recursive representation for the object
    Args:
        source (Unknown)    : Object to be built into a dictionary
        attr_list  (list)   : List of the object's attributes to include
        depth (int)         : if == 0 : return object summary only as string
                              if != 1 : return list[ object summary, dict{ attribute :,repr to depth -1]
                                       a starting depth -1 will fully expand all attributes
        deepdive (boolean)  : if True append attributes returned from dir() to list
        recursion (int)     : number of levels of recursion allowable for this representation
    Returns:
        list: summary of source object in element [0]
              object definition in element[1]
    Raises:
        ReprBuildError: if a valid list of attributes is not found
    Additional Information:
    """
    attr_depths = {}
    if isinstance(attr_list, dict):
        attr_depths = attr_list
    elif isinstance(attr_list, list):
        for cur_member in attr_list:
            attr_depths[cur_member] = depth
    elif attr_list is not None:
        raise ReprBuildError("member_list not of type list or dict")
    if deepdive:
        __obj_dict__ = getattr(source, "__dict__", None)
        if __obj_dict__ is not None:
            for i in __obj_dict__:
                attr_depths[i] = depth
        for i in dir(source):
            attr_depths[i] = depth

    if recursion > MAXRECURSION:
        member_dict = f"<Recursion limit of {MAXRECURSION} exceeded>"
    else:
        member_dict = {}
        for cur_member, cur_depth in attr_depths.items():
            cur_defn = build_attribute_defn(
                source,
                cur_member,
                depth=cur_depth,
                deepdive=deepdive,
                recursion=recursion + 1,
            )
            if cur_defn is not None:
                member_dict[cur_member] = cur_defn

    if is_valid_repr(member_dict):
        return member_dict
    else:
        return [_get_summary(source), member_dict]


def build_repr(source, summary: [Optional] = None, **kwargs):
    """Create a recursive representation for the source object
    Args:
        source (Type)    : Object to be built into a dictionary
        summary (str)    : Additional information to add to the summary
        **kwargs (params)   :
            attr_list  (list)   : List of the object's attributes to include
                                  If no supplied the _repr_attrs attribute of the source will determine
                                  the representation to be built
            depth (int)         : if == 0 : return representation without recursion
                                  if != 1 : recursively build representations for included attributes,
                                            decrementing depth at each level of recursion
                                            A starting depth -1 will fully expand all attributes
            deepdive (boolean)  : if True append attributes returned from dir() to the representation
    Returns:
        str: string representation of the representation definition
    Raises:
    Additional Information:
    """
    obj_defn = build_object_defn(source, **kwargs)
    if not is_valid_repr(obj_defn):
        obj_defn = [_get_summary(source), obj_defn]
    if summary is not None and isinstance(summary, str):
        obj_defn[0] = obj_defn[0] + "," + summary
    return repr(obj_defn)
