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

"""Module providing definitions of common Qobj classes."""
from types import SimpleNamespace


class QobjDictField(SimpleNamespace):
    """A class used to represent a dictionary field in Qobj

    Exists as a backwards compatibility shim around a dictionary for Qobjs
    previously constructed using marshmallow.
    """

    def __init__(self, **kwargs):
        """Instantiate a new Qobj dict field object.

        Args:
            kwargs: arbitrary keyword arguments that can be accessed as
                attributes of the object.
        """
        self.__dict__.update(kwargs)

    def to_dict(self):
        """Return a dictionary format representation of the QASM Qobj.

        Returns:
            dict: The dictionary form of the QobjHeader.
        """
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        """Create a new QobjHeader object from a dictionary.

        Args:
            data (dict): A dictionary representing the QobjHeader to create. It
                will be in the same format as output by :func:`to_dict`.

        Returns:
            QobjDictFieldr: The QobjDictField from the input dictionary.
        """

        return cls(**data)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.__dict__ == other.__dict__:
                return True
        return False


class QobjHeader(QobjDictField):
    """A class used to represent a dictionary header in Qobj objects."""

    pass


class QobjExperimentHeader(QobjHeader):
    """A class representing a header dictionary for a Qobj Experiment."""

    pass
