"""
Exception for errors raised by the CouplingGraph object.

Author: Andrew Cross
"""


class CouplingGraphError(Exception):
    """Base class for errors raised by the coupling graph object."""

    def __init__(self, *msg):
        """Set the error message."""
        self.msg = ' '.join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)
