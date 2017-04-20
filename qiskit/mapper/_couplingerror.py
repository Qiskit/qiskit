"""
Exception for errors raised by the Coupling object.

Author: Andrew Cross
"""


class CouplingError(Exception):
    """Base class for errors raised by the coupling graph object."""

    def __init__(self, *msg):
        """Set the error message."""
        self.msg = ' '.join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)
