"""
Exception for errors raised while parsing OPENQASM.

Author: Jim Challenger
"""


class QasmException(Exception):
    """Base class for errors raised while parsing OPENQASM."""

    def __init__(self, *msg):
        """Set the error message."""
        self.msg = ' '.join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)
