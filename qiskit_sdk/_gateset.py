"""
Unitary gate collection.

Author: Andrew Cross
"""
from ._gate import Gate
from ._qiskitexception import QISKitException


class GateSet(object):
    """Unitary gate collection."""

    def __init__(self):
        """New gate set."""
        self.gs = set([])

    def add(self, g):
        """Add gate to gate set."""
        if type(g) is not Gate:
            raise QISKitException("attempt to add non-Gate to GateSet")
        self.gs.add(g)

    def invert(self):
        """Invert all gates in this."""
        for g in self.gs:
            g.invert()

    def control(self, *qregs):
        """Add controls to all gates in this."""
        for g in self.gs:
            g.control(*qregs)

    def doif(self, c, val):
        """Add classical control register to all gates in this."""
        for g in self.gs:
            g.doif(c, val)
