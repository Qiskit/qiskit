import numpy

from .maybe_numba import HAVE_NUMBA, vectorize

if HAVE_NUMBA:

    @vectorize
    def abs2(z):
        return z * numpy.conj(z)

    #   return z.real * z.real + z.imag * z.imag # No differene here.
else:

    def abs2(z):
        return abs(z) ** 2
