def _noop_jit(f, *args, **kwargs):
    """A a decorator that does nothing"""
    return f


def _have_numba():
    try:
        import numba

        return True
    except ImportError:
        return False


# True if importing numba succeeded
HAVE_NUMBA = _have_numba()

if HAVE_NUMBA:
    from numba import vectorize, jit, njit, int64
    from numba.typed import List
else:
    njit = _noop_jit
    jit = _noop_jit
    vectorize = _noop_jit
    int64 = None
    List = None
