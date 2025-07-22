from symengine import cse, sqrt, symbols

def test_cse_single():
    x, y, x0 = symbols("x, y, x0")
    e = pow(x + y, 2) + sqrt(x + y)
    substs, reduced = cse([e])
    assert substs == [(x0, x + y)]
    assert reduced == [sqrt(x0) + x0**2]


def test_multiple_expressions():
    w, x, y, z, x0 = symbols("w, x, y, z, x0")
    e1 = (x + y)*z
    e2 = (x + y)*w
    substs, reduced = cse([e1, e2])
    assert substs == [(x0, x + y)]
    assert reduced == [x0*z, x0*w]
