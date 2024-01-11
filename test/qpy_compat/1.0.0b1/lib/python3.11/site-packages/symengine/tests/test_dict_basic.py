from symengine.test_utilities import raises

from symengine import symbols, DictBasic, sin, Integer


def test_DictBasic():
    x, y, z = symbols("x y z")
    d = DictBasic({x: 2, y: z})

    assert str(d) == "{x: 2, y: z}" or str(d) == "{y: z, x: 2}"
    assert d[x] == 2

    raises(KeyError, lambda: d[2*z])
    if 2*z in d:
        assert False

    d[2*z] = x
    assert d[2*z] == x
    if 2*z not in d:
        assert False
    assert set(d.items()) == {(2*z, x), (x, Integer(2)), (y, z)}

    del d[x]
    assert set(d.keys()) == {2*z, y}
    assert set(d.values()) == {x, z}

    e = y + sin(2*z)
    assert e.subs(d) == z + sin(x)
