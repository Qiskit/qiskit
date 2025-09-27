from .lib.symengine_wrapper import Symbol, Basic
from itertools import combinations, permutations, product, product as cartes
import re as _re
import string
import sys


_range = _re.compile('([0-9]*:[0-9]+|[a-zA-Z]?:[a-zA-Z])')


def symbols(names, **args):
    r"""
    Transform strings into instances of :class:`Symbol` class.
    :func:`symbols` function returns a sequence of symbols with names taken
    from ``names`` argument, which can be a comma or whitespace delimited
    string, or a sequence of strings::
        >>> from symengine import symbols
        >>> x, y, z = symbols('x,y,z')
        >>> a, b, c = symbols('a b c')
    The type of output is dependent on the properties of input arguments::
        >>> symbols('x')
        x
        >>> symbols('x,')
        (x,)
        >>> symbols('x,y')
        (x, y)
        >>> symbols(('a', 'b', 'c'))
        (a, b, c)
        >>> symbols(['a', 'b', 'c'])
        [a, b, c]
        >>> symbols(set(['a', 'b', 'c']))
        set([a, b, c])
    If an iterable container is needed for a single symbol, set the ``seq``
    argument to ``True`` or terminate the symbol name with a comma::
        >>> symbols('x', seq=True)
        (x,)
    To reduce typing, range syntax is supported to create indexed symbols.
    Ranges are indicated by a colon and the type of range is determined by
    the character to the right of the colon. If the character is a digit
    then all contiguous digits to the left are taken as the nonnegative
    starting value (or 0 if there is no digit left of the colon) and all
    contiguous digits to the right are taken as 1 greater than the ending
    value::
        >>> symbols('x:10')
        (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)
        >>> symbols('x5:10')
        (x5, x6, x7, x8, x9)
        >>> symbols('x5(:2)')
        (x50, x51)
        >>> symbols('x5:10,y:5')
        (x5, x6, x7, x8, x9, y0, y1, y2, y3, y4)
        >>> symbols(('x5:10', 'y:5'))
        ((x5, x6, x7, x8, x9), (y0, y1, y2, y3, y4))
    If the character to the right of the colon is a letter, then the single
    letter to the left (or 'a' if there is none) is taken as the start
    and all characters in the lexicographic range *through* the letter to
    the right are used as the range::
        >>> symbols('x:z')
        (x, y, z)
        >>> symbols('x:c')  # null range
        ()
        >>> symbols('x(:c)')
        (xa, xb, xc)
        >>> symbols(':c')
        (a, b, c)
        >>> symbols('a:d, x:z')
        (a, b, c, d, x, y, z)
        >>> symbols(('a:d', 'x:z'))
        ((a, b, c, d), (x, y, z))
    Multiple ranges are supported; contiguous numerical ranges should be
    separated by parentheses to disambiguate the ending number of one
    range from the starting number of the next::
        >>> symbols('x:2(1:3)')
        (x01, x02, x11, x12)
        >>> symbols(':3:2')  # parsing is from left to right
        (00, 01, 10, 11, 20, 21)
    Only one pair of parentheses surrounding ranges are removed, so to
    include parentheses around ranges, double them. And to include spaces,
    commas, or colons, escape them with a backslash::
        >>> symbols('x((a:b))')
        (x(a), x(b))
        >>> symbols('x(:1\,:2)')  # or 'x((:1)\,(:2))'
        (x(0,0), x(0,1))
    """
    result = []

    if isinstance(names, str):
        marker = 0
        literals = [r'\,', r'\:', r'\ ']
        for i in range(len(literals)):
            lit = literals.pop(0)
            if lit in names:
                while chr(marker) in names:
                    marker += 1
                lit_char = chr(marker)
                marker += 1
                names = names.replace(lit, lit_char)
                literals.append((lit_char, lit[1:]))

        def literal(s):
            if literals:
                for c, l in literals:
                    s = s.replace(c, l)
            return s

        names = names.strip()
        as_seq = names.endswith(',')
        if as_seq:
            names = names[:-1].rstrip()
        if not names:
            raise ValueError('no symbols given')

        # split on commas
        names = [n.strip() for n in names.split(',')]
        if not all(n for n in names):
            raise ValueError('missing symbol between commas')
        # split on spaces
        for i in range(len(names) - 1, -1, -1):
            names[i: i + 1] = names[i].split()

        cls = args.pop('cls', Symbol)
        seq = args.pop('seq', as_seq)

        for name in names:
            if not name:
                raise ValueError('missing symbol')

            if ':' not in name:
                symbol = cls(literal(name), **args)
                result.append(symbol)
                continue

            split = _range.split(name)
            # remove 1 layer of bounding parentheses around ranges
            for i in range(len(split) - 1):
                if i and ':' in split[i] and split[i] != ':' and \
                        split[i - 1].endswith('(') and \
                        split[i + 1].startswith(')'):
                    split[i - 1] = split[i - 1][:-1]
                    split[i + 1] = split[i + 1][1:]
            for i, s in enumerate(split):
                if ':' in s:
                    if s[-1].endswith(':'):
                        raise ValueError('missing end range')
                    a, b = s.split(':')
                    if b[-1] in string.digits:
                        a = 0 if not a else int(a)
                        b = int(b)
                        split[i] = [str(c) for c in range(a, b)]
                    else:
                        a = a or 'a'
                        split[i] = [string.ascii_letters[c] for c in range(
                            string.ascii_letters.index(a),
                            string.ascii_letters.index(b) + 1)]  # inclusive
                    if not split[i]:
                        break
                else:
                    split[i] = [s]
            else:
                seq = True
                if len(split) == 1:
                    names = split[0]
                else:
                    names = [''.join(s) for s in cartes(*split)]
                if literals:
                    result.extend([cls(literal(s), **args) for s in names])
                else:
                    result.extend([cls(s, **args) for s in names])

        if not seq and len(result) <= 1:
            if not result:
                return ()
            return result[0]

        return tuple(result)
    else:
        for name in names:
            result.append(symbols(name, **args))

        return type(names)(result)


def var(names, **args):
    """
    Create symbols and inject them into the global namespace.

    INPUT:
-            s -- a string, either a single variable name, or
-                 a space separated list of variable names, or
-                 a list of variable names.

    This calls :func:`symbols` with the same arguments and puts the results
    into the *global* namespace. It's recommended not to use :func:`var` in
    library code, where :func:`symbols` has to be used::

    Examples
    ========

    >>> from symengine import var

    >>> var('x')
    x
    >>> x
    x

    >>> var('a,ab,abc')
    (a, ab, abc)
    >>> abc
    abc

    See :func:`symbols` documentation for more details on what kinds of
    arguments can be passed to :func:`var`.

    """
    def traverse(symbols, frame):
        """Recursively inject symbols to the global namespace. """
        for symbol in symbols:
            if isinstance(symbol, Basic):
                frame.f_globals[symbol.__str__()] = symbol
            # Once we hace an undefined function class
            # implemented, put a check for function here
            else:
                traverse(symbol, frame)

    from inspect import currentframe
    frame = currentframe().f_back

    try:
        syms = symbols(names, **args)

        if syms is not None:
            if isinstance(syms, Basic):
                frame.f_globals[syms.__str__()] = syms
            # Once we hace an undefined function class
            # implemented, put a check for function here
            else:
                traverse(syms, frame)
    finally:
        del frame  # break cyclic dependencies as stated in inspect docs

    return syms


class NotIterable:
    """
    Use this as mixin when creating a class which is not supposed to return
    true when iterable() is called on its instances. I.e. avoid infinite loop
    when calling e.g. list() on the instance
    """
    pass


def iterable(i, exclude=(str, dict, NotIterable)):
    """
    Return a boolean indicating whether ``i`` is SymPy iterable.
    True also indicates that the iterator is finite, i.e. you e.g.
    call list(...) on the instance.

    When SymPy is working with iterables, it is almost always assuming
    that the iterable is not a string or a mapping, so those are excluded
    by default. If you want a pure Python definition, make exclude=None. To
    exclude multiple items, pass them as a tuple.
    """
    try:
        iter(i)
    except TypeError:
        return False
    if exclude:
        return not isinstance(i, exclude)
    return True


def is_sequence(i):
    """
    Return a boolean indicating whether ``i`` is a sequence in the SymPy
    sense. If anything that fails the test below should be included as
    being a sequence for your application, set 'include' to that object's
    type; multiple types should be passed as a tuple of types.
    """
    return hasattr(i, '__getitem__') and iterable(i)
