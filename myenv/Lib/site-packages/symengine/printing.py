from .lib.symengine_wrapper import ccode, sympify, Basic, repr_latex as _repr_latex

class CCodePrinter:

    def doprint(self, expr, assign_to=None):
        if not isinstance(assign_to, (Basic, type(None), str)):
            raise TypeError("{} cannot assign to object of type {}".format(
                    type(self).__name__, type(assign_to)))

        expr = sympify(expr)
        if not assign_to:
            if expr.is_Matrix:
                raise RuntimeError("Matrices need a assign_to parameter")
            return ccode(expr)

        assign_to = str(assign_to)
        if not expr.is_Matrix:
            return f"{assign_to} = {ccode(expr)};"

        code_lines = []
        for i, element in enumerate(expr):
            code_line = f'{assign_to}[{i}] = {element};'
            code_lines.append(code_line)
        return '\n'.join(code_lines)


def init_printing(pretty_print=True, use_latex=True):
    if pretty_print:
        if not use_latex:
            raise RuntimeError("Only latex is supported for pretty printing")
        _repr_latex[0] = True
    else:
        _repr_latex[0] = False
