# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""

"""
import math
from antlr4 import CommonTokenStream, InputStream, ParseTreeWalker
from qiskit.qasm2 import (
    expressionLexer,
    expressionParser,
    expressionListener,
    Qasm2Expression,
    QasmError,
)
from .qasm2expression import ExpReg

# This class defines a complete listener for a parse tree produced by expressionParser.
class Qasm2ExpressionListener(expressionListener):
    """ """

    def __init__(
        self,
        input_src: str,
        param_dict: dict = None,
    ) -> None:
        """


        Parameters
        ----------
        input_src : str
            DESCRIPTION.
        param_dict : dict, optional
            DESCRIPTION. The default is None.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """
        self.qasm2expr = Qasm2Expression(input_src, param_dict)

    def do_expr(self) -> float:
        """
        Parse source from the expression source code
        ``input_src`` and return the result.

        Returns
        -------
        float
            result of evaluating the input expression.

        """
        _i = InputStream(self.qasm2expr.input_src)
        lexer = expressionLexer(_i)
        stream = CommonTokenStream(lexer)
        stream.fill()
        _p = expressionParser(stream)
        tree = _p.start()
        walker = ParseTreeWalker()
        walker.walk(self, tree)
        return self.qasm2expr.result

    # Enter a parse tree produced by expressionParser#start.
    def enterStart(self, ctx: expressionParser.StartContext):
        if ctx.getChildCount() != 1:  # The listener only good for one expr
            raise QasmError(
                "Qasm2ExpressionListener:enterStart child count is {} not 1".format(
                    ctx.getChildCount()
                )
            )
        self.qasm2expr.expregstack.push(ExpReg())

    # Exit a parse tree produced by expressionParser#start.
    def exitStart(self, ctx: expressionParser.StartContext):
        _tos = self.qasm2expr.expregstack.pop()
        self.qasm2expr.result = _tos.result

    # Enter a parse tree produced by expressionParser#UMINUS.
    def enterUMINUS(self, ctx: expressionParser.UMINUSContext):
        self.qasm2expr.expregstack.push(ExpReg("-", [ctx.getChild(1)]))

    # Exit a parse tree produced by expressionParser#UMINUS.
    def exitUMINUS(self, ctx: expressionParser.UMINUSContext):
        _tos = self.qasm2expr.expregstack.pop()
        _parent = self.qasm2expr.expregstack.peek()
        _tos.result = -(_tos.operands[0])
        if _parent.operands:
            _parent.operands[_parent.index] = _tos.result
            _parent.index += 1
        else:
            _parent.result = _tos.result

    # Enter a parse tree produced by expressionParser#UNARYOPGRP.
    def enterUNARYOPGRP(self, ctx: expressionParser.UNARYOPGRPContext):
        self.qasm2expr.expregstack.push(
            ExpReg(ctx.getChild(0).getText(), [ctx.getChild(1)])
        )

    # Exit a parse tree produced by expressionParser#UNARYOPGRP.
    def exitUNARYOPGRP(self, ctx: expressionParser.UNARYOPGRPContext):
        _tos = self.qasm2expr.expregstack.pop()
        _parent = self.qasm2expr.expregstack.peek()
        if _tos.op == "exp":
            _tos.result = math.exp(_tos.operands[0])
        elif _tos.op == "ln":
            _tos.result = math.log(_tos.operands[0])
        elif _tos.op == "sin":
            _tos.result = math.sin(_tos.operands[0])
        elif _tos.op == "cos":
            _tos.result = math.cos(_tos.operands[0])
        elif _tos.op == "tan":
            _tos.result = math.tan(_tos.operands[0])
        elif _tos.op == "sqrt":
            _tos.result = math.sqrt(_tos.operands[0])
        else:
            raise QasmError(
                "Qasm2ExpressionListener::exitMULOPGRP unknown operator {}.".format(
                    _tos.op
                )
            )
        if _parent.operands:
            _parent.operands[_parent.index] = _tos.result
            _parent.index += 1
        else:
            _parent.result = _tos.result

    # Enter a parse tree produced by expressionParser#EADDOPIGRP.
    def enterEADDOPIGRP(self, ctx: expressionParser.EADDOPIGRPContext):
        pass

    # Exit a parse tree produced by expressionParser#EADDOPIGRP.
    def exitEADDOPIGRP(self, ctx: expressionParser.EADDOPIGRPContext):
        pass

    # Enter a parse tree produced by expressionParser#IMULOPIGRP.
    def enterIMULOPIGRP(self, ctx: expressionParser.IMULOPIGRPContext):
        pass

    # Exit a parse tree produced by expressionParser#IMULOPIGRP.
    def exitIMULOPIGRP(self, ctx: expressionParser.IMULOPIGRPContext):
        pass

    # Enter a parse tree produced by expressionParser#IADDOPIGRP.
    def enterIADDOPIGRP(self, ctx: expressionParser.IADDOPIGRPContext):
        pass

    # Exit a parse tree produced by expressionParser#IADDOPIGRP.
    def exitIADDOPIGRP(self, ctx: expressionParser.IADDOPIGRPContext):
        pass

    # Enter a parse tree produced by expressionParser#IMULOPEGRP.
    def enterIMULOPEGRP(self, ctx: expressionParser.IMULOPEGRPContext):
        pass

    # Exit a parse tree produced by expressionParser#IMULOPEGRP.
    def exitIMULOPEGRP(self, ctx: expressionParser.IMULOPEGRPContext):
        pass

    # Enter a parse tree produced by expressionParser#IADDOPEGRP.
    def enterIADDOPEGRP(self, ctx: expressionParser.IADDOPEGRPContext):
        pass

    # Exit a parse tree produced by expressionParser#IADDOPEGRP.
    def exitIADDOPEGRP(self, ctx: expressionParser.IADDOPEGRPContext):
        pass

    # Enter a parse tree produced by expressionParser#PARENGRP.
    def enterPARENGRP(self, ctx: expressionParser.PARENGRPContext):
        pass

    # Exit a parse tree produced by expressionParser#PARENGRP.
    def exitPARENGRP(self, ctx: expressionParser.PARENGRPContext):
        pass

    # Enter a parse tree produced by expressionParser#EMULOPIGRP.
    def enterEMULOPIGRP(self, ctx: expressionParser.EMULOPIGRPContext):
        pass

    # Exit a parse tree produced by expressionParser#EMULOPIGRP.
    def exitEMULOPIGRP(self, ctx: expressionParser.EMULOPIGRPContext):
        pass

    # Enter a parse tree produced by expressionParser#MULOPGRP.
    def enterMULOPGRP(self, ctx: expressionParser.MULOPGRPContext):
        self.qasm2expr.expregstack.push(
            ExpReg(ctx.getChild(1).getText(), [ctx.getChild(0), ctx.getChild(2)])
        )

    # Exit a parse tree produced by expressionParser#MULOPGRP.
    def exitMULOPGRP(self, ctx: expressionParser.MULOPGRPContext):
        _tos = self.qasm2expr.expregstack.pop()
        _parent = self.qasm2expr.expregstack.peek()
        if _tos.op == "*":
            _tos.result = _tos.operands[0] * _tos.operands[1]
        elif _tos.op == "/":
            _tos.result = _tos.operands[0] / _tos.operands[1]
        elif _tos.op == "%":
            _tos.result = _tos.operands[0] % _tos.operands[1]
        elif _tos.op == "^":
            _tos.result = _tos.operands[0] ** _tos.operands[1]
        else:
            raise QasmError(
                "Qasm2ExpressionListener::exitMULOPGRP unknown operator {}.".format(
                    _tos.op
                )
            )
        if _parent.operands:
            _parent.operands[_parent.index] = _tos.result
            _parent.index += 1
        else:
            _parent.result = _tos.result

    # Enter a parse tree produced by expressionParser#NUMBERGRP.
    def enterNUMBERGRP(self, ctx: expressionParser.NUMBERGRPContext):
        _num = ExpReg._resolve_number(ctx.getText())
        _expreg = ExpReg(None, [_num], _num)  # born already resolved
        _expreg.index = 1
        self.qasm2expr.expregstack.push(_expreg)

    # Exit a parse tree produced by expressionParser#NUMBERGRP.
    def exitNUMBERGRP(self, ctx: expressionParser.NUMBERGRPContext):
        _tos = self.qasm2expr.expregstack.pop()
        _parent = self.qasm2expr.expregstack.peek()
        if _parent.operands:
            _parent.operands[_parent.index] = _tos.result
            _parent.index += 1
        else:
            _parent.result = _tos.result

    # Enter a parse tree produced by expressionParser#ADDOPGRP.
    def enterADDOPGRP(self, ctx: expressionParser.ADDOPGRPContext):
        self.qasm2expr.expregstack.push(
            ExpReg(
                ctx.getChild(1).getText(),
                [ctx.getChild(0).getText(), ctx.getChild(2).getText()],
            )
        )

    # Exit a parse tree produced by expressionParser#ADDOPGRP.
    def exitADDOPGRP(self, ctx: expressionParser.ADDOPGRPContext):
        _tos = self.qasm2expr.expregstack.pop()
        _parent = self.qasm2expr.expregstack.peek()
        if _tos.op == "+":
            _tos.result = _tos.operands[0] + _tos.operands[1]
        elif _tos.op == "-":
            _tos.result = _tos.operands[0] - _tos.operands[1]
        else:
            raise QasmError(
                "Qasm2ExpressionListener::exitADDOPGRP unknown operator {}.".format(
                    _tos.op
                )
            )
        if _parent.operands:
            _parent.operands[_parent.index] = _tos.result
            _parent.index += 1
        else:
            _parent.result = _tos.result

    # Enter a parse tree produced by expressionParser#addop.
    def enterAddop(self, ctx: expressionParser.AddopContext):
        pass

    # Exit a parse tree produced by expressionParser#addop.
    def exitAddop(self, ctx: expressionParser.AddopContext):
        pass

    # Enter a parse tree produced by expressionParser#mulop.
    def enterMulop(self, ctx: expressionParser.MulopContext):
        pass

    # Exit a parse tree produced by expressionParser#mulop.
    def exitMulop(self, ctx: expressionParser.MulopContext):
        pass

    # Enter a parse tree produced by expressionParser#unaryop.
    def enterUnaryop(self, ctx: expressionParser.UnaryopContext):
        pass

    # Exit a parse tree produced by expressionParser#unaryop.
    def exitUnaryop(self, ctx: expressionParser.UnaryopContext):
        pass
