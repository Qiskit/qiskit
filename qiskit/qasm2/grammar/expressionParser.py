# Generated from expression.g4 by ANTLR 4.9.2
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys

if sys.version_info[1] > 5:
    from typing import TextIO
else:
    from typing.io import TextIO


def serializedATN():
    with StringIO() as buf:
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3\24")
        buf.write("Q\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\3\2\3\2\3\2")
        buf.write("\3\2\7\2\21\n\2\f\2\16\2\24\13\2\3\2\5\2\27\n\2\3\3\3")
        buf.write("\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3")
        buf.write("\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\5\3\64")
        buf.write("\n\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3")
        buf.write("\3\3\3\3\3\3\3\7\3F\n\3\f\3\16\3I\13\3\3\4\3\4\3\5\3\5")
        buf.write("\3\6\3\6\3\6\2\3\4\7\2\4\6\b\n\2\5\4\2\4\4\7\7\3\2\b\13")
        buf.write("\3\2\f\21\2Y\2\26\3\2\2\2\4\63\3\2\2\2\6J\3\2\2\2\bL\3")
        buf.write("\2\2\2\nN\3\2\2\2\f\27\5\4\3\2\r\22\5\4\3\2\16\17\7\3")
        buf.write("\2\2\17\21\5\4\3\2\20\16\3\2\2\2\21\24\3\2\2\2\22\20\3")
        buf.write("\2\2\2\22\23\3\2\2\2\23\27\3\2\2\2\24\22\3\2\2\2\25\27")
        buf.write("\3\2\2\2\26\f\3\2\2\2\26\r\3\2\2\2\26\25\3\2\2\2\27\3")
        buf.write("\3\2\2\2\30\31\b\3\1\2\31\32\7\4\2\2\32\64\5\4\3\16\33")
        buf.write("\34\7\24\2\2\34\35\5\b\5\2\35\36\7\24\2\2\36\64\3\2\2")
        buf.write('\2\37 \7\24\2\2 !\5\b\5\2!"\5\4\3\n"\64\3\2\2\2#$\7')
        buf.write("\24\2\2$%\5\6\4\2%&\7\24\2\2&\64\3\2\2\2'(\7\24\2\2(")
        buf.write(")\5\6\4\2)*\5\4\3\7*\64\3\2\2\2+,\5\n\6\2,-\5\4\3\5-\64")
        buf.write("\3\2\2\2./\7\5\2\2/\60\5\4\3\2\60\61\7\6\2\2\61\64\3\2")
        buf.write("\2\2\62\64\7\22\2\2\63\30\3\2\2\2\63\33\3\2\2\2\63\37")
        buf.write("\3\2\2\2\63#\3\2\2\2\63'\3\2\2\2\63+\3\2\2\2\63.\3\2")
        buf.write("\2\2\63\62\3\2\2\2\64G\3\2\2\2\65\66\f\r\2\2\66\67\5\b")
        buf.write("\5\2\678\5\4\3\168F\3\2\2\29:\f\f\2\2:;\5\6\4\2;<\5\4")
        buf.write("\3\r<F\3\2\2\2=>\f\t\2\2>?\5\b\5\2?@\7\24\2\2@F\3\2\2")
        buf.write("\2AB\f\6\2\2BC\5\6\4\2CD\7\24\2\2DF\3\2\2\2E\65\3\2\2")
        buf.write("\2E9\3\2\2\2E=\3\2\2\2EA\3\2\2\2FI\3\2\2\2GE\3\2\2\2G")
        buf.write("H\3\2\2\2H\5\3\2\2\2IG\3\2\2\2JK\t\2\2\2K\7\3\2\2\2LM")
        buf.write("\t\3\2\2M\t\3\2\2\2NO\t\4\2\2O\13\3\2\2\2\7\22\26\63E")
        buf.write("G")
        return buf.getvalue()


class expressionParser(Parser):

    grammarFileName = "expression.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [DFA(ds, i) for i, ds in enumerate(atn.decisionToState)]

    sharedContextCache = PredictionContextCache()

    literalNames = [
        "<INVALID>",
        "','",
        "'-'",
        "'('",
        "')'",
        "'+'",
        "'*'",
        "'/'",
        "'%'",
        "'^'",
        "'sin'",
        "'cos'",
        "'tan'",
        "'sqrt'",
        "'exp'",
        "'ln'",
    ]

    symbolicNames = [
        "<INVALID>",
        "<INVALID>",
        "<INVALID>",
        "<INVALID>",
        "<INVALID>",
        "<INVALID>",
        "<INVALID>",
        "<INVALID>",
        "<INVALID>",
        "<INVALID>",
        "<INVALID>",
        "<INVALID>",
        "<INVALID>",
        "<INVALID>",
        "<INVALID>",
        "<INVALID>",
        "NUMBER",
        "WS",
        "Identifier",
    ]

    RULE_start = 0
    RULE_expr = 1
    RULE_addop = 2
    RULE_mulop = 3
    RULE_unaryop = 4

    ruleNames = ["start", "expr", "addop", "mulop", "unaryop"]

    EOF = Token.EOF
    T__0 = 1
    T__1 = 2
    T__2 = 3
    T__3 = 4
    T__4 = 5
    T__5 = 6
    T__6 = 7
    T__7 = 8
    T__8 = 9
    T__9 = 10
    T__10 = 11
    T__11 = 12
    T__12 = 13
    T__13 = 14
    T__14 = 15
    NUMBER = 16
    WS = 17
    Identifier = 18

    def __init__(self, input: TokenStream, output: TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.9.2")
        self._interp = ParserATNSimulator(
            self, self.atn, self.decisionsToDFA, self.sharedContextCache
        )
        self._predicates = None

    class StartContext(ParserRuleContext):
        __slots__ = "parser"

        def __init__(self, parser, parent: ParserRuleContext = None, invokingState: int = -1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self, i: int = None):
            if i is None:
                return self.getTypedRuleContexts(expressionParser.ExprContext)
            else:
                return self.getTypedRuleContext(expressionParser.ExprContext, i)

        def getRuleIndex(self):
            return expressionParser.RULE_start

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterStart"):
                listener.enterStart(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitStart"):
                listener.exitStart(self)

    def start(self):

        localctx = expressionParser.StartContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_start)
        self._la = 0  # Token type
        try:
            self.state = 20
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 1, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 10
                self.expr(0)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 11
                self.expr(0)
                self.state = 16
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == expressionParser.T__0:
                    self.state = 12
                    self.match(expressionParser.T__0)
                    self.state = 13
                    self.expr(0)
                    self.state = 18
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)

                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)

                pass

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ExprContext(ParserRuleContext):
        __slots__ = "parser"

        def __init__(self, parser, parent: ParserRuleContext = None, invokingState: int = -1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return expressionParser.RULE_expr

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class UMINUSContext(ExprContext):
        def __init__(
            self, parser, ctx: ParserRuleContext
        ):  # actually a expressionParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self):
            return self.getTypedRuleContext(expressionParser.ExprContext, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterUMINUS"):
                listener.enterUMINUS(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitUMINUS"):
                listener.exitUMINUS(self)

    class UNARYOPGRPContext(ExprContext):
        def __init__(
            self, parser, ctx: ParserRuleContext
        ):  # actually a expressionParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def unaryop(self):
            return self.getTypedRuleContext(expressionParser.UnaryopContext, 0)

        def expr(self):
            return self.getTypedRuleContext(expressionParser.ExprContext, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterUNARYOPGRP"):
                listener.enterUNARYOPGRP(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitUNARYOPGRP"):
                listener.exitUNARYOPGRP(self)

    class EADDOPIGRPContext(ExprContext):
        def __init__(
            self, parser, ctx: ParserRuleContext
        ):  # actually a expressionParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self):
            return self.getTypedRuleContext(expressionParser.ExprContext, 0)

        def addop(self):
            return self.getTypedRuleContext(expressionParser.AddopContext, 0)

        def Identifier(self):
            return self.getToken(expressionParser.Identifier, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterEADDOPIGRP"):
                listener.enterEADDOPIGRP(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitEADDOPIGRP"):
                listener.exitEADDOPIGRP(self)

    class IMULOPIGRPContext(ExprContext):
        def __init__(
            self, parser, ctx: ParserRuleContext
        ):  # actually a expressionParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self, i: int = None):
            if i is None:
                return self.getTokens(expressionParser.Identifier)
            else:
                return self.getToken(expressionParser.Identifier, i)

        def mulop(self):
            return self.getTypedRuleContext(expressionParser.MulopContext, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterIMULOPIGRP"):
                listener.enterIMULOPIGRP(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitIMULOPIGRP"):
                listener.exitIMULOPIGRP(self)

    class IADDOPIGRPContext(ExprContext):
        def __init__(
            self, parser, ctx: ParserRuleContext
        ):  # actually a expressionParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self, i: int = None):
            if i is None:
                return self.getTokens(expressionParser.Identifier)
            else:
                return self.getToken(expressionParser.Identifier, i)

        def addop(self):
            return self.getTypedRuleContext(expressionParser.AddopContext, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterIADDOPIGRP"):
                listener.enterIADDOPIGRP(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitIADDOPIGRP"):
                listener.exitIADDOPIGRP(self)

    class IMULOPEGRPContext(ExprContext):
        def __init__(
            self, parser, ctx: ParserRuleContext
        ):  # actually a expressionParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self):
            return self.getToken(expressionParser.Identifier, 0)

        def mulop(self):
            return self.getTypedRuleContext(expressionParser.MulopContext, 0)

        def expr(self):
            return self.getTypedRuleContext(expressionParser.ExprContext, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterIMULOPEGRP"):
                listener.enterIMULOPEGRP(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitIMULOPEGRP"):
                listener.exitIMULOPEGRP(self)

    class IADDOPEGRPContext(ExprContext):
        def __init__(
            self, parser, ctx: ParserRuleContext
        ):  # actually a expressionParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self):
            return self.getToken(expressionParser.Identifier, 0)

        def addop(self):
            return self.getTypedRuleContext(expressionParser.AddopContext, 0)

        def expr(self):
            return self.getTypedRuleContext(expressionParser.ExprContext, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterIADDOPEGRP"):
                listener.enterIADDOPEGRP(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitIADDOPEGRP"):
                listener.exitIADDOPEGRP(self)

    class PARENGRPContext(ExprContext):
        def __init__(
            self, parser, ctx: ParserRuleContext
        ):  # actually a expressionParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self):
            return self.getTypedRuleContext(expressionParser.ExprContext, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterPARENGRP"):
                listener.enterPARENGRP(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitPARENGRP"):
                listener.exitPARENGRP(self)

    class EMULOPIGRPContext(ExprContext):
        def __init__(
            self, parser, ctx: ParserRuleContext
        ):  # actually a expressionParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self):
            return self.getTypedRuleContext(expressionParser.ExprContext, 0)

        def mulop(self):
            return self.getTypedRuleContext(expressionParser.MulopContext, 0)

        def Identifier(self):
            return self.getToken(expressionParser.Identifier, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterEMULOPIGRP"):
                listener.enterEMULOPIGRP(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitEMULOPIGRP"):
                listener.exitEMULOPIGRP(self)

    class MULOPGRPContext(ExprContext):
        def __init__(
            self, parser, ctx: ParserRuleContext
        ):  # actually a expressionParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self, i: int = None):
            if i is None:
                return self.getTypedRuleContexts(expressionParser.ExprContext)
            else:
                return self.getTypedRuleContext(expressionParser.ExprContext, i)

        def mulop(self):
            return self.getTypedRuleContext(expressionParser.MulopContext, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterMULOPGRP"):
                listener.enterMULOPGRP(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitMULOPGRP"):
                listener.exitMULOPGRP(self)

    class NUMBERGRPContext(ExprContext):
        def __init__(
            self, parser, ctx: ParserRuleContext
        ):  # actually a expressionParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def NUMBER(self):
            return self.getToken(expressionParser.NUMBER, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterNUMBERGRP"):
                listener.enterNUMBERGRP(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitNUMBERGRP"):
                listener.exitNUMBERGRP(self)

    class ADDOPGRPContext(ExprContext):
        def __init__(
            self, parser, ctx: ParserRuleContext
        ):  # actually a expressionParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self, i: int = None):
            if i is None:
                return self.getTypedRuleContexts(expressionParser.ExprContext)
            else:
                return self.getTypedRuleContext(expressionParser.ExprContext, i)

        def addop(self):
            return self.getTypedRuleContext(expressionParser.AddopContext, 0)

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterADDOPGRP"):
                listener.enterADDOPGRP(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitADDOPGRP"):
                listener.exitADDOPGRP(self)

    def expr(self, _p: int = 0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = expressionParser.ExprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 2
        self.enterRecursionRule(localctx, 2, self.RULE_expr, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 49
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 2, self._ctx)
            if la_ == 1:
                localctx = expressionParser.UMINUSContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx

                self.state = 23
                self.match(expressionParser.T__1)
                self.state = 24
                self.expr(12)
                pass

            elif la_ == 2:
                localctx = expressionParser.IMULOPIGRPContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 25
                self.match(expressionParser.Identifier)
                self.state = 26
                self.mulop()
                self.state = 27
                self.match(expressionParser.Identifier)
                pass

            elif la_ == 3:
                localctx = expressionParser.IMULOPEGRPContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 29
                self.match(expressionParser.Identifier)
                self.state = 30
                self.mulop()
                self.state = 31
                self.expr(8)
                pass

            elif la_ == 4:
                localctx = expressionParser.IADDOPIGRPContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 33
                self.match(expressionParser.Identifier)
                self.state = 34
                self.addop()
                self.state = 35
                self.match(expressionParser.Identifier)
                pass

            elif la_ == 5:
                localctx = expressionParser.IADDOPEGRPContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 37
                self.match(expressionParser.Identifier)
                self.state = 38
                self.addop()
                self.state = 39
                self.expr(5)
                pass

            elif la_ == 6:
                localctx = expressionParser.UNARYOPGRPContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 41
                self.unaryop()
                self.state = 42
                self.expr(3)
                pass

            elif la_ == 7:
                localctx = expressionParser.PARENGRPContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 44
                self.match(expressionParser.T__2)
                self.state = 45
                self.expr(0)
                self.state = 46
                self.match(expressionParser.T__3)
                pass

            elif la_ == 8:
                localctx = expressionParser.NUMBERGRPContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 48
                self.match(expressionParser.NUMBER)
                pass

            self._ctx.stop = self._input.LT(-1)
            self.state = 69
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 4, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 67
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 3, self._ctx)
                    if la_ == 1:
                        localctx = expressionParser.MULOPGRPContext(
                            self,
                            expressionParser.ExprContext(self, _parentctx, _parentState),
                        )
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 51
                        if not self.precpred(self._ctx, 11):
                            from antlr4.error.Errors import FailedPredicateException

                            raise FailedPredicateException(self, "self.precpred(self._ctx, 11)")
                        self.state = 52
                        self.mulop()
                        self.state = 53
                        self.expr(12)
                        pass

                    elif la_ == 2:
                        localctx = expressionParser.ADDOPGRPContext(
                            self,
                            expressionParser.ExprContext(self, _parentctx, _parentState),
                        )
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 55
                        if not self.precpred(self._ctx, 10):
                            from antlr4.error.Errors import FailedPredicateException

                            raise FailedPredicateException(self, "self.precpred(self._ctx, 10)")
                        self.state = 56
                        self.addop()
                        self.state = 57
                        self.expr(11)
                        pass

                    elif la_ == 3:
                        localctx = expressionParser.EMULOPIGRPContext(
                            self,
                            expressionParser.ExprContext(self, _parentctx, _parentState),
                        )
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 59
                        if not self.precpred(self._ctx, 7):
                            from antlr4.error.Errors import FailedPredicateException

                            raise FailedPredicateException(self, "self.precpred(self._ctx, 7)")
                        self.state = 60
                        self.mulop()
                        self.state = 61
                        self.match(expressionParser.Identifier)
                        pass

                    elif la_ == 4:
                        localctx = expressionParser.EADDOPIGRPContext(
                            self,
                            expressionParser.ExprContext(self, _parentctx, _parentState),
                        )
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 63
                        if not self.precpred(self._ctx, 4):
                            from antlr4.error.Errors import FailedPredicateException

                            raise FailedPredicateException(self, "self.precpred(self._ctx, 4)")
                        self.state = 64
                        self.addop()
                        self.state = 65
                        self.match(expressionParser.Identifier)
                        pass

                self.state = 71
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 4, self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx

    class AddopContext(ParserRuleContext):
        __slots__ = "parser"

        def __init__(self, parser, parent: ParserRuleContext = None, invokingState: int = -1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return expressionParser.RULE_addop

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterAddop"):
                listener.enterAddop(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitAddop"):
                listener.exitAddop(self)

    def addop(self):

        localctx = expressionParser.AddopContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_addop)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 72
            _la = self._input.LA(1)
            if not (_la == expressionParser.T__1 or _la == expressionParser.T__4):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class MulopContext(ParserRuleContext):
        __slots__ = "parser"

        def __init__(self, parser, parent: ParserRuleContext = None, invokingState: int = -1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return expressionParser.RULE_mulop

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterMulop"):
                listener.enterMulop(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitMulop"):
                listener.exitMulop(self)

    def mulop(self):

        localctx = expressionParser.MulopContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_mulop)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 74
            _la = self._input.LA(1)
            if not (
                (
                    ((_la) & ~0x3F) == 0
                    and (
                        (1 << _la)
                        & (
                            (1 << expressionParser.T__5)
                            | (1 << expressionParser.T__6)
                            | (1 << expressionParser.T__7)
                            | (1 << expressionParser.T__8)
                        )
                    )
                    != 0
                )
            ):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class UnaryopContext(ParserRuleContext):
        __slots__ = "parser"

        def __init__(self, parser, parent: ParserRuleContext = None, invokingState: int = -1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return expressionParser.RULE_unaryop

        def enterRule(self, listener: ParseTreeListener):
            if hasattr(listener, "enterUnaryop"):
                listener.enterUnaryop(self)

        def exitRule(self, listener: ParseTreeListener):
            if hasattr(listener, "exitUnaryop"):
                listener.exitUnaryop(self)

    def unaryop(self):

        localctx = expressionParser.UnaryopContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_unaryop)
        self._la = 0  # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 76
            _la = self._input.LA(1)
            if not (
                (
                    ((_la) & ~0x3F) == 0
                    and (
                        (1 << _la)
                        & (
                            (1 << expressionParser.T__9)
                            | (1 << expressionParser.T__10)
                            | (1 << expressionParser.T__11)
                            | (1 << expressionParser.T__12)
                            | (1 << expressionParser.T__13)
                            | (1 << expressionParser.T__14)
                        )
                    )
                    != 0
                )
            ):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    def sempred(self, localctx: RuleContext, ruleIndex: int, predIndex: int):
        if self._predicates == None:
            self._predicates = dict()
        self._predicates[1] = self.expr_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def expr_sempred(self, localctx: ExprContext, predIndex: int):
        if predIndex == 0:
            return self.precpred(self._ctx, 11)

        if predIndex == 1:
            return self.precpred(self._ctx, 10)

        if predIndex == 2:
            return self.precpred(self._ctx, 7)

        if predIndex == 3:
            return self.precpred(self._ctx, 4)
