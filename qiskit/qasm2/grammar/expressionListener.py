# Generated from expression.g4 by ANTLR 4.9.2
from antlr4 import *

if __name__ is not None and "." in __name__:
    from .expressionParser import expressionParser
else:
    from expressionParser import expressionParser

# This class defines a complete listener for a parse tree produced by expressionParser.
class expressionListener(ParseTreeListener):

    # Enter a parse tree produced by expressionParser#start.
    def enterStart(self, ctx: expressionParser.StartContext):
        pass

    # Exit a parse tree produced by expressionParser#start.
    def exitStart(self, ctx: expressionParser.StartContext):
        pass

    # Enter a parse tree produced by expressionParser#UMINUS.
    def enterUMINUS(self, ctx: expressionParser.UMINUSContext):
        pass

    # Exit a parse tree produced by expressionParser#UMINUS.
    def exitUMINUS(self, ctx: expressionParser.UMINUSContext):
        pass

    # Enter a parse tree produced by expressionParser#UNARYOPGRP.
    def enterUNARYOPGRP(self, ctx: expressionParser.UNARYOPGRPContext):
        pass

    # Exit a parse tree produced by expressionParser#UNARYOPGRP.
    def exitUNARYOPGRP(self, ctx: expressionParser.UNARYOPGRPContext):
        pass

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
        pass

    # Exit a parse tree produced by expressionParser#MULOPGRP.
    def exitMULOPGRP(self, ctx: expressionParser.MULOPGRPContext):
        pass

    # Enter a parse tree produced by expressionParser#NUMBERGRP.
    def enterNUMBERGRP(self, ctx: expressionParser.NUMBERGRPContext):
        pass

    # Exit a parse tree produced by expressionParser#NUMBERGRP.
    def exitNUMBERGRP(self, ctx: expressionParser.NUMBERGRPContext):
        pass

    # Enter a parse tree produced by expressionParser#ADDOPGRP.
    def enterADDOPGRP(self, ctx: expressionParser.ADDOPGRPContext):
        pass

    # Exit a parse tree produced by expressionParser#ADDOPGRP.
    def exitADDOPGRP(self, ctx: expressionParser.ADDOPGRPContext):
        pass

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


del expressionParser
