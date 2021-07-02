#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 20:42:25 2021

@author: jax
"""

from qiskit.qasm2 import qasm2Visitor
from qiskit.qasm2 import qasm2Parser


class Qasm2Visitor(qasm2Visitor):
    def visitProgram(self, ctx: qasm2Parser.ProgramContext):
        print("I'm visiting program!")
        print("ProgramContext is {}".format(ctx))
        print("ProgramContext chidren {}".format(ctx.getChildren()))
        print(ctx.toStringTree())
        print("ProgramContext text: {}".format(ctx.getText()))
        return self.visitChildren(ctx)

    def visitHeader(self, ctx: qasm2Parser.HeaderContext):
        print("I'm visiting header!")
        print("HeaderContext is {}".format(ctx))
        print("HeaderContext chidren {}".format(ctx.getChildren()))
        print(ctx.toStringTree())
        print("HeaderContext version: {}".format(ctx.version()))
        return self.visitChildren(ctx)

    def visitVersion(self, ctx: qasm2Parser.VersionContext):
        print("I'm visiting version!")
        print("VersionContext is {}".format(ctx))
        print("VersionContext chidren {}".format(ctx.getChildren()))
        print(ctx.toStringTree())
        print("VersionContext text: {}".format(ctx.getText()))
        print("Version number: {}".format(str(ctx.RealNumber())))
        print("VersionContext payload: {}".format(ctx.getPayload()))
        return self.visitChildren(ctx)
