#!/usr/bin/env python
# Author: Jim Challenger
import os
import sys
from . import _qasm_yy as qasm
import traceback


class Qasm(object):

    def __init__(self, filename = None, data = None):
        if ( filename == None and data == None ):
            raise qasm.QasmException("Missing input file and/or data")
        if ( filename != None and data != None ):
            raise qasm.QasmException("File and data must not both be specified initializing qasm")

        self._filename = filename
        self._data = data

    def print_tokens(self):
        if ( self._filename ):
            self._data = open(self._filename).read()

        qasm_p = qasm.QasmParser(self._filename)
        return qasm_p.print_tokens()
        
    def parse(self):
        if ( self._filename ):
            self._data = open(self._filename).read()

        qasm_p = qasm.QasmParser(self._filename)
        qasm_p.parse_debug(False)
        return qasm_p.parse(self._data)

def main(args):
    
    try:
        q = Qasm(filename=args[0])
        ast = q.parse()
        print('---------------------------------------- PARSE TREE ----------------------------------------')
        ast.to_string(0)
        print('-------------------------------------- END PARSE TREE --------------------------------------')


    except qasm.QasmException as e:
        print('--------------------------------------------------------------------------------')
        print(e.msg)
        print('--------------------------------------------------------------------------------')

    except Exception as e:
        print('--------------------------------------------------------------------------------')
        print(sys.exc_info()[0], 'Exception parsing qasm file')
        traceback.print_exc()
        print('--------------------------------------------------------------------------------')
       

if __name__ == '__main__':
    main(sys.argv[1:])
