# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring

"""
Stand-alone tool to draw a QASM circuit based on matplotlib_drawer
"""

import json
from argparse import ArgumentParser

from qiskit.tools.matplotlibdrawer import MatplotlibDrawer


def options():
    parser = ArgumentParser()
    parser.add_argument('--qasm', action='store', help='input QASM file')
    parser.add_argument('--style', action='store', help='style file')
    parser.add_argument('--scale', action='store', help='scaling factor', type=float, default=1.0)
    parser.add_argument('--out', action='store', help='output figure file (pdf, png or svg)')
    parser.add_argument('--json', action='store', help='output JSON file of AST')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    args = parser.parse_args()
    if args.verbose:
        print('options:', args)
    if not args.qasm:
        parser.print_usage()
    return args


def main():
    args = options()
    if not args.qasm:
        return
    drawer = MatplotlibDrawer(style=args.style, scale=args.scale)
    drawer.load_qasm_file(args.qasm)
    # output json
    if args.json:
        with open(args.json, 'w') as outfile:
            json.dump(drawer.ast, outfile, sort_keys=True, indent=2)
    # draw quantum circuit
    drawer.draw(args.out, verbose=args.verbose)


if __name__ == '__main__':
    main()
