#!/usr/bin/env python3
# This code is part of Qiskit.
#
# (C) Copyright IBM 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import argparse
import multiprocessing
import os
import sys
import re

# regex for character encoding from PEP 263
pep263 = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)")

def discover_files(code_paths):
    out_paths = []
    for path in code_paths:
        if os.path.isfile(path):
            out_paths.append(path)
        else:
            for directory in os.walk(path):
                dir_path = directory[0]
                for subfile in directory[2]:
                    if subfile.endswith('.py') or subfile.endswith('.pyx'):
                        out_paths.append(os.path.join(dir_path, subfile))
    return out_paths


def validate_header(file_path):
    header = """# This code is part of Qiskit.
#
"""
    apache_text = """#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
    count = 0
    with open(file_path, encoding='utf8') as fd:
        lines = fd.readlines()
    start = 0
    for index, line in enumerate(lines):
        count += 1
        if count > 5:
            return file_path, False, "Header not found in first 5 lines"
        if count<=2 and pep263.match(line):
            return file_path, False, "Unnecessary encoding specification (PEP 263, 3120)"
        if line == "# This code is part of Qiskit.\n":
            start = index
            break
    if ''.join(lines[start:start + 2]) != header:
        return (file_path, False,
                "Header up to copyright line does not match: %s" % header)
    if not lines[start + 2].startswith("# (C) Copyright IBM 20"):
        return (file_path, False,
                "Header copyright line not found")
    if ''.join(lines[start + 3:start + 11]) != apache_text:
        return (file_path, False,
                "Header apache text string doesn't match:\n %s" % apache_text)
    return (file_path, True, None)


def main():
    default_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'qiskit')
    parser = argparse.ArgumentParser(description="Check file headers.")
    parser.add_argument("paths", type=str, nargs='*',
                        default=[default_path],
                        help='Paths to scan by default uses ../qiskit from the'
                             ' script')
    args = parser.parse_args()
    files = discover_files(args.paths)
    pool = multiprocessing.Pool()
    res = pool.map(validate_header, files)
    failed_files = [x for x in res if x[1] is False]
    if len(failed_files) > 0:
        for failed_file in failed_files:
            sys.stderr.write("%s failed header check because:\n" % failed_file[0])
            sys.stderr.write("%s\n\n" % failed_file[2])
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
