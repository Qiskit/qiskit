# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os

CWD = os.path.dirname(os.path.abspath(__file__))


class Results:
    def __init__(self, names, directory):
        self.names = names
        self.directory = directory

    def _repr_html_(self):
        ret = "<div>"
        for name in self.names:
            ret += '<table><tr>'
            ret += '<td colspan=2><tt> %s </tt></td></tr>' % name
            ret += '<tr><td><img src="%s"</td>' % name
            reference = os.path.join(self.directory, 'references', name)
            if os.path.exists(os.path.join(CWD, reference)):
                ret += '<td><img src="%s"</td>' % reference
            else:
                ret += '<td style="text-align:center">' \
                       'Add <a download="%s" href="%s">this image</a> ' \
                       'to %s and push</td>' % (name, reference, reference)
            ret += '</tr></table>'
        ret += "</div>"
        return ret


if __name__ == '__main__':
    names = []
    for file in os.listdir(CWD):
        if file.endswith(".png"):
            names.append(file)
    results = Results(names, 'mpl')
