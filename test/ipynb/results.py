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
import json
from PIL import Image, ImageChops, ImageDraw

SWD = os.path.dirname(os.path.abspath(__file__))


def _get_black_pixels(image):
    black_and_white_version = image.convert('1')
    black_pixels = black_and_white_version.histogram()[0]
    return black_pixels


def similarity_ratio(current, expected):
    diff_name = current.split('.')
    diff_name.insert(-1, 'diff')
    diff_name = '.'.join(diff_name)
    current = Image.open(current)
    expected = Image.open(expected)

    diff = ImageChops.difference(expected, current).convert('L')
    black_or_b(diff, current, expected).save(diff_name, "PNG")
    black_pixels = _get_black_pixels(diff)
    total_pixels = diff.size[0] * diff.size[1]
    return black_pixels / total_pixels, diff_name


def new_gray(size, color):
    img = Image.new('L', size)
    dr = ImageDraw.Draw(img)
    dr.rectangle((0, 0) + size, color)
    return img


def black_or_b(diff_image, image, reference, opacity=0.85):
    """Copied from https://stackoverflow.com/a/30307875 """
    thresholded_diff = diff_image
    for repeat in range(3):
        thresholded_diff = ImageChops.add(thresholded_diff, thresholded_diff)
    size = diff_image.size
    mask = new_gray(size, int(255 * (opacity)))
    shade = new_gray(size, 0)
    new = reference.copy()
    new.paste(shade, mask=mask)
    new.paste(image, mask=thresholded_diff)
    return new


class Results:
    def __init__(self, names, directory):
        self.names = names
        self.directory = directory
        self.data = {}
        datafilename = os.path.join(SWD, directory, 'result_test.json')
        if os.path.exists(datafilename):
            with open(datafilename, 'r') as datafile:
                self.data = json.load(datafile)

    def _repr_html_(self):
        ret = "<div>"
        for name in self.names:
            fullpath_name = os.path.join(self.directory, name)
            fullpath_reference = os.path.join(self.directory, 'references', name)
            ret += '<table><tr>'
            ratio, diff_name = similarity_ratio(fullpath_name, fullpath_reference)
            ret += '<td colspan=2><tt> %s <b>%s</b> </tt> </td>' % (self.data[name], name)
            ret += '<td> ratio: %s </td></tr>' % ratio
            ret += '<tr><td><img src="%s"</td>' % fullpath_name
            if os.path.exists(os.path.join(SWD, fullpath_reference)):
                ret += '<td><img src="%s"</td>' % fullpath_reference
                ret += '<td><img src="%s"</td>' % diff_name
            else:
                ret += '<td style="text-align:center">' \
                       'Add <a download="%s" href="%s">this image</a> ' \
                       'to %s and push</td>' % (name, fullpath_reference, fullpath_reference)
            ret += '</tr></table>'
        ret += "</div>"
        return ret


if __name__ == '__main__':
    names = []
    for file in os.listdir(os.path.join(SWD, 'mpl')):
        if file.endswith(".png") and not file.endswith(".diff.png"):
            names.append(file)
    results = Results(names, 'mpl')
