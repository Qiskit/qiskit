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
import zipfile

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
    drawing = ImageDraw.Draw(img)
    drawing.rectangle((0, 0) + size, color)
    return img


def black_or_b(diff_image, image, reference, opacity=0.85):
    """Copied from https://stackoverflow.com/a/30307875 """
    thresholded_diff = diff_image
    for _ in range(3):
        thresholded_diff = ImageChops.add(thresholded_diff, thresholded_diff)
    size = diff_image.size
    mask = new_gray(size, int(255 * (opacity)))
    shade = new_gray(size, 0)
    new = reference.copy()
    new.paste(shade, mask=mask)
    if image.size != new.size:
        image = image.resize(new.size)
    if image.size != thresholded_diff.size:
        thresholded_diff = thresholded_diff.resize(image.size)
    new.paste(image, mask=thresholded_diff)
    return new


def zipfiles(files, zipname):
    with zipfile.ZipFile(zipname, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            zipf.write(file, arcname=os.path.basename(file))


class Results:
    def __init__(self, names, directory):
        self.names = names
        self.directory = directory
        self.data = {}
        self.exact_match = []
        self.mismatch = []
        self.missing = []
        datafilename = os.path.join(SWD, directory, 'result_test.json')
        if os.path.exists(datafilename):
            with open(datafilename, 'r') as datafile:
                self.data = json.load(datafile)

    @staticmethod
    def passed_result_html(result, reference, diff, title):
        ret = '<details><summary style="background-color:lightgreen;"> %s </summary>' % title
        ret += '<table>'
        ret += '<tr><td><img src="%s"</td>' % result
        ret += '<td><img src="%s"</td>' % reference
        ret += '<td><img src="%s"</td>' % diff
        ret += '</tr></table></details>'
        return ret

    @staticmethod
    def failed_result_html(result, reference, diff, title):
        ret = '<details open><summary style="background-color:lightcoral;"> %s </summary>' % title
        ret += '<table>'
        ret += '<tr><td><img src="%s"</td>' % result
        ret += '<td><img src="%s"</td>' % reference
        ret += '<td><img src="%s"</td>' % diff
        ret += '</tr></table></details>'
        return ret

    @staticmethod
    def no_reference_html(result, title):
        ret = '<details><summary style="background-color:lightgrey;"> %s </summary>' % title
        ret += '<table><tr><td><img src="%s"</td>' % result
        ret += '</tr></table></details>'
        return ret

    def diff_images(self):
        for name in self.names:
            ratio = diff_name = title = None
            fullpath_name = os.path.join(self.directory, name)
            fullpath_reference = os.path.join(self.directory, 'references', name)

            if os.path.exists(os.path.join(SWD, fullpath_reference)):
                ratio, diff_name = similarity_ratio(fullpath_name, fullpath_reference)
                title = '<tt><b>%s</b> | %s </tt> | ratio: %s' % (name,
                                                                  self.data[name]['testname'],
                                                                  ratio)
                if ratio == 1:
                    self.exact_match.append(fullpath_name)
                else:
                    self.mismatch.append(fullpath_name)
            else:
                self.missing.append(fullpath_name)

            self.data[name]['ratio'] = ratio
            self.data[name]['diff_name'] = diff_name
            self.data[name]['title'] = title

    def summary(self):
        ret = ''

        if len(self.mismatch) >= 2:
            zipfiles(self.mismatch, 'mpl/mismatch.zip')
            ret += '<div><a href="mpl/mismatch.zip">' \
                   'Download %s mismatch results as a zip</a></div>' % len(self.mismatch)

        if len(self.missing) >= 2:
            zipfiles(self.missing, 'mpl/missing.zip')
            ret += '<div><a href="mpl/missing.zip">' \
                   'Download %s missing results as a zip</a></div>' % len(self.missing)

        return ret

    def _repr_html_(self):
        ret = self.summary()
        ret += "<div>"
        for name in self.names:
            fullpath_name = os.path.join(self.directory, name)
            fullpath_reference = os.path.join(self.directory, 'references', name)
            if os.path.exists(os.path.join(SWD, fullpath_reference)):
                if self.data[name]['ratio'] == 1:
                    ret += Results.passed_result_html(fullpath_name, fullpath_reference,
                                                      self.data[name]['diff_name'],
                                                      self.data[name]['title'])
                else:
                    ret += Results.failed_result_html(fullpath_name, fullpath_reference,
                                                      self.data[name]['diff_name'],
                                                      self.data[name]['title'])
            else:
                title = 'Download <a download="%s" href="%s">this image</a> to <tt>%s</tt>' \
                        ' and add/push to the repo</td>' % (name, fullpath_name, fullpath_reference)
                ret += Results.no_reference_html(fullpath_name, title)
        ret += "</div>"
        return ret


if __name__ == '__main__':
    result_files = []
    for file in os.listdir(os.path.join(SWD, 'mpl')):
        if file.endswith(".png") and not file.endswith(".diff.png"):
            result_files.append(file)
    results = Results(sorted(result_files), 'mpl')
    results.diff_images()
