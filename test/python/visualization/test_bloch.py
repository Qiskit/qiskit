# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for qiskit.visualization.bloch.Bloch"""

import os
import tempfile
import unittest

from ddt import ddt, data, unpack

from qiskit.utils import optionals
from test import QiskitTestCase  # pylint: disable=wrong-import-order

if optionals.HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt

    from qiskit.visualization.bloch import Bloch, _hide_tick_lines_and_labels


# Mirrors the ``ketex`` template used internally by ``Bloch.set_label_convention``
# for the "polarization jones"/"polarization jones letters" conventions.
_KETEX = "$\\left.|%s\\right\\rangle$"


@ddt
@unittest.skipUnless(optionals.HAS_MATPLOTLIB, "matplotlib not available.")
class TestBloch(QiskitTestCase):
    """Tests for the ``Bloch`` sphere plotting helper."""

    def setUp(self):
        super().setUp()
        self.addCleanup(plt.close, "all")

    def test_save_creates_file(self):
        """Bloch.save() renders the sphere and writes an image file.

        This test exercises most of the module: ``save`` calls
        ``render``, which fans out into ``plot_back``, ``plot_front``,
        ``plot_axes``, ``plot_axes_labels``, ``plot_vectors`` (including the
        ``Arrow3D`` artist, since ``savefig`` forces a real draw), and
        ``plot_annotations``.
        """
        bloch = Bloch()
        bloch.add_vectors([0, 1, 0])
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "bloch.png")
            bloch.save(name=path)
            self.assertTrue(os.path.isfile(path))

    def test_init_defaults(self):
        """Default constructor arguments populate the expected attributes."""
        bloch = Bloch()
        self.assertEqual(bloch.view, [-60, 30])
        self.assertEqual(bloch.figsize, [5, 5])
        self.assertEqual(bloch.sphere_color, "#FFDDDD")
        self.assertEqual(bloch.frame_color, "gray")
        self.assertEqual(
            bloch.vector_color, ["#dc267f", "#648fff", "#fe6100", "#785ef0", "#ffb000"]
        )
        self.assertEqual(bloch.point_color, ["b", "r", "g", "#CC6600"])
        self.assertEqual(bloch.points, [])
        self.assertEqual(bloch.vectors, [])
        self.assertEqual(bloch.annotations, [])
        self.assertEqual(bloch.point_style, [])
        self.assertFalse(bloch._rendered)

    def test_init_with_figsize_and_view(self):
        """Explicit ``figsize``/``view`` arguments are stored as given."""
        bloch = Bloch(figsize=[3, 3], view=[10, 20])
        self.assertEqual(bloch.figsize, [3, 3])
        self.assertEqual(bloch.view, [10, 20])

    def test_str(self):
        """``__str__`` reports the data and property summary."""
        bloch = Bloch()
        bloch.add_points([0, 0, 1])
        bloch.add_vectors([0, 1, 0])
        text = str(bloch)
        self.assertIn("Number of points:  1", text)
        self.assertIn("Number of vectors: 1", text)
        self.assertIn("frame_color:     gray", text)
        self.assertIn("sphere_color:    #FFDDDD", text)

    def test_clear(self):
        """``clear`` empties all data lists."""
        bloch = Bloch()
        bloch.add_points([0, 0, 1])
        bloch.add_vectors([0, 1, 0])
        bloch.add_annotation([0, 0, 1], "test")
        bloch.clear()
        self.assertEqual(bloch.points, [])
        self.assertEqual(bloch.vectors, [])
        self.assertEqual(bloch.annotations, [])
        self.assertEqual(bloch.point_style, [])

    @data(
        ("original", ["$x$", ""], ["$y$", ""], ["$\\left|0\\right>$", "$\\left|1\\right>$"]),
        ("xyz", ["$x$", ""], ["$y$", ""], ["$z$", ""]),
        ("sx sy sz", ["$s_x$", ""], ["$s_y$", ""], ["$s_z$", ""]),
        ("01", ["", ""], ["", ""], ["$\\left|0\\right>$", "$\\left|1\\right>$"]),
        (
            "polarization jones",
            [
                _KETEX % "\\nearrow\\hspace{-1.46}\\swarrow",
                _KETEX % "\\nwarrow\\hspace{-1.46}\\searrow",
            ],
            [_KETEX % "\\circlearrowleft", _KETEX % "\\circlearrowright"],
            [_KETEX % "\\leftrightarrow", _KETEX % "\\updownarrow"],
        ),
        (
            "polarization jones letters",
            [_KETEX % "D", _KETEX % "A"],
            [_KETEX % "L", _KETEX % "R"],
            [_KETEX % "H", _KETEX % "V"],
        ),
        (
            "polarization stokes",
            ["$\\leftrightarrow$", "$\\updownarrow$"],
            ["$\\nearrow\\hspace{-1.46}\\swarrow$", "$\\nwarrow\\hspace{-1.46}\\searrow$"],
            ["$\\circlearrowleft$", "$\\circlearrowright$"],
        ),
    )
    @unpack
    def test_set_label_convention(self, convention, xlabel, ylabel, zlabel):
        """Each supported convention sets the expected axis labels."""
        bloch = Bloch()
        bloch.set_label_convention(convention)
        self.assertEqual(bloch.xlabel, xlabel)
        self.assertEqual(bloch.ylabel, ylabel)
        self.assertEqual(bloch.zlabel, zlabel)

    def test_set_label_convention_invalid(self):
        """An unknown convention name raises ``ValueError``."""
        bloch = Bloch()
        with self.assertRaises(ValueError):
            bloch.set_label_convention("not-a-real-convention")

    def test_add_points_single_default(self):
        """A bare 3-coordinate point is reshaped and stored with style 's'."""
        bloch = Bloch()
        bloch.add_points([0, 0, 1])
        self.assertEqual(bloch.point_style, ["s"])
        self.assertEqual(len(bloch.points), 1)
        self.assertEqual(bloch.points[0].shape, (3, 2))

    def test_add_points_multiple_s(self):
        """Multiple points with the default meth keep style 's'."""
        bloch = Bloch()
        bloch.add_points([[0, 1], [0, 0], [1, 0]])
        self.assertEqual(bloch.point_style, ["s"])
        self.assertEqual(bloch.points[0].shape, (3, 2))

    def test_add_points_multicolor(self):
        """``meth='m'`` stores the points with style 'm'."""
        bloch = Bloch()
        bloch.add_points([[0, 1], [0, 0], [1, 0]], meth="m")
        self.assertEqual(bloch.point_style, ["m"])

    def test_add_points_line(self):
        """``meth='l'`` stores the points with style 'l'."""
        bloch = Bloch()
        bloch.add_points([[0, 1], [0, 0], [1, 0]], meth="l")
        self.assertEqual(bloch.point_style, ["l"])

    def test_add_vectors_single(self):
        """A single 3-element vector is appended as one entry."""
        bloch = Bloch()
        bloch.add_vectors([0, 0, 1])
        self.assertEqual(len(bloch.vectors), 1)

    def test_add_vectors_multiple(self):
        """A list of vectors is appended as multiple entries."""
        bloch = Bloch()
        bloch.add_vectors([[0, 0, 1], [1, 0, 0]])
        self.assertEqual(len(bloch.vectors), 2)

    def test_add_annotation(self):
        """A valid position/text pair is stored on the annotations list."""
        bloch = Bloch()
        bloch.add_annotation([0, 0, 1], "test", color="red")
        self.assertEqual(len(bloch.annotations), 1)
        annotation = bloch.annotations[0]
        self.assertEqual(annotation["position"], [0, 0, 1])
        self.assertEqual(annotation["text"], "test")
        self.assertEqual(annotation["opts"], {"color": "red"})

    def test_add_annotation_invalid_type(self):
        """A position that isn't a 3-element sequence raises ``TypeError``."""
        bloch = Bloch()
        with self.assertRaises(TypeError):
            bloch.add_annotation([0, 0], "test")

    def test_make_sphere(self):
        """``make_sphere`` is a thin alias for ``render``."""
        bloch = Bloch()
        bloch.make_sphere()
        self.assertTrue(bloch._rendered)

    def test_plot_points_sorts_by_distance(self):
        """Points at varying distance from the origin exercise the sort-by-radius branch."""
        bloch = Bloch()
        bloch.add_points([[0, 0.5], [0, 0], [1, 0]])
        bloch.render()
        self.assertTrue(bloch._rendered)

    def test_render_default(self):
        """Rendering without a background creates a figure and 3D axes."""
        bloch = Bloch()
        bloch.render()
        self.assertTrue(bloch._rendered)
        self.assertIsNotNone(bloch.fig)
        self.assertIsNotNone(bloch.axes)

    def test_render_background_true(self):
        """``background=True`` still renders successfully."""
        bloch = Bloch(background=True)
        bloch.add_vectors([0, 0, 1])
        bloch.render()
        self.assertTrue(bloch._rendered)

    def test_render_twice_clears_axes(self):
        """A second render call clears and redraws the existing axes."""
        bloch = Bloch()
        bloch.render()
        bloch.render()
        self.assertTrue(bloch._rendered)

    def test_render_external_fig_and_axes(self):
        """User-supplied figure/axes are reused instead of created internally."""
        fig = plt.figure(figsize=(2, 2))
        axes = fig.add_subplot(111, projection="3d")
        bloch = Bloch(fig=fig, axes=axes)
        bloch.render()
        self.assertIs(bloch.fig, fig)
        self.assertIs(bloch.axes, axes)

    def test_render_with_points_vectors_and_annotations(self):
        """A sphere combining every point style, vectors, and an annotation renders."""
        bloch = Bloch()
        bloch.add_points([0, 0, 1], meth="s")
        bloch.add_points([[0, 1], [0, 0], [1, 0]], meth="m")
        bloch.add_points([[0, 1], [0, 0], [1, 0]], meth="l")
        bloch.add_vectors([0, 1, 0])
        bloch.add_vectors([1, 0, 0])
        bloch.add_annotation([0, 0, 1], "test")
        bloch.render()
        self.assertTrue(bloch._rendered)

    def test_render_vector_style_line(self):
        """An empty ``vector_style`` selects the simple-line vector branch.

        This branch is currently broken on every supported Matplotlib version:
        ``plot_vectors`` passes the z-data both positionally and as the ``zs=``
        keyword to ``Axes3D.plot``, which raises ``TypeError``. This test
        documents that known bug rather than masking it; once it's fixed
        (tracked separately), this test should be updated to assert the
        sphere renders successfully instead.
        """
        bloch = Bloch()
        bloch.vector_style = ""
        bloch.add_vectors([0, 1, 0])
        with self.assertRaises(TypeError):
            bloch.render()

    def test_show(self):
        """``show`` renders the sphere, then hands off to ``plt.show``.

        This is currently broken on every backend: ``show`` calls
        ``plt.show(self.fig)``, but ``pyplot.show`` (and the backend ``show``
        implementations it forwards to) only accept a keyword-only ``block``
        argument, never a positional figure. This test documents that known
        bug rather than masking it; once it's fixed (tracked separately),
        this test should be updated to assert ``show`` succeeds instead.
        """
        bloch = Bloch()
        bloch.add_vectors([0, 0, 1])
        with self.assertRaises(TypeError):
            bloch.show()
        self.assertTrue(bloch._rendered)

    def test_save_default_name_increments_savenum(self):
        """Saving without an explicit name auto-numbers files in ``dirc``."""
        bloch = Bloch()
        bloch.add_vectors([0, 0, 1])
        with tempfile.TemporaryDirectory() as tmp_dir:
            cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                bloch.save(dirc="frames")
                bloch.save(dirc="frames")
            finally:
                os.chdir(cwd)
            self.assertEqual(bloch.savenum, 2)
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "frames", "bloch_0.png")))
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "frames", "bloch_1.png")))

    def test_save_no_args_writes_to_cwd(self):
        """Saving with neither ``name`` nor ``dirc`` writes ``bloch_<n>.png`` to cwd."""
        bloch = Bloch()
        bloch.add_vectors([0, 0, 1])
        with tempfile.TemporaryDirectory() as tmp_dir:
            cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                bloch.save()
            finally:
                os.chdir(cwd)
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "bloch_0.png")))

    def test_hide_tick_lines_and_labels(self):
        """The module-level helper hides all ticklines/labels on an axis."""
        fig = plt.figure()
        axes = fig.add_subplot(111, projection="3d")
        _hide_tick_lines_and_labels(axes.xaxis)
        for item in axes.xaxis.get_ticklines() + axes.xaxis.get_ticklabels():
            self.assertFalse(item.get_visible())


if __name__ == "__main__":
    unittest.main(verbosity=2)
