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

"""Unit tests for pulse waveforms."""

import unittest
from unittest.mock import patch
import numpy as np

from qiskit.circuit import Parameter
from qiskit.pulse.library import (
    SymbolicPulse,
    Waveform,
    Constant,
    Gaussian,
    GaussianSquare,
    Drag,
    gaussian,
    gaussian_square,
    drag as pl_drag,
)

from qiskit.pulse import functional_pulse, PulseError
from qiskit.test import QiskitTestCase
from qiskit.utils import optionals as _optional

if _optional.HAS_SYMENGINE:
    import symengine as sym
else:
    import sympy as sym


class TestWaveform(QiskitTestCase):
    """Waveform tests."""

    def test_sample_pulse(self):
        """Test pulse initialization."""
        n_samples = 100
        samples = np.linspace(0, 1.0, n_samples, dtype=np.complex128)
        name = "test"
        sample_pulse = Waveform(samples, name=name)

        self.assertEqual(sample_pulse.samples.dtype, np.complex128)
        np.testing.assert_almost_equal(sample_pulse.samples, samples)

        self.assertEqual(sample_pulse.duration, n_samples)
        self.assertEqual(sample_pulse.name, name)

    def test_type_casting(self):
        """Test casting of input samples to numpy array."""
        n_samples = 100
        samples_f64 = np.linspace(0, 1.0, n_samples, dtype=np.float64)

        sample_pulse_f64 = Waveform(samples_f64)
        self.assertEqual(sample_pulse_f64.samples.dtype, np.complex128)

        samples_c64 = np.linspace(0, 1.0, n_samples, dtype=np.complex64)

        sample_pulse_c64 = Waveform(samples_c64)
        self.assertEqual(sample_pulse_c64.samples.dtype, np.complex128)

        samples_list = np.linspace(0, 1.0, n_samples).tolist()

        sample_pulse_list = Waveform(samples_list)
        self.assertEqual(sample_pulse_list.samples.dtype, np.complex128)

    def test_pulse_limits(self):
        """Test that limits of pulse norm of one are enforced properly."""

        # test norm is correct for complex128 numpy data
        unit_pulse_c128 = np.exp(1j * 2 * np.pi * np.linspace(0, 1, 1000), dtype=np.complex128)
        # test does not raise error
        try:
            Waveform(unit_pulse_c128)
        except PulseError:
            self.fail("Waveform incorrectly failed on approximately unit norm samples.")

        invalid_const = 1.1
        with self.assertRaises(PulseError):
            Waveform(invalid_const * np.exp(1j * 2 * np.pi * np.linspace(0, 1, 1000)))

        with patch("qiskit.pulse.library.pulse.Pulse.limit_amplitude", new=False):
            wave = Waveform(invalid_const * np.exp(1j * 2 * np.pi * np.linspace(0, 1, 1000)))
            self.assertGreater(np.max(np.abs(wave.samples)), 1.0)

        # Test case where data is converted to python types with complex as a list
        # with form [re, im] and back to a numpy array.
        # This is how the transport layer handles samples in the qobj so it is important
        # to test.
        unit_pulse_c64 = np.exp(1j * 2 * np.pi * np.linspace(0, 1, 1000), dtype=np.complex64)
        sample_components = np.stack(
            np.transpose([np.real(unit_pulse_c64), np.imag(unit_pulse_c64)])
        )
        pulse_list = sample_components.tolist()
        recombined_pulse = [sample[0] + sample[1] * 1j for sample in pulse_list]

        # test does not raise error
        try:
            Waveform(recombined_pulse)
        except PulseError:
            self.fail("Waveform incorrectly failed to approximately unit norm samples.")


class TestParametricPulses(QiskitTestCase):
    """Tests for all subclasses of ParametricPulse."""

    def test_construction(self):
        """Test that parametric pulses can be constructed without error."""
        Gaussian(duration=25, sigma=4, amp=0.5j)
        GaussianSquare(duration=150, amp=0.2, sigma=8, width=140)
        GaussianSquare(duration=150, amp=0.2, sigma=8, risefall_sigma_ratio=2.5)
        Constant(duration=150, amp=0.1 + 0.4j)
        Drag(duration=25, amp=0.2 + 0.3j, sigma=7.8, beta=4)

    # This test should be removed once deprecation of complex amp is completed.
    def test_complex_amp_deprecation(self):
        """Test that deprecation warnings and errors are raised for complex amp,
        and that pulses are equivalent."""

        # Test deprecation warnings and errors:
        with self.assertWarns(PendingDeprecationWarning):
            Gaussian(duration=25, sigma=4, amp=0.5j)
        with self.assertWarns(PendingDeprecationWarning):
            GaussianSquare(duration=125, sigma=4, amp=0.5j, width=100)
        with self.assertRaises(PulseError):
            Gaussian(duration=25, sigma=4, amp=0.5j, angle=1)
        with self.assertRaises(PulseError):
            GaussianSquare(duration=125, sigma=4, amp=0.5j, width=100, angle=0.1)

        # Test that new and old API pulses are the same:
        gauss_pulse_complex_amp = Gaussian(duration=25, sigma=4, amp=0.5j)
        gauss_pulse_amp_angle = Gaussian(duration=25, sigma=4, amp=0.5, angle=np.pi / 2)
        np.testing.assert_almost_equal(
            gauss_pulse_amp_angle.get_waveform().samples,
            gauss_pulse_complex_amp.get_waveform().samples,
        )

    def test_gaussian_pulse(self):
        """Test that Gaussian sample pulse matches the pulse library."""
        gauss = Gaussian(duration=25, sigma=4, amp=0.5j)
        sample_pulse = gauss.get_waveform()
        self.assertIsInstance(sample_pulse, Waveform)
        pulse_lib_gauss = gaussian(duration=25, sigma=4, amp=0.5j, zero_ends=True).samples
        np.testing.assert_almost_equal(sample_pulse.samples, pulse_lib_gauss)

    def test_gaussian_square_pulse(self):
        """Test that GaussianSquare sample pulse matches the pulse library."""
        gauss_sq = GaussianSquare(duration=125, sigma=4, amp=0.5j, width=100)
        sample_pulse = gauss_sq.get_waveform()
        self.assertIsInstance(sample_pulse, Waveform)
        pulse_lib_gauss_sq = gaussian_square(
            duration=125, sigma=4, amp=0.5j, width=100, zero_ends=True
        ).samples
        np.testing.assert_almost_equal(sample_pulse.samples, pulse_lib_gauss_sq)
        gauss_sq = GaussianSquare(duration=125, sigma=4, amp=0.5j, risefall_sigma_ratio=3.125)
        sample_pulse = gauss_sq.get_waveform()
        self.assertIsInstance(sample_pulse, Waveform)
        pulse_lib_gauss_sq = gaussian_square(
            duration=125, sigma=4, amp=0.5j, width=100, zero_ends=True
        ).samples
        np.testing.assert_almost_equal(sample_pulse.samples, pulse_lib_gauss_sq)

    def test_gauss_square_extremes(self):
        """Test that the gaussian square pulse can build a gaussian."""
        duration = 125
        sigma = 4
        amp = 0.5j
        gaus_square = GaussianSquare(duration=duration, sigma=sigma, amp=amp, width=0)
        gaus = Gaussian(duration=duration, sigma=sigma, amp=amp)
        np.testing.assert_almost_equal(
            gaus_square.get_waveform().samples, gaus.get_waveform().samples
        )
        gaus_square = GaussianSquare(duration=duration, sigma=sigma, amp=amp, width=121)
        const = Constant(duration=duration, amp=amp)
        np.testing.assert_almost_equal(
            gaus_square.get_waveform().samples[2:-2], const.get_waveform().samples[2:-2]
        )

    def test_gauss_square_passes_validation_after_construction(self):
        """Test that parameter validation is consistent before and after construction.

        This previously used to raise an exception: see gh-7882."""
        pulse = GaussianSquare(duration=125, sigma=4, amp=0.5j, width=100)
        pulse.validate_parameters()

    def test_drag_pulse(self):
        """Test that the Drag sample pulse matches the pulse library."""
        drag = Drag(duration=25, sigma=4, amp=0.5j, beta=1)
        sample_pulse = drag.get_waveform()
        self.assertIsInstance(sample_pulse, Waveform)
        pulse_lib_drag = pl_drag(duration=25, sigma=4, amp=0.5j, beta=1, zero_ends=True).samples
        np.testing.assert_almost_equal(sample_pulse.samples, pulse_lib_drag)

    def test_drag_validation(self):
        """Test drag parameter validation, specifically the beta validation."""
        duration = 25
        sigma = 4
        amp = 0.5j
        beta = 1
        wf = Drag(duration=duration, sigma=sigma, amp=amp, beta=beta)
        samples = wf.get_waveform().samples
        self.assertTrue(max(np.abs(samples)) <= 1)
        with self.assertRaises(PulseError):
            wf = Drag(duration=duration, sigma=sigma, amp=1.2, beta=beta)
        beta = sigma**2
        with self.assertRaises(PulseError):
            wf = Drag(duration=duration, sigma=sigma, amp=amp, beta=beta)
        # If sigma is high enough, side peaks fall out of range and norm restriction is met
        sigma = 100
        wf = Drag(duration=duration, sigma=sigma, amp=amp, beta=beta)

    def test_drag_beta_validation(self):
        """Test drag beta parameter validation."""

        def check_drag(duration, sigma, amp, beta):
            wf = Drag(duration=duration, sigma=sigma, amp=amp, beta=beta)
            samples = wf.get_waveform().samples
            self.assertTrue(max(np.abs(samples)) <= 1)

        check_drag(duration=50, sigma=16, amp=1, beta=2)
        check_drag(duration=50, sigma=16, amp=1, beta=4)
        check_drag(duration=50, sigma=16, amp=0.5, beta=20)
        check_drag(duration=50, sigma=16, amp=-1, beta=2)
        check_drag(duration=50, sigma=16, amp=1, beta=-2)
        check_drag(duration=50, sigma=16, amp=1, beta=6)
        check_drag(duration=50, sigma=16, amp=-0.5j, beta=25)
        with self.assertRaises(PulseError):
            check_drag(duration=50, sigma=16, amp=1, beta=20)
        with self.assertRaises(PulseError):
            check_drag(duration=50, sigma=4, amp=0.8, beta=20)
        with self.assertRaises(PulseError):
            check_drag(duration=50, sigma=4, amp=0.8, beta=-20)

    def test_constant_samples(self):
        """Test the constant pulse and its sampled construction."""
        const = Constant(duration=150, amp=0.1 + 0.4j)
        self.assertEqual(const.get_waveform().samples[0], 0.1 + 0.4j)
        self.assertEqual(len(const.get_waveform().samples), 150)

    def test_parameters(self):
        """Test that the parameters can be extracted as a dict through the `parameters`
        attribute."""
        drag = Drag(duration=25, amp=0.2, sigma=7.8, beta=4, angle=0.2)
        self.assertEqual(set(drag.parameters.keys()), {"duration", "amp", "sigma", "beta", "angle"})
        const = Constant(duration=150, amp=1)
        self.assertEqual(set(const.parameters.keys()), {"duration", "amp", "angle"})

    def test_repr(self):
        """Test the repr methods for parametric pulses."""
        gaus = Gaussian(duration=25, amp=0.7, sigma=4, angle=0.3)
        self.assertEqual(repr(gaus), "Gaussian(duration=25, amp=0.7, sigma=4, angle=0.3)")
        gaus = Gaussian(
            duration=25, amp=0.1 + 0.7j, sigma=4
        )  # Should be removed once the deprecation of complex
        # amp is completed.
        self.assertEqual(repr(gaus), "Gaussian(duration=25, amp=(0.1+0.7j), sigma=4, angle=0)")
        gaus_square = GaussianSquare(duration=20, sigma=30, amp=1.0, width=3)
        self.assertEqual(
            repr(gaus_square), "GaussianSquare(duration=20, amp=1.0, sigma=30, width=3, angle=0)"
        )
        gaus_square = GaussianSquare(
            duration=20, sigma=30, amp=1.0, angle=0.2, risefall_sigma_ratio=0.1
        )
        self.assertEqual(
            repr(gaus_square),
            "GaussianSquare(duration=20, amp=1.0, sigma=30, width=14.0, angle=0.2)",
        )
        drag = Drag(duration=5, amp=0.5, sigma=7, beta=1)
        self.assertEqual(repr(drag), "Drag(duration=5, amp=0.5, sigma=7, beta=1, angle=0)")
        const = Constant(duration=150, amp=0.1, angle=0.3)
        self.assertEqual(repr(const), "Constant(duration=150, amp=0.1, angle=0.3)")

    def test_param_validation(self):
        """Test that parametric pulse parameters are validated when initialized."""
        with self.assertRaises(PulseError):
            Gaussian(duration=25, sigma=0, amp=0.5j)
        with self.assertRaises(PulseError):
            GaussianSquare(duration=150, amp=0.2, sigma=8)
        with self.assertRaises(PulseError):
            GaussianSquare(duration=150, amp=0.2, sigma=8, width=100, risefall_sigma_ratio=5)
        with self.assertRaises(PulseError):
            GaussianSquare(duration=150, amp=0.2, sigma=8, width=160)
        with self.assertRaises(PulseError):
            GaussianSquare(duration=150, amp=0.2, sigma=8, risefall_sigma_ratio=10)
        with self.assertRaises(PulseError):
            Constant(duration=150, amp=0.9 + 0.8j)
        with self.assertRaises(PulseError):
            Drag(duration=25, amp=0.2 + 0.3j, sigma=-7.8, beta=4)

    def test_hash_generation(self):
        """Test if pulse generate unique hash."""
        test_hash = [
            hash(GaussianSquare(duration=688, amp=0.1 + 0.1j, sigma=64, width=432))
            for _ in range(10)
        ]

        ref_hash = [test_hash[0] for _ in range(10)]

        self.assertListEqual(test_hash, ref_hash)

    def test_gaussian_limit_amplitude(self):
        """Test that the check for amplitude less than or equal to 1 can be disabled."""
        with self.assertRaises(PulseError):
            Gaussian(duration=100, sigma=1.0, amp=1.1 + 0.8j)

        with patch("qiskit.pulse.library.pulse.Pulse.limit_amplitude", new=False):
            waveform = Gaussian(duration=100, sigma=1.0, amp=1.1 + 0.8j)
            self.assertGreater(np.abs(waveform.amp), 1.0)

    def test_gaussian_limit_amplitude_per_instance(self):
        """Test that the check for amplitude per instance."""
        with self.assertRaises(PulseError):
            Gaussian(duration=100, sigma=1.0, amp=1.1 + 0.8j)

        waveform = Gaussian(duration=100, sigma=1.0, amp=1.1 + 0.8j, limit_amplitude=False)
        self.assertGreater(np.abs(waveform.amp), 1.0)

    def test_gaussian_square_limit_amplitude(self):
        """Test that the check for amplitude less than or equal to 1 can be disabled."""
        with self.assertRaises(PulseError):
            GaussianSquare(duration=100, sigma=1.0, amp=1.1 + 0.8j, width=10)

        with patch("qiskit.pulse.library.pulse.Pulse.limit_amplitude", new=False):
            waveform = GaussianSquare(duration=100, sigma=1.0, amp=1.1 + 0.8j, width=10)
            self.assertGreater(np.abs(waveform.amp), 1.0)

    def test_gaussian_square_limit_amplitude_per_instance(self):
        """Test that the check for amplitude per instance."""
        with self.assertRaises(PulseError):
            GaussianSquare(duration=100, sigma=1.0, amp=1.1 + 0.8j, width=10)

        waveform = GaussianSquare(
            duration=100, sigma=1.0, amp=1.1 + 0.8j, width=10, limit_amplitude=False
        )
        self.assertGreater(np.abs(waveform.amp), 1.0)

    def test_drag_limit_amplitude(self):
        """Test that the check for amplitude less than or equal to 1 can be disabled."""
        with self.assertRaises(PulseError):
            Drag(duration=100, sigma=1.0, beta=1.0, amp=1.1 + 0.8j)

        with patch("qiskit.pulse.library.pulse.Pulse.limit_amplitude", new=False):
            waveform = Drag(duration=100, sigma=1.0, beta=1.0, amp=1.1 + 0.8j)
            self.assertGreater(np.abs(waveform.amp), 1.0)

    def test_drag_limit_amplitude_per_instance(self):
        """Test that the check for amplitude per instance."""
        with self.assertRaises(PulseError):
            Drag(duration=100, sigma=1.0, beta=1.0, amp=1.1 + 0.8j)

        waveform = Drag(duration=100, sigma=1.0, beta=1.0, amp=1.1 + 0.8j, limit_amplitude=False)
        self.assertGreater(np.abs(waveform.amp), 1.0)

    def test_constant_limit_amplitude(self):
        """Test that the check for amplitude less than or equal to 1 can be disabled."""
        with self.assertRaises(PulseError):
            Constant(duration=100, amp=1.1 + 0.8j)

        with patch("qiskit.pulse.library.pulse.Pulse.limit_amplitude", new=False):
            waveform = Constant(duration=100, amp=1.1 + 0.8j)
            self.assertGreater(np.abs(waveform.amp), 1.0)

    def test_constant_limit_amplitude_per_instance(self):
        """Test that the check for amplitude per instance."""
        with self.assertRaises(PulseError):
            Constant(duration=100, amp=1.1 + 0.8j)

        waveform = Constant(duration=100, amp=1.1 + 0.8j, limit_amplitude=False)
        self.assertGreater(np.abs(waveform.amp), 1.0)

    def test_get_parameters(self):
        """Test getting pulse parameters as attribute."""
        drag_pulse = Drag(duration=100, amp=0.1, sigma=40, beta=3)
        self.assertEqual(drag_pulse.duration, 100)
        self.assertEqual(drag_pulse.amp, 0.1)
        self.assertEqual(drag_pulse.sigma, 40)
        self.assertEqual(drag_pulse.beta, 3)

        with self.assertRaises(AttributeError):
            # pylint: disable=pointless-statement
            drag_pulse.non_existing_parameter

    def test_envelope_cache(self):
        """Test speed up of instantiation with lambdify envelope cache."""
        drag_instance1 = Drag(duration=100, amp=0.1, sigma=40, beta=3)
        drag_instance2 = Drag(duration=100, amp=0.1, sigma=40, beta=3)
        self.assertTrue(drag_instance1._envelope_lam is drag_instance2._envelope_lam)

    def test_constraints_cache(self):
        """Test speed up of instantiation with lambdify constraints cache."""
        drag_instance1 = Drag(duration=100, amp=0.1, sigma=40, beta=3)
        drag_instance2 = Drag(duration=100, amp=0.1, sigma=40, beta=3)
        self.assertTrue(drag_instance1._constraints_lam is drag_instance2._constraints_lam)

    def test_deepcopy(self):
        """Test deep copying instance."""
        import copy

        drag = Drag(duration=100, amp=0.1, sigma=40, beta=3)
        drag_copied = copy.deepcopy(drag)

        self.assertNotEqual(id(drag), id(drag_copied))

        orig_wf = drag.get_waveform()
        copied_wf = drag_copied.get_waveform()

        np.testing.assert_almost_equal(orig_wf.samples, copied_wf.samples)

    def test_fully_parametrized_pulse(self):
        """Test instantiating a pulse with parameters."""
        amp = Parameter("amp")
        duration = Parameter("duration")
        sigma = Parameter("sigma")
        beta = Parameter("beta")

        # doesn't raise an error
        drag = Drag(duration=duration, amp=amp, sigma=sigma, beta=beta)

        with self.assertRaises(PulseError):
            drag.get_waveform()

    # pylint: disable=invalid-name
    def test_custom_pulse(self):
        """Test defining a custom pulse which is not in the form of amp * F(t)."""
        t, t1, t2, amp1, amp2 = sym.symbols("t, t1, t2, amp1, amp2")
        envelope = sym.Piecewise((amp1, sym.And(t > t1, t < t2)), (amp2, sym.true))

        custom_pulse = SymbolicPulse(
            pulse_type="Custom",
            duration=100,
            parameters={"t1": 30, "t2": 80, "amp1": 0.1j, "amp2": -0.1},
            envelope=envelope,
        )
        waveform = custom_pulse.get_waveform()
        reference = np.concatenate([-0.1 * np.ones(30), 0.1j * np.ones(50), -0.1 * np.ones(20)])
        np.testing.assert_array_almost_equal(waveform.samples, reference)

    def test_no_subclass(self):
        """Test no dedicated pulse subclass is created."""

        gaussian_pulse = Gaussian(160, 0.1, 40)
        self.assertIs(type(gaussian_pulse), SymbolicPulse)

        gaussian_square_pulse = GaussianSquare(800, 0.1, 64, 544)
        self.assertIs(type(gaussian_square_pulse), SymbolicPulse)

        drag_pulse = Drag(160, 0.1, 40, 1.5)
        self.assertIs(type(drag_pulse), SymbolicPulse)

        constant_pulse = Constant(800, 0.1)
        self.assertIs(type(constant_pulse), SymbolicPulse)

    def test_gaussian_deprecated_type_check(self):
        """Test isinstance check works with deprecation."""
        gaussian_pulse = Gaussian(160, 0.1, 40)

        self.assertTrue(isinstance(gaussian_pulse, SymbolicPulse))
        with self.assertWarns(PendingDeprecationWarning):
            self.assertTrue(isinstance(gaussian_pulse, Gaussian))
            self.assertFalse(isinstance(gaussian_pulse, GaussianSquare))
            self.assertFalse(isinstance(gaussian_pulse, Drag))
            self.assertFalse(isinstance(gaussian_pulse, Constant))

    def test_gaussian_square_deprecated_type_check(self):
        """Test isinstance check works with deprecation."""
        gaussian_square_pulse = GaussianSquare(800, 0.1, 64, 544)

        self.assertTrue(isinstance(gaussian_square_pulse, SymbolicPulse))
        with self.assertWarns(PendingDeprecationWarning):
            self.assertFalse(isinstance(gaussian_square_pulse, Gaussian))
            self.assertTrue(isinstance(gaussian_square_pulse, GaussianSquare))
            self.assertFalse(isinstance(gaussian_square_pulse, Drag))
            self.assertFalse(isinstance(gaussian_square_pulse, Constant))

    def test_drag_deprecated_type_check(self):
        """Test isinstance check works with deprecation."""
        drag_pulse = Drag(160, 0.1, 40, 1.5)

        self.assertTrue(isinstance(drag_pulse, SymbolicPulse))
        with self.assertWarns(PendingDeprecationWarning):
            self.assertFalse(isinstance(drag_pulse, Gaussian))
            self.assertFalse(isinstance(drag_pulse, GaussianSquare))
            self.assertTrue(isinstance(drag_pulse, Drag))
            self.assertFalse(isinstance(drag_pulse, Constant))

    def test_constant_deprecated_type_check(self):
        """Test isinstance check works with deprecation."""
        constant_pulse = Constant(160, 0.1, 40, 1.5)

        self.assertTrue(isinstance(constant_pulse, SymbolicPulse))
        with self.assertWarns(PendingDeprecationWarning):
            self.assertFalse(isinstance(constant_pulse, Gaussian))
            self.assertFalse(isinstance(constant_pulse, GaussianSquare))
            self.assertFalse(isinstance(constant_pulse, Drag))
            self.assertTrue(isinstance(constant_pulse, Constant))


class TestFunctionalPulse(QiskitTestCase):
    """Waveform tests."""

    # pylint: disable=invalid-name, unexpected-keyword-arg
    def test_gaussian(self):
        """Test gaussian pulse."""

        @functional_pulse
        def local_gaussian(duration, amp, t0, sig):
            x = np.linspace(0, duration - 1, duration)
            return amp * np.exp(-((x - t0) ** 2) / sig**2)

        pulse_wf_inst = local_gaussian(duration=10, amp=1, t0=5, sig=1, name="test_pulse")
        _y = 1 * np.exp(-((np.linspace(0, 9, 10) - 5) ** 2) / 1**2)

        self.assertListEqual(list(pulse_wf_inst.samples), list(_y))

        # check name
        self.assertEqual(pulse_wf_inst.name, "test_pulse")

        # check duration
        self.assertEqual(pulse_wf_inst.duration, 10)

    # pylint: disable=invalid-name
    def test_variable_duration(self):
        """Test generation of sample pulse with variable duration."""

        @functional_pulse
        def local_gaussian(duration, amp, t0, sig):
            x = np.linspace(0, duration - 1, duration)
            return amp * np.exp(-((x - t0) ** 2) / sig**2)

        _durations = np.arange(10, 15, 1)

        for _duration in _durations:
            pulse_wf_inst = local_gaussian(duration=_duration, amp=1, t0=5, sig=1)
            self.assertEqual(len(pulse_wf_inst.samples), _duration)


if __name__ == "__main__":
    unittest.main()
