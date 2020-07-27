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

"""Unit tests for pulse waveforms."""

import unittest
import numpy as np

from qiskit.pulse.library import (Waveform, Constant, ConstantPulse, Gaussian,
                                  GaussianSquare, Drag, gaussian)

from qiskit.pulse import functional_pulse, PulseError
from qiskit.test import QiskitTestCase


class TestWaveform(QiskitTestCase):
    """Waveform tests."""

    def test_sample_pulse(self):
        """Test pulse initialization."""
        n_samples = 100
        samples = np.linspace(0, 1., n_samples, dtype=np.complex128)
        name = 'test'
        sample_pulse = Waveform(samples, name=name)

        self.assertEqual(sample_pulse.samples.dtype, np.complex128)
        np.testing.assert_almost_equal(sample_pulse.samples, samples)

        self.assertEqual(sample_pulse.duration, n_samples)
        self.assertEqual(sample_pulse.name, name)

    def test_type_casting(self):
        """Test casting of input samples to numpy array."""
        n_samples = 100
        samples_f64 = np.linspace(0, 1., n_samples, dtype=np.float64)

        sample_pulse_f64 = Waveform(samples_f64)
        self.assertEqual(sample_pulse_f64.samples.dtype, np.complex128)

        samples_c64 = np.linspace(0, 1., n_samples, dtype=np.complex64)

        sample_pulse_c64 = Waveform(samples_c64)
        self.assertEqual(sample_pulse_c64.samples.dtype, np.complex128)

        samples_list = np.linspace(0, 1., n_samples).tolist()

        sample_pulse_list = Waveform(samples_list)
        self.assertEqual(sample_pulse_list.samples.dtype, np.complex128)

    def test_pulse_limits(self):
        """Test that limits of pulse norm of one are enforced properly."""

        # test norm is correct for complex128 numpy data
        unit_pulse_c128 = np.exp(1j*2*np.pi*np.linspace(0, 1, 1000), dtype=np.complex128)
        # test does not raise error
        try:
            Waveform(unit_pulse_c128)
        except PulseError:
            self.fail('Waveform incorrectly failed on approximately unit norm samples.')

        invalid_const = 1.1
        with self.assertRaises(PulseError):
            Waveform(invalid_const*np.exp(1j*2*np.pi*np.linspace(0, 1, 1000)))

        # Test case where data is converted to python types with complex as a list
        # with form [re, im] and back to a numpy array.
        # This is how the transport layer handles samples in the qobj so it is important
        # to test.
        unit_pulse_c64 = np.exp(1j*2*np.pi*np.linspace(0, 1, 1000), dtype=np.complex64)
        sample_components = np.stack(np.transpose([np.real(unit_pulse_c64),
                                                   np.imag(unit_pulse_c64)]))
        pulse_list = sample_components.tolist()
        recombined_pulse = [sample[0]+sample[1]*1j for sample in pulse_list]

        # test does not raise error
        try:
            Waveform(recombined_pulse)
        except PulseError:
            self.fail('Waveform incorrectly failed to approximately unit norm samples.')


class TestParametricPulses(QiskitTestCase):
    """Tests for all subclasses of ParametricPulse."""

    def test_construction(self):
        """Test that parametric pulses can be constructed without error."""
        Gaussian(duration=25, sigma=4, amp=0.5j)
        GaussianSquare(duration=150, amp=0.2, sigma=8, width=140)
        Constant(duration=150, amp=0.1 + 0.4j)
        Drag(duration=25, amp=0.2 + 0.3j, sigma=7.8, beta=4)

    def test_sampled_pulse(self):
        """Test that we can convert to a sampled pulse."""
        gauss = Gaussian(duration=25, sigma=4, amp=0.5j)
        sample_pulse = gauss.get_waveform()
        self.assertIsInstance(sample_pulse, Waveform)
        pulse_lib_gaus = gaussian(duration=25, sigma=4,
                                  amp=0.5j, zero_ends=False).samples
        np.testing.assert_almost_equal(sample_pulse.samples, pulse_lib_gaus)

    def test_gauss_samples(self):
        """Test that the gaussian samples match the formula."""
        duration = 25
        sigma = 4
        amp = 0.5j
        # formulaic
        times = np.array(range(25), dtype=np.complex_)
        times = times - (duration / 2) + 0.5
        gauss = amp * np.exp(-(times / sigma)**2 / 2)
        # wf
        wf = Gaussian(duration=duration, sigma=sigma, amp=amp)
        samples = wf.get_waveform().samples
        np.testing.assert_almost_equal(samples, gauss)

    def test_gauss_square_samples(self):
        """Test that the gaussian square samples match the formula."""
        duration = 125
        sigma = 4
        amp = 0.5j
        # formulaic
        times = np.array(range(25), dtype=np.complex_)
        times = times - (25 / 2) + 0.5
        gauss = amp * np.exp(-(times / sigma)**2 / 2)
        # wf
        wf = GaussianSquare(duration=duration, sigma=sigma, amp=amp, width=100)
        samples = wf.get_waveform().samples
        np.testing.assert_almost_equal(samples[50], amp)
        np.testing.assert_almost_equal(samples[100], amp)
        np.testing.assert_almost_equal(samples[:10], gauss[:10])

    def test_gauss_square_extremes(self):
        """Test that the gaussian square pulse can build a gaussian."""
        duration = 125
        sigma = 4
        amp = 0.5j
        gaus_square = GaussianSquare(duration=duration, sigma=sigma, amp=amp, width=0)
        gaus = Gaussian(duration=duration, sigma=sigma, amp=amp)
        np.testing.assert_almost_equal(gaus_square.get_waveform().samples,
                                       gaus.get_waveform().samples)
        gaus_square = GaussianSquare(duration=duration, sigma=sigma, amp=amp, width=121)
        const = Constant(duration=duration, amp=amp)
        np.testing.assert_almost_equal(gaus_square.get_waveform().samples[2:-2],
                                       const.get_waveform().samples[2:-2])

    def test_drag_samples(self):
        """Test that the drag samples match the formula."""
        duration = 25
        sigma = 4
        amp = 0.5j
        beta = 1
        # formulaic
        times = np.array(range(25), dtype=np.complex_)
        times = times - (25 / 2) + 0.5
        gauss = amp * np.exp(-(times / sigma)**2 / 2)
        gauss_deriv = -(times / sigma**2) * gauss
        drag = gauss + 1j * beta * gauss_deriv
        # wf
        wf = Drag(duration=duration, sigma=sigma, amp=amp, beta=beta)
        samples = wf.get_waveform().samples
        np.testing.assert_almost_equal(samples, drag)

    def test_drag_validation(self):
        """Test drag parameter validation, specifically the beta validation."""
        duration = 25
        sigma = 4
        amp = 0.5j
        beta = 1
        wf = Drag(duration=duration, sigma=sigma, amp=amp, beta=beta)
        samples = wf.get_waveform().samples
        self.assertTrue(max(np.abs(samples)) <= 1)
        beta = sigma ** 2
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

    def test_constant_samples(self):
        """Test the constant pulse and its sampled construction."""
        const = Constant(duration=150, amp=0.1 + 0.4j)
        self.assertEqual(const.get_waveform().samples[0], 0.1 + 0.4j)
        self.assertEqual(len(const.get_waveform().samples), 150)

    def test_parameters(self):
        """Test that the parameters can be extracted as a dict through the `parameters`
        attribute."""
        drag = Drag(duration=25, amp=0.2 + 0.3j, sigma=7.8, beta=4)
        self.assertEqual(set(drag.parameters.keys()), {'duration', 'amp', 'sigma', 'beta'})
        const = Constant(duration=150, amp=1)
        self.assertEqual(set(const.parameters.keys()), {'duration', 'amp'})

    def test_repr(self):
        """Test the repr methods for parametric pulses."""
        gaus = Gaussian(duration=25, amp=0.7, sigma=4)
        self.assertEqual(repr(gaus), 'Gaussian(duration=25, amp=(0.7+0j), sigma=4)')
        gaus_square = GaussianSquare(duration=20, sigma=30, amp=1.0, width=3)
        self.assertEqual(repr(gaus_square),
                         'GaussianSquare(duration=20, amp=(1+0j), sigma=30, width=3)')
        drag = Drag(duration=5, amp=0.5, sigma=7, beta=1)
        self.assertEqual(repr(drag), 'Drag(duration=5, amp=(0.5+0j), sigma=7, beta=1)')
        const = Constant(duration=150, amp=0.1 + 0.4j)
        self.assertEqual(repr(const), 'Constant(duration=150, amp=(0.1+0.4j))')

    def test_complex_param_is_complex(self):
        """Check that complex param 'amp' is cast to complex."""
        const = Constant(duration=150, amp=1)
        self.assertIsInstance(const.amp, complex)

    def test_param_validation(self):
        """Test that parametric pulse parameters are validated when initialized."""
        with self.assertRaises(PulseError):
            Gaussian(duration=25, sigma=0, amp=0.5j)
        with self.assertRaises(PulseError):
            GaussianSquare(duration=150, amp=0.2, sigma=8, width=160)
        with self.assertRaises(PulseError):
            Constant(duration=150, amp=0.9 + 0.8j)
        with self.assertRaises(PulseError):
            Drag(duration=25, amp=0.2 + 0.3j, sigma=-7.8, beta=4)
        with self.assertRaises(PulseError):
            Drag(duration=25, amp=0.2 + 0.3j, sigma=7.8, beta=4j)

    def test_deprecated_parametric_pulses(self):
        """Test deprecated parametric pulses."""
        with self.assertWarns(DeprecationWarning):
            const = ConstantPulse(duration=150, amp=0.1 + 0.4j)
            self.assertEqual(const.get_waveform().samples[0], 0.1 + 0.4j)
            self.assertEqual(len(const.get_waveform().samples), 150)


# pylint: disable=invalid-name,unexpected-keyword-arg


class TestFunctionalPulse(QiskitTestCase):
    """Waveform tests."""

    def test_gaussian(self):
        """Test gaussian pulse.
        """

        @functional_pulse
        def local_gaussian(duration, amp, t0, sig):
            x = np.linspace(0, duration - 1, duration)
            return amp * np.exp(-(x - t0) ** 2 / sig ** 2)

        pulse_wf_inst = local_gaussian(duration=10, amp=1, t0=5, sig=1, name='test_pulse')
        _y = 1 * np.exp(-(np.linspace(0, 9, 10) - 5)**2 / 1**2)

        self.assertListEqual(list(pulse_wf_inst.samples), list(_y))

        # check name
        self.assertEqual(pulse_wf_inst.name, 'test_pulse')

        # check duration
        self.assertEqual(pulse_wf_inst.duration, 10)

    def test_variable_duration(self):
        """Test generation of sample pulse with variable duration.
        """

        @functional_pulse
        def local_gaussian(duration, amp, t0, sig):
            x = np.linspace(0, duration - 1, duration)
            return amp * np.exp(-(x - t0) ** 2 / sig ** 2)

        _durations = np.arange(10, 15, 1)

        for _duration in _durations:
            pulse_wf_inst = local_gaussian(duration=_duration, amp=1, t0=5, sig=1)
            self.assertEqual(len(pulse_wf_inst.samples), _duration)


if __name__ == '__main__':
    unittest.main()
