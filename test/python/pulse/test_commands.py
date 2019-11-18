# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name,unexpected-keyword-arg

"""Test cases for the pulse command group."""

import unittest
import numpy as np

from qiskit.pulse import (SamplePulse, Acquire, FrameChange, PersistentValue,
                          Snapshot, Kernel, Discriminator, functional_pulse,
                          Delay, PulseError, ConstantPulse, Gaussian, GaussianSquare,
                          Drag, pulse_lib)
from qiskit.test import QiskitTestCase


class TestSamplePulse(QiskitTestCase):
    """SamplePulse tests."""

    def test_sample_pulse(self):
        """Test pulse initialization."""
        n_samples = 100
        samples = np.linspace(0, 1., n_samples, dtype=np.complex128)
        name = 'test'
        sample_pulse = SamplePulse(samples, name=name)

        self.assertEqual(sample_pulse.samples.dtype, np.complex128)
        np.testing.assert_almost_equal(sample_pulse.samples, samples)

        self.assertEqual(sample_pulse.duration, n_samples)
        self.assertEqual(sample_pulse.name, name)

    def test_type_casting(self):
        """Test casting of input samples to numpy array."""
        n_samples = 100
        samples_f64 = np.linspace(0, 1., n_samples, dtype=np.float64)

        sample_pulse_f64 = SamplePulse(samples_f64)
        self.assertEqual(sample_pulse_f64.samples.dtype, np.complex128)

        samples_c64 = np.linspace(0, 1., n_samples, dtype=np.complex64)

        sample_pulse_c64 = SamplePulse(samples_c64)
        self.assertEqual(sample_pulse_c64.samples.dtype, np.complex128)

        samples_list = np.linspace(0, 1., n_samples).tolist()

        sample_pulse_list = SamplePulse(samples_list)
        self.assertEqual(sample_pulse_list.samples.dtype, np.complex128)

    def test_pulse_limits(self):
        """Test that limits of pulse norm of one are enforced properly."""

        # test norm is correct for complex128 numpy data
        unit_pulse_c128 = np.exp(1j*2*np.pi*np.linspace(0, 1, 1000), dtype=np.complex128)
        # test does not raise error
        try:
            SamplePulse(unit_pulse_c128)
        except PulseError:
            self.fail('SamplePulse incorrectly failed on approximately unit norm samples.')

        invalid_const = 1.1
        with self.assertRaises(PulseError):
            SamplePulse(invalid_const*np.exp(1j*2*np.pi*np.linspace(0, 1, 1000)))

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
            SamplePulse(recombined_pulse)
        except PulseError:
            self.fail('SamplePulse incorrectly failed to approximately unit norm samples.')


class TestAcquire(QiskitTestCase):
    """Acquisition tests."""

    def test_can_construct_valid_acquire_command(self):
        """Test if valid acquire command can be constructed.
        """
        kernel_opts = {
            'start_window': 0,
            'stop_window': 10
        }
        kernel = Kernel(name='boxcar', **kernel_opts)

        discriminator_opts = {
            'neighborhoods': [{'qubits': 1, 'channels': 1}],
            'cal': 'coloring',
            'resample': False
        }
        discriminator = Discriminator(name='linear_discriminator', **discriminator_opts)

        acq_command = Acquire(duration=10, kernel=kernel, discriminator=discriminator)

        self.assertEqual(acq_command.duration, 10)
        self.assertEqual(acq_command.discriminator.name, 'linear_discriminator')
        self.assertEqual(acq_command.discriminator.params, discriminator_opts)
        self.assertEqual(acq_command.kernel.name, 'boxcar')
        self.assertEqual(acq_command.kernel.params, kernel_opts)
        self.assertTrue(acq_command.name.startswith('acq'))

    def test_can_construct_acquire_command_with_default_values(self):
        """Test if an acquire command can be constructed with default discriminator and kernel.
        """
        acq_command_a = Acquire(duration=10)
        acq_command_b = Acquire(duration=10)

        self.assertEqual(acq_command_a.duration, 10)
        self.assertEqual(acq_command_a.discriminator, None)
        self.assertEqual(acq_command_a.kernel, None)
        self.assertTrue(acq_command_a.name.startswith('acq'))
        self.assertNotEqual(acq_command_a.name, acq_command_b.name)
        self.assertEqual(acq_command_b.name, 'acq' + str(int(acq_command_a.name[3:]) + 1))


class TestFrameChangeCommand(QiskitTestCase):
    """FrameChange tests."""

    def test_default(self):
        """Test default frame change.
        """
        fc_command = FrameChange(phase=1.57)

        self.assertEqual(fc_command.phase, 1.57)
        self.assertEqual(fc_command.duration, 0)
        self.assertTrue(fc_command.name.startswith('fc'))


class TestPersistentValueCommand(QiskitTestCase):
    """PersistentValue tests."""

    def test_default(self):
        """Test default persistent value.
        """
        pv_command = PersistentValue(value=0.5 - 0.5j)

        self.assertEqual(pv_command.value, 0.5-0.5j)
        self.assertEqual(pv_command.duration, 0)
        self.assertTrue(pv_command.name.startswith('pv'))


class TestSnapshotCommand(QiskitTestCase):
    """Snapshot tests."""

    def test_default(self):
        """Test default snapshot.
        """
        snap_command = Snapshot(label='test_name', snapshot_type='state')

        self.assertEqual(snap_command.name, "test_name")
        self.assertEqual(snap_command.type, "state")
        self.assertEqual(snap_command.duration, 0)


class TestDelayCommand(QiskitTestCase):
    """Delay tests."""

    def test_delay(self):
        """Test delay."""
        delay_command = Delay(10, name='test_name')

        self.assertEqual(delay_command.name, "test_name")
        self.assertEqual(delay_command.duration, 10)


class TestFunctionalPulse(QiskitTestCase):
    """SamplePulse tests."""

    def test_gaussian(self):
        """Test gaussian pulse.
        """

        @functional_pulse
        def gaussian(duration, amp, t0, sig):
            x = np.linspace(0, duration - 1, duration)
            return amp * np.exp(-(x - t0) ** 2 / sig ** 2)

        pulse_command = gaussian(duration=10, name='test_pulse', amp=1, t0=5, sig=1)
        _y = 1 * np.exp(-(np.linspace(0, 9, 10) - 5)**2 / 1**2)

        self.assertListEqual(list(pulse_command.samples), list(_y))

        # check name
        self.assertEqual(pulse_command.name, 'test_pulse')

        # check duration
        self.assertEqual(pulse_command.duration, 10)

    def test_variable_duration(self):
        """Test generation of sample pulse with variable duration.
        """

        @functional_pulse
        def gaussian(duration, amp, t0, sig):
            x = np.linspace(0, duration - 1, duration)
            return amp * np.exp(-(x - t0) ** 2 / sig ** 2)

        _durations = np.arange(10, 15, 1)

        for _duration in _durations:
            pulse_command = gaussian(duration=_duration, amp=1, t0=5, sig=1)
            self.assertEqual(len(pulse_command.samples), _duration)


class TestKernel(QiskitTestCase):
    """Kernel tests."""

    def test_can_construct_kernel_with_default_values(self):
        """Test if Kernel can be constructed with default name and params."""
        kernel = Kernel()

        self.assertEqual(kernel.name, None)
        self.assertEqual(kernel.params, {})


class TestDiscriminator(QiskitTestCase):
    """Discriminator tests."""

    def test_can_construct_discriminator_with_default_values(self):
        """Test if Discriminator can be constructed with default name and params."""
        discriminator = Discriminator()

        self.assertEqual(discriminator.name, None)
        self.assertEqual(discriminator.params, {})


class TestParametricPulses(QiskitTestCase):
    """Tests for all subclasses of ParametricPulse."""

    def test_construction(self):
        """Test that parametric pulses can be constructed without error."""
        gauss = Gaussian(duration=25, sigma=16, amp=0.5j)
        gaussquare = GaussianSquare(duration=150, amp=0.2, sigma=8, width=140)
        const = ConstantPulse(duration=150, amp=0.1 + 0.4j)
        drag = Drag(duration=25, amp=0.2 + 0.3j, sigma=7.8, beta=4)

    def test_sampled_pulse(self):
        """Test that we can convert to a sampled pulse."""
        gauss = Gaussian(duration=25, sigma=16, amp=0.5j)
        sample_pulse = gauss.get_sample_pulse()
        self.assertIsInstance(sample_pulse, SamplePulse)
        pulse_lib_gaus = pulse_lib.gaussian(duration=25, sigma=16,
                                            amp=0.5j, zero_ends=False).samples
        np.testing.assert_almost_equal(sample_pulse.samples, pulse_lib_gaus)

    def test_samples(self):
        """Test that the gaussian samples match the formula."""
        duration = 25
        sigma = 16
        amp = 0.5j
        # formulaic
        times = np.array(range(25), dtype=np.complex_)
        times = times - (duration / 2) + 0.5
        gauss = amp * np.exp(-(times / sigma)**2 / 2)
        # command
        command = Gaussian(duration=duration, sigma=sigma, amp=amp)
        samples = command.get_sample_pulse().samples
        np.testing.assert_almost_equal(gauss, samples)

    def test_constant(self):
        """Test the constant pulse and its sampled construction."""
        const = ConstantPulse(duration=150, amp=0.1 + 0.4j)
        self.assertEqual(const.get_sample_pulse().samples[0], 0.1 + 0.4j)
        self.assertEqual(len(const.get_sample_pulse().samples), 150)

    def test_get_params(self):
        """Test that the parameters can be extracted as a dict through `get_params`."""
        drag = Drag(duration=25, amp=0.2 + 0.3j, sigma=7.8, beta=4)
        self.assertEqual(set(drag.get_params().keys()), {'duration', 'amp', 'sigma', 'beta'})
        const = ConstantPulse(duration=150, amp=1)
        self.assertEqual(set(const.get_params().keys()), {'duration', 'amp'})

    def test_complex_param_is_complex(self):
        """Check that complex param 'amp' is cast to complex."""
        const = ConstantPulse(duration=150, amp=1)
        self.assertIsInstance(const.amp, complex)


if __name__ == '__main__':
    unittest.main()
