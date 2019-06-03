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

"""Reference circuits used by the tests."""

import math

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister


class ReferenceCircuits:
    """Container for reference circuits used by the tests."""

    @staticmethod
    def bell():
        """Return a Bell circuit."""
        qr = QuantumRegister(2, name='qr')
        cr = ClassicalRegister(2, name='qc')
        qc = QuantumCircuit(qr, cr, name='bell')
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr, cr)

        return qc

    @staticmethod
    def bell_no_measure():
        """Return a Bell circuit."""
        qr = QuantumRegister(2, name='qr')
        cr = ClassicalRegister(2, name='cr')
        qc = QuantumCircuit(qr, cr, name='bell_no_measure')
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])

        return qc

    @staticmethod
    def h_measure_h():
        """Return circuit with two Hadamard gates and a measurement in between."""
        qr = QuantumRegister(2, name='qr')
        cr = ClassicalRegister(2, name='cr')
        qc = QuantumCircuit(qr, cr, name='h_measure_h')
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        qc.h(qr[0])

        return qc

    @staticmethod
    def h_measure_h_double():
        """Return circuit with two Hadamard gates and a measurement after each one."""
        qr = QuantumRegister(2, name='qr')
        cr = ClassicalRegister(2, name='cr')
        qc = QuantumCircuit(qr, cr, name='h_measure_h_double')
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])

        return qc

    @staticmethod
    def rx_measure_rx():
        """Return circuit with two R_x(pi/3) gates and a measurement after each one."""
        qr = QuantumRegister(2, name='qr')
        cr = ClassicalRegister(2, name='cr')
        qc = QuantumCircuit(qr, cr, name='rx_measure_rx')
        qc.rx(math.pi/3, qr[0])
        qc.measure(qr[0], cr[0])
        qc.rx(math.pi/3, qr[0])
        qc.measure(qr[0], cr[0])

        return qc
