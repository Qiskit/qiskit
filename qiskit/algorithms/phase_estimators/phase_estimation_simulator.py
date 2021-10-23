import numpy
import scipy


class PhaseEstimationSimulator:
    """Simulate quantum phase estimation given a list of eigenphases and
    a list of expansion coefficients.

    This class simulates the parallel QPE algorithm using a classical discrete
    Fourier transform, but no quantum circuits.

    The (parallel) QPE algorithm takes as input a unitary, a vector that it
    operates on, and the desired number of qubits in the phase-readout register.
    The output is an array of the probabilites of measuring each bitstring in
    the readout register. The bitstrings are mapped to a discrete set of
    phases. In the ideal case, the output depends on the input via the spectrum
    of the unitary (the phases) and the moduli of the coefficients of the
    expansion of the input vector in the corresponding eigenvectors. In
    particular, phase information in the expansion coefficients does not enter
    into the bitstring probabilities.

    `PhaseEstimationSimulator` takes as input a list of eigenphases and a
    corresponding list of expansion coefficients, which are stored as object
    properties of the class. It is only required to supply nonzero expansion
    coefficients. The method `estimate_phases` takes as input the number of
    qubits in the phase-readout register and returns an array of the bitstring
    probabilities. Alternatively, the method `from_eigenproblem` constructs and
    stores the arrays of phases and coeffcients from a unitary and input vector.

    The module-level function `_array_to_qiskit_endian` converts the array of
    output probabilties to the order of amplitudes returned by the qiskit
    statevector simulator.

    This simulator is faster than constructing and simulating a circuit. It is
    far simpler, as well, and is based only on well-tested components of numpy
    and scipy.

    """

    def __init__(self, phases=None, coeffs=None):
        self.phases = phases
        self.coeffs = coeffs

    def from_eigenproblem(self, unitary, input_vector):
        """Compute the phases and coefficients from a unitary and input vector.

        The probabilities returned by the QPE algorithm depend only on the
        eigen-phases and coeffcients of the input vector in the basis of the
        unitary. This function computes and stores these phases from the input
        unitary and vector.
        """
        vals, vecs = scipy.linalg.eig(unitary)
        vecs = vecs.transpose() # put eigenvecs in columns
        # Convert eigenvalue of unitary to a phase.
        coeffs = [numpy.vdot(input_vector, vec) for vec in vecs]
        norm_fac = numpy.sqrt(sum(x * x.conjugate() for x in coeffs))
        coeffs = [x / norm_fac for x in coeffs]
        phases = (numpy.angle(vals.astype(complex)) / (2 * numpy.pi)).real
        self.phases = phases
        self.coeffs = coeffs

    def estimate_phases(self, num_qubits):
        """Return an array of probabilities of measurement of each bitstring.

        The probabilities are computed from eigen-phases and expansion coefficients
        that are either set previously or computed via `from_eigenproblem`.
        """
        n = 2 ** num_qubits
        prob = numpy.zeros(n)
        for phi, c in zip(self.phases, self.coeffs):
            if c != 0:
                amps = _phase_estimation_amplitudes(phi, num_qubits)
                prob += numpy.abs(c)**2 * numpy.abs(amps)**2
        return prob


def _intermediate_phase_vec(phi, num_qubits):
    n = 2 ** num_qubits
    inds = numpy.arange(n, 0, -1)
    v = numpy.exp(2 * numpy.pi * 1.0j * phi * inds) / numpy.sqrt(n)
    return v

def _phase_estimation_amplitudes(phi, num_qubits):
    v = _intermediate_phase_vec(phi, num_qubits)
    return numpy.fft.ifft(v) * numpy.sqrt(len(v))

def _ind_to_qiskit_endian(ind, n):
    return int(numpy.binary_repr(ind, width=n)[::-1], base=2)

def _array_to_qiskit_endian(a):
    """Permutate array `a` to qiskit qubit register endianness.

    Endian refers here to bit order, not byte order. The indices
    into `a` are assumed to correspond to bitstrings given by their
    binary representations. The indices of the output array correspond
    to bit-wise reversal of their binary representation.

    For example, assume `a` is an array of bitstring probabilties computed by
    QPE simulation, with bitstrings in the order "0...0", "0...1", etc.
    Then the output array corresponds to the probabilities output by a QPE
    algorithm run on the statevector simulator.
    """
    n = int(numpy.log2(len(a)))
    alist = [a[_ind_to_qiskit_endian(i, n)] for i in range(0, len(a))]
    return numpy.array(alist)
