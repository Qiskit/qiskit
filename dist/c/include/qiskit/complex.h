#ifndef QISKIT__COMPLEX_H
#define QISKIT__COMPLEX_H

/// A complex double.
///
/// See also ``qk_complex64_to_native`` and ``qk_complex64_from_native`` to convert
/// this struct to (or from) a compiler-native complex number representation.
typedef struct {
    /// The real part.
    double re;
    /// The imaginary part.
    double im;
} QkComplex64;

// Complex number typedefs conversions.
#ifdef __cplusplus
#include <complex>
static std::complex<double> qk_complex64_to_native(QkComplex64 *value) {
    return std::complex<double>(value->re, value->im);
}
static QkComplex64 qk_complex64_from_native(std::complex<double> *value) {
    return (QkComplex64){value->real(), value->imag()};
}
#else //__cplusplus
#include <complex.h>

#ifdef _MSC_VER
static _Dcomplex qk_complex64_to_native(QkComplex64 *value) {
    return (_Dcomplex){value->re, value->im};
}
static QkComplex64 qk_complex64_from_native(_Dcomplex *value) {
    return (QkComplex64){creal(*value), cimag(*value)};
}
#else
/// @ingroup QkComplex64
/// Convert a ``QkComplex64`` to a compiler-native complex number representation.
///
/// @param value A pointer to the ``QkComplex64`` to convert.
/// @return A native representation of the complex number.
///
/// # Example
///
/// Assuming a GNU/clang compiler with ``complex double`` as native complex number, we have
///
///     QkComplex64 qk_value = {1, 1}; // represents 1 + i
///     complex double value = qk_complex64_to_native(&qk_value);
///
/// # Safety
///
/// Behavior is undefined if ``value`` is not a valid, non-null pointer to a ``QkComplex64``.
static double complex qk_complex64_to_native(QkComplex64 *value) {
    return value->re + I * value->im;
}

/// @ingroup QkComplex64
/// Convert a compiler-native complex number to a ``QkComplex64``.
///
/// @param value A pointer to the native complex number.
/// @return The ``QkComplex64`` representation.
///
/// # Example
///
/// Assuming a GNU/clang compiler with ``complex double`` as native complex number, we have
///
///     complex double value = 1 + I;
///     QkComplex64 qk_value = qk_complex64_from_native(&value);
///
/// # Safety
///
/// Behavior is undefined if ``value`` is not a valid, non-null pointer to a complex number.
static QkComplex64 qk_complex64_from_native(complex double *value) {
    return (QkComplex64){creal(*value), cimag(*value)};
}
#endif // _MSC_VER

#endif //__cplusplus

#endif // QISKIT__COMPLEX_H
