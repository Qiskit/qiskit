#ifndef QISKIT__COMPLEX_H
#define QISKIT__COMPLEX_H

/**
 * A complex double.
 *
 * See also ``qk_complex64_to_native`` and ``qk_complex64_from_native`` to convert
 * this struct to (or from) a compiler-native complex number representation.
 */
typedef struct {
    /* The real part. */
    double re;
    /* The imaginary part. */
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
static double _Complex qk_complex64_to_native(QkComplex64 *value) {
    return value->re + I * value->im;
}
static QkComplex64 qk_complex64_from_native(double _Complex *value) {
    return (QkComplex64){creal(*value), cimag(*value)};
}
#endif // _MSC_VER

#endif //__cplusplus

#endif // QISKIT__COMPLEX_H
