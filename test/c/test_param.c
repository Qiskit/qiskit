// This code is part of Qiskit.
//
// (C) Copyright IBM 2025.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#include "common.h"
#include <complex.h>
#include <qiskit.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int test_param_new(void) {
    QkParam *p = qk_param_new("a");
    char* str = qk_param_to_string(p);
    qk_param_free(p);

    if (strcmp(str, "a") != 0) {
        printf("The parameter %s is not a\n", str);
        qk_str_free(str);
        return EqualityError;
    }
    qk_str_free(str);
    return Ok;
}

int test_param_binary_ops(void) {
    QkParam *a = qk_param_new("a");
    QkParam *b = qk_param_new("b");
    QkParam *ret;
    char* str;

    // add
    ret = qk_param_add(a, b);
    str = qk_param_to_string(ret);
    qk_param_free(ret);
    if (strcmp(str, "a + b") != 0) {
        printf("qk_param_add %s is not a + b\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        return EqualityError;
    }
    qk_str_free(str);

    // sub
    ret = qk_param_sub(a, b);
    str = qk_param_to_string(ret);
    qk_param_free(ret);
    if (strcmp(str, "a - b") != 0) {
        printf("qk_param_sub %s is not a - b\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        return EqualityError;
    }
    qk_str_free(str);

    // mul
    ret = qk_param_mul(a, b);
    str = qk_param_to_string(ret);
    qk_param_free(ret);
    if (strcmp(str, "a*b") != 0) {
        printf("qk_param_mul %s is not a*b\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        return EqualityError;
    }
    qk_str_free(str);

    // div
    ret = qk_param_div(a, b);
    str = qk_param_to_string(ret);
    qk_param_free(ret);
    if (strcmp(str, "a/b") != 0) {
        printf("qk_param_div %s is not a/b\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        return EqualityError;
    }
    qk_str_free(str);

    // pow
    ret = qk_param_pow(a, b);
    str = qk_param_to_string(ret);
    qk_param_free(ret);
    if (strcmp(str, "a**b") != 0) {
        printf("qk_param_pow %s is not a**b\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        return EqualityError;
    }
    qk_str_free(str);
    qk_param_free(a);
    qk_param_free(b);

    return Ok;
}

int test_param_unary_ops(void) {
    QkParam *a = qk_param_new("a");
    QkParam *b = qk_param_new("b");
    QkParam *c = qk_param_add(a, b);

    QkParam *ret;
    char* str;

    // sin
    ret = qk_param_sin(c);
    str = qk_param_to_string(ret);
    qk_param_free(ret);
    if (strcmp(str, "sin(a + b)") != 0) {
        printf("qk_param_sin %s is not sin(a + b)\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    // cos
    ret = qk_param_cos(c);
    str = qk_param_to_string(ret);
    qk_param_free(ret);
    if (strcmp(str, "cos(a + b)") != 0) {
        printf("qk_param_cos %s is not cos(a + b)\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    // tan
    ret = qk_param_tan(c);
    str = qk_param_to_string(ret);
    qk_param_free(ret);
    if (strcmp(str, "tan(a + b)") != 0) {
        printf("qk_param_tan %s is not tan(a + b)\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    // asin
    ret = qk_param_asin(c);
    str = qk_param_to_string(ret);
    qk_param_free(ret);
    if (strcmp(str, "asin(a + b)") != 0) {
        printf("qk_param_asin %s is not asin(a + b)\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    // acos
    ret = qk_param_acos(c);
    str = qk_param_to_string(ret);
    qk_param_free(ret);
    if (strcmp(str, "acos(a + b)") != 0) {
        printf("qk_param_acos %s is not acos(a + b)\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    // atan
    ret = qk_param_atan(c);
    str = qk_param_to_string(ret);
    qk_param_free(ret);
    if (strcmp(str, "atan(a + b)") != 0) {
        printf("qk_param_atan %s is not atan(a + b)\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    // log
    ret = qk_param_log(c);
    str = qk_param_to_string(ret);
    qk_param_free(ret);
    if (strcmp(str, "log(a + b)") != 0) {
        printf("qk_param_log %s is not log(a + b)\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    // exp
    ret = qk_param_exp(c);
    str = qk_param_to_string(ret);
    qk_param_free(ret);
    if (strcmp(str, "exp(a + b)") != 0) {
        printf("qk_param_exp %s is not exp(a + b)\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    // abs
    ret = qk_param_abs(c);
    str = qk_param_to_string(ret);
    qk_param_free(ret);
    if (strcmp(str, "abs(a + b)") != 0) {
        printf("qk_param_abs %s is not abs(a + b)\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    // sign
    ret = qk_param_sign(c);
    str = qk_param_to_string(ret);
    qk_param_free(ret);
    if (strcmp(str, "sign(a + b)") != 0) {
        printf("qk_param_sign %s is not sign(a + b)\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    // neg
    ret = qk_param_neg(c);
    str = qk_param_to_string(ret);
    qk_param_free(ret);
    if (strcmp(str, "-a - b") != 0) {
        printf("qk_param_neg %s is not -a - b\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    // conj
    ret = qk_param_conj(c);
    str = qk_param_to_string(ret);
    qk_param_free(ret);
    if (strcmp(str, "conj(a) + conj(b)") != 0) {
        printf("qk_param_conj %s is not conj(a) + conj(b)\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    qk_param_free(a);
    qk_param_free(b);
    qk_param_free(c);
    return Ok;
}

int test_param_with_value(void) {
    QkParam *a = qk_param_new("a");
    QkParam *v = qk_param_from_double(2.5);
    QkParam *ret;
    char* str;

    // add
    ret = qk_param_add(a, v);
    str = qk_param_to_string(ret);
    qk_param_free(ret);
    if (strcmp(str, "2.5 + a") != 0) {
        printf("qk_param_add %s is not 2.5 + a\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(v);
        return EqualityError;
    }
    qk_str_free(str);

    qk_param_free(a);
    qk_param_free(v);
    return Ok;
}

int test_param_bind(void) {
    QkParam *a = qk_param_new("a");
    QkParam *b = qk_param_new("b");
    const QkParam* params[2] = {a, b};
    double values[2] = {1.5, 2.2};

    QkParam *c = qk_param_add(a, b);
    QkParam *d = qk_param_bind(c, params, values, 2);

    qk_param_free(a);
    qk_param_free(b);
    qk_param_free(c);

    double ret;
    if (!qk_param_as_real(d, &ret)) {
        char* str = qk_param_to_string(d);
        printf("Parameter has some unbound symbols : %s\n", str);
        qk_str_free(str);
        qk_param_free(d);
        return RuntimeError;
    }

    if (ret != values[0] + values[1]) {
        printf("bound parameter %f is not %f\n", ret, values[0] + values[1]);
        qk_param_free(d);
        return EqualityError;
    }
    qk_param_free(d);
    return Ok;
}

int test_circuit(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_param_new);
    num_failed += RUN_TEST(test_param_binary_ops);
    num_failed += RUN_TEST(test_param_unary_ops);
    num_failed += RUN_TEST(test_param_with_value);
    num_failed += RUN_TEST(test_param_bind);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
