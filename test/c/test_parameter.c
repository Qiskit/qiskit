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

int test_parameter_new(void) {
    QkParam *p = qk_parameter_new("a");
    char* str = qk_parameter_to_string(p);
    qk_parameter_free(p);

    if (strcmp(str, "a") != 0) {
        printf("The parameter %s is not a", str);
        qk_str_free(str);
        return EqualityError;
    }
    qk_str_free(str);
    return Ok;
}

int test_parameter_binary_ops(void) {
    QkParam *a = qk_parameter_new("a");
    QkParam *b = qk_parameter_new("b");
    QkParam *ret;
    char* str;

    // add
    ret = qk_parameter_add(a, b);
    str = qk_parameter_to_string(ret);
    qk_parameter_free(ret);
    if (strcmp(str, "a+b") != 0) {
        printf("qk_parameter_add %s is not a+b", str);
        qk_str_free(str);
        qk_parameter_free(a);
        qk_parameter_free(b);
        return EqualityError;
    }
    qk_str_free(str);

    // sub
    ret = qk_parameter_sub(a, b);
    str = qk_parameter_to_string(ret);
    qk_parameter_free(ret);
    if (strcmp(str, "a-b") != 0) {
        printf("qk_parameter_sub %s is not a-b", str);
        qk_str_free(str);
        qk_parameter_free(a);
        qk_parameter_free(b);
        return EqualityError;
    }
    qk_str_free(str);

    // mul
    ret = qk_parameter_mul(a, b);
    str = qk_parameter_to_string(ret);
    qk_parameter_free(ret);
    if (strcmp(str, "a*b") != 0) {
        printf("qk_parameter_mul %s is not a*b", str);
        qk_str_free(str);
        qk_parameter_free(a);
        qk_parameter_free(b);
        return EqualityError;
    }
    qk_str_free(str);

    // div
    ret = qk_parameter_div(a, b);
    str = qk_parameter_to_string(ret);
    qk_parameter_free(ret);
    if (strcmp(str, "a/b") != 0) {
        printf("qk_parameter_div %s is not a/b", str);
        qk_str_free(str);
        qk_parameter_free(a);
        qk_parameter_free(b);
        return EqualityError;
    }
    qk_str_free(str);

    // pow
    ret = qk_parameter_pow(a, b);
    str = qk_parameter_to_string(ret);
    qk_parameter_free(ret);
    if (strcmp(str, "a**b") != 0) {
        printf("qk_parameter_pow %s is not a**b", str);
        qk_str_free(str);
        qk_parameter_free(a);
        qk_parameter_free(b);
        return EqualityError;
    }
    qk_str_free(str);
    qk_parameter_free(a);
    qk_parameter_free(b);

    return Ok;
}

int test_parameter_unary_ops(void) {
    QkParam *a = qk_parameter_new("a");
    QkParam *b = qk_parameter_new("b");
    QkParam *c = qk_parameter_add(a, b);

    QkParam *ret;
    char* str;

    // sin
    ret = qk_parameter_sin(c);
    str = qk_parameter_to_string(ret);
    qk_parameter_free(ret);
    if (strcmp(str, "sin(a+b)") != 0) {
        printf("qk_parameter_sin %s is not sin(a+b)", str);
        qk_str_free(str);
        qk_parameter_free(a);
        qk_parameter_free(b);
        qk_parameter_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    // cos
    ret = qk_parameter_cos(c);
    str = qk_parameter_to_string(ret);
    qk_parameter_free(ret);
    if (strcmp(str, "cos(a+b)") != 0) {
        printf("qk_parameter_cos %s is not cos(a+b)", str);
        qk_str_free(str);
        qk_parameter_free(a);
        qk_parameter_free(b);
        qk_parameter_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    // tan
    ret = qk_parameter_tan(c);
    str = qk_parameter_to_string(ret);
    qk_parameter_free(ret);
    if (strcmp(str, "tan(a+b)") != 0) {
        printf("qk_parameter_tan %s is not tan(a+b)", str);
        qk_str_free(str);
        qk_parameter_free(a);
        qk_parameter_free(b);
        qk_parameter_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    // asin
    ret = qk_parameter_asin(c);
    str = qk_parameter_to_string(ret);
    qk_parameter_free(ret);
    if (strcmp(str, "asin(a+b)") != 0) {
        printf("qk_parameter_asin %s is not asin(a+b)", str);
        qk_str_free(str);
        qk_parameter_free(a);
        qk_parameter_free(b);
        qk_parameter_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    // acos
    ret = qk_parameter_acos(c);
    str = qk_parameter_to_string(ret);
    qk_parameter_free(ret);
    if (strcmp(str, "acos(a+b)") != 0) {
        printf("qk_parameter_acos %s is not acos(a+b)", str);
        qk_str_free(str);
        qk_parameter_free(a);
        qk_parameter_free(b);
        qk_parameter_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    // atan
    ret = qk_parameter_atan(c);
    str = qk_parameter_to_string(ret);
    qk_parameter_free(ret);
    if (strcmp(str, "atan(a+b)") != 0) {
        printf("qk_parameter_atan %s is not atan(a+b)", str);
        qk_str_free(str);
        qk_parameter_free(a);
        qk_parameter_free(b);
        qk_parameter_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    // log
    ret = qk_parameter_log(c);
    str = qk_parameter_to_string(ret);
    qk_parameter_free(ret);
    if (strcmp(str, "log(a+b)") != 0) {
        printf("qk_parameter_log %s is not log(a+b)", str);
        qk_str_free(str);
        qk_parameter_free(a);
        qk_parameter_free(b);
        qk_parameter_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    // exp
    ret = qk_parameter_exp(c);
    str = qk_parameter_to_string(ret);
    qk_parameter_free(ret);
    if (strcmp(str, "exp(a+b)") != 0) {
        printf("qk_parameter_exp %s is not exp(a+b)", str);
        qk_str_free(str);
        qk_parameter_free(a);
        qk_parameter_free(b);
        qk_parameter_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    // abs
    ret = qk_parameter_abs(c);
    str = qk_parameter_to_string(ret);
    qk_parameter_free(ret);
    if (strcmp(str, "abs(a+b)") != 0) {
        printf("qk_parameter_abs %s is not abs(a+b)", str);
        qk_str_free(str);
        qk_parameter_free(a);
        qk_parameter_free(b);
        qk_parameter_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    // sign
    ret = qk_parameter_sign(c);
    str = qk_parameter_to_string(ret);
    qk_parameter_free(ret);
    if (strcmp(str, "sign(a+b)") != 0) {
        printf("qk_parameter_sign %s is not sign(a+b)", str);
        qk_str_free(str);
        qk_parameter_free(a);
        qk_parameter_free(b);
        qk_parameter_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    // neg
    ret = qk_parameter_neg(c);
    str = qk_parameter_to_string(ret);
    qk_parameter_free(ret);
    if (strcmp(str, "-(a+b)") != 0) {
        printf("qk_parameter_neg %s is not -(a+b)", str);
        qk_str_free(str);
        qk_parameter_free(a);
        qk_parameter_free(b);
        qk_parameter_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    // conj
    ret = qk_parameter_conj(c);
    str = qk_parameter_to_string(ret);
    qk_parameter_free(ret);
    if (strcmp(str, "conj(a+b)") != 0) {
        printf("qk_parameter_conj %s is not conj(a+b)", str);
        qk_str_free(str);
        qk_parameter_free(a);
        qk_parameter_free(b);
        qk_parameter_free(c);
        return EqualityError;
    }
    qk_str_free(str);

    qk_parameter_free(a);
    qk_parameter_free(b);
    qk_parameter_free(c);
    return Ok;
}

int test_parameter_with_value(void) {
    QkParam *a = qk_parameter_new("a");
    QkParam *v = qk_parameter_from_value(2.5);
    QkParam *ret;
    char* str;

    // add
    ret = qk_parameter_add(a, v);
    str = qk_parameter_to_string(ret);
    qk_parameter_free(ret);
    if (strcmp(str, "2.5+a") != 0) {
        printf("qk_parameter_add %s is not 2.5+a", str);
        qk_str_free(str);
        qk_parameter_free(a);
        qk_parameter_free(v);
        return EqualityError;
    }
    qk_str_free(str);

    qk_parameter_free(a);
    qk_parameter_free(v);
    return Ok;
}

int test_parameter_bind(void) {
    QkParam *a = qk_parameter_new("a");
    QkParam *b = qk_parameter_new("b");
    QkParam* params[2] = {a, b};
    double values[2] = {1.5, 2.2};

    QkParam *c = qk_parameter_add(a, b);
    QkParam *d = qk_parameter_bind(c, params, values, 2);

    qk_parameter_free(a);
    qk_parameter_free(b);
    qk_parameter_free(c);

    double ret;
    if (!qk_parameter_as_real(d, &ret)) {
        char* str = qk_parameter_to_string(d);
        printf("Parameter has some unbound symbols : %s", str);
        qk_str_free(str);
        qk_parameter_free(d);
        return RuntimeError;
    }

    if (ret != values[0] + values[1]) {
        printf("bound parameter %f is not %f", ret, values[0] + values[1]);
        qk_parameter_free(d);
        return EqualityError;
    }
    qk_parameter_free(d);
    return Ok;
}


int test_circuit(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_parameter_new);
    num_failed += RUN_TEST(test_parameter_binary_ops);
    num_failed += RUN_TEST(test_parameter_unary_ops);
    num_failed += RUN_TEST(test_parameter_with_value);
    num_failed += RUN_TEST(test_parameter_bind);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
