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
#include <qiskit.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int test_param_new(void) {
    QkParam *p = qk_param_new_symbol("a");
    char *str = qk_param_str(p);
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
    QkParam *a = qk_param_new_symbol("a");
    QkParam *b = qk_param_new_symbol("b");
    QkParam *ret = qk_param_zero();
    char *str;

    // add
    qk_param_add(ret, a, b);
    str = qk_param_str(ret);
    if (strcmp(str, "a + b") != 0) {
        printf("qk_param_add %s is not a + b\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(ret);
        return EqualityError;
    }
    qk_str_free(str);

    // sub
    qk_param_sub(ret, a, b);
    str = qk_param_str(ret);
    if (strcmp(str, "a - b") != 0) {
        printf("qk_param_sub %s is not a - b\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(ret);
        return EqualityError;
    }
    qk_str_free(str);

    // mul
    qk_param_mul(ret, a, b);
    str = qk_param_str(ret);
    if (strcmp(str, "a*b") != 0) {
        printf("qk_param_mul %s is not a*b\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(ret);
        return EqualityError;
    }
    qk_str_free(str);

    // div
    qk_param_div(ret, a, b);
    str = qk_param_str(ret);
    if (strcmp(str, "a/b") != 0) {
        printf("qk_param_div %s is not a/b\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(ret);
        return EqualityError;
    }
    qk_str_free(str);

    // pow
    qk_param_pow(ret, a, b);
    str = qk_param_str(ret);
    if (strcmp(str, "a**b") != 0) {
        printf("qk_param_pow %s is not a**b\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(ret);
        return EqualityError;
    }
    qk_str_free(str);
    qk_param_free(a);
    qk_param_free(b);
    qk_param_free(ret);

    return Ok;
}

int test_param_unary_ops(void) {
    QkParam *a = qk_param_new_symbol("a");
    QkParam *b = qk_param_new_symbol("b");
    QkParam *c = qk_param_zero();
    qk_param_add(c, a, b);

    QkParam *ret = qk_param_zero();
    char *str;

    // sin
    qk_param_sin(ret, c);
    str = qk_param_str(ret);
    if (strcmp(str, "sin(a + b)") != 0) {
        printf("qk_param_sin %s is not sin(a + b)\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        qk_param_free(ret);
        return EqualityError;
    }
    qk_str_free(str);

    // cos
    qk_param_cos(ret, c);
    str = qk_param_str(ret);
    if (strcmp(str, "cos(a + b)") != 0) {
        printf("qk_param_cos %s is not cos(a + b)\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        qk_param_free(ret);
        return EqualityError;
    }
    qk_str_free(str);

    // tan
    qk_param_tan(ret, c);
    str = qk_param_str(ret);
    if (strcmp(str, "tan(a + b)") != 0) {
        printf("qk_param_tan %s is not tan(a + b)\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        qk_param_free(ret);
        return EqualityError;
    }
    qk_str_free(str);

    // asin
    qk_param_asin(ret, c);
    str = qk_param_str(ret);
    if (strcmp(str, "asin(a + b)") != 0) {
        printf("qk_param_asin %s is not asin(a + b)\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        qk_param_free(ret);
        return EqualityError;
    }
    qk_str_free(str);

    // acos
    qk_param_acos(ret, c);
    str = qk_param_str(ret);
    if (strcmp(str, "acos(a + b)") != 0) {
        printf("qk_param_acos %s is not acos(a + b)\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        qk_param_free(ret);
        return EqualityError;
    }
    qk_str_free(str);

    // atan
    qk_param_atan(ret, c);
    str = qk_param_str(ret);
    if (strcmp(str, "atan(a + b)") != 0) {
        printf("qk_param_atan %s is not atan(a + b)\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        qk_param_free(ret);
        return EqualityError;
    }
    qk_str_free(str);

    // log
    qk_param_log(ret, c);
    str = qk_param_str(ret);
    if (strcmp(str, "log(a + b)") != 0) {
        printf("qk_param_log %s is not log(a + b)\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        qk_param_free(ret);
        return EqualityError;
    }
    qk_str_free(str);

    // exp
    qk_param_exp(ret, c);
    str = qk_param_str(ret);
    if (strcmp(str, "exp(a + b)") != 0) {
        printf("qk_param_exp %s is not exp(a + b)\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        qk_param_free(ret);
        return EqualityError;
    }
    qk_str_free(str);

    // abs
    qk_param_abs(ret, c);
    str = qk_param_str(ret);
    if (strcmp(str, "abs(a + b)") != 0) {
        printf("qk_param_abs %s is not abs(a + b)\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        qk_param_free(ret);
        return EqualityError;
    }
    qk_str_free(str);

    // sign
    qk_param_sign(ret, c);
    str = qk_param_str(ret);
    if (strcmp(str, "sign(a + b)") != 0) {
        printf("qk_param_sign %s is not sign(a + b)\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        qk_param_free(ret);
        return EqualityError;
    }
    qk_str_free(str);

    // neg
    qk_param_neg(ret, c);
    str = qk_param_str(ret);
    if (strcmp(str, "-a - b") != 0) {
        printf("qk_param_neg %s is not -a - b\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        qk_param_free(ret);
        return EqualityError;
    }
    qk_str_free(str);

    // conj
    qk_param_conjugate(ret, c);
    str = qk_param_str(ret);
    if (strcmp(str, "conj(a) + conj(b)") != 0) {
        printf("qk_param_conj %s is not conj(a) + conj(b)\n", str);
        qk_str_free(str);
        qk_param_free(a);
        qk_param_free(b);
        qk_param_free(c);
        qk_param_free(ret);
        return EqualityError;
    }
    qk_str_free(str);

    qk_param_free(a);
    qk_param_free(b);
    qk_param_free(c);
    qk_param_free(ret);
    return Ok;
}

int test_param_with_value(void) {
    QkParam *a = qk_param_new_symbol("a");
    QkParam *v = qk_param_from_double(2.5);
    QkParam *ret = qk_param_zero();
    char *str;

    // add
    qk_param_add(ret, a, v);
    str = qk_param_str(ret);
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
    QkParam *a = qk_param_new_symbol("a");
    QkParam *b = qk_param_new_symbol("b");
    const QkParam *params[2] = {a, b};
    double values[2] = {1.5, 2.2};

    QkParam *c = qk_param_zero();
    qk_param_add(c, a, b);
    QkParam *d = qk_param_zero();
    qk_param_bind(d, c, params, values, 2);

    qk_param_free(a);
    qk_param_free(b);
    qk_param_free(c);

    double ret;
    if (!qk_param_as_real(&ret, d)) {
        char *str = qk_param_str(d);
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

int test_param(void) {
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
