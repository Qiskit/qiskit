#include "common.h"
#include <complex.h>
#include <math.h>
#include <qiskit.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

QkExprNode *inner_build_test_expression();
static int test_expression(void) {
    QkExprNode *expr = inner_build_test_expression();
    printf("NODE TYPE %d\n", qk_expr_node_kind(expr));

    return RuntimeError;
}

int test_classical_expr(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_expression);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}

