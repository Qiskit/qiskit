# `qiskit-cext`

This crate contains the bindings for Qiskit's C API.

## Building the library

The C bindings are compiled into a shared library, which can be built along with the header file
by running
```bash
make c
```
in the root of the repository. The header file, `qiskit.h`, is generated using
[cbindgen](https://github.com/mozilla/cbindgen) and stored in `dist/c/include`.
Similarly, the `libqiskit` shared library is stored in `dist/c/lib`.

You can ask Make to build only the header file with `make cheader`, or only the
shared-object library with `make clib`.
Instead of ``make clib`` the shared C library can also be compiled via
```bash
cargo rustc --release --crate-type cdylib -p qiskit-cext
```
note that the `crate-type` should be defined explicitly to build the `cdylib` instead of the `rlib` default.

The following example uses the header to build a 100-qubit observable:
```c
#include <complex.h>
#include <qiskit.h>
#include <stdint.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    // build a 100-qubit empty observable
    uint32_t num_qubits = 100;
    QkObs *obs = qk_obs_zero(num_qubits);

    // add the term 2 * (X0 Y1 Z2) to the observable
    complex double coeff = 2;
    QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
    uint32_t indices[3] = {0, 1, 2};
    QkObsTerm term = {coeff, 3, bit_terms, indices, num_qubits};
    qk_obs_add_term(obs, &term);

    // print some properties
    printf("num_qubits: %u\n", qk_obs_num_qubits(obs));
    printf("num_terms: %lu\n", qk_obs_num_terms(obs));

    // free the memory allocated for the observable
    qk_obs_free(obs);

    return 0;
}
```
Refer to the C API documentation for more information and examples.

## Compiling

The above program can be compiled by including the header and linking to the `qiskit` library, which
are located in the standard directory configuration whose root is `dist/c`.
```bash
make c
gcc program.c -I$/path/to/dist/c/include -lqiskit -L/path/to/dist/c/lib
```

The example program will then output
```bash
./a.out
```
```text
num_qubits: 100
num_terms: 1
```
