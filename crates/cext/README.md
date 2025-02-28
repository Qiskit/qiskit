# `qiskit-cext`

This crate contains the bindings for Qiskit's C API. 

## Building the library

The C bindings are compiled into a shared library, which can be built along with the header file
by running
```
$ make cheader
```
in the root of the repository. The header file, `qiskit.h`, is generated using 
[cbindgen](https://github.com/mozilla/cbindgen) and stored in `dist/c/include`.

The following example uses the header to build a 100-qubit observable:
```c
#include <complex.h>
#include <qiskit.h>
#include <stdint.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    // build a 100-qubit empty observable
    uint32_t num_qubits = 100;
    QkSparseObservable *obs = qk_obs_zero(num_qubits);

    // add the term 2 * (X0 Y1 Z2) to the observable
    complex double coeff = 2;
    QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
    uint32_t indices[3] = {0, 1, 2};
    QkSparseTerm term = {coeff, 3, bit_terms, indices, num_qubits};
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

The above program can be compiled by including the header and linking to the `qiskit_cext` library,
which is located in `target/release`. (The exact name depends on the platform, e.g.,
`target/release/libqiskit_cext.dylib` on MacOS.) 

```bash
make cheader
gcc <program.c> -lqiskit_cext -L /path/to/target/release  -I /path/to/dist/c/include
```

For which the example program will output
```bash
./a.out
```
```text
num_qubits: 100
num_terms: 1
```
