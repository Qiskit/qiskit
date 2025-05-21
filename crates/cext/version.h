#define QISKIT_VERSION_MAJOR 2
#define QISKIT_VERSION_MINOR 1
#define QISKIT_VERSION_PATCH 0

#define QISKIT_VERSION_NUMERIC(M, m, p) ((M) << 16 | (m) << 8 | (p))
#define QISKIT_VERSION                                                                             \
    (QISKIT_VERSION_MAJOR << 16 | QISKIT_VERSION_MINOR << 8 | QISKIT_VERSION_PATCH)