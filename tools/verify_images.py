#!/usr/bin/env python3
# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility script to verify that all images have alt text"""

from pathlib import Path
import multiprocessing
import sys
import glob

# Dictionary to allowlist lines of code that the checker will not error
# Format: {"file_path": [list_of_line_numbers]}
ALLOWLIST_MISSING_ALT_TEXT = {
    "qiskit/primitives/statevector_estimator.py": [51],
    "qiskit/pulse/builder.py": [41, 63, 92, 275, 323, 1167, 1221, 2045],
    "qiskit/pulse/library/symbolic_pulses.py": [332],
    "qiskit/transpiler/layout.py": [457, 472, 500],
    "qiskit/transpiler/__init__.py": [35, 332, 392, 447, 534, 559, 610, 630, 655, 702, 720, 739, 762, 809, 818, 890, 902, 948, 959, 978],
    "qiskit/transpiler/passes/utils/filter_op_nodes.py": [40],
    "qiskit/transpiler/passes/utils/remove_barriers.py": [27],
    "qiskit/transpiler/passes/scheduling/dynamical_decoupling.py": [49],
    "qiskit/transpiler/passes/scheduling/padding/dynamical_decoupling.py": [55],
    "qiskit/transpiler/passes/routing/star_prerouting.py": [91],
    "qiskit/providers/fake_provider/__init__.py": [31],
    "qiskit/quantum_info/states/statevector.py": [175],
    "qiskit/qasm3/__init__.py": [128],
    "qiskit/converters/dag_to_circuit.py": [36],
    "qiskit/circuit/controlledgate.py": [72, 86],
    "qiskit/circuit/operation.py": [36],
    "qiskit/circuit/quantumcircuit.py": [837, 903, 2171, 3427, 4308, 4323],
    "qiskit/circuit/__init__.py": [66],
    "qiskit/circuit/parameter.py": [43],
    "qiskit/circuit/random/utils.py": [39, 297],
    "qiskit/circuit/library/overlap.py": [130],
    "qiskit/circuit/library/iqp.py": [40, 49, 99, 108, 156],
    "qiskit/circuit/library/graph_state.py": [43, 107],
    "qiskit/circuit/library/phase_estimation.py": [78, 133],
    "qiskit/circuit/library/grover_operator.py": [105, 119, 134, 151],
    "qiskit/circuit/library/quantum_volume.py": [37, 45, 152],
    "qiskit/circuit/library/hidden_linear_function.py": [57, 125],
    "qiskit/circuit/library/fourier_checking.py": [72, 126],
    "qiskit/circuit/library/__init__.py": [28],
    "qiskit/circuit/library/boolean_logic/quantum_xor.py": [55, 90],
    "qiskit/circuit/library/boolean_logic/quantum_and.py": [34, 45, 118, 131],
    "qiskit/circuit/library/boolean_logic/inner_product.py": [57, 123],
    "qiskit/circuit/library/boolean_logic/quantum_or.py": [35, 46, 119, 132],
    "qiskit/circuit/library/basis_change/qft.py": [40, 50, 66],
    "qiskit/circuit/library/generalized_gates/gms.py": [47],
    "qiskit/circuit/library/generalized_gates/permutation.py": [56, 64, 118, 129],
    "qiskit/circuit/library/generalized_gates/gr.py": [47, 101, 149, 197],
    "qiskit/circuit/library/generalized_gates/mcmt.py": [131],
    "qiskit/circuit/library/arithmetic/piecewise_chebyshev.py": [39],
    "qiskit/circuit/library/n_local/real_amplitudes.py": [70, 79, 86, 93],
    "qiskit/circuit/library/n_local/evolved_operator_ansatz.py": [63, 215, 228],
    "qiskit/circuit/library/n_local/n_local.py": [150, 162, 171, 180, 192],
    "qiskit/circuit/library/n_local/pauli_two_design.py": [61, 149],
    "qiskit/circuit/library/n_local/qaoa_ansatz.py": [49],
    "qiskit/circuit/library/n_local/excitation_preserving.py": [72, 85],
    "qiskit/circuit/library/n_local/efficient_su2.py": [73, 86],
    "qiskit/synthesis/arithmetic/multipliers/rg_qft_multiplier.py": [37],
    "qiskit/synthesis/arithmetic/multipliers/hrs_cumulative_multiplier.py": [29],
    "qiskit/visualization/dag_visualization.py": [101],
    "qiskit/visualization/gate_map.py": [84, 1004, 1170, 1272],
    "qiskit/visualization/state_visualization.py": [76, 216, 223, 292, 306, 408, 424, 645, 661, 824, 838],
    "qiskit/visualization/counts_visualization.py": [113, 217],
    "qiskit/visualization/__init__.py": [49, 62, 103, 144, 157, 172],
    "qiskit/visualization/timeline/interface.py": [300, 316, 332],
    "qiskit/visualization/circuit/circuit_visualization.py": [191],
    "qiskit/visualization/pulse_v2/interface.py": [314, 332, 350],
}


def is_image(line: str) -> bool:
    return line.strip().startswith((".. image:", ".. plot:"))


def is_option(line: str) -> bool:
    return line.strip().startswith(":")

def in_allowlist(filename: str, line_num: int) -> bool:
    return line_num in ALLOWLIST_MISSING_ALT_TEXT.get(filename, [])


def validate_image(file_path: str) -> tuple[str, list[str]]:
    """Validate all the images of a single file"""
    invalid_images: list[str] = []

    lines = Path(file_path).read_text().splitlines()

    line_index = 0
    image_found = False
    image_line = -1
    options: list[str] = []

    while line_index < len(lines):
        line = lines[line_index].strip()

        if image_found and not is_option(line) and not is_valid_image(options):
            invalid_images.append(f"- Error in line {image_line}: {lines[image_line-1].strip()}")
            image_found = False
            options = []
            continue

        if image_found and is_option(line):
            options.append(line)
            line_index += 1
            continue

        image_found = is_image(line) and not in_allowlist(file_path, line_index + 1)
        image_line = line_index + 1
        line_index += 1

    return (file_path, invalid_images)


def is_valid_image(options: list[str]) -> bool:
    alt_exists = any(option.startswith(":alt:") for option in options)
    nofigs_exists = any(option.startswith(":nofigs:") for option in options)

    # Only `.. plot::`` directives without the `:nofigs:` option are required to have alt text.
    # Meanwhile, all `.. image::` directives need alt text and they don't have a `:nofigs:` option.
    return alt_exists or nofigs_exists

def main() -> None:
    files = glob.glob("qiskit/**/*.py", recursive=True)

    with multiprocessing.Pool() as pool:
        results = pool.map(validate_image, files)

    failed_files = [x for x in results if len(x[1])]

    if not len(failed_files):
        print("✅ All images have alt text")
        sys.exit(0)

    print("💔 Some images are missing the alt text", file=sys.stderr)

    for filename, image_errors in failed_files:
        print(f"\nErrors found in {filename}:", file=sys.stderr)

        for image_error in image_errors:
            print(image_error, file=sys.stderr)

    print(
        "\nAlt text is crucial for making documentation accessible to all users. It should serve the same purpose as the images on the page, conveying the same meaning rather than describing visual characteristics. When an image contains words that are important to understanding the content, the alt text should include those words as well.",
        file=sys.stderr,
    )

    sys.exit(1)


if __name__ == "__main__":
    main()