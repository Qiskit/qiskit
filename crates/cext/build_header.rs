// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

extern crate cbindgen;

/// This function generates the C header for Qiskit from the qiskit-cext crate.
fn main() {
    let mut config = cbindgen::Config::default();
    config.include_version = true; // include cbindgen version in header
    config.style = cbindgen::Style::Type; // define enums/structs as: typedef struct { .. } Name;
    config.language = cbindgen::Language::C;

    config.sys_includes = vec!["complex.h".to_string()]; // C includes to add
    let after_includes = r##"#ifdef QISKIT_C_PYTHON_INTERFACE
    #include <Python.h>
#endif

// Complex number typedefs -- note these are memory aligned but
// not calling convention compatible.
typedef float complex QkComplex32;
typedef double complex QkComplex64;

// Always expose [cfg(feature = "cbinding")] -- workaround for
// https://github.com/mozilla/cbindgen/issues/995
#define QISKIT_WITH_CBINDINGS
"##;
    config.after_includes = Some(after_includes.to_string()); // additional header lines

    config.defines = [
        // map cfg(feature=..) to #ifdef
        (
            "feature = cbinding".to_string(),
            "QISKIT_WITH_CBINDINGS".to_string(),
        ),
        (
            "feature = python_binding".to_string(),
            "QISKIT_C_PYTHON_INTERFACE".to_string(),
        ),
    ]
    .into();
    // maybe enable pragma once? are there compilers we target that do not support this?

    // Enum naming configurations. We prefix enums with names for now.
    let mut enumeration = cbindgen::EnumConfig::default();
    enumeration.prefix_with_name = true;
    config.enumeration = enumeration;

    // Struct/enum renames. We also use this to manually prefix the objects with Qk, to avoid
    // global renames of C Python objects or other dependencies.
    let mut export = cbindgen::ExportConfig::default();
    export.rename = [
        ("SparseObservable".to_string(), "QkObs".to_string()),
        ("CSparseTerm".to_string(), "QkObsTerm".to_string()),
        ("BitTerm".to_string(), "QkBitTerm".to_string()),
        ("Complex64".to_string(), "QkComplex64".to_string()),
    ]
    .into();
    config.export = export;

    // Define Rust dependencies to parse.
    let mut parse = cbindgen::ParseConfig::default();
    parse.parse_deps = true;
    parse.include = Some(vec!["qiskit-accelerate".to_string()]);
    config.parse = parse;

    // Ensure the include directory exists, and then set the full header path.
    let mut path = "../../dist/c/include".to_string();
    ::std::fs::create_dir_all(&path).expect("Failed creating dist/c/include.");
    path.push_str("/qiskit.h");

    // Build the header.
    cbindgen::Builder::new()
        .with_crate(".")
        .with_config(config)
        .generate()
        .expect("Unable to generate C bindings.")
        .write_to_file(path);
}
