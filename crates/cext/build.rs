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

use std::str::FromStr;

extern crate cbindgen;

/// This function generates the C header for Qiskit from the qiskit-cext crate.
fn generate_qiskit_header() {
    // Trigger this script if the header was changed/removed.
    #![allow(clippy::print_stdout)]
    println!("cargo:rerun-if-changed=../../target/qiskit.h");

    // Pull the config from the cbindgen.toml file.
    let config = cbindgen::Config::from_file("cbindgen.toml").unwrap();

    // Ensure target path exists and then set the full filename of qiskit.h.
    let mut path = ::std::path::PathBuf::from_str("../../target/").unwrap();
    ::std::fs::create_dir_all(&path).expect("Could not create target directory.");
    path.push("qiskit.h");

    // Build the header.
    cbindgen::Builder::new()
        .with_crate(".")
        .with_config(config)
        .generate()
        .expect("Unable to generate C bindings.")
        .write_to_file(path);
}

// Get the Python library directory and library name that PyO3 is using, and store it into a
// configuration file.
fn write_python_config() {
    let interpreter_config = pyo3_build_config::get();
    let pyo3_lib_config = format!(
        "PYO3_PYTHON_LIB_DIR={}\nPYO3_PYTHON_LIB_NAME={}\n",
        interpreter_config.lib_dir.clone().unwrap_or("".to_string()),
        interpreter_config
            .lib_name
            .clone()
            .unwrap_or("".to_string())
    );

    // This path is relative to the current file, i.e. we write into the root's target dir.
    let pyo3_config_file = "../../target/pyo3_python.config";
    match ::std::fs::write(pyo3_config_file, pyo3_lib_config) {
        Ok(_) => (),
        Err(_) => println!("cargo:warning=Failed to write Python config."),
    };
}

fn main() {
    generate_qiskit_header();
    write_python_config();
}
