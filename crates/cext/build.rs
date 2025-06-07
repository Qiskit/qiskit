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

fn main() {
    generate_qiskit_header();
}
