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

use std::env;
use std::fs;
use std::path::Path;
use std::str::FromStr;

extern crate cbindgen;

/// This function generates version_constants.rs with integer version numbers
fn write_version_constants() {
    //Obtain version constants from environment variables.
    let major = env::var("CARGO_PKG_VERSION_MAJOR").unwrap();
    let minor = env::var("CARGO_PKG_VERSION_MINOR").unwrap();
    let patch = env::var("CARGO_PKG_VERSION_PATCH").unwrap();

    // Read the contents of cbindgen.toml and update version numbers.
    let cbindgen_path = Path::new("cbindgen.toml");
    let contents = fs::read_to_string(&cbindgen_path).expect("Failed to read cbindgen.toml");
    let contents = contents
        .replace("@QISKIT_VERSION_MAJOR@", &major)
        .replace("@QISKIT_VERSION_MINOR@", &minor)
        .replace("@QISKIT_VERSION_PATCH@", &patch);
    fs::write(&cbindgen_path, contents).expect("Failed to write cbindgen.generated.toml");
}

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
    write_version_constants();
    generate_qiskit_header();
}
