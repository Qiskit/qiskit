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

use std::{fs::File, io::Write, str::FromStr};

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
    let mut buffer: Vec<u8> = Vec::new();
    cbindgen::Builder::new()
        .with_crate(".")
        .with_config(config)
        .generate()
        .expect("Unable to generate C bindings.")
        .write(&mut buffer);

    // Undo cbindgen's renaming of QkPyObject to PyObject. We currently only have a single
    // target-replacement pair, this could be promoted to a HashMap if we have more than that.
    let target = "QkPyObject";
    let replacement = "PyObject";
    let header = String::from_utf8(buffer)
        .expect("Unable to parse C bindings.")
        .replace(target, replacement);

    // When writing the header, we're checking that all bytes have been written.
    let mut header_file = File::create(&path).expect("The qiskit.h path should exist.");
    let header_bytes = header.as_bytes();
    let expected_bytes_written = header_bytes.len();
    let bytes_written = header_file
        .write(header.as_bytes())
        .expect("Unable to write header.");
    if bytes_written != expected_bytes_written {
        panic!("Unable to write header (partial write detected).");
    }
}

fn main() {
    generate_qiskit_header();
}
