/*
Copyright (c) 2017 IBM Corporation. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/**
 * @file    misc.h
 * @brief   miscellaneous functions
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _misc_hpp_
#define _misc_hpp_

#include <algorithm>
#include <bitset>
#include <complex>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "types.hpp"

/*******************************************************************************
 *
 * Function headers
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// I/O Handling
//------------------------------------------------------------------------------

/**
 * Converts a string to all lowercase characters
 * @param str: string to be converted to lowercase
 */
void to_lowercase(std::string &str);

/**
 * Opens a file and returns it as a string
 * @param infile: the name to be converted to a buffer
 * @return: contents of file as a std::string
 */
std::string read_file(std::string infile);

/**
 * Writes a string to a file on disk
 * @param outfile: the output file name
 * @param contents: the name to be converted to a buffer
 */
void write_file(std::string outfile, std::string contents);

/**
 * Reads an input stream until end of file is reached, and returns the stream as
 * a string. Note that this adds an additional newline character at the end of
 * the input.
 * @param input: the input stream (eg std::cin)
 * @return: a string containing the input stream contents
 */
std::string read_stream(std::istream &input);

/**
 * Reads an input stream until seperating the stream into multiple strings when
 * a specified file break string is reached. The output is returned as a vector
 * of each of the resulting strings. Note this adds a newline character to the
 * end of each file.
 * @param input: the input stream (eg std::cin)
 * @param file_break: the line to use as a virtual file break
 * @return: a vector of strings containing the input stream
 */
std::vector<std::string> read_stream(std::istream &input,
                                     std::string file_break);

//------------------------------------------------------------------------------
// Map overloads
//------------------------------------------------------------------------------

cket_t &operator+=(cket_t &m1, const cket_t &m2);
cket_t operator+(const cket_t &m1, const cket_t &m2);
cket_t &operator*=(cket_t &m1, const complex_t z);
cket_t operator*(const cket_t &m1, const complex_t z);
cket_t operator*(const complex_t z, const cket_t &m1);

rket_t &operator+=(rket_t &m1, const rket_t &m2);
rket_t operator+(const rket_t &m1, const rket_t &m2);
rket_t &operator*=(rket_t &m1, const double z);
rket_t operator*(const rket_t &m1, const double z);
rket_t operator*(const double z, const rket_t &m1);

//------------------------------------------------------------------------------
// Vector operations
//------------------------------------------------------------------------------
/**
 * Operator overloads for vectors
 */
rvector_t &operator+=(rvector_t &v1, const rvector_t &v2);
cvector_t &operator+=(cvector_t &v1, const cvector_t &v2);
rvector_t operator+(const rvector_t &v1, const rvector_t &v2);
cvector_t operator+(const cvector_t &v1, const cvector_t &v2);
rvector_t &operator-=(rvector_t &v1, const rvector_t &v2);
cvector_t &operator-=(cvector_t &v1, const cvector_t &v2);
rvector_t operator-(const rvector_t &v1, const rvector_t &v2);
cvector_t operator-(const cvector_t &v1, const cvector_t &v2);
rvector_t &operator*=(rvector_t &v1, const double a);
cvector_t &operator*=(cvector_t &v1, const complex_t z);
rvector_t operator*(const rvector_t &v1, const double a);
rvector_t operator*(const double a, const rvector_t &v1);
cvector_t operator*(const cvector_t &v1, const complex_t z);
cvector_t operator*(const complex_t z, const cvector_t &v1);
/**
 * Computes the inner product (v1^*.v2) between two numeric vectors.
 * @param v1: the lhs vector (complex conjugated)
 * @param v2: the rhs vector
 * @return: value of the inner product
 */
template <typename T>
std::complex<T> inner_product(const std::vector<std::complex<T>> &v1,
                              const std::vector<std::complex<T>> &v2);

/**
 * Renormalizes a numeric vector.
 * @param v: the vector
 */
template <typename T> void renormalize(std::vector<T> &v);

/**
 * Renormalizes a numeric matrix.
 * @param mat: the matrix
 */
template <typename T> void renormalize(matrix<T> &mat);

/**
 * Returns the real part of a complex vector
 * @param vec: a complex vector
 * @return: vector of the real parts of vec
 */
template <typename T>
std::vector<T> real(const std::vector<std::complex<T>> &vec);

/**
 * Returns the imaginary part of a complex vector
 * @param vec: a complex vector
 * @return: vector of the imaginary parts of vec
 */
template <typename T>
std::vector<T> imag(const std::vector<std::complex<T>> &vec);

/**
 * Returns the outer product of two vectors
 * @param ket the left (ket) vector
 * @param bra the right (bra) vector
 * @return: a matrix
 */
template <typename T>
matrix<T> outer_product(const std::vector<T> &ket, const std::vector<T> &bra);
template <typename T>
std::map<std::string, T> outer_product(const std::map<std::string, T> &ket,
                                       const std::map<std::string, T> &bra,
                                       double epsilon = 0.);
//------------------------------------------------------------------------------
// Chop small values
//------------------------------------------------------------------------------

/**
 * Sets matrix entries below with abosolute value below threshold to zero.
 * @param mat: a numeric matrix
 * @param epsilon: threshold value
 * @returns: a reference to the original input with truncated values
 */
double &chop(double &val, double epsilon);
complex_t &chop(complex_t &val, double epsilon);
template <typename T> matrix<T> &chop(matrix<T> &mat, double epsilon);
template <typename T> std::vector<T> &chop(std::vector<T> &vec, double epsilon);
template <typename T1, typename T2>
std::map<T1, T2> &chop(std::map<T1, T2> &map, double epsilon);

//------------------------------------------------------------------------------
// Vectorize / devectorize matrices
//------------------------------------------------------------------------------

/**
 * Flattens a matrix into a vector by stacking matrix columns
 * @param mat: a complex matrix
 * @return: a complex vector
 */
cvector_t vectorize(const cmatrix_t &mat);

/**
 * Converts a column-vectorized square matrix back into a square matrix
 * @param vec: a vectorized square matrix
 * @return: a square complex matrix
 */
cmatrix_t devectorize(const cvector_t &vec);

//------------------------------------------------------------------------------
// Convert integers to dit-strings
//------------------------------------------------------------------------------

/**
 * Converts an integer into a integer base string representation.
 * @param n: integer
 * @param base: representation base (default is 2 for bitstrings)
 * @return: a dit-string
 */
std::string int2string(uint_t n, uint_t base = 2);

/**
 * Converts an integer into a integer base string representation and pads with
 * zeros to a fixed length.
 * @param n: integer
 * @param base: representation base (default is 2 for bitstrings)
 * @param length: length of the padded string
 * @return: a fixed length dit-string
 */
std::string int2string(uint_t n, uint_t base, uint_t length);

/**
 * Converts an integer into vector of ints, where the least significant dit is
 * the first element of the vector.
 * @param n: integer
 * @param base: representation base (default is 2)
 * @param minlen: minimum length of the returned vector
 * @return: a vector of ints
 */
std::vector<uint_t> int2reg(uint_t n, uint_t base = 2);
std::vector<uint_t> int2reg(uint_t n, uint_t base, uint_t minlen);

/**
 * Write me
 * @param str
 * @param regs
 * @return
 */

std::string format_bitstr(std::string str, const creg_t &regs);

/**
 * Converts an n-qudit complex vector into a sparse state vector representation
 * as a map of non-zero values indexed by the standard basis dit-string.
 * @param psi: a complex vector
 * @param dit: subsystem dimension
 * @param epsilon: threshold for truncating small values to zero
 * @return: a map representation of non-zero state vector components
 */
cket_t vec2ket(const cvector_t &psi, uint_t dit, double epsilon,
               const creg_t &regs);

/**
 * Converts an n-qudit complex vector into a sparse state vector representation
 * as a map of non-zero values indexed by the standard basis dit-string.
 * @param psi: a complex vector
 * @param dit: subsystem dimension
 * @param epsilon: threshold for truncating small values to zero
 * @return: a map representation of non-zero state vector components
 */
cket_t vec2ket(const cvector_t &psi, uint_t dit, double epsilon);

/**
 * Converts an n-qubit complex vector into a sparse state vector representation
 * as a map of non-zero values indexed by the standard basis bitstring.
 * @param psi: a complex vector
 * @param epsilon: threshold for truncating small values to zero
 * @return: a map representation of non-zero state vector components
 */
cket_t vec2ket(const cvector_t &psi, double epsilon = 0.);

//------------------------------------------------------------------------------
// Convert hex to binary vectors
//------------------------------------------------------------------------------

/**
 * Convet a hexadecimal string to a binary vector.
 * @param str: a hex string.
 * @returns: a binary vector.
 */
std::vector<uint_t> hex2reg(std::string str);

//------------------------------------------------------------------------------
// Pad unitary matrices to larger dimensions
//------------------------------------------------------------------------------

/**
 * Pads a single system unitary matrix to a unitary matrix on a larger dimension
 * system. The padded matrix acts trivially on the added subspace.
 * @param U1: a single system matrix
 * @param dit: dimension for the enlarged matrix (must be >= current dim)
 * @return: a dit x dit matrix
 */
cmatrix_t qudit_unitary1(const cmatrix_t &U1, uint_t dit);

/**
 * Pads a two-system unitary matrix to a unitary matrix on a larger dimension
 * system. The padded matrix acts trivially on the added subspace.
 * @param U2: a two-system bipartite matrix
 * @param dit: subsystem dim for the enlarged matrix (must be >= current dim)
 * @return: a dit^2 x dit^2 matrix
 */
cmatrix_t qudit_unitary2(const cmatrix_t &U2, uint_t dit);

/*******************************************************************************
 *
 * Function implementations
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// I/O Handling
//------------------------------------------------------------------------------

void to_lowercase(std::string &str) {
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
}

std::string read_file(std::string infile) {

  std::ifstream file(infile); // open file
  if (!file) {                // check file opened correctly
    throw std::runtime_error(
        std::string("failed to open input file \"" + infile + "\""));
  }
  std::stringstream buffer;
  buffer << file.rdbuf(); // read file
  file.close();           // close file

  return buffer.str();
}

void write_file(std::string outfile, std::string contents) {

  // open output file
  std::ofstream file(outfile); // open file
  if (!file) {                 // check file opened correctly
    throw std::runtime_error(std::string("failed to open output file: ") +
                             outfile);
  }
  file << contents;
  file.close();
}

std::string read_stream(std::istream &input) {

  std::stringstream buffer;
  for (std::string line; std::getline(input, line);)
    buffer << line << std::endl;
  std::string file = buffer.str(); // convert to string
  file.pop_back();                 // remove last newline char

  return file;
}

std::vector<std::string> read_stream(std::istream &input,
                                     std::string file_break) {

  std::stringstream buffer;
  std::vector<std::string> files;

  for (std::string line; std::getline(input, line);) {
    if (line == file_break) {
      std::string file = buffer.str();
      file.pop_back();           // remove last newline char
      files.push_back(file);     // store file
      buffer.str(std::string()); // clear buffer
    } else
      buffer << line << std::endl;
  } // store last  file
  std::string file = buffer.str();
  file.pop_back();       // remove last newline char
  files.push_back(file); // store file

  return files;
}

//------------------------------------------------------------------------------
// Map overloads
//------------------------------------------------------------------------------

cket_t &operator+=(cket_t &m1, const cket_t &m2) {
  for (const auto &p : m2)
    m1[p.first] += p.second;
  return m1;
}

cket_t operator+(const cket_t &m1, const cket_t &m2) {
  auto ret = m1;
  ret += m2;
  return ret;
}

cket_t &operator*=(cket_t &m1, const complex_t z) {
  for (const auto &p : m1)
    m1[p.first] *= z;
  return m1;
}

cket_t operator*(const cket_t &m1, const complex_t z) {
  auto ret = m1;
  ret *= z;
  return ret;
}

cket_t operator*(const complex_t z, const cket_t &m1) { return m1 * z; }

rket_t &operator+=(rket_t &m1, const rket_t &m2) {
  for (const auto &p : m2)
    m1[p.first] += p.second;
  return m1;
}
rket_t operator+(const rket_t &m1, const rket_t &m2) {
  auto ret = m1;
  ret += m2;
  return ret;
}
rket_t &operator*=(rket_t &m1, const double z) {
  for (const auto &p : m1)
    m1[p.first] *= z;
  return m1;
}

rket_t operator*(const rket_t &m1, const double z) {
  auto ret = m1;
  ret *= z;
  return ret;
}

rket_t operator*(const double z, const rket_t &m1) { return m1 * z; }

//------------------------------------------------------------------------------
// Vector operations
//------------------------------------------------------------------------------

rvector_t &operator+=(rvector_t &v1, const rvector_t &v2) {
  if (v1.empty()) // allow adding to empty vector
    v1.resize(v2.size());
  if (v1.size() == v2.size()) {
    for (size_t j = 0; j < v1.size(); j++)
      v1[j] += v2[j];
    return v1;
  } else {
    throw std::runtime_error(std::string("vectors are different lengths"));
  }
}

cvector_t &operator+=(cvector_t &v1, const cvector_t &v2) {
  if (v1.empty()) // allow adding to empty vector
    v1.resize(v2.size());
  if (v1.size() == v2.size()) {
    for (size_t j = 0; j < v1.size(); j++)
      v1[j] += v2[j];
    return v1;
  } else {
    throw std::runtime_error(std::string("vectors are different lengths"));
  }
}

rvector_t &operator-=(rvector_t &v1, const rvector_t &v2) {
  if (v1.empty()) // allow adding to empty vector
    v1.resize(v2.size());
  if (v1.size() == v2.size()) {
    for (size_t j = 0; j < v1.size(); j++)
      v1[j] -= v2[j];
    return v1;
  } else {
    throw std::runtime_error(std::string("vectors are different lengths"));
  }
}

cvector_t &operator-=(cvector_t &v1, const cvector_t &v2) {
  if (v1.empty()) // allow adding to empty vector
    v1.resize(v2.size());
  if (v1.size() == v2.size()) {
    for (size_t j = 0; j < v1.size(); j++)
      v1[j] -= v2[j];
    return v1;
  } else {
    throw std::runtime_error(std::string("vectors are different lengths"));
  }
}

rvector_t &operator*=(rvector_t &v1, const double a) {
  for (size_t j = 0; j < v1.size(); j++)
    v1[j] *= a;
  return v1;
}

cvector_t &operator*=(cvector_t &v1, const complex_t z) {
  for (size_t j = 0; j < v1.size(); j++)
    v1[j] *= z;
  return v1;
}

rvector_t operator+(const rvector_t &v1, const rvector_t &v2) {
  rvector_t ret = v1;
  ret += v2;
  return ret;
}

cvector_t operator+(const cvector_t &v1, const cvector_t &v2) {
  cvector_t ret = v1;
  ret += v2;
  return ret;
}

rvector_t operator-(const rvector_t &v1, const rvector_t &v2) {
  rvector_t ret = v1;
  ret -= v2;
  return ret;
}

cvector_t operator-(const cvector_t &v1, const cvector_t &v2) {
  cvector_t ret = v1;
  ret -= v2;
  return ret;
}

rvector_t operator*(const rvector_t &v1, const double a) {
  rvector_t ret = v1;
  ret *= a;
  return ret;
}

cvector_t operator*(const cvector_t &v1, const complex_t z) {
  cvector_t ret = v1;
  ret *= z;
  return ret;
}

rvector_t operator*(const double a, const rvector_t &v1) { return v1 * a; }

cvector_t operator*(const complex_t z, const cvector_t &v1) { return v1 * z; }

template <typename T>
std::complex<T> inner_product(const std::vector<std::complex<T>> &v1,
                              const std::vector<std::complex<T>> &v2) {
  std::complex<T> n = 0.;
  if (v1.size() != v2.size()) {
    throw std::runtime_error(
        std::string("(inner_product) vectors are different lengths"));
  } else {
    for (unsigned long j = 0; j != v1.size(); j++)
      n += std::conj(v1[j]) * v2[j];
  }
  return n;
}

template <typename T> void renormalize(std::vector<T> &vec) {
  double norm = 0.;
  for (const auto &e : vec)
    norm += std::abs(std::conj(e) * e);
  for (auto &e : vec)
    e *= 1 / std::sqrt(norm);
}

template <typename T> void renormalize(matrix<T> &mat) {
  T scale = 1. / MOs::Trace(mat);
  mat = scale * mat;
}

template <typename T>
std::vector<T> real(const std::vector<std::complex<T>> &vec) {
  std::vector<T> re(vec.size());
  for (uint_t j = 0; j != vec.size(); j++)
    re[j] = std::real(vec[j]);
  return re;
}

template <typename T>
std::vector<T> imag(const std::vector<std::complex<T>> &vec) {
  std::vector<T> im(vec.size());
  for (uint_t j = 0; j != vec.size(); j++)
    im[j] = std::imag(vec[j]);
  return im;
}

template <typename T>
matrix<T> outer_product(const std::vector<T> &ket, const std::vector<T> &bra) {
  const uint_t d1 = ket.size();
  const uint_t d2 = bra.size();
  matrix<T> ret(d1, d2);
  for (uint_t i = 0; i < d1; i++)
    for (uint_t j = 0; j < d2; j++) {
      ret(i, j) = ket[i] * conj(bra[j]);
    }
  return ret;
}

template <typename T>
std::map<std::string, T> outer_product(const std::map<std::string, T> &ket,
                                       const std::map<std::string, T> &bra,
                                       double epsilon) {
  std::map<std::string, T> ret;
  for (const auto &p1 : ket)
    for (const auto &p2 : bra) {
      T val = ket.second * std::conj(bra.second);
      chop(val, epsilon);
      if (std::abs(val) > 0) {
        std::string key = ket.first + '|' + bra.first;
        ret[key] = val;
      }
    }
  return ret;
}
//------------------------------------------------------------------------------
// Chop small values
//------------------------------------------------------------------------------

double &chop(double &val, double epsilon) {
  if (std::abs(val) < epsilon)
    val = 0.;
  return val;
}

complex_t &chop(complex_t &val, double epsilon) {
  if (std::abs(val.real()) < epsilon)
    val.real(0.);
  if (std::abs(val.imag()) < epsilon)
    val.imag(0.);
  return val;
}

template <typename T>
std::vector<T> &chop(std::vector<T> &vec, double epsilon) {
  if (epsilon > 0.)
    for (auto &v : vec)
      chop(v, epsilon);
  return vec;
}

template <typename T1, typename T2>
std::map<T1, T2> &chop(std::map<T1, T2> &map, double epsilon) {
  if (epsilon > 0.)
    for (auto &p : map) {
      chop(p.second, epsilon);
      if (std::abs(p.second) < epsilon)
        map.erase(p.first);
    }
  return map;
}

template <typename T> matrix<T> &chop(matrix<T> &mat, double epsilon) {
  if (epsilon > 0.)
    for (uint_t col = 0; col != mat.GetColumns(); col++)
      for (uint_t row = 0; row != mat.GetRows(); row++)
        chop(mat(row, col), epsilon);
  return mat;
}

//------------------------------------------------------------------------------
// Vectorize / devectorize matrices
//------------------------------------------------------------------------------

cmatrix_t devectorize(const cvector_t &vec) {
  uint_t dim = std::sqrt(vec.size());
  if (vec.size() != dim * dim) {
    throw std::runtime_error(
        std::string("(devectorize) vector is not a vectorized square matrix"));
  }
  cmatrix_t mat(dim, dim);
  for (uint_t col = 0; col != dim; col++)
    for (uint_t row = 0; row != dim; row++)
      mat(row, col) = vec[col * dim + row];
  return mat;
}

cvector_t vectorize(const cmatrix_t &mat) {
  uint_t ncol = mat.GetColumns();
  uint_t nrow = mat.GetRows();
  cvector_t vec(ncol * nrow);
  for (uint_t col = 0; col != ncol; col++)
    for (uint_t row = 0; row != nrow; row++)
      vec[col * nrow + row] = mat(row, col);
  return vec;
}

//------------------------------------------------------------------------------
// Convert integers to dit-strings
//------------------------------------------------------------------------------

std::string int2string(uint_t n, uint_t base) {
  if (n < base)
    return std::to_string(n);
  else
    return int2string(n / base, base) + std::to_string(n % base);
}

std::string int2string(uint_t n, uint_t base, uint_t minlen) {
  std::string s = int2string(n, base);
  auto l = s.size();
  if (l < minlen)
    s = std::string(minlen - l, '0') + s;
  return s;
}

//------------------------------------------------------------------------------
// Convert integers to dit-vectors
//------------------------------------------------------------------------------

std::vector<uint_t> int2reg(uint_t n, uint_t base) {
  std::vector<uint_t> ret;
  while (n >= base) {
    ret.push_back(n % base);
    n /= base;
  }
  ret.push_back(n); // last case n < base;
  return ret;
}

std::vector<uint_t> int2reg(uint_t n, uint_t base, uint_t minlen) {
  std::vector<uint_t> ret = int2reg(n, base);
  if (ret.size() < minlen) // pad vector with zeros
    ret.resize(minlen);
  return ret;
}

//------------------------------------------------------------------------------
// Convert hexadecimal string to binary vector
//------------------------------------------------------------------------------

std::vector<uint_t> hex2reg(std::string str) {
  std::vector<uint_t> reg;
  std::string prefix = str.substr(0, 2);
  if (prefix == "0x" || prefix == "0X") { // Hexadecimal
    str.erase(0, 2); // remove '0x';
    size_t length = (str.size() % 8) + 32 * (str.size() / 8);
    reg.reserve(length);
    while (str.size() > 8) {
      unsigned long hex = stoul(str.substr(str.size() - 8), 0, 16);
      std::vector<uint_t> tmp = int2reg(hex, 2, 32);
      std::move(tmp.begin(), tmp.end(), back_inserter(reg));
      str.erase(str.size() - 8);
    }
    if (str.size() > 0) {
      std::vector<uint_t> tmp = int2reg(stoul(str, 0, 16), 2, 0);
      std::move(tmp.begin(), tmp.end(), back_inserter(reg));
    }
    return reg;
  } else {
    throw std::runtime_error(std::string("invalid hexadecimal"));
  }
}

//------------------------------------------------------------------------------
// Convert vectors to ket-form
//------------------------------------------------------------------------------

std::string format_bitstr(std::string str, const creg_t &regs) {
  if (regs.empty())
    return str;
  else {
    // check sizes
    unsigned long n = 0;
    for (const auto &sz : regs)
      n += sz;
    if (n != str.length()) {
      throw std::runtime_error(
          std::string("string length is different to specified reg sizes."));
    }

    std::string ret = "";
    unsigned long shift = 0;
    for (const auto &sz : regs) {
      for (uint_t j = 0; j != sz; ++j)
        ret += str[shift + j]; // CS bit-ordering
      ret += " ";
      shift += sz;
    }
    if (!ret.empty())
      ret.pop_back();
    return ret;
  }
}

cket_t vec2ket(const cvector_t &psi, uint_t dit, double epsilon,
               const creg_t &regs) {

  cvector_t vec = psi;
  chop(vec, epsilon);

  // check vector is of length dit^d
  double n = std::log2(psi.size()) / std::log2(dit);
  if (std::abs(trunc(n) - n) > 1e-5) {
    throw std::runtime_error(
        std::string("vector is not a tensor product of qudit states."));
  }

  cket_t ketmap;
  for (uint_t k = 0; k != vec.size(); ++k) {
    if (std::abs(vec[k]) > 0.) {
      std::string bitstr = int2string(k, dit);
      // pad zeros onto front to reach required length
      bitstr = std::string(trunc(n) - bitstr.length(), '0') + bitstr;
      bitstr = format_bitstr(bitstr, regs);
      // assign key-value pair
      ketmap.insert({bitstr, vec[k]});
    }
  }
  return ketmap;
}

cket_t vec2ket(const cvector_t &psi, uint_t dit, double epsilon) {
  creg_t vec;
  return vec2ket(psi, dit, epsilon, vec);
}

cket_t vec2ket(const cvector_t &psi, double epsilon) {
  return vec2ket(psi, 2, epsilon);
}

//------------------------------------------------------------------------------
// Pad unitary matrices to larger dimensions
//------------------------------------------------------------------------------

cmatrix_t qudit_unitary1(const cmatrix_t &U1, uint_t dit) {

  if (dit == 2)
    return U1;
  else {
    cmatrix_t U(dit, dit);
    MOs::Identity(U);

    for (uint_t i = 0; i != 2; ++i)
      for (uint_t j = 0; j != 2; ++j)
        U(i, j) = U1(i, j);

    return U;
  }
}

cmatrix_t qudit_unitary2(const cmatrix_t &U2, uint_t dit) {

  if (dit == 2)
    return U2;
  else {
    uint_t d = dit * dit;
    cmatrix_t U(d, d);
    MOs::Identity(U);

    for (uint_t i = 0; i != 2; ++i)
      for (uint_t j = 0; j != 2; ++j) {
        U(i, j) = U2(i, j);
        U(i, j + dit) = U2(i, j + 2);
        U(i + dit, j) = U2(i + 2, j);
        U(i + dit, j + dit) = U2(i + 2, j + 2);
      }
    return U;
  }
}

//------------------------------------------------------------------------------
#endif
