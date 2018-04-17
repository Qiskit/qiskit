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
 * @file    BinaryVector.hpp
 * @brief   BinaryVector class
 * @author  Sergey Bravyi <sbravyi@us.ibm.com>
 */

#ifndef _binary_vector_h_
#define _binary_vector_h_

#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>
#include <stdint.h>

namespace BV {
  
  // Types
  using uint_t = uint64_t;

/*******************************************************************************
 *
 * BinaryVector Class
 *
 ******************************************************************************/
class BinaryVector {
private:
  uint_t m_length;
  std::vector<uint_t> m_data;

public:
  const static size_t blockSize = 64; // 64-bit blocks

  BinaryVector() : m_length(0), m_data(0){};

  explicit BinaryVector(uint_t length)
      : m_length(length), m_data((length - 1) / blockSize + 1, 0){};

  BinaryVector(std::vector<uint_t> mdata)
      : m_length(mdata.size()), m_data(mdata){};

  explicit BinaryVector(std::string);

  bool setLength(uint_t length);

  void setVector(std::string);
  void setValue(bool value, uint_t pos);

  void set0(uint_t pos) { setValue(0, pos); };
  void set1(uint_t pos) { setValue(1, pos); };

  void flipAt(uint_t pos);

  BinaryVector &operator+=(const BinaryVector &rhs);

  bool operator[](const uint_t pos) const;

  void swap(BinaryVector &rhs);

  uint_t getLength() const { return m_length; };

  inline void makeZero() { m_data.assign((m_length - 1) / blockSize + 1, 0ul); }

  bool isZero() const;

  bool isSame(const BinaryVector &rhs) const;
  bool isSame(const BinaryVector &rhs, bool pad) const;

  std::vector<uint_t> nonzeroIndices() const;
  inline std::vector<uint_t> getData() const { return m_data; };
};

/*******************************************************************************
 *
 * Related Functions
 *
 ******************************************************************************/

inline bool operator==(const BinaryVector &lhs, const BinaryVector &rhs) {
  return lhs.isSame(rhs, true);
}

inline int64_t gauss_eliminate(std::vector<BinaryVector> &M,
                            const int64_t start_col = 0)
// returns the rank of M.
// M[] has length nrows.
// each M[i] must have the same length ncols.
{
  const int64_t nrows = M.size();
  const int64_t ncols = M.front().getLength();
  int64_t rank = 0;
  int64_t k, r, i;
  for (k = start_col; k < ncols; k++) {
    i = -1;
    for (r = rank; r < nrows; r++) {
      if (M[r][k] == 0)
        continue;
      if (i == -1) {
        i = r;
        rank++;
      } else {
        M[r] += M[i];
      }
    }
    if (i >= rank) {
      M[i].swap(M[rank - 1]);
    }
  }
  return rank;
}

inline std::vector<uint_t> string_to_bignum(std::string val, uint_t blockSize,
                                            uint_t base) {
  std::vector<uint_t> ret;
  if (blockSize * log2(base) > 64) {
    throw std::runtime_error(
        std::string("block size is greater than 64-bits for current case"));
  }
  auto n = val.size();
  auto blocks = n / blockSize;
  auto tail = n % blockSize;
  for (uint_t j = 0; j != blocks; ++j)
    ret.push_back(
        stoull(val.substr(n - (j + 1) * blockSize, blockSize), 0, blockSize));
  if (tail > 0)
    ret.push_back(stoull(val.substr(0, tail), 0, blockSize));
  return ret;
}

inline std::vector<uint_t> string_to_bignum(std::string val) {
  std::string type = val.substr(0, 2);
  if (type == "0b" || type == "0B")
    // Binary string
    return string_to_bignum(val.substr(2, val.size() - 2), 64, 2);
  else if (type == "0x" || type == "0X")
    // Hexidecimal string
    return string_to_bignum(val.substr(2, val.size() - 2), 16, 16);
  else {
    // Decimal string
    throw std::runtime_error(
        std::string("string must be binary (0b) or hex (0x)"));
  }
}

/*******************************************************************************
 *
 * BinaryVector Class Methods
 *
 ******************************************************************************/

BinaryVector::BinaryVector(std::string val) {
  m_data = string_to_bignum(val);
  m_length = m_data.size();
}

bool BinaryVector::setLength(uint_t length) {
  if (length == 0)
    return false;
  if (m_length > 0)
    return false;
  m_length = length;
  m_data.assign((length - 1) / blockSize + 1, 0ul);
  return true;
}

void BinaryVector::setValue(bool value, uint_t pos) {
  auto q = pos / blockSize;
  auto r = pos % blockSize;
  if (value)
    m_data[q] |= (1 << r);
  else
    m_data[q] &= ~(1 << r);
}

void BinaryVector::flipAt(const uint_t pos) {
  auto q = pos / blockSize;
  auto r = pos % blockSize;
  m_data[q] ^= (1 << r);
}

BinaryVector &BinaryVector::operator+=(const BinaryVector &rhs) {
  const auto size = m_data.size();
  for (size_t i = 0; i < size; i++)
    m_data[i] ^= rhs.m_data[i];
  return (*this);
}

bool BinaryVector::operator[](const uint_t pos) const {
  auto q = pos / blockSize;
  auto r = pos % blockSize;
  return ((m_data[q] & (1 << r)) != 0);
}

void BinaryVector::swap(BinaryVector &rhs) {
  uint_t tmp;
  tmp = rhs.m_length;
  rhs.m_length = m_length;
  m_length = tmp;

  m_data.swap(rhs.m_data);
}

bool BinaryVector::isZero() const {
  const size_t size = m_data.size();
  for (size_t i = 0; i < size; i++)
    if (m_data[i])
      return false;
  return true;
}

bool BinaryVector::isSame(const BinaryVector &rhs) const {
  if (m_length != rhs.m_length)
    return false;
  const size_t size = m_data.size();
  for (size_t q = 0; q < size; q++) {
    if (m_data[q] != rhs.m_data[q])
      return false;
  }
  return true;
}

bool BinaryVector::isSame(const BinaryVector &rhs, bool pad) const {
  if (!pad)
    return isSame(rhs);
  else {
    const auto sz0 = m_data.size();
    const auto sz1 = rhs.m_data.size();
    const auto sz = (sz0 > sz1) ? sz1 : sz0;

    // Check vectors agree on overlap
    for (size_t q = 0; q < sz; q++)
      if (m_data[q] != rhs.m_data[q])
        return false;
    // Check padding of larger vector is trivial
    for (size_t q = sz; q < sz0; q++)
      if (m_data[q] != 0)
        return false;
    for (size_t q = sz; q < sz1; q++)
      if (rhs.m_data[q] != 0)
        return false;

    return true;
  }
}

std::vector<uint_t> BinaryVector::nonzeroIndices() const {
  std::vector<uint_t> result;
  size_t i = 0;
  while (i < m_data.size()) {
    while (m_data[i] == 0) {
      i++;
      if (i == m_data.size())
        return result; // empty
    }
    auto m = m_data[i];
    size_t r = 0;
    while (r < blockSize) {
      while ((m & (1 << r)) == 0) {
        r++;
      }
      if (r >= blockSize)
        break;
      result.push_back((uint_t)(i)*blockSize + r);
      r++;
    }
    i++;
  }
  return result;
}

} // end namespace BV
#endif