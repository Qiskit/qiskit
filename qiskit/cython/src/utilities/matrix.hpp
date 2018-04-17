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
 * @file    matrix.h
 * @brief   C++ BLAS matrix library
 * @author  Jay Gambetta <jay.gambetta@us.ibm.com>
 */

/*
Dependences: BLAS
Brief Discription: This is my Matrix class. It works only with real/complex
matrices and stores the entries in  column-major-order. In column-major storage
the columns are stored one after the other. The linear offset p from the
beginning of the array to any given element A(i,j) can then be computed as:
   p = j*Nrows+i
where Nrows is the number of rows in the matrix. Hence, one scrolls down
rows and moves to a new column once the last row is reached. More precisely, if
i wanted to know the i and j associtated with a given p then i would use
  i=p %Nrows
  j= floor(p/Nrows)

Multiplication is done with the C wrapper of the fortran blas library.
*/

#ifndef Matrix_h
#define Matrix_h

#include <complex>
#include <iostream>
#include <vector>

/*******************************************************************************
 *
 * BLAS headers
 *
 ******************************************************************************/

const char Trans[] = {'N', 'T', 'C'};
/*  Trans (input) CHARACTER*1.
                On entry, TRANSA specifies the form of op( A ) to be used in the
   matrix multiplication as follows:
                        = 'N' no transpose;
                        = 'T' transpose of A;
                        = 'C' hermitian conjugate of A.
*/
const char UpLo[] = {'U', 'L'};
/*  UpLo    (input) CHARACTER*1
                        = 'U':  Upper triangle of A is stored;
                        = 'L':  Lower triangle of A is stored.
*/
const char Jobz[] = {'V', 'N'};
/*  Jobz    (input) CHARACTER*1
                        = 'N':  Compute eigenvalues only;
                        = 'V':  Compute eigenvalues and eigenvectors.
*/
const char Range[] = {'A', 'V', 'I'};
/*  Range   (input) CHARACTER*1
                                = 'A': all eigenvalues will be found.
                                = 'V': all eigenvalues in the half-open interval
   (VL,VU] will be found.
                                = 'I': the IL-th through IU-th eigenvalues will
   be found.
*/

#ifdef __cplusplus
extern "C" {
#endif

//===========================================================================
// Prototypes for level 3 BLAS
//===========================================================================

// Single-Precison Real Matrix-Vector Multiplcation
void sgemv_(const char *TransA, const size_t *M, const size_t *N,
            const float *alpha, const float *A, const size_t *lda,
            const float *x, const size_t *incx, const float *beta, float *y,
            const size_t *lincy);
// Double-Precison Real Matrix-Vector Multiplcation
void dgemv_(const char *TransA, const size_t *M, const size_t *N,
            const double *alpha, const double *A, const size_t *lda,
            const double *x, const size_t *incx, const double *beta, double *y,
            const size_t *lincy);
// Single-Precison Complex Matrix-Vector Multiplcation
void cgemv_(const char *TransA, const size_t *M, const size_t *N,
            const std::complex<float> *alpha, const std::complex<float> *A,
            const size_t *lda, const std::complex<float> *x, const size_t *incx,
            const std::complex<float> *beta, std::complex<float> *y,
            const size_t *lincy);
// Double-Precison Real Matrix-Vector Multiplcation
void zgemv_(const char *TransA, const size_t *M, const size_t *N,
            const std::complex<double> *alpha, const std::complex<double> *A,
            const size_t *lda, const std::complex<double> *x,
            const size_t *incx, const std::complex<double> *beta,
            std::complex<double> *y, const size_t *lincy);
// Single-Precison Real Matrix-Matrix Multiplcation
void sgemm_(const char *TransA, const char *TransB, const size_t *M,
            const size_t *N, const size_t *K, const float *alpha,
            const float *A, const size_t *lda, const float *B,
            const size_t *lba, const float *beta, float *C, size_t *ldc);
// Double-Precison Real Matrix-Matrix Multiplcation
void dgemm_(const char *TransA, const char *TransB, const size_t *M,
            const size_t *N, const size_t *K, const double *alpha,
            const double *A, const size_t *lda, const double *B,
            const size_t *lba, const double *beta, double *C, size_t *ldc);
// Single-Precison Complex Matrix-Matrix Multiplcation
void cgemm_(const char *TransA, const char *TransB, const size_t *M,
            const size_t *N, const size_t *K, const std::complex<float> *alpha,
            const std::complex<float> *A, const size_t *lda,
            const std::complex<float> *B, const size_t *ldb,
            const std::complex<float> *beta, std::complex<float> *C,
            size_t *ldc);
// Double-Precison Complex Matrix-Matrix Multiplcation
void zgemm_(const char *TransA, const char *TransB, const size_t *M,
            const size_t *N, const size_t *K, const std::complex<double> *alpha,
            const std::complex<double> *A, const size_t *lda,
            const std::complex<double> *B, const size_t *ldb,
            const std::complex<double> *beta, std::complex<double> *C,
            size_t *ldc);
#ifdef __cplusplus
}
#endif

/*******************************************************************************
 *
 * Matrix Class
 *
 ******************************************************************************/

enum OutputStyle { Column, List, Matrix };

template <class T> // define a class template
class matrix {
  // friend functions get to use the private varibles of the class as well as
  // have different classes as inputs
  template <class S>
  friend std::ostream &
  operator<<(std::ostream &output,
             const matrix<S> &A); // overloading << to output a matrix
  template <class S>
  friend std::istream &
  operator>>(std::istream &input,
             const matrix<S> &A); // overloading >> to read in a matrix

  // Multiplication (does not catch an error of a S1 = real and S2 being
  // complex)
  template <class S1, class S2>
  friend matrix<S1>
  operator*(const S2 &beta,
            const matrix<S1> &A); // multiplication by a scalar beta*A
  template <class S1, class S2>
  friend matrix<S1>
  operator*(const matrix<S1> &A,
            const S2 &beta); // multiplication by a scalar A*beta
  // Single-Precison Matrix Multiplication
  friend matrix<float>
  operator*(const matrix<float> &A,
            const matrix<float> &B); // real matrix multiplication A*B
  friend matrix<std::complex<float>> operator*(
      const matrix<std::complex<float>> &A,
      const matrix<std::complex<float>> &B); // complex matrix multplication A*B
  friend matrix<std::complex<float>>
  operator*(const matrix<float> &A,
            const matrix<std::complex<float>>
                &B); // real-complex matrix multplication A*B
  friend matrix<std::complex<float>>
  operator*(const matrix<std::complex<float>> &A,
            const matrix<float> &B); // real-complex matrix multplication A*B
  // Double-Precision Matrix Multiplication
  friend matrix<double>
  operator*(const matrix<double> &A,
            const matrix<double> &B); // real matrix multiplication A*B
  friend matrix<std::complex<double>>
  operator*(const matrix<std::complex<double>> &A,
            const matrix<std::complex<double>>
                &B); // complex matrix multplication A*B
  friend matrix<std::complex<double>>
  operator*(const matrix<double> &A,
            const matrix<std::complex<double>>
                &B); // real-complex matrix multplication A*B
  friend matrix<std::complex<double>>
  operator*(const matrix<std::complex<double>> &A,
            const matrix<double> &B); // real-complex matrix multplication A*B
  // Single-Precision Matrix-Vector Multiplication
  friend std::vector<float> operator*(const matrix<float> &A,
                                      const std::vector<float> &v);
  friend std::vector<std::complex<float>>
  operator*(const matrix<std::complex<float>> &A,
            const std::vector<std::complex<float>> &v);
  // Double-Precision Matrix-Vector Multiplication
  friend std::vector<double> operator*(const matrix<double> &A,
                                       const std::vector<double> &v);
  friend std::vector<std::complex<double>>
  operator*(const matrix<std::complex<double>> &A,
            const std::vector<std::complex<double>> &v);

public:
  // Constructors of matrix class
  matrix();
  matrix(size_t rows, size_t cols);
  explicit matrix(size_t size); // Makes a square matrix and rows = sqrt(size) columns =
                                // sqrt(dims)
  matrix(const matrix<T> &m);
  matrix(const matrix<T> &m, const char uplo);
  
  // Initialize an empty matrix() to matrix(size_t  rows, size_t cols)
  void initialize(size_t rows, size_t cols);
  // Clear used memory
  void clear();

  // Destructor
  virtual ~matrix();

  // Assignment operator
  matrix<T> &operator=(const matrix<T> &m);
  template <class S>
  matrix<T> &operator=(const matrix<S> &m); // Still would like to have real
                                            // assigend by complex -- take real
                                            // part

  // Addressing elements by vector representation
  T &operator[](size_t element);
  T operator[](size_t element) const;
  // Addressing elements by matrix representation
  T &operator()(size_t row, size_t col);
  T operator()(size_t row, size_t col) const;

  // overloading functions.
  matrix<T> operator+(const matrix<T> &A);
  matrix<T> operator-(const matrix<T> &A);
  matrix<T> operator+(const matrix<T> &A) const;
  matrix<T> operator-(const matrix<T> &A) const;
  matrix<T> &operator+=(const matrix<T> &A);
  matrix<T> &operator-=(const matrix<T> &A);

  // Member Functions
  size_t GetColumns() const; // gives the number of columns
  size_t GetRows() const;    // gives the number of rows
  size_t GetLD() const;      // gives the leading dimension -- number of rows
  size_t size() const;       // gives the size of the underlying vector
  void resize(size_t row, size_t col); // sets the size of the underlying vector
  void SetOutputStyle(enum OutputStyle outputstyle); // sets the style the
                                                     // matrix is display by <<
  T *GetMat() const; // gives you the address of element 0 then *(c+i) gives you
                     // the ith element

protected:
  size_t rows_, cols_, size_, LD_;
  // rows_ and cols_ are the rows and columns of the matrix
  // size_ = rows*colums dimensions of the vector representation
  // LD is the leading dimeonsion and for Column major order is in general eqaul
  // to rows
  enum OutputStyle outputstyle_;
  // outputstyle_ is the output style used by <<
  T *mat_;
  // the ptr to the vector containing the matrix
};

/*******************************************************************************
 *
 * Matrix Operations
 *
 ******************************************************************************/

namespace MOs {

// Common Matricies
template <class T> void Null(matrix<T> &A);
template <class T> void Identity(matrix<T> &I);
void Pauli(matrix<std::complex<double>> &sx, matrix<std::complex<double>> &sy,
           matrix<std::complex<double>> &sz);
void Pauli(matrix<std::complex<float>> &sx, matrix<std::complex<float>> &sy,
           matrix<std::complex<float>> &sz);

// operations on Matrics
template <class T> matrix<T> Transpose(const matrix<T> &A);
template <class T>
matrix<std::complex<T>> Dagger(const matrix<std::complex<T>> &A);
template <class T>
matrix<std::complex<T>> Conjugate(const matrix<std::complex<T>> &A);
template <class T> T Trace(const matrix<T> &A);
template <class T> matrix<T> TraceOutA(const matrix<T> &, size_t);
template <class T> matrix<T> TraceOutB(const matrix<T> &, size_t);
template <class T>
matrix<T> TensorProduct(const matrix<T> &A, const matrix<T> &B);
}

template <class T> inline void MOs::Null(matrix<T> &A) {
  // set zero to all elements of this matrix
  for (size_t i = 0; i < A.GetRows(); i++)
    for (size_t j = 0; j < A.GetColumns(); j++)
      A(i, j) = T(0);
  return;
}

template <class T> inline void MOs::Identity(matrix<T> &I) {
  // The Idenity operator
  size_t dim = I.GetRows();
  if (dim != I.GetColumns())
    std::cerr << "error: Routine MOs::Identity: Matrix is not square"
              << std::endl;
  for (size_t i = 0; i < dim; i++)
    for (size_t j = 0; j < dim; j++)
      I(i, j) = i == j ? T(1) : T(0);
  return;
}

inline void MOs::Pauli(matrix<std::complex<double>> &sx,
                       matrix<std::complex<double>> &sy,
                       matrix<std::complex<double>> &sz) {
  // Pauli x
  sx(0, 0) = std::complex<double>(0);
  sx(1, 0) = std::complex<double>(1);
  sx(0, 1) = std::complex<double>(1);
  sx(1, 1) = std::complex<double>(0);

  // Pauli x
  sy(0, 0) = std::complex<double>(0);
  sy(1, 0) = std::complex<double>(0, 1);
  sy(0, 1) = std::complex<double>(0, -1);
  sy(1, 1) = std::complex<double>(0);

  // Pauli z
  sz(0, 0) = std::complex<double>(1);
  sz(1, 0) = std::complex<double>(0);
  sz(0, 1) = std::complex<double>(0);
  sz(1, 1) = std::complex<double>(-1);

  return;
}

inline void MOs::Pauli(matrix<std::complex<float>> &sx,
                       matrix<std::complex<float>> &sy,
                       matrix<std::complex<float>> &sz) {
  // Pauli x
  sx(0, 0) = std::complex<float>(0);
  sx(1, 0) = std::complex<float>(1);
  sx(0, 1) = std::complex<float>(1);
  sx(1, 1) = std::complex<float>(0);

  // Pauli x
  sy(0, 0) = std::complex<float>(0);
  sy(1, 0) = std::complex<float>(0, 1);
  sy(0, 1) = std::complex<float>(0, -1);
  sy(1, 1) = std::complex<float>(0);

  // Pauli z
  sz(1, 0) = std::complex<float>(0);
  sz(0, 0) = std::complex<float>(1);
  sz(0, 1) = std::complex<float>(0);
  sz(1, 1) = std::complex<float>(-1);

  return;
}

template <class T> inline matrix<T> MOs::Transpose(const matrix<T> &A) {
  // Transposes a Matrix
  size_t rows = A.GetRows(), cols = A.GetColumns();
  matrix<T> temp(cols, rows);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      temp(j, i) = A(i, j);
    }
  }
  return temp;
}

template <class T>
inline matrix<std::complex<T>> MOs::Dagger(const matrix<std::complex<T>> &A) {
  // Take the Hermitian conjugate of a complex matrix
  size_t cols = A.GetColumns(), rows = A.GetRows();
  matrix<std::complex<T>> temp(cols, rows);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      temp(j, i) = conj(A(i, j));
    }
  }
  return temp;
}

template <class T>
inline matrix<std::complex<T>>
MOs::Conjugate(const matrix<std::complex<T>> &A) {
  // Take the complex conjugate of a complex matrix
  size_t rows = A.GetRows(), cols = A.GetColumns();
  matrix<std::complex<T>> temp(rows, cols);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      temp(i, j) = conj(A(i, j));
    }
  }
  return temp;
}

template <class T> inline T MOs::Trace(const matrix<T> &A) {
  // Finds the trace of a matrix
  size_t rows = A.GetRows(), cols = A.GetColumns();
  if (rows != cols)
    std::cerr << "error: Routine MOs::Trace: Matrix is not square" << std::endl;
  T temp = 0.0;
  for (size_t i = 0; i < rows; i++) {
    temp = temp + A(i, i);
  }
  return temp;
}

template <class T>
inline matrix<T> MOs::TraceOutA(const matrix<T> &rho, size_t dimA) {
  // Traces out first system (dimension dimA) of composite Hilbert space
  size_t rows = rho.GetRows(), cols = rho.GetColumns();
  if (rows != cols) {
    std::cerr << "error: Routine MOs::TraceOutA: Matrix is not square"
              << std::endl;
    exit(1);
  }
  if (rows % dimA != 0) {
    std::cerr << "error: Routine MOs::TraceOutA: dim(rho)/dim(system B) is not "
                 "an integer"
              << std::endl;
    exit(1);
  }
  size_t dimB = rows / dimA;
  matrix<T> rhoB(dimB, dimB);
  T temp = 0.0;
  for (size_t i = 0; i < dimB; i++) {
    for (size_t j = 0; j < dimB; j++) {
      for (size_t k = 0; k < dimA; k++) {
        temp = temp + rho(i + dimB * k, j + dimB * k);
      }
      rhoB(i, j) = temp;
      temp = 0.0;
    }
  }
  return rhoB;
}

template <class T>
inline matrix<T> MOs::TraceOutB(const matrix<T> &rho, size_t dimB) {
  // Traces out second system (dimension dimB) of composite Hilbert space
  size_t rows = rho.GetRows(), cols = rho.GetColumns();
  if (rows != cols) {
    std::cerr << "error: Routine MOs::TraceOutB: Matrix is not square"
              << std::endl;
    exit(1);
  }
  if (rows % dimB != 0) {
    std::cerr << "error: Routine MOs::TraceOutB: dim(rho)/dim(system B) is not "
                 "an integer"
              << std::endl;
    exit(1);
  }
  size_t dimA = rows / dimB;
  matrix<T> rhoA(dimA, dimA);
  T temp = 0.0;
  for (size_t i = 0; i < dimA; i++) {
    size_t offsetX = i * dimB;
    for (size_t j = 0; j < dimA; j++) {
      size_t offsetY = j * dimB;
      for (size_t k = 0; k < dimB; k++) {
        temp = temp + rho(offsetX + k, offsetY + k);
      }
      rhoA(i, j) = temp;
      temp = 0.0;
    }
  }
  return rhoA;
}

template <class T>
inline matrix<T> MOs::TensorProduct(const matrix<T> &A, const matrix<T> &B) {
  // Works out the TensorProduct of two matricies A tensor B
  // Note that if A is i x j and B is p x q then A \otimes B is an ip x jq
  // rmatrix
  size_t rows1 = A.GetRows(), rows2 = B.GetRows(), cols1 = A.GetColumns(),
         cols2 = B.GetColumns();
  size_t rows_new = rows1 * rows2, cols_new = cols1 * cols2, n, m;
  matrix<T> temp(rows_new, cols_new);
  // a11 B, a12 B ... a1j B
  // ai1 B, ai2 B ... aij B
  for (size_t i = 0; i < rows1; i++) {
    for (size_t j = 0; j < cols1; j++) {
      for (size_t p = 0; p < rows2; p++) {
        for (size_t q = 0; q < cols2; q++) {
          n = i * rows2 + p;
          m = j * cols2 + q; //  0 (0 + 1)  + 1*dimb=2 + (0 + 1 )  (j*dimb+q)
          temp(n, m) = A(i, j) * B(p, q);
        }
      }
    }
  }
  return temp;
}

/*******************************************************************************
 *
 * Matrix class: methods
 *
 ******************************************************************************/

template <class T>
inline matrix<T>::matrix()
    : rows_(0), cols_(0), size_(0), LD_(0), outputstyle_(Matrix), mat_(nullptr) {}
// constructs an empty matrix using the ....
template <class T>
inline matrix<T>::matrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), size_(rows * cols), LD_(rows),
      outputstyle_(Column), mat_(new T[size_]) {}
// constructs an empty matrix of size rows, cols and sets outputstyle to zero
template <class T>
inline matrix<T>::matrix(size_t dim2)
    : size_(dim2), outputstyle_(Column), mat_(new T[size_]) {
  // constructs a square matrix of dims sqrt(size)*sqrt(size)
  // PLEASE DO NOT CHANGE THIS AS IT IS USED BY ODESOLVER
  //  size_t dim2 = rhovec.GetRows();
  // size_t dim = (size_t)std::floor(sqrt((double)dim2+1.0));
  if (size_ == 0)
    rows_ = size_;
  else if (size_ == 1)
    rows_ = size_;
  else if (size_ == 2) {
    std::cerr << "Error: matrix constructor matrix(dim): the number you enterd "
                 "does not have a interger sqrt."
              << std::endl;
    exit(1);
  } else
    for (rows_ = 2; rows_ < size_; ++rows_) {
      if (size_ == rows_ * rows_)
        break;
      if (rows_ * rows_ > size_) {
        std::cerr << "Error: matrix constructor matrix(dim): the number you "
                     "enterd does not have a interger sqrt"
                  << std::endl;
        exit(1);
      }
    }

  cols_ = rows_;
  LD_ = rows_;
}
// constructs an empty matrix of size rows, cols and sets outputstyle to zero
template <class T>
inline matrix<T>::matrix(const matrix<T> &rhs)
    : rows_(rhs.rows_), cols_(rhs.cols_), size_(rhs.size_), LD_(rows_),
      outputstyle_(rhs.outputstyle_), mat_(new T[size_]) {
  // Copy constructor, copies the matrix to another matrix
  for (size_t p = 0; p < size_; p++) {
    mat_[p] = rhs.mat_[p];
  }
}
template <class T>
inline matrix<T>::matrix(const matrix<T> &rhs, const char uplo)
    : rows_(rhs.rows_), cols_(rhs.cols_), size_(rhs.size_), LD_(rows_),
      outputstyle_(rhs.outputstyle_), mat_(new T[size_]) {
  // Copy constructor, copies the matric to another matrix but only the upper or
  // lower triangle
  if (uplo == 'U') {
    for (size_t i = 0; i < rows_; i++)
      for (size_t j = i; j < cols_; j++)
        mat_[j * rows_ + i] = rhs.mat_[j * rows_ + i];
  } else if (uplo == 'L') {
    for (size_t i = 0; i < rows_; i++)
      for (size_t j = 0; j <= i; j++)
        mat_[j * rows_ + i] = rhs.mat_[j * rows_ + i];
  } else {
    std::cerr << "Error: matrix copy constructor did not have a valid option "
                 "-- these are Lower, Upper"
              << std::endl;
    exit(1);
  }
}
template <class T> inline void matrix<T>::initialize(size_t rows, size_t cols) {
  if (rows_ != rows || cols_ != cols) { // if the rows are different size delete
    // re-construct the matrix
    if (mat_ != 0)
      delete[](mat_);
    rows_ = rows;
    cols_ = cols;
    size_ = rows_ * cols_;
    LD_ = rows;
    mat_ = new T[size_];
    // std::cout << "Assignement (reassigment), size " << size_ << " rows_ " <<
    // rows_ << " cols " << cols_ << " LD "<< LD_ <<  std::endl;
  }
}
template <class T> inline void matrix<T>::clear() {
  if (!mat_ || !size_)
    return;
  rows_ = cols_ = size_ = 0;
  delete[](mat_);
  mat_ = 0;
}

template <class T> inline void matrix<T>::resize(size_t rows, size_t cols) {
  if (rows_ == rows && cols_ == cols)
    return;
  T *tempmat = new T[size_ = rows * cols];

  for (size_t i = 0; i < rows; i++)
    for (size_t j = 0; j < cols; j++)
      if (i < rows_ && j < cols_)
        tempmat[j * rows + i] = mat_[j * rows_ + i];
      else
        tempmat[j * rows + i] = 0.0;
  LD_ = rows_ = rows;
  cols_ = cols;
  delete[](mat_);
  mat_ = tempmat;
}

template <class T> inline matrix<T>::~matrix() {
  // destructor, deletes the matrix from memory when we leave the scope
  if (mat_ != nullptr)
    delete[](mat_);
}
template <class T>
inline matrix<T> &matrix<T>::operator=(const matrix<T> &rhs) {
  // overloading the assignement operator
  // postcondition: normal assignment via copying has been performed;
  //    if matrix and rhs were different sizes, matrix
  //    has been resized to match the size of rhs
  //#ifdef DEBUG
  if (rows_ != rhs.rows_ || cols_ != rhs.cols_) { // if the rows are different
    // size delete re-construct
    // the matrix
    if (mat_ != 0)
      delete[](mat_);
    rows_ = rhs.rows_;
    cols_ = rhs.cols_;
    size_ = rows_ * cols_;
    LD_ = rhs.LD_;
    mat_ = new T[size_];
  }
  //#endif
  for (size_t p = 0; p < size_; p++) { // the copying of the matrix
    // std::cout << p << std::endl;
    mat_[p] = T(rhs.mat_[p]);
  }
  return *this;
}
template <class T>
template <class S>
inline matrix<T> &matrix<T>::operator=(const matrix<S> &rhs) {
  // overloading the assignement operator
  // postcondition: normal assignment via copying has been performed;
  //    if matrix and rhs were different sizes, matrix
  //    has been resized to match the size of rhs
  //#ifdef DEBUG
  if (rows_ != rhs.GetRows() ||
      cols_ != rhs.GetColumns()) { // if the rows are different size delete
    // re-construct the matrix
    if (mat_ != 0)
      delete[](mat_);
    rows_ = rhs.GetRows();
    cols_ = rhs.GetColumns();
    size_ = rows_ * cols_;
    LD_ = rhs.GetLD();
    mat_ = new T[size_];
  }
  //#endif
  for (size_t p = 0; p < size_; p++) { // the copying of the matrix
    mat_[p] = T(rhs[p]);
  }
  return *this;
}
template <class T> inline T &matrix<T>::operator[](size_t p) {
// returns the pth element of the vector representing the matrix,  remember []
// is the operator acting on the object matrix (read backwards).
#ifdef DEBUG
  if (p >= size_) {
    std::cerr
        << "error: matrix class operator []: Matrix subscript out of bounds"
        << std::endl;
    exit(1);
  }
#endif
  return mat_[p];
}
template <class T> inline T matrix<T>::operator[](size_t p) const {
// returns the pth element of the vector representing the matrix if the operator
// is acting on a const
#ifdef DEBUG
  if (p >= size_) {
    std::cerr << "Error: matrix class operator [] const: Matrix subscript out "
                 "of bounds"
              << std::endl;
    exit(1);
  }
#endif
  return mat_[p];
}
template <class T> inline T &matrix<T>::operator()(size_t i, size_t j) {
// return the data at position i,j, remember () is the operator acting on the
// object matrix (read backwards).
#ifdef DEBUG
  if (i >= rows_ || j >= cols_) {
    std::cerr
        << "Error: matrix class operator (): Matrices subscript out of bounds"
        << std::endl;
    exit(1);
  }
#endif
  return mat_[j * rows_ + i];
}
template <class T> inline T matrix<T>::operator()(size_t i, size_t j) const {
// return the data at i,j if the operator is acting on a const
#ifdef DEBUG
  if (i >= rows_ || j >= cols_) {
    std::cerr << "Error: matrix class operator ()   const: Matrices subscript "
                 "out of bounds"
              << std::endl;
    exit(1);
  }
#endif
  return mat_[j * rows_ + i];
}
template <class T> inline size_t matrix<T>::GetRows() const {
  // returns the rows of the matrix
  return rows_;
}
template <class T> inline size_t matrix<T>::GetColumns() const {
  // returns the colums of the matrix
  return cols_;
}
template <class T> inline size_t matrix<T>::GetLD() const {
  // returns the leading dimension
  return LD_;
}
template <class T> inline size_t matrix<T>::size() const {
  // returns the size of the underlying vector
  return size_;
}
template <class T> inline T *matrix<T>::GetMat() const {
  // returns the ptr for the matrix data
  return mat_;
}
template <class T>
inline void matrix<T>::SetOutputStyle(enum OutputStyle outputstyle) {
  // sets the outputstyle
  outputstyle_ = outputstyle;
}
template <class T> inline matrix<T> matrix<T>::operator+(const matrix<T> &A) {
// overloads the + for matrix addition, can this be more efficient
#ifdef DEBUG
  if (rows_ != A.rows_ || cols_ != A.cols_) {
    std::cerr
        << "Error: matrix class operator +: Matrices are not the same size"
        << std::endl;
    exit(1);
  }
#endif
  matrix<T> temp(rows_, cols_);
  for (unsigned int p = 0; p < size_; p++) {
    temp.mat_[p] = mat_[p] + A.mat_[p];
  }
  return temp;
}
template <class T> inline matrix<T> matrix<T>::operator-(const matrix<T> &A) {
// overloads the - for matrix substraction, can this be more efficient
#ifdef DEBUG
  if (rows_ != A.rows_ || cols_ != A.cols_) {
    std::cerr
        << "Error: matrix class operator -: Matrices are not the same size"
        << std::endl;
    exit(1);
  }
#endif
  matrix<T> temp(rows_, cols_);
  for (unsigned int p = 0; p < size_; p++) {
    temp.mat_[p] = mat_[p] - A.mat_[p];
  }
  return temp;
}
template <class T>
inline matrix<T> matrix<T>::operator+(const matrix<T> &A) const {
// overloads the + for matrix addition if it is a const matrix, can this be more
// efficient
#ifdef DEBUG
  if (rows_ != A.rows_ || cols_ != A.cols_) {
    std::cerr << "Error: matrix class operator + const: Matrices are not the "
                 "same size"
              << std::endl;
    exit(1);
  }
#endif
  matrix<T> temp(rows_, cols_);
  for (unsigned int p = 0; p < size_; p++) {
    temp.mat_[p] = mat_[p] + A.mat_[p];
  }
  return temp;
}
template <class T>
inline matrix<T> matrix<T>::operator-(const matrix<T> &A) const {
// overloads the - for matrix substraction, can this be more efficient
#ifdef DEBUG
  if (rows_ != A.rows_ || cols_ != A.cols_) {
    std::cerr << "Error: matrix class operator - const: Matrices are not the "
                 "same size"
              << std::endl;
    exit(1);
  }
#endif
  matrix<T> temp(rows_, cols_);
  for (unsigned int p = 0; p < size_; p++) {
    temp.mat_[p] = mat_[p] - A.mat_[p];
  }
  return temp;
}
template <class T> inline matrix<T> &matrix<T>::operator+=(const matrix<T> &A) {
// overloads the += for matrix addition and assignment, can this be more
// efficient
#ifdef DEBUG
  if (rows_ != A.rows_ || cols_ != A.cols_) {
    std::cerr
        << "Error: matrix class operator +=: Matrices are not the same size"
        << std::endl;
    exit(1);
  }
#endif
  for (size_t p = 0; p < size_; p++) {
    mat_[p] += A.mat_[p];
  }
  return *this;
}
template <class T> inline matrix<T> &matrix<T>::operator-=(const matrix<T> &A) {
// overloads the -= for matrix subtraction and assignement, can this be more
// efficient
#ifdef DEBUG
  if (rows_ != A.rows_ || cols_ != A.cols_) {
    std::cerr
        << "Error: matrix class operator -=: Matrices are not the same size"
        << std::endl;
    exit(1);
  }
#endif
  for (size_t p = 0; p < size_; p++) {
    mat_[p] -= A.mat_[p];
  }
  return *this;
}

/*******************************************************************************
 *
 * Matrix class: Friend Functions
 *
 ******************************************************************************/
template <class T>
std::ostream &operator<<(std::ostream &output, const matrix<T> &A) {
  // overloads << and has three output styles
  // Column is in a column vector
  // List is a row of the actual column vector
  // Matrix is the matrix
  if (A.outputstyle_ == List) {
    for (size_t p = 0; p < A.size_; p++) {
      output << A.mat_[p] << '\t';
    }
  } else if (A.outputstyle_ == Matrix) {
    for (size_t i = 0; i < A.rows_; i++) {
      for (size_t j = 0; j < A.cols_; j++) {
        output << A.mat_[j * A.rows_ + i] << "\t";
      }
      output << std::endl;
    }
  }

  else if (A.outputstyle_ == Column) {
    for (size_t p = 0; p < A.size_; p++) {
      output << A.mat_[p] << '\n';
    }
  } else {
    std::cerr << "Error: matrix operator << is not assigned with a valid "
                 "option -- these are Column, List, Matrix"
              << std::endl;
    exit(1);
  }

  return output;
}
template <class T>
std::istream &operator>>(std::istream &input, const matrix<T> &A) {
  // overloads the >> to read in a row into column format
  for (size_t j = 0; j < A.cols_; j++) {
    for (size_t i = 0; i < A.rows_; i++) {
      input >> A.mat_[j * A.rows_ + i];
    }
  }
  return input;
}
template <class S1, class S2>
matrix<S1> operator*(const matrix<S1> &A, const S2 &beta) {
  // overloads A*beta
  size_t rows = A.rows_, cols = A.cols_;
  matrix<S1> temp(rows, cols);
  for (size_t j = 0; j < cols; j++) {
    for (size_t i = 0; i < rows; i++) {
      temp(i, j) = beta * A(i, j);
    }
  }
  return temp;
}
template <class S1, class S2>
matrix<S1> operator*(const S2 &beta, const matrix<S1> &A) {
  // overloads beta*A
  size_t rows = A.rows_, cols = A.cols_;
  matrix<S1> temp(rows, cols);
  for (size_t j = 0; j < cols; j++) {
    for (size_t i = 0; i < rows; i++) {
      temp(i, j) = beta * A(i, j);
    }
  }
  return temp;
}

// Operator overloading with BLAS functions
inline matrix<double> operator*(const matrix<double> &A,
                                const matrix<double> &B) {
  // overloads A*B for real matricies and uses the blas dgemm routine
  // cblas_dgemm(CblasXMajor,op,op,N,M,K,alpha,A,LDA,B,LDB,beta,C,LDC)
  // C-> alpha*op(A)*op(B) +beta C
  matrix<double> C(A.rows_, B.cols_);
  double alpha = 1.0, beta = 0.0;
  dgemm_(&Trans[0], &Trans[0], &A.rows_, &B.cols_, &A.cols_, &alpha, A.mat_,
         &A.LD_, B.mat_, &B.LD_, &beta, C.mat_, &C.LD_);
  // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows_, B.cols_,
  // A.cols_, 1.0, A.mat_, A.LD_, B.mat_, B.LD_, 0.0, C.mat_, C.LD_);
  return C;
}
inline matrix<float> operator*(const matrix<float> &A, const matrix<float> &B) {
  // overloads A*B for real matricies and uses the blas sgemm routine
  // cblas_sgemm(CblasXMajor,op,op,N,M,K,alpha,A,LDA,B,LDB,beta,C,LDC)
  // C-> alpha*op(A)*op(B) +beta C
  matrix<float> C(A.rows_, B.cols_);
  float alpha = 1.0, beta = 0.0;
  sgemm_(&Trans[0], &Trans[0], &A.rows_, &B.cols_, &A.cols_, &alpha, A.mat_,
         &A.LD_, B.mat_, &B.LD_, &beta, C.mat_, &C.LD_);
  // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows_, B.cols_,
  // A.cols_, 1.0, A.mat_, A.LD_, B.mat_, B.LD_, 0.0, C.mat_, C.LD_);
  return C;
}
inline matrix<std::complex<float>>
operator*(const matrix<std::complex<float>> &A,
          const matrix<std::complex<float>> &B) {
  // overloads A*B for complex matricies and uses the blas zgemm routine
  // cblas_zgemm(CblasXMajor,op,op,N,M,K,alpha,A,LDA,B,LDB,beta,C,LDC)
  // C-> alpha*op(A)*op(B) +beta C
  matrix<std::complex<float>> C(A.rows_, B.cols_);
  std::complex<float> alpha = 1.0, beta = 0.0;
  cgemm_(&Trans[0], &Trans[0], &A.rows_, &B.cols_, &A.cols_, &alpha, A.mat_,
         &A.LD_, B.mat_, &B.LD_, &beta, C.mat_, &C.LD_);
  // cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows_, B.cols_,
  // A.cols_, &alpha, A.mat_, A.LD_, B.mat_, B.LD_, &beta, C.mat_, C.LD_);
  return C;
}
inline matrix<std::complex<double>>
operator*(const matrix<std::complex<double>> &A,
          const matrix<std::complex<double>> &B) {
  // overloads A*B for complex matricies and uses the blas zgemm routine
  // cblas_zgemm(CblasXMajor,op,op,N,M,K,alpha,A,LDA,B,LDB,beta,C,LDC)
  // C-> alpha*op(A)*op(B) +beta C
  matrix<std::complex<double>> C(A.rows_, B.cols_);
  std::complex<double> alpha = 1.0, beta = 0.0;
  zgemm_(&Trans[0], &Trans[0], &A.rows_, &B.cols_, &A.cols_, &alpha, A.mat_,
         &A.LD_, B.mat_, &B.LD_, &beta, C.mat_, &C.LD_);
  // cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows_, B.cols_,
  // A.cols_, &alpha, A.mat_, A.LD_, B.mat_, B.LD_, &beta, C.mat_, C.LD_);
  return C;
}
inline matrix<std::complex<float>>
operator*(const matrix<float> &A, const matrix<std::complex<float>> &B) {
  // overloads A*B for complex matricies and uses the blas zgemm routine
  // cblas_zgemm(CblasXMajor,op,op,N,M,K,alpha,A,LDA,B,LDB,beta,C,LDC)
  // C-> alpha*op(A)*op(B) +beta C
  matrix<std::complex<float>> C(A.rows_, B.cols_), Ac(A.rows_, A.cols_);
  Ac = A;
  std::complex<float> alpha = 1.0, beta = 0.0;
  cgemm_(&Trans[0], &Trans[0], &Ac.rows_, &B.cols_, &Ac.cols_, &alpha, Ac.mat_,
         &Ac.LD_, B.mat_, &B.LD_, &beta, C.mat_, &C.LD_);
  // cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Ac.rows_, B.cols_,
  // Ac.cols_, &alpha, Ac.mat_, Ac.LD_, B.mat_, B.LD_, &beta, C.mat_, C.LD_);
  return C;
}
inline matrix<std::complex<double>>
operator*(const matrix<double> &A, const matrix<std::complex<double>> &B) {
  // overloads A*B for complex matricies and uses the blas zgemm routine
  // cblas_zgemm(CblasXMajor,op,op,N,M,K,alpha,A,LDA,B,LDB,beta,C,LDC)
  // C-> alpha*op(A)*op(B) +beta C
  matrix<std::complex<double>> C(A.rows_, B.cols_), Ac(A.rows_, A.cols_);
  Ac = A;
  std::complex<double> alpha = 1.0, beta = 0.0;
  zgemm_(&Trans[0], &Trans[0], &Ac.rows_, &B.cols_, &Ac.cols_, &alpha, Ac.mat_,
         &Ac.LD_, B.mat_, &B.LD_, &beta, C.mat_, &C.LD_);
  // cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Ac.rows_, B.cols_,
  // Ac.cols_, &alpha, Ac.mat_, Ac.LD_, B.mat_, B.LD_, &beta, C.mat_, C.LD_);
  return C;
}
inline matrix<std::complex<float>>
operator*(const matrix<std::complex<float>> &A, const matrix<float> &B) {
  // overloads A*B for complex matricies and uses the blas zgemm routine
  // cblas_zgemm(CblasXMajor,op,op,N,M,K,alpha,A,LDA,B,LDB,beta,C,LDC)
  // C-> alpha*op(A)*op(B) +beta C
  matrix<std::complex<float>> C(A.rows_, B.cols_), Bc(B.rows_, B.cols_);
  Bc = B;
  std::complex<float> alpha = 1.0, beta = 0.0;
  cgemm_(&Trans[0], &Trans[0], &A.rows_, &Bc.cols_, &A.cols_, &alpha, A.mat_,
         &A.LD_, Bc.mat_, &Bc.LD_, &beta, C.mat_, &C.LD_);
  // cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows_, Bc.cols_,
  // A.cols_, &alpha, A.mat_, A.LD_, Bc.mat_, Bc.LD_, &beta, C.mat_, C.LD_);
  return C;
}
inline matrix<std::complex<double>>
operator*(const matrix<std::complex<double>> &A, const matrix<double> &B) {
  // overloads A*B for complex matricies and uses the blas zgemm routine
  // cblas_zgemm(CblasXMajor,op,op,N,M,K,alpha,A,LDA,B,LDB,beta,C,LDC)
  // C-> alpha*op(A)*op(B) +beta C
  matrix<std::complex<double>> C(A.rows_, B.cols_), Bc(B.rows_, B.cols_);
  Bc = B;
  std::complex<double> alpha = 1.0, beta = 0.0;
  zgemm_(&Trans[0], &Trans[0], &A.rows_, &Bc.cols_, &A.cols_, &alpha, A.mat_,
         &A.LD_, Bc.mat_, &Bc.LD_, &beta, C.mat_, &C.LD_);
  // cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows_, Bc.cols_,
  // A.cols_, &alpha, A.mat_, A.LD_, Bc.mat_, Bc.LD_, &beta, C.mat_, C.LD_);
  return C;
}

// Single-Precision Real
inline std::vector<float> operator*(const matrix<float> &A,
                                    const std::vector<float> &x) {
  // overload A*v for complex matrixies and will used a blas function
  std::vector<float> y(A.rows_);
  float alpha = 1.0, beta = 0.0;
  const size_t incx = 1, incy = 1;
  sgemv_(&Trans[0], &A.rows_, &A.cols_, &alpha, A.mat_, &A.LD_, x.data(), &incx,
         &beta, y.data(), &incy);
  return y;
}
// Double-Precision Real
inline std::vector<double> operator*(const matrix<double> &A,
                                     const std::vector<double> &x) {
  // overload A*v for complex matrixies and will used a blas function
  std::vector<double> y(A.rows_);
  double alpha = 1.0, beta = 0.0;
  const size_t incx = 1, incy = 1;
  dgemv_(&Trans[0], &A.rows_, &A.cols_, &alpha, A.mat_, &A.LD_, x.data(), &incx,
         &beta, y.data(), &incy);
  return y;
}
// Single-Precision Complex
inline std::vector<std::complex<float>>
operator*(const matrix<std::complex<float>> &A,
          const std::vector<std::complex<float>> &x) {
  // overload A*v for complex matrixies and will used a blas function
  std::vector<std::complex<float>> y(A.rows_);
  std::complex<float> alpha = 1.0, beta = 0.0;
  const size_t incx = 1, incy = 1;
  cgemv_(&Trans[0], &A.rows_, &A.cols_, &alpha, A.mat_, &A.LD_, x.data(), &incx,
         &beta, y.data(), &incy);
  return y;
}
// Double-Precision Complex
inline std::vector<std::complex<double>>
operator*(const matrix<std::complex<double>> &A,
          const std::vector<std::complex<double>> &x) {
  // overload A*v for complex matrixies and will used a blas function
  std::vector<std::complex<double>> y(A.rows_);
  std::complex<double> alpha = 1.0, beta = 0.0;
  const size_t incx = 1, incy = 1;
  zgemv_(&Trans[0], &A.rows_, &A.cols_, &alpha, A.mat_, &A.LD_, x.data(), &incx,
         &beta, y.data(), &incy);
  return y;
}

//------------------------------------------------------------------------------
// end _matrix_h_
//------------------------------------------------------------------------------
#endif
