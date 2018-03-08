#include <array>
#include <chrono>
#include <complex>
#include <iostream>
#include <vector>

// ostream overload for pairs
template <typename T1, typename T2>
std::ostream &operator<<(std::ostream &out, const std::pair<T1, T2> &p) {
  out << "(" << p.first << ", " << p.second << ")";
  return out;
}

// ostream overload for vectors
template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
  out << "[";
  size_t last = v.size() - 1;
  for (size_t i = 0; i < v.size(); ++i) {
    out << v[i];
    if (i != last)
      out << ", ";
  }
  out << "]";
  return out;
}

// ostream overload for vectors
template <typename T, size_t N>
std::ostream &operator<<(std::ostream &out, const std::array<T, N> &v) {
  out << "[";
  for (size_t i = 0; i < N; ++i) {
    out << v[i];
    if (i != N - 1)
      out << ", ";
  }
  out << "]";
  return out;
}

#include "qubit_vector.hpp"

using namespace std;

using myclock_t = std::chrono::system_clock;
using uint_t = uint64_t;
using int_t = int64_t;
using cvector_t = std::vector<complex_t>;
using rvector_t = std::vector<double>;


bool approx_equal(complex_t a, complex_t b, double threshold=1e-10) {
  return !(std::abs(a - b) > threshold);
}
bool approx_equal(double a, double b, double threshold=1e-10) {
  return !(std::abs(a - b) > threshold);
}

template <typename T>
bool approx_equal(std::vector<T> a, std::vector<T> b, double threshold=1e-10) {
  if (a.size() != b.size())
    return false;
  double diff = 0.;
  for (uint_t j=0; j < a.size(); j++)
    diff += std::abs(a[j] - b[j]);
  return !(diff > threshold);
}

void test_condition(bool value, std::string msg = "") {
  std::string pass = (value) ? "PASSED: " : "FAILED: ";
  std::clog << pass << msg << std::endl;
}

/*******************************************************************************
   *
   * QubitVector class methods to test
   * 
   * Constructors: 
   *  - QubitVector(uint_t)
   *  - QubitVector(cvector_t)
   *  - QubitVector(rvector_t)
   * 
   * Operators:
   *  - assignment = (cvector_t)
   *  - assignment = (rvector_t)
   *  - element access []
   *  - scalar multiplication *= (complex_t)
   *  - scalar multiplication *= (double)
   *  - scalar multiplication * (complex_t)
   *  - scalar multiplication * (double)
   *  - vector addition += (QubitVector)
   *  - vector addition + (QubitVector)
   *  - vector subtraction -= (QubitVector)
   *  - vector subtraction - (QubitVector)
   * 
   * Utility methods:
   *  - size()
   *  - qubits()
   *  - vector()
   *  - set_omp_threads()
   *  - set_omp_threshold()
   * 
   * Algebra:
   *  - norm()
   *  - dot()
   *  - renormalize()
   *  - initialize()
   *  - initialize_plus()
   * 
   * Z-measurement outcome probability vectirs:
   *  - probabilities()
   *  - probabilities(uint)
   *  - probabilities(vector<uint>)
   *  - probabilities(array<uint>)
   * 
   * Z-measurement outcome probability values
   *  - probabilities(uint)
   *  - probabilities(uint, uint)
   *  - probabilities(vector<uint>, uint)
   *  - probabilities(array<uint>, uint)
   * 
   * Matrices:
   *  - apply_matrix(uint_t, cvector)  // square matrix length input
   *  - apply_matrix(uint_t, cvector)  // matrix diagonal length input
   *  - apply_matrix(uint_t, int, cvector)  // square matrix length input
   *  - apply_matrix(uint_t, int, cvector)  // matrix diagonal length input
   *  - apply_matrix(array<int>, cvector)  // square matrix length input
   *  - apply_matrix(array<int>, cvector)  // matrix diagonal length input
   *  - apply_matrix(vector<int>, cvector)  // square matrix length input
   *  - apply_matrix(vector<int>, cvector)  // matrix diagonal length input
   * 
   * Norms with matrix multiplication:
   *  - norm(uint, cvector)
   *  - norm(vector<uint>, cvector)
   *  - norm(array<uint>, cvector)
   * 
   * Expectation values:
   *  - expectation_value(uint, cvector)
   *  - expectation_value(vector<uint>, cvector)
   *  - expectation_value(array<uint>, cvector)
   * 
   ******************************************************************************/

int main(int argc, char **argv) {

  // USEFUL STATES
  const complex_t I(0, 1);
  const complex_t one(1, 0);
  const complex_t zero(0, 0);
  const complex_t ort2(1./std::sqrt(2.));

  const cvector_t UI({1., 0., 0., 1.});
  const cvector_t UX({0., 1., 1., 0.});
  const cvector_t UY({0., I, -I, 0.});
  const cvector_t UZ({1., 0., 0., -1.});

  const cvector_t empty({0., 0.});
  const cvector_t xp({ort2, ort2});
  const cvector_t xm({ort2, -ort2});
  const cvector_t yp({ort2, I * ort2});
  const cvector_t ym({ort2, -I * ort2});
  const cvector_t zp({1., 0.});
  const cvector_t zm({0., 1.});



  //------------------------------------------------------------------------------
  // CONSTRUCTORS
  //------------------------------------------------------------------------------

  clog << "\nCONSTRUCTOR TESTS" << endl;
  {  
    QubitVector test;
    test_condition(test.size() == 1, "size (default constructor)");
    test_condition(test.qubits() == 0, "qubits (default constructor)");
    test_condition(approx_equal(test.vector(), cvector_t({zero})), "vector (default constructor)");
  }
  {
    vector<size_t> dims({1, 2, 4, 8, 16, 32, 64, 128, 256, 512});
    bool size = true;
    bool qubits = true;
    bool vector = true;
    for (uint_t j=0; j < 10; j++) {
      QubitVector test(j);   
      size &= (test.size() == dims[j]);
      qubits &= (test.qubits() == j);
      vector &= (approx_equal(test.vector(), cvector_t(dims[j], 0.)));
    }
    test_condition(size, "size (uint_t constructor)");
    test_condition(qubits == 1, "qubits (uint_t constructor)");
    test_condition(vector, "vector (uint_t constructor)");
  }
  {
    QubitVector test(xp);
    test_condition(test.size() == 2, "size (cvector_t constructor)");
    test_condition(test.qubits() == 1, "qubits (cvector_t constructor)");
    test_condition(approx_equal(test.vector(), xp), "vector (cvector_t constructor)");
  }
  {
    QubitVector test(rvector_t({0., 1.}));
    test_condition(test.size() == 2, "size (rvector_t constructor)");
    test_condition(test.qubits() == 1, "qubits (rvector_t constructor)");
    test_condition(approx_equal(test.vector(), zm), "vector (rvector_t constructor)");
  }


 
  //------------------------------------------------------------------------------
  // OPERATORS
  //------------------------------------------------------------------------------

  clog << "\nOPERATOR TESTS" << endl;
  {
    // setter operators
    QubitVector test;
    test = xp;
    bool pass_equal = true;
    pass_equal &= test.size() == 2;
    pass_equal &= test.qubits() == 1;
    pass_equal &= approx_equal(test.vector(), xp);
    test_condition(pass_equal, "operator= (cvector)");
    test = rvector_t({{0.,1.}});
    pass_equal = true;
    pass_equal &= test.size() == 2;
    pass_equal &= test.qubits() == 1;
    pass_equal &= approx_equal(test.vector(), zm);
    test_condition(pass_equal, "operator= (rvector)");
  }
  {
    cvector_t v1(4, 1.0);
    cvector_t v2(4, -1.0);
    cvector_t v3(4, I);
    cvector_t v4(4, 2.0);

    // *= operators
    {
      QubitVector test = v1;
      bool pass_mult = true;
      test = v1;
      test *= -one;
      pass_mult &= (test.vector() == v2);
      test = v1;
      test *= I;
      pass_mult &= (test.vector() == v3);
      test = v1;
      test *= 2.0;
      pass_mult &= (test.vector() == v4);
      test_condition(pass_mult, "operator*=");
    }
    // * operators
    {
    bool pass_mult = true;
    QubitVector test = v1;
    pass_mult &= ((I * test).vector() == v3);
    pass_mult &= ((test * I).vector() == v3);
    pass_mult &= ((-one * test).vector() == v2);
    pass_mult &= ((test * -one).vector() == v2);
    pass_mult &= ((2.0 * test).vector() == v4);
    pass_mult &= ((test * 2.0).vector() == v4);
    test_condition(pass_mult, "operator*");
    }
    // += operators
    {
      bool pass_equal = true;
      QubitVector testA(v1);
      QubitVector testB(v2);
      QubitVector refP(2);
      testA += testB;
      pass_equal &= testA.size() == refP.size();
      pass_equal &= testA.qubits() == refP.qubits();
      pass_equal &= approx_equal(testA.vector(), refP.vector());
      test_condition(pass_equal, "operator+=");
    }
    // + operators
    {
      bool pass_equal = true;
      QubitVector testA(v1);
      QubitVector testB(v2);
      QubitVector refP(2);
      QubitVector testAB = testA + testB;
      pass_equal &= testAB.size() == refP.size();
      pass_equal &= testAB.qubits() == refP.qubits();
      pass_equal &= approx_equal(testAB.vector(), refP.vector());
      test_condition(pass_equal, "operator+");
    }
    // -= operators
    {
      bool pass_minus = true;
      QubitVector testA(v1);
      QubitVector testB(v2);
      QubitVector refM(v4);
      testA -= testB;
      pass_minus &= testA.size() == refM.size();
      pass_minus &= testA.qubits() == refM.qubits();
      pass_minus &= approx_equal(testA.vector(), refM.vector());
      test_condition(pass_minus, "operator-=");
    }
    // - operators
    {
      bool pass_minus = true;
      QubitVector testA(v1);
      QubitVector testB(v2);
      QubitVector refM(v4);
      QubitVector testAB = testA - testB;
      pass_minus &= testAB.size() == refM.size();
      pass_minus &= testAB.qubits() == refM.qubits();
      pass_minus &= approx_equal(testAB.vector(), refM.vector());
      test_condition(pass_minus, "operator-");
    }

  }
 
  //------------------------------------------------------------------------------
  // UTILITY
  //------------------------------------------------------------------------------

  clog << "\nUTILITY TESTS" << endl;
  {
    // initialize
    vector<size_t> dims({1, 2, 4, 8, 16, 32, 64, 128, 256, 512});
    bool pass_init_z = true;
    bool pass_init_x = true;
    for (uint_t j=0; j < 4; j++) {
      QubitVector test(j);
      complex_t val = 1.0 / std::pow(2.0, 0.5 * j);
      cvector_t vec_z(dims[j], 0.), vec_x(dims[j], val);
      vec_z[0] = 1.0;
      test.initialize();
      pass_init_z &= approx_equal(test.vector(), vec_z);
      test.initialize_plus();
      pass_init_x &= approx_equal(test.vector(), vec_x);
    }
    test_condition(pass_init_z, "initialize");
    test_condition(pass_init_x, "initialize_plus");
  
    // renormalize
    bool pass_renorm = true;
    for (uint_t j=0; j < 10; j++) {  
      QubitVector test(rvector_t(1ULL << j, 1.0));
      complex_t val = 1.0 / std::pow(2.0, 0.5 * j);
      cvector_t vec_x(dims[j], val);
      test.renormalize();
      pass_renorm &= approx_equal(test.vector(), vec_x);
    }
    test_condition(pass_renorm, "renormalize");
    
    // Dot
    {
      bool pass_dot = true;
      QubitVector test(zm);
      pass_dot &= approx_equal(test.dot(empty), zero);
      pass_dot &= approx_equal(test.dot(xp), ort2);
      pass_dot &= approx_equal(test.dot(xm), -ort2);
      pass_dot &= approx_equal(test.dot(yp), I * ort2);
      pass_dot &= approx_equal(test.dot(ym), -I * ort2);
      pass_dot &= approx_equal(test.dot(zp), zero);
      pass_dot &= approx_equal(test.dot(zm), one);
      test_condition(pass_dot, "dot");
    }

    // conj
    {
      QubitVector test(yp);
      test.conj();
      test_condition(approx_equal(test.vector(), ym), "conj");
    }

    // norm
    {
      bool pass = true;
      QubitVector test(0);
      pass &= approx_equal(test.norm(), 0.0);
      test = xp;
      pass &= approx_equal(test.norm(), 1.0);
      test = cvector_t({0.5, 0.0, 0.25, 0.0});
      pass &= approx_equal(test.norm(), 0.3125);
      test_condition(pass, "norm");
    }
  }

  
  //------------------------------------------------------------------------------
  // GATES
  //------------------------------------------------------------------------------
  clog << "\nAPPLY MATRIX TESTS" << endl;
  {
    // apply_x
    {
      bool pass = true;
      for (uint_t j=1; j < 4; j++) {
        const uint_t dim = 1ULL << j;
        // j qubit states with last qubit in xp, xm, yp, xm, zp, zm and remaining qubits in zp
        cvector_t xp0(dim, 0.), xm0(dim, 0.), yp0(dim, 0.), ym0(dim, 0.), zp0(dim, 0), zm0(dim, 0);
        xp0[0] = ort2; xp0[1] = ort2;
        xm0[0] = ort2; xm0[1] = -ort2;
        yp0[0] = ort2; yp0[1] = I * ort2;
        ym0[0] = ort2; ym0[1] = -I * ort2;
        zp0[0] = one;
        zm0[1] = one;

        QubitVector test(j);
        test = zp0;
        test.apply_x(0);
        pass &= approx_equal(test.vector(), zm0);
        test = zm0;
        test.apply_x(0);
        pass &= approx_equal(test.vector(), zp0);
        test = xp0;
        test.apply_x(0);
        pass &= approx_equal(test.vector(), xp0);
        test = xm0;
        test.apply_x(0);
        test *= -one; // correct for global phase
        pass &= approx_equal(test.vector(), xm0);
        test = yp0;
        test.apply_x(0);
        test *= -I; // correct for global phase
        pass &= approx_equal(test.vector(), ym0);
        test = ym0;
        test.apply_x(0);
        test *= I; // correct for global phase
        pass &= approx_equal(test.vector(), yp0);
      }
      test_condition(pass, "apply_x");
    }
    // apply_y
    {
      bool pass = true;
      for (uint_t j=1; j < 4; j++) {
        const uint_t dim = 1ULL << j;
        // j qubit states with last qubit in xp, xm, yp, xm, zp, zm and remaining qubits in zp
        cvector_t xp0(dim, 0.), xm0(dim, 0.), yp0(dim, 0.), ym0(dim, 0.), zp0(dim, 0), zm0(dim, 0);
        xp0[0] = ort2; xp0[1] = ort2;
        xm0[0] = ort2; xm0[1] = -ort2;
        yp0[0] = ort2; yp0[1] = I * ort2;
        ym0[0] = ort2; ym0[1] = -I * ort2;
        zp0[0] = one;
        zm0[1] = one;
        QubitVector test(j);
        test = zp0;
        test.apply_y(0);
        test *= -I; // correct for global phase
        pass &= approx_equal(test.vector(), zm0);
        test = zm0;
        test.apply_y(0);
        test *= I; // correct for global phase
        pass &= approx_equal(test.vector(), zp0);
        test = xp0;
        test.apply_y(0);
        test *= I; // correct for global phase
        pass &= approx_equal(test.vector(), xm0);
        test = xm0;
        test.apply_y(0);
        test *= -I; // correct for global phase
        pass &= approx_equal(test.vector(), xp0);
        test = yp0;
        test.apply_y(0);
        pass &= approx_equal(test.vector(), yp0);
        test = ym0;
        test.apply_y(0);
        test *= -one; // correct for global phase
        pass &= approx_equal(test.vector(), ym0);
      }
      test_condition(pass, "apply_y");
    }
    // apply_z
    {
      bool pass = true;
      for (uint_t j=1; j < 4; j++) {
        const uint_t dim = 1ULL << j;
        // j qubit states with last qubit in xp, xm, yp, xm, zp, zm and remaining qubits in zp
        cvector_t xp0(dim, 0.), xm0(dim, 0.), yp0(dim, 0.), ym0(dim, 0.), zp0(dim, 0), zm0(dim, 0);
        xp0[0] = ort2; xp0[1] = ort2;
        xm0[0] = ort2; xm0[1] = -ort2;
        yp0[0] = ort2; yp0[1] = I * ort2;
        ym0[0] = ort2; ym0[1] = -I * ort2;
        zp0[0] = one;
        zm0[1] = one;
        QubitVector test(j);
        test = zp0;
        test.apply_z(0);
        pass &= approx_equal(test.vector(), zp0);
        test = zm0;
        test.apply_z(0);
        test *= -one; // correct for global phase
        pass &= approx_equal(test.vector(), zm0);
        test = xp0;
        test.apply_z(0);
        pass &= approx_equal(test.vector(), xm0);
        test = xm;
        test.apply_z(0);
        pass &= approx_equal(test.vector(), xp);
        test = yp0;
        test.apply_z(0);
        pass &= approx_equal(test.vector(), ym0);
        test = ym0;
        test.apply_z(0);
        pass &= approx_equal(test.vector(), yp0);
      }
      test_condition(pass, "apply_z");
    }
    // CNOT gate
    {
      bool pass = true;
      const uint_t dim = 1ULL << 2;
      // j qubit states with last qubit in xp, xm, yp, xm, zp, zm and remaining qubits in zp
      cvector_t xp0(dim, 0.), xm0(dim, 0.), yp0(dim, 0.), ym0(dim, 0.), zp0(dim, 0), zm0(dim, 0);
      xp0[0] = ort2; xp0[1] = ort2;
      xm0[0] = ort2; xm0[1] = -ort2;
      yp0[0] = ort2; yp0[1] = I * ort2;
      ym0[0] = ort2; ym0[1] = -I * ort2;
      zp0[0] = one;
      zm0[1] = one;
      cvector_t zmzm({zero, zero, zero, one});
      cvector_t bellxp({ort2, zero, zero, ort2});
      cvector_t bellxm({ort2, zero, zero, -ort2});
      cvector_t bellyp({ort2, zero, zero, I * ort2});
      cvector_t bellym({ort2, zero, zero, -I * ort2});

      QubitVector test(2);
      test = zp0;
      test.apply_cnot(0, 1);
      pass &= approx_equal(test.vector(), zp0);
      test = zm0;
      test.apply_cnot(0, 1);
      pass &= approx_equal(test.vector(), zmzm);
      test = xp0;
      test.apply_cnot(0, 1);
      pass &= approx_equal(test.vector(), bellxp);
      test = xm0;
      test.apply_cnot(0, 1);
      pass &= approx_equal(test.vector(), bellxm);
      test = yp0;
      test.apply_cnot(0, 1);
      pass &= approx_equal(test.vector(), bellyp);
      test = ym0;
      test.apply_cnot(0, 1);
      pass &= approx_equal(test.vector(), bellym);

      test = zp0;
      test.apply_cnot(1, 0);
      pass &= approx_equal(test.vector(), zp0);
      test = zm0;
      test.apply_cnot(1, 0);
      pass &= approx_equal(test.vector(), zm0);
      test = xp0;
      test.apply_cnot(1, 0);
      pass &= approx_equal(test.vector(), xp0);
      test = xm0;
      test.apply_cnot(1, 0);
      pass &= approx_equal(test.vector(), xm0);
      test = yp0;
      test.apply_cnot(1, 0);
      pass &= approx_equal(test.vector(), yp0);
      test = ym0;
      test.apply_cnot(1, 0);
      pass &= approx_equal(test.vector(), ym0);
      test_condition(pass, "apply_cnot");
    }
    
    // Single qubit full matrix (X gate)
    {
      bool pass = true;
      for (uint_t j=1; j < 4; j++) {
        const uint_t dim = 1ULL << j;
        // j qubit states with last qubit in xp, xm, yp, xm, zp, zm and remaining qubits in zp
        cvector_t xp0(dim, 0.), xm0(dim, 0.), yp0(dim, 0.), ym0(dim, 0.), zp0(dim, 0), zm0(dim, 0);
        xp0[0] = ort2; xp0[1] = ort2;
        xm0[0] = ort2; xm0[1] = -ort2;
        yp0[0] = ort2; yp0[1] = I * ort2;
        ym0[0] = ort2; ym0[1] = -I * ort2;
        zp0[0] = one;
        zm0[1] = one;

        QubitVector test(j);
        test = zp0;
        test.apply_matrix(0, UX);
        pass &= approx_equal(test.vector(), zm0);
        test = zm0;
        test.apply_matrix(0, UX);
        pass &= approx_equal(test.vector(), zp0);
        test = xp0;
        test.apply_matrix(0, UX);
        pass &= approx_equal(test.vector(), xp0);
        test = xm0;
        test.apply_matrix(0, UX);
        test *= -one; // correct for global phase
        pass &= approx_equal(test.vector(), xm0);
        test = yp0;
        test.apply_matrix(0, UX);
        test *= -I; // correct for global phase
        pass &= approx_equal(test.vector(), ym0);
        test = ym0;
        test.apply_matrix(0, UX);
        test *= I; // correct for global phase
        pass &= approx_equal(test.vector(), yp0);
      }
      test_condition(pass, "apply_matrix (1-qubit full)");
    }

    // Single qubit diagonal matrix (Z-gate)
    {
      bool pass = true;
      for (uint_t j=1; j < 4; j++) {
        const uint_t dim = 1ULL << j;
        // j qubit states with last qubit in xp, xm, yp, xm, zp, zm and remaining qubits in zp
        cvector_t xp0(dim, 0.), xm0(dim, 0.), yp0(dim, 0.), ym0(dim, 0.), zp0(dim, 0), zm0(dim, 0);
        xp0[0] = ort2; xp0[1] = ort2;
        xm0[0] = ort2; xm0[1] = -ort2;
        yp0[0] = ort2; yp0[1] = I * ort2;
        ym0[0] = ort2; ym0[1] = -I * ort2;
        zp0[0] = one;
        zm0[1] = one;

        QubitVector test(j);
        cvector_t Zdiag({one, -one});
        test = zp0;
        test.apply_matrix(0, Zdiag);
        pass &= approx_equal(test.vector(), zp0);
        test = zm0;
        test.apply_matrix(0, Zdiag);
        test *= -one; // correct for global phase
        pass &= approx_equal(test.vector(), zm0);
        test = xp0;
        test.apply_matrix(0, Zdiag);
        pass &= approx_equal(test.vector(), xm0);
        test = xm0;
        test.apply_matrix(0, Zdiag);
        pass &= approx_equal(test.vector(), xp0);
        test = yp0;
        test.apply_matrix(0, Zdiag);
        pass &= approx_equal(test.vector(), ym0);
        test = ym0;
        test.apply_matrix(0, Zdiag);
        pass &= approx_equal(test.vector(), yp0);
      }
      test_condition(pass, "apply_matrix (1-qubit diagonal)");
    }
  
    // apply_matrix (2-qubit full)
    {
      bool pass = true;
      const uint_t dim = 1ULL << 2;
      // j qubit states with last qubit in xp, xm, yp, xm, zp, zm and remaining qubits in zp
      cvector_t xp0(dim, 0.), xm0(dim, 0.), yp0(dim, 0.), ym0(dim, 0.), zp0(dim, 0), zm0(dim, 0);
      xp0[0] = ort2; xp0[1] = ort2;
      xm0[0] = ort2; xm0[1] = -ort2;
      yp0[0] = ort2; yp0[1] = I * ort2;
      ym0[0] = ort2; ym0[1] = -I * ort2;
      zp0[0] = one;
      zm0[1] = one;
      cvector_t zmzm({zero, zero, zero, one});
      cvector_t bellxp({ort2, zero, zero, ort2});
      cvector_t bellxm({ort2, zero, zero, -ort2});
      cvector_t bellyp({ort2, zero, zero, I * ort2});
      cvector_t bellym({ort2, zero, zero, -I * ort2});
      cvector_t UCX({one, zero, zero, zero,
                    zero, zero, zero, one,
                    zero, zero, one, zero,
                    zero, one, zero, zero});
      QubitVector test(2);
      test = zp0;
      test.apply_matrix(0, 1, UCX);
      pass &= approx_equal(test.vector(), zp0);
      test = zm0;
      test.apply_matrix(0, 1, UCX);
      pass &= approx_equal(test.vector(), zmzm);
      test = xp0;
      test.apply_matrix(0, 1, UCX);
      pass &= approx_equal(test.vector(), bellxp);
      test = xm0;
      test.apply_matrix(0, 1, UCX);
      pass &= approx_equal(test.vector(), bellxm);
      test = yp0;
      test.apply_matrix(0, 1, UCX);
      pass &= approx_equal(test.vector(), bellyp);
      test = ym0;
      test.apply_matrix(0, 1, UCX);
      pass &= approx_equal(test.vector(), bellym);

      test = zp0;
      test.apply_matrix(1, 0, UCX);
      pass &= approx_equal(test.vector(), zp0);
      test = zm0;
      test.apply_matrix(1, 0, UCX);
      pass &= approx_equal(test.vector(), zm0);
      test = xp0;
      test.apply_matrix(1, 0, UCX);
      pass &= approx_equal(test.vector(), xp0);
      test = xm0;
      test.apply_matrix(1, 0, UCX);
      pass &= approx_equal(test.vector(), xm0);
      test = yp0;
      test.apply_matrix(1, 0, UCX);
      pass &= approx_equal(test.vector(), yp0);
      test = ym0;
      test.apply_matrix(1, 0, UCX);
      pass &= approx_equal(test.vector(), ym0);
      
      test_condition(pass, "apply_matrix (2-qubit full)");
    }

    {
      bool pass = true;
      QubitVector test(2);
      test.initialize_plus();
      test.apply_cz(0, 1);
      pass &= approx_equal(test.vector(), cvector_t({0.5, 0.5, 0.5, -0.5}));
      test.initialize_plus();
      test.apply_cz(1, 0);
      pass &= approx_equal(test.vector(), cvector_t({0.5, 0.5, 0.5, -0.5}));
      test.initialize();
      pass &= approx_equal(test.vector(), cvector_t({one, zero, zero, zero}));
      test_condition(pass, "apply_cz");
    }

    {
      bool pass = true;
      cvector_t cz_diag = {one, one, one, -one};
      QubitVector test(2);
      
      test.initialize_plus();
      test.apply_matrix(0, 1, cz_diag);
      pass &= approx_equal(test.vector(), cvector_t({0.5, 0.5, 0.5, -0.5}));
      test.initialize_plus();
      test.apply_matrix(1, 0, cz_diag);
      pass &= approx_equal(test.vector(), cvector_t({0.5, 0.5, 0.5, -0.5}));
      test.initialize();
      test.apply_matrix(0, 1, cz_diag);
      pass &= approx_equal(test.vector(), cvector_t({one, zero, zero, zero}));
      test_condition(pass, "apply_matrix(2-qubit diagonal)");
    }

  }

  clog << "\nPROBABILITY TESTS" << endl;
  {
  QubitVector test(1);
    bool pass = true;
    pass &= approx_equal(test.probability(0, 0), 0.);
    pass &= approx_equal(test.probability(0, 1), 0.);
    test.initialize();
    pass &= approx_equal(test.probability(0, 0), 1.);
    pass &= approx_equal(test.probability(0, 1), 0.);
    test = cvector_t({ort2, ort2}); // Xp
    pass &= approx_equal(test.probability(0, 0), 0.5);
    pass &= approx_equal(test.probability(0, 1), 0.5);
    test = cvector_t({ort2, -I * ort2}); // Ym
    pass &= approx_equal(test.probability(0, 0), 0.5);
    pass &= approx_equal(test.probability(0, 1), 0.5);
    test_condition(pass, "probability (1-qubit)");
  }
  {
  QubitVector test(1);
    bool pass = true;
    pass &= approx_equal(test.probabilities(0), rvector_t({0.0, 0.0}));
    test.initialize();
    pass &= approx_equal(test.probabilities(0), rvector_t({1.0, 0.0}));
    test = cvector_t({ort2, ort2}); // Xp
    pass &= approx_equal(test.probabilities(0), rvector_t({0.5, 0.5}));
    test = cvector_t({ort2, -I * ort2}); // Ym
    pass &= approx_equal(test.probabilities(0), rvector_t({0.5, 0.5}));
    test_condition(pass, "probabilities (1-qubit)");
  }
  
  // END
  return 0;
}