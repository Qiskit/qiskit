#ifndef SYMENGINE_PYWRAPPER_H
#define SYMENGINE_PYWRAPPER_H

#include <Python.h>
#include <symengine/number.h>
#include <symengine/constants.h>
#include <symengine/functions.h>

namespace SymEngine {

std::string pickle_dumps(const PyObject *);
PyObject* pickle_loads(const std::string &);

/*
 * PySymbol is a subclass of Symbol that keeps a reference to a Python object.
 * When subclassing a Symbol from Python, the information stored in subclassed
 * object is lost because all the arithmetic and function evaluations happen on
 * the C++ side. The object returned by `(x + 1) - 1` is wrapped in the Python
 * class Symbol and therefore the fact that `x` is a subclass of Symbol is lost.
 *
 * By subclassing in the C++ side and keeping a python object reference, the
 * subclassed python object can be returned instead of wrapping in a Python
 * class Symbol.
 *
 * TODO: Python object and C++ object both keep a reference to each other as one
 * must be alive when the other is alive. This creates a cyclic reference and
 * should be fixed.
*/

class PySymbol : public Symbol {
private:
    PyObject* obj;
    std::string bytes;
public:
    const bool store_pickle;
    PySymbol(const std::string& name, PyObject* obj, bool store_pickle) :
            Symbol(name), obj(obj), store_pickle(store_pickle) {
        if (store_pickle) {
            bytes = pickle_dumps(obj);
        } else {
            Py_INCREF(obj);
        }
    }
    PyObject* get_py_object() const {
        if (store_pickle) {
            return pickle_loads(bytes);
        } else {
            Py_INCREF(obj);
            return obj;
        }
    }
    virtual ~PySymbol() {
        if (not store_pickle) {
            // TODO: This is never called because of the cyclic reference.
            Py_DECREF(obj);
        }
    }
};

/*
 * This module provides classes to wrap Python objects defined in SymPy
 * or Sage into SymEngine.
 *
 * PyModule is a python module (SymPy or Sage) that provides Python callbacks.
 * These callback functions for conversion, evaluation and differentiation are
 * defined in the Cython module `symengine_wrapper.pyx` and passed to C++.
 *
 * PyNumber is for numeric types where addition, subtraction, multiplication
 * and division with other numeric types produce a numeric type.
 *
 * PyFunction is an instance of a function defined in SymPy or Sage and contains
 * a PyFunctionClass instance which holds the callback functions needed for
 * interaction with other SymEngine functions
 *
 * C++ Evaluation methods like eval_double, eval_mpfr calls the eval_ method to
 * convert PyNumber and PyFunction to known SymEngine types.
 */

//! Class to store the Python objects and Cython callback functions specific
//  to a Python module. eg: SymPy or Sage
class PyModule : public EnableRCPFromThis<PyModule> {
public:
    // Callback function to convert a SymEngine object to Python
    PyObject* (*to_py_)(const RCP<const Basic>);
    // Callback function to convert a Python object to SymEngine
    RCP<const Basic> (*from_py_)(PyObject*);
    // Callback function to evaluate a Python object to a bits number of
    // precision and return a SymEngine Number
    RCP<const Number> (*eval_)(PyObject*, long bits);
    // Callback function to differentiate a Python object with respect to
    // a SymEngine symbol and get a SymEngine object
    RCP<const Basic> (*diff_)(PyObject*, RCP<const Basic>);
    // Common constants in Python
    PyObject *one, *zero, *minus_one;
public:
    PyModule(PyObject* (*)(const RCP<const Basic> x), RCP<const Basic> (*)(PyObject*),
             RCP<const Number> (*)(PyObject*, long), RCP<const Basic> (*)(PyObject*, RCP<const Basic>));
    ~PyModule();
    PyObject* get_zero() const { return zero; }
    PyObject* get_one() const { return one; }
    PyObject* get_minus_one() const { return minus_one; }
};

//! Python numeric types that do not have direct counterparts in SymEngine are
//  wrapped using this method. Eg: Sage's real_mpfi.
//  Arithmetic operations are done by calling Python/C API's PyNumber_* methods
//  after converting SymEngine::Number to Python module type. Arithmetic
//  operations always returns a PyNumber type.
class PyNumber : public NumberWrapper {
private:
    //! Python reference to the object being wrapped
    PyObject* pyobject_;
    //! Python module that this object belongs to
    RCP<const PyModule> pymodule_;
public:
    PyNumber(PyObject* pyobject, const RCP<const PyModule> &pymodule);
    ~PyNumber() {
        Py_DECREF(pyobject_);
    }
    PyObject* get_py_object() const { return pyobject_; }
    RCP<const PyModule> get_py_module() const { return pymodule_; }
    //! \return true if `0`
    virtual bool is_zero() const;
    //! \return true if `1`
    virtual bool is_one() const;
    //! \return true if `-1`
    virtual bool is_minus_one() const;
    //! \return true if negative
    virtual bool is_negative() const;
    //! \return true if positive
    virtual bool is_positive() const;
    //! \return true if complex
    virtual bool is_complex() const;
    //! return true if the number is an exact representation
    //  false if the number is an approximation
    virtual bool is_exact() const { return true; };

    //! Addition
    virtual RCP<const Number> add(const Number &other) const;
    //! Subtraction
    virtual RCP<const Number> sub(const Number &other) const;
    virtual RCP<const Number> rsub(const Number &other) const;
    //! Multiplication
    virtual RCP<const Number> mul(const Number &other) const;
    //! Division
    virtual RCP<const Number> div(const Number &other) const;
    virtual RCP<const Number> rdiv(const Number &other) const;
    //! Power
    virtual RCP<const Number> pow(const Number &other) const;
    virtual RCP<const Number> rpow(const Number &other) const;

    virtual RCP<const Number> eval(long bits) const;
    virtual std::string __str__() const;
    virtual int compare(const Basic &o) const;
    virtual bool __eq__(const Basic &o) const;
    virtual hash_t __hash__() const;
};

/*! Class to represent the parent class for a PyFunction. Stores
 *  a python reference `pyobject_` to a python callable object.
 *  A PyFunction instance is an instance of the parent PyFunctionClass's
 *  `pyobject_`.
 * */
class PyFunctionClass : public EnableRCPFromThis<PyFunctionClass> {
private:
    //! Callable python object to construct an instance of this class
    PyObject *pyobject_;
    //! Name of the function
    std::string name_;
    //! Hash of the python function
    mutable hash_t hash_;
    //! PyModule that this python function belongs to
    RCP<const PyModule> pymodule_;
public:
    PyFunctionClass(PyObject *pyobject, std::string name, const RCP<const PyModule> &pymodule);
    PyObject* get_py_object() const { return pyobject_; }
    RCP<const PyModule> get_py_module() const { return pymodule_; }
    std::string get_name() const { return name_; }
    //! Create an instance of this class with arguments `vec`.
    PyObject* call(const vec_basic &vec) const;
    bool __eq__(const PyFunctionClass &x) const;
    int compare(const PyFunctionClass &x) const;
    hash_t hash() const;
};

/*! Class to represent the parent class for a PyFunction. Stores
 *  a python reference `pyobject_` to a python callable object.
 *  A PyFunction instance is an instance of the parent PyFunctionClass's
 *  `pyobject_`.
 * */
class PyFunction : public FunctionWrapper {
private:
    RCP<const PyFunctionClass> pyfunction_class_;
    PyObject *pyobject_;
public:
    PyFunction(const vec_basic &vec, const RCP<const PyFunctionClass> &pyfunc_class,
               PyObject *pyobject);
    ~PyFunction();

    PyObject *get_py_object() const;
    RCP<const PyFunctionClass> get_pyfunction_class() const;
    //! Create an instance of similar type with arguments `x`.
    virtual RCP<const Basic> create(const vec_basic &x) const;
    //! Eval the number to bits precision and return a SymEngine::Number type
    virtual RCP<const Number> eval(long bits) const;
    /*! Evaluate the derivative w.r.t. `x` by calling the callback function
     *  of the module that this function belongs to.
     * */
    virtual RCP<const Basic> diff_impl(const RCP<const Symbol> &x) const;
    virtual int compare(const Basic &o) const;
    virtual bool __eq__(const Basic &o) const;
    virtual hash_t __hash__() const;
};

std::string wrapper_dumps(const Basic &x);
RCP<const Basic> wrapper_loads(const std::string &s);

}

#endif //SYMENGINE_PYWRAPPER_H
