#cython: language_level=3

cimport symengine
from symengine cimport RCP, map_basic_basic, rcp_const_basic
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool as cppbool

cdef class Basic(object):
    cdef rcp_const_basic thisptr

cdef class MatrixBase(object):
    cdef symengine.MatrixBase* thisptr

cdef class PyFunctionClass(object):
    cdef RCP[const symengine.PyFunctionClass] thisptr

cdef class PyModule(object):
    cdef RCP[const symengine.PyModule] thisptr

cdef class _DictBasic(object):
    cdef map_basic_basic c

cdef class DictBasicIter(object):
    cdef map_basic_basic.iterator begin
    cdef map_basic_basic.iterator end
    cdef init(self, map_basic_basic.iterator begin, map_basic_basic.iterator end)

cdef object c2py(rcp_const_basic o)

cdef class _Lambdify(object):
    cdef size_t args_size, tot_out_size
    cdef list out_shapes
    cdef readonly bint real
    cdef readonly size_t n_exprs
    cdef public str order
    cdef vector[int] accum_out_sizes
    cdef object numpy_dtype

    cdef _init(self, symengine.vec_basic& args_, symengine.vec_basic& outs_, cppbool cse)
    cdef _load(self, const string &s)
    cpdef eval_real(self, inp, out)
    cpdef eval_complex(self, inp, out)
    cpdef unsafe_eval(sef, inp, out, unsigned nbroadcast=*)

cdef class LambdaDouble(_Lambdify):
    cdef unique_ptr[symengine.LambdaRealDoubleVisitor] lambda_visitor
    cdef _init(self, symengine.vec_basic& args_, symengine.vec_basic& outs_, cppbool cse)
    cpdef unsafe_real(self, double[::1] inp, double[::1] out, int inp_offset=*, int out_offset=*)
    cpdef as_scipy_low_level_callable(self)
    cpdef as_ctypes(self)
    cpdef unsafe_real(self,
                      double[::1] inp, double[::1] out,
                      int inp_offset=*, int out_offset=*)

cdef class LambdaComplexDouble(_Lambdify):
    cdef unique_ptr[symengine.LambdaComplexDoubleVisitor] lambda_visitor
    cdef _init(self, symengine.vec_basic& args_, symengine.vec_basic& outs_, cppbool cse)
    cpdef unsafe_complex(self, double complex[::1] inp, double complex[::1] out, int inp_offset=*, int out_offset=*)

cdef class _LLVMLambdify(_Lambdify):
    cdef int opt_level

cdef class LLVMDouble(_LLVMLambdify):
    cdef unique_ptr[symengine.LLVMDoubleVisitor] lambda_visitor
    cdef _init(self, symengine.vec_basic& args_, symengine.vec_basic& outs_, cppbool cse)
    cdef _load(self, const string &s)
    cpdef unsafe_real(self, double[::1] inp, double[::1] out, int inp_offset=*, int out_offset=*)
    cpdef as_scipy_low_level_callable(self)
    cpdef as_ctypes(self)

cdef class LLVMFloat(_LLVMLambdify):
    cdef unique_ptr[symengine.LLVMFloatVisitor] lambda_visitor
    cdef _init(self, symengine.vec_basic& args_, symengine.vec_basic& outs_, cppbool cse)
    cdef _load(self, const string &s)
    cpdef unsafe_real(self, float[::1] inp, float[::1] out, int inp_offset=*, int out_offset=*)

