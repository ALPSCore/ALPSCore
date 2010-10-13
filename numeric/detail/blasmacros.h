#include <boost/numeric/bindings/blas.hpp>
#include<vector>

// provide overloads for types where blas can be used        

namespace blas{

#define IMPLEMENT_FOR_REAL_BLAS_TYPES(F) F(float) F(double)
#define IMPLEMENT_FOR_COMPLEX_BLAS_TYPES(F) \
F(std::complex<float>) \
F(std::complex<double>)
#define IMPLEMENT_FOR_ALL_BLAS_TYPES(F) \
IMPLEMENT_FOR_REAL_BLAS_TYPES(F) \
IMPLEMENT_FOR_COMPLEX_BLAS_TYPES(F) 
#define PLUS_ASSIGN(T) \
void plus_assign(std::vector<T>::iterator first1, std::vector<T>::iterator last1, std::vector<T>::const_iterator first2) \
{ boost::numeric::bindings::blas::detail::axpy(last1-first1, 1., &*first2, 1, &*first1, 1);}
    IMPLEMENT_FOR_ALL_BLAS_TYPES(PLUS_ASSIGN)
#undef MINUS_ASSIGN
    

#define IMPLEMENT_FOR_REAL_BLAS_TYPES(F) F(float) F(double)
#define IMPLEMENT_FOR_COMPLEX_BLAS_TYPES(F) \
F(std::complex<float>) \
F(std::complex<double>)
#define IMPLEMENT_FOR_ALL_BLAS_TYPES(F) \
IMPLEMENT_FOR_REAL_BLAS_TYPES(F) \
IMPLEMENT_FOR_COMPLEX_BLAS_TYPES(F) 
#define MINUS_ASSIGN(T) \
void minus_assign(std::vector<T>::iterator first1, std::vector<T>::iterator last1, std::vector<T>::const_iterator first2) \
{ boost::numeric::bindings::blas::detail::axpy(last1-first1, -1., &*first2, 1, &*first1, 1);}
    IMPLEMENT_FOR_ALL_BLAS_TYPES(MINUS_ASSIGN)
#undef MINUS_ASSIGN
    
#define IMPLEMENT_FOR_REAL_BLAS_TYPES(F) F(float) F(double)
#define IMPLEMENT_FOR_COMPLEX_BLAS_TYPES(F) \
F(std::complex<float>) \
F(std::complex<double>)
#define IMPLEMENT_FOR_ALL_BLAS_TYPES(F) \
IMPLEMENT_FOR_REAL_BLAS_TYPES(F) \
IMPLEMENT_FOR_COMPLEX_BLAS_TYPES(F) 
#define MULTIPLIES_ASSIGN(T) \
void multiplies_assign(std::vector<T>::iterator start1, std::vector<T>::iterator end1, T lambda)                            \
    { boost::numeric::bindings::blas::detail::scal(end1-start1, lambda, &*start1, 1);}
    IMPLEMENT_FOR_ALL_BLAS_TYPES(MULTIPLIES_ASSIGN)
#undef MULTIPLIES_ASSIGN
    
#define IMPLEMENT_FOR_REAL_BLAS_TYPES(F) F(float) F(double)
#define IMPLEMENT_FOR_COMPLEX_BLAS_TYPES(F) \
F(std::complex<float>) \
F(std::complex<double>)
#define IMPLEMENT_FOR_ALL_BLAS_TYPES(F) \
IMPLEMENT_FOR_REAL_BLAS_TYPES(F) \
IMPLEMENT_FOR_COMPLEX_BLAS_TYPES(F) 
#define SCALAR_PRODUCT(T) \
inline T scalar_product(const std::vector<T> v1, const std::vector<T> v2)                                              \
    { return boost::numeric::bindings::blas::detail::dot(v1.size(), &v1[0],1,&v2[0],1);}
    IMPLEMENT_FOR_ALL_BLAS_TYPES(SCALAR_PRODUCT)
#undef SCALAR_PRODUCT
    
    
} // namespace