/*****************************************************************************
 *
 * ALPS DMFT Project - BLAS Compatibility headers
 *  General Matrix Class with BLAS bindings.
 *
 * Copyright (C) 2005 - 2009 by 
 *                              Emanuel Gull <gull@phys.columbia.edu>,
 *                              Brigitte Surer <surerb@phys.ethz.ch>
 *
 *
* This software is part of the ALPS Applications, published under the ALPS
* Application License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Application License along with
* the ALPS Applications; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

#ifndef GENBLAS_MATRIX
#define GENBLAS_MATRIX

#include "./blasheader.hpp"
#include "./vector.hpp"
#include <cmath>
#include <cassert>
#include <complex>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <sys/types.h>
#include <vector>

#include <alps/hdf5.hpp>

#ifdef USE_MATRIX_DISPATCH //use dispatch to tiny matrix functions for small matrices
#undef __APPLE_CC__

#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

#define MSIZE_P (1)(2)(4) //use tiny matrix dispatcher for these sizes.
#define MSIZE_Q (1)(2)(4)
#define MSIZE_R (1)(2)(4)

#define MLOOP(unused, product) \
BOOST_PP_CAT(\
BOOST_PP_CAT(\
BOOST_PP_CAT(\
BOOST_PP_CAT(\
BOOST_PP_CAT(\
void matmult_, BOOST_PP_SEQ_ELEM(0, product)\
)\
, _\
)\
, BOOST_PP_SEQ_ELEM(1, product)\
)\
, _\
)\
, BOOST_PP_SEQ_ELEM(2, product)\
)(const double * A, const double * B, double * C);
//that produces the function definitions:
BOOST_PP_SEQ_FOR_EACH_PRODUCT(MLOOP, (MSIZE_P)(MSIZE_Q)(MSIZE_R))
//macros that will be used in the switch statement.
#define CASE_MACRO_R(ignored, nr, np_and_nq) \
BOOST_PP_CAT( \
BOOST_PP_CAT( \
BOOST_PP_CAT( \
BOOST_PP_CAT( \
BOOST_PP_CAT( \
case BOOST_PP_SEQ_ELEM(nr, MSIZE_R): matmult_ , BOOST_PP_SEQ_ELEM(BOOST_PP_SEQ_ELEM(0, np_and_nq),MSIZE_P) \
), \
_ \
), \
BOOST_PP_SEQ_ELEM(BOOST_PP_SEQ_ELEM(1, np_and_nq),MSIZE_Q)\
), \
_\
),\
BOOST_PP_SEQ_ELEM(nr, MSIZE_R)(&values_[0], &M2.values_[0], &Mres.values_[0]); break; \
)
#define CASE_MACRO_Q(ignored, nq, np) \
case BOOST_PP_SEQ_ELEM(nq,MSIZE_Q): switch(size2_){ BOOST_PP_REPEAT(BOOST_PP_SEQ_SIZE(MSIZE_R), CASE_MACRO_R, (np)(nq)) default: use_blas=true; } break;
#define CASE_MACRO_P(ignored, np, ignored2) \
case BOOST_PP_SEQ_ELEM(np,MSIZE_P): switch(Mres.size2_){ BOOST_PP_REPEAT(BOOST_PP_SEQ_SIZE(MSIZE_Q), CASE_MACRO_Q, np) default: use_blas=true;} break;
#endif

template<typename T>
inline bool deviates (T entry) {
    return (fabs(entry)>1.e-15);
}
namespace blas{
  //simple BLAS matrix that uses BLAS calls for rank one and matrix vector.
  //contains size 1 and size 2 that do not have to be the same.
  //std::ostream &operator<<(std::ostream &os, const general_matrix &M); //forward declaration
    template<typename T>
    class general_matrix{
    public:
        typedef T value_type;
        general_matrix(int size1, int size2)
        : size1_(size1), size2_(size2), total_memory_size_(size1*size2), values_(size1*size2, T(0))
        {
        }
        general_matrix(int size1=0)
        : size1_(size1), size2_(size1), total_memory_size_(size1*size1), values_(size1*size1, T(0))
        {
        }    
        
        template <class M> 
        general_matrix(const M& mat)
        : size1_(mat.size1()), size2_(mat.size2()), total_memory_size_(size1_*size2_), values_(mat.values())
        {
        }
        
        friend void swap(general_matrix& x,general_matrix& y)
        {
            std::swap(x.values_, y.values_);
            std::swap(x.size1_, y.size1_);
            std::swap(x.size2_, y.size2_);
            std::swap(x.total_memory_size_, y.total_memory_size_);
        }
        
        general_matrix& operator=(const general_matrix &rhs)
        {
            general_matrix temp(rhs);
            swap(temp, *this);
            return *this;
        }
        
        inline const std::vector<T> values() const
        {
            return values_; 
        }
        
        inline std::vector<T> &values() 
        {
            return values_; 
        }
        
        inline T &operator()(const unsigned i, const unsigned j)
        {
            assert((i < size1_) && (j < size2_));
            return values_[i*size2_+j];
        }
        
        inline const T &operator()(const unsigned i, const unsigned j) const 
        {
            assert((i < size1_) && (j < size2_));
            return values_[i*size2_+j];
        }
        
        inline void set(const unsigned i, const unsigned j, const T value)
        {
            assert((i < size1_) && (j < size2_));
            values_[i*size2_+j] = value;
        }
        
        inline const fortran_int_t &size1() const 
        {
            return size1_;
        }
  
        inline const fortran_int_t &size2() const
        { 
            return size2_;
        }
        
        inline const fortran_int_t &size() const
        { 
            assert(size1_ == size2_);
            return size1_;
        }
        
        inline void resize(fortran_int_t size1, fortran_int_t size2)
        {
            values_.resize(size1*size2); //set to zero?
            size1_=size1;
            size2_=size2;
            total_memory_size_=size1*size2;
        }
        
        inline void matrix_right_multiply(const general_matrix<T> &M2,  general_matrix<T> &Mres, T beta=0) const
        { 
            std::cout<<"please use the optimized version!"<<std::endl;
            abort();
        }
        
        void clear()
        {
            values_.clear();
            values_.resize(size1_*size2_,T(0));
        }
        
        bool operator==(const general_matrix &other) const 
        {
            if((size1_!=other.size1_) || (size2_!=other.size2_) ) return false;
            std::vector<T> diff(total_memory_size_);
            transform(values_.begin(), values_.end(), other.values_.begin(), diff.begin(), std::minus<double>());
            typename std::vector<T>::iterator it= find_if(diff.begin(),diff.end(),deviates<T>);
            return (it == diff.end());
        }
        bool operator!=(const general_matrix &other)const 
        {
            return !(*this == other);
        }
        
        //multiply matrix by value.
        general_matrix<T> &operator *=(double lambda)
        {
            fortran_int_t inc=1;
            fortran_int_t total_size=size1_*size2_;
            dscal_(&total_size, &lambda, &values_[0], &inc);
            return *this;
        }
        
        general_matrix<T> & operator+=(const general_matrix &rhs) 
        {
            assert((rhs.size1() == size1_) && (rhs.size2() == size2_));
            std::vector<T> temp = rhs.values();
            transform(values_.begin(), values_.end(), temp.begin(), values_.begin(), std::plus<T>());
            return *this;
        }
        
        general_matrix<T> & operator-=(const general_matrix &rhs) 
        {
            assert((rhs.size1() == size1_) && (rhs.size2() == size2_));
            std::vector<T> temp = rhs.values();
            transform(values_.begin(), values_.end(), temp.begin(), values_.begin(), std::minus<T>());
            return *this;
        }
        
        const general_matrix<T> operator+(const general_matrix &other) const 
        {
            
            general_matrix result(*this);     
            result += other;           
            return result;              
        }
        
        const general_matrix<T> operator-(const general_matrix &other) const 
        {
            
            general_matrix result(*this);     
            result -= other;           
            return result;              
        }
        
        double max() const
        {
            return *max_element(values_.begin(), values_.end());        
        }
        
        double min() const
        {
            return *min_element(values_.begin(), values_.end());        
        }
        
        //compute M=D*M, where D is a diagonal matrix represented by
        //the
        //vector. Mij*=Dii
        void left_multiply_diagonal_matrix(const blas::vector &diagonal_matrix)
        {
            assert(size1_==diagonal_matrix.size()); 
            fortran_int_t inc=1;
            for(int i=0;i<size1_;++i){ //for each col: multiply col. with constant
                dscal_(&size2_, &diagonal_matrix(i), &values_[i*size2_], &inc);
            }     
        }  
        
        //same, but M=M*D. Slow because of much larger stride.. Mij *=Djj
        void right_multiply_diagonal_matrix(const blas::vector &diagonal_matrix)
        {
            assert(size2_==diagonal_matrix.size()); //must have as many rows as diag has size.
            for(int i=0;i<size1_;++i){ 
                dscal_(&size2_, &diagonal_matrix(i), &values_[i], &size2_);
            }     
        }
        
        inline void right_multiply(const vector &v1, vector &v2) const
        { //perform v2[i]=M[ij]v1[j]
            assert(size1_==size2_);
        //call the BLAS routine for matrix vector multiplication:
            char trans='T';
            double alpha=1., beta=0.;       //no need to multiply a constant or add a vector
            fortran_int_t inc=1;
            dgemv_(&trans, &size1_, &size1_, &alpha, &values_[0], &size1_, &(v1(0)), &inc, &beta, &(v2(0)), &inc);
        }

        inline void multiply_row(const int i, const T&val)
        { 
            fortran_int_t inc=1;
            dscal_(&size2_, &val, &values_[i*size2_], &inc);
        }
        
        inline void multiply_column(const int j, const T&val)
        { 
            dscal_(&size1_, &val, &values_[j], &size2_);
        }
        
        inline double trace()const
        {
            assert(size1_==size2_);
            double tr=0.;
            for(int i=0;i<size1_;++i) tr+=operator()(i,i);
            return tr;
        }
    
        void set_to_identity()
        {
            assert(size1_==size2_);
            clear();
            for(int i=0;i<size1_;++i){
                operator()(i,i)=T(1);
            }
        }
    
        void transpose() 
        {
            if(size1_==0 || size2_==0) return;
            std::vector<T> values(total_memory_size_);
            for(int i=0;i<size1_;++i){
                for(int j=0;j<size2_;++j){
                    values[j*size1_+i]=values_[i*size2_+j];
                }
            }
            std::swap(size1_, size2_);
            values_=values;
        }
        
        general_matrix operator*(const general_matrix M2)const
        {
            general_matrix M(size1_, M2.size2_);
            matrix_right_multiply(M2, M);
            return M;
        }
        
        general_matrix &operator*=(const general_matrix M2)
        {
            general_matrix M(size1_, M2.size2_);
            matrix_right_multiply(M2, M);
            swap(M,*this);
            return *this;
        }
    
        general_matrix operator*(const blas::vector &v)const
        {
            general_matrix M(*this);
            right_multiply_diagonal_matrix(v);
            return M;
        }
        
        general_matrix &operator*=(const blas::vector &v){
            right_multiply_diagonal_matrix(v);
            return *this;
        }
        
        general_matrix &invert(){
            throw(std::logic_error(std::string("you linked the general case for invert. Please use the specializations.")));
        }

        void save(alps::hdf5::archive &ar) const
        {
            using namespace alps;
            ar << make_pvp("", &values_.front(), std::vector<std::size_t>(2, size1_,size2_));
        }

        void load(alps::hdf5::archive &ar)
        {
            using namespace alps;
            resize(ar.extent("")[0]);
            ar >> make_pvp("", &values_.front(), std::vector<std::size_t>(2, size1_,size2_));
        }

    private:
        fortran_int_t size1_; //current size of matrix
        fortran_int_t size2_; //current size of matrix
        fortran_int_t total_memory_size_; //total memory allocated for this matrix
        std::vector<T> values_;
  };
  
  //this is crap!! reimplement!
#ifndef USE_MATRIX_DISPATCH
    template<> inline void general_matrix<double>::matrix_right_multiply(const general_matrix<double> &M2,  general_matrix<double> &Mres, double beta) const
    {
        //call the BLAS routine for matrix matrix multiplication:
        //std::cerr<<"sizes are: "<<size1_<<" "<<Mres.size1()<<" "<<Mres.size2()<<std::endl;
        assert(Mres.size1()==size1_);
        assert(Mres.size2()==M2.size2());
        assert(size2_==M2.size1());
        if(size1_ < 4 || size2_ < 4 || M2.size2_ < 4){ //do it by hand! a test showed this to be the optimal size on my intel.
            int s2=Mres.size2();
            double *mres_ij=&Mres(0,0);
            for(int i=0;i<size1_;++i){
                for(int j=0;j<s2;++j){
                    (*mres_ij)*=beta;
                    const double *thism_ik=&operator()(i,0);
                    const double *m2_kj=&M2(0,j);
                    for(int k=0;k<size2_;++k){
                        (*mres_ij)+=(*thism_ik)*(*m2_kj);
                        m2_kj+=s2;
                        thism_ik++;
                    }
                    mres_ij++;
                }
            }
        }
        else{
            double one_double=1.;
            char notrans='N';
            //std::cout<<"invoking dgemm_"<<std::endl;
            dgemm_(&notrans, &notrans, &M2.size2_, &size1_, &size2_, &one_double, &M2.values_[0], &M2.size2_, &values_[0],
                   &size2_, &beta, &Mres.values_[0], &M2.size2_);
        }
    }
#else
    template<> inline void general_matrix<double>::matrix_right_multiply(const general_matrix<double> &M2,  general_matrix<double> &Mres, double beta) const
    {
        //call the BLAS routine for matrix matrix multiplication:
        assert(Mres.size1()==size1_);
        assert(Mres.size2()==M2.size2());
        assert(size2_==M2.size1());
        double one_double=1.;
        char notrans='N';
        bool use_blas=false;
        switch(Mres.size1()){ //boost pp dispatcher for matrix multiplication - see also dispatcher.
                BOOST_PP_REPEAT(BOOST_PP_SEQ_SIZE(MSIZE_P), CASE_MACRO_P, )
            default: use_blas=true; break;
        }
        if(use_blas){
            dgemm_(&notrans, &notrans, &M2.size2_, &size1_, &size2_, &one_double, &M2.values_[0], &M2.size2_, &values_[0],
                   &size2_, &beta, &Mres.values_[0], &M2.size2_);
        }
    }
#endif //matrix dispatch
    template<> inline void general_matrix<std::complex<double> >::matrix_right_multiply(const general_matrix<std::complex<double> > &M2,  general_matrix<std::complex<double> >  &Mres, std::complex<double> beta) const
    {
        //call the BLAS routine for matrix matrix multiplication:
        assert(Mres.size1()==size1_);
        assert(Mres.size2()==M2.size2());
        assert(size2_==M2.size1());
        std::complex<double> one_double=1.;
        char notrans='N';
        if(size1_ < 4 && size2_ < 4 && M2.size2()< 4){ //do it by hand! a test showed this to be the optimal size on my intel.
            general_matrix M2t(M2);
            M2t.transpose();
            int s2=Mres.size2();
            for(int i=0;i<size1_*s2;++i){ Mres.values_[i]*=beta; }
            for(int i=0;i<size1_;++i){
                for(int k=0;k<size2_;++k){
                    std::complex<double> visk=values_[i*size2_+k];
                    std::complex<double> *vjsk=&M2t.values_[k];
                    std::complex<double> *visj=&Mres.values_[i*s2];
                    for(int j=0;j<s2;++j){
                        *visj+=visk*(*vjsk);
                        visj++;
                        vjsk+=size2_;
                    }
                }
            }
        }
        else{
            zgemm_(&notrans, &notrans, &M2.size2_, &size1_, &size2_, &one_double, &M2.values_[0], &M2.size2_, &values_[0],
                   &size2_, &beta, &Mres.values_[0], &M2.size2_);
        }
    }
  
    template<> inline double general_matrix<std::complex<double> >::trace() const
    {
        assert(size1_==size2_);
        std::complex<double> trace=0;
        for(int i=0;i<size1_;++i){
            trace+=operator()(i,i);
        }
        if(trace.imag() > 1.e-10) std::cerr<<"careful, trace is complex: "<<trace<<std::endl;
        return trace.real();
    }
  
    template<> inline double general_matrix<std::complex<double> >::max()const
    {
        int max_index;
        fortran_int_t total_size=size1_*size2_;
        fortran_int_t inc=1;
        if(total_size==0) return 0;
        if(total_size==1) return std::abs(values_[0]);
        else
            max_index=izamax_(&total_size, &values_[0],&inc);
        return std::abs(values_[max_index-1]);
    }
    
    template<typename T> inline std::ostream &operator<<(std::ostream &os, const general_matrix<T> &M)
    {
        os<<"[ ";
        for(int i=0;i<M.size1();++i){
            os<<"[ ";
            for(int j=0;j<M.size2();++j){
                os<<M(i,j)<<" ";
            }
            os<<" ]"<<std::endl;
        }
        os<<"]"<<std::endl;
        return os;
    }
    
    template<> inline std::ostream &operator<<(std::ostream &os, const general_matrix<std::complex<double> > &M)
    {
        for(int i=0;i<M.size1();++i){
            for(int j=0;j<M.size2();++j){
                os<<M(i,j).real()<<"+i*"<<M(i,j).imag()<<" ";
            }
            os<<std::endl;
        }
        os<<std::endl;
        return os;
    }
    
    template<> inline std::ostream &operator<<(std::ostream &os, const general_matrix<double > &M)
    {
        for(int i=0;i<M.size1();++i){
            for(int j=0;j<M.size2();++j){
                os<<M(i,j)<<" ";
            }
            os<<std::endl;
        }
        os<<std::endl;
        return os;
    }
    
    template<> inline general_matrix<std::complex<double> >& general_matrix<std::complex<double> >::invert()
    {
        general_matrix<std::complex<double> > B(size1_, size1_);
        fortran_int_t *ipiv=new fortran_int_t[size1_];
        fortran_int_t info;
        B.set_to_identity();
        FORTRAN_ID(zgesv)(&size1_, &size1_, &values_[0], &size1_, ipiv, &(B(0,0)), &size1_, &info);
        delete[] ipiv;
        if(info){ throw(std::logic_error("in dgesv: info was not zero.")); }
        swap(B, *this);
        return *this;
    }
    
    template<> inline general_matrix<double >& general_matrix<double >::invert()
    {
        general_matrix<double> B(size1_, size1_);
        fortran_int_t *ipiv=new fortran_int_t[size1_];
        fortran_int_t info;
        B.set_to_identity();
        FORTRAN_ID(dgesv)(&size1_, &size1_, &values_[0], &size1_, ipiv, &(B(0,0)), &size1_, &info);
        delete[] ipiv;
        if(info){ throw(std::logic_error("in dgesv: info was not zero.")); }
        swap(B, *this);
        //std::cout<<"B: "<<B<<" this: "<<*this<<std::endl;
        return *this;
    }

    template<typename T> inline general_matrix<T> operator*(const T &lambda, const general_matrix<T> &M)
    {
        general_matrix<T> M2(M);
        M2*=lambda;
        return M2;
    }
  
    template<typename T> inline general_matrix<T> operator*(const general_matrix<T> &M, const T &lambda)
    {
        general_matrix<T> M2(M);
        M2*=lambda;
        return M2;
    }
  
    template<typename T> inline general_matrix<T>operator*(const blas::vector &v, const general_matrix<T> &M)
    {
        general_matrix<T> M2(M);
        M2.left_multiply_diagonal_matrix(v);
        return M2;
    }
  
    typedef general_matrix<double> double_matrix;
    typedef general_matrix<std::complex< double> > complex_double_matrix;
} //namespace

#endif
