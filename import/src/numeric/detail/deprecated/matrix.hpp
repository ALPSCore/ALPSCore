/*****************************************************************************
 *
 * ALPS DMFT Project - BLAS Compatibility headers
 *  Square Matrix Class
 *
 * Copyright (C) 2005 - 2010 by 
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

#ifndef BLAS_MATRIX
#define BLAS_MATRIX

#include "./blasheader.hpp"
#include "./vector.hpp"
#include "./general_matrix.hpp"
#include <cmath>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <sys/types.h> //on osx for uint
#include <vector>
#include <numeric>

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/pointer.hpp>

#ifdef UBLAS
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
typedef boost::numeric::ublas::matrix<double,boost::numeric::ublas::column_major> dense_matrix;
typedef boost::numeric::ublas::matrix<std::complex<double>,boost::numeric::ublas::column_major> complex_dense_matrix;
#endif

namespace blas{
  //simple BLAS matrix that uses BLAS calls for rank one and matrix vector.
  //std::ostream &operator<<(std::ostream &os, const matrix &M); //forward declaration
   
    class matrix{
        public:
        matrix(int size=0, double initial_value=0.)
        : values_(size*size, initial_value), size_(size)
        {
        }
        
        void swap(matrix& rhs)
        {
            std::swap(values_, rhs.values_);
            std::swap(size_, rhs.size_);
        }
        friend void swap(matrix& x,matrix& y)
        {
            std::swap(x.values_, y.values_);
            std::swap(x.size_, y.size_);
        }
        
        // we keep this to ensure the strong exception guarantee, the default
        // one would only satisfy the weak one
        matrix& operator=(matrix rhs)
        {
            swap(rhs);
            return *this;
        }
        
        inline double &operator()(const unsigned int i, const unsigned int j)
        {
            assert((i < size_) && (j < size_));
            return values_[i*size_+j];
        }
        
        inline const double &operator()(const unsigned int i, const unsigned int j) const 
        {
            assert((i < size_) && (j < size_));
            return values_[i*size_+j];
        }
        
        inline const int size() const
        {
            return size_;
        }
        
        inline const int size1() const
        {
            return size_;
        }
        
        inline const int size2() const
        {
            return size_;
        }
        inline const std::vector<double> values() const
        {
            return values_; 
        }
        void resize(int new_size)
        {
            if(new_size==size_) return;
            else if(new_size>size_){
                values_.resize(new_size*new_size);
                for(int i=size_-1;i>0;--i){
                    std::copy(values_.rbegin()+new_size*new_size-(i+1)*size_, values_.rbegin()+new_size*new_size-i*size_, values_.rbegin()+new_size*(new_size-i)-size_);
                }
            }
            else{
                for(int i=1;i<size_;++i){
                    std::copy(values_.begin()+i*size_, values_.begin()+i*size_+new_size, values_.begin()+i*new_size);
                }
                values_.resize(new_size*new_size);
            }
            size_ = new_size;
        }
        void resize(int size1, int size2)
        {
             assert((size1 == size2));
             resize(size1);
        }
        void resize_nocopy(int new_size)
        {
            values_.clear();
            values_.resize(new_size*new_size,0.);
            size_ = new_size;
            
        }
        inline void add_outer_product(const blas::vector &v1, const blas::vector &v2, double alpha=1.)
        {
            fortran_int_t inc=1;
            if(size_>1){
                FORTRAN_ID(dger)(&size_, &size_, &alpha,&v2.values()[0], &inc, &v1.values()[0], &inc, &values_[0], &size_); 
            }else if(size_==1){
                values_[0]+=alpha*v1(0)*v2(0);  
            }else
                return;
        }
       
        inline double sum()
        { 
            return std::accumulate(values_.begin(), values_.end(), 0.);
        }
        
        inline void insert_row_column_last(blas::vector &row, blas::vector &col, double Mkk)
        {
            fortran_int_t oldsize=size_;
            resize(size_+1);
            fortran_int_t one=1;
            FORTRAN_ID(dcopy)(&oldsize, &(col.values()[0]), &one, &(values_[oldsize     ]), &size_); //copy in row (careful: col. major)
            FORTRAN_ID(dcopy)(&oldsize, &(row.values()[0]), &one, &(values_[oldsize*size_]), &one         );   //copy in column
            operator()(oldsize, oldsize)=Mkk;
        }
        
        inline void getrow(int k, double *row) const
        {
            assert((k < size_));
            fortran_int_t one=1;
            FORTRAN_ID(dcopy)(&size_, &(values_[k*size_]), &one, row, &one);
        }
        
        blas::vector getrow(int k) const
        {
            assert((k < size_));
            blas::vector row(size_);
            fortran_int_t one=1;
            FORTRAN_ID(dcopy)(&size_, &(values_[k*size_]), &one, &row.values()[0], &one);
            return row;
        }
        
        inline void getcol(int k, double *col) const
        {
            assert((k < size_));
            fortran_int_t one=1;
            FORTRAN_ID(dcopy)(&size_, &(values_[k]), &size_, col, &one);
        }
        
        blas::vector getcol(int k) const
        {
            assert((k < size_));
            blas::vector col(size_);
            fortran_int_t one=1;
            FORTRAN_ID(dcopy)(&size_, &(values_[k]), &size_,  &col.values()[0], &one);
            return col;
        }
        
        
        inline void setrow(int k, const double *row)
        {
            assert((k < size_));
            fortran_int_t one=1;
            FORTRAN_ID(dcopy)(&size_, row, &one, &(values_[k*size_]), &one);
        }
        
        inline void setrow(int k, const blas::vector row){
            assert((k < size_));
            fortran_int_t one=1;
            FORTRAN_ID(dcopy)(&size_, &row.values()[0], &one, &(values_[k*size_]), &one);
        }
    
        inline void setcol(int k, const double *col)
        {
            assert((k < size_));
            fortran_int_t one=1;
            FORTRAN_ID(dcopy)(&size_, col, &one, &(values_[k]), &size_);
        }
        
        inline void setcol(int k, const blas::vector col){
            assert((k < size_));
            fortran_int_t one=1;
            FORTRAN_ID(dcopy)(&size_, &col.values()[0], &one, &(values_[k]), &size_);
        }
        
        inline void insert_row_column(blas::vector &row, blas::vector &col, double Mkk, int k)
        {
            assert((k <= size_));
            resize(size_+1);
            std::vector<double> new_values(size_*size_);
            
            for(int j=size_-2;j>=0;--j){
                int j_new=j<k?j:j+1;
                for(int i=size_-2;i>=0;--i){
                    int i_new=i<k?i:i+1;
                    new_values[i_new*size_+j_new]=operator()(i, j);
                }
                new_values[k*size_+j_new]=row(j);
                new_values[j_new*size_+k]=col(j);
            }
            new_values[k*size_+k]=Mkk;
            values_=new_values;
        }
        
        inline void remove_row_column(int c)
        {
            assert((c < size_));
            int new_size=size_-1;
            blas::matrix new_matrix(new_size);
            for(int i=0, k=0;i<new_size;++i,++k){
                if(k==c) ++k;
                for(int j=0,l=0;j<new_size;++j,++l){
                    if(c==l) ++l;
                    new_matrix(i,j)=operator()(k, l);
                }
            }
            swap(new_matrix);
        }
        
        inline void remove_row_column_last(){
            size_--;
        }
        
        inline void swap_columns(int c1, int c2)
        {
            assert((c1 < size_) && (c2 < size_));
            if(c1==c2) return;
            FORTRAN_ID(dswap)(&size_, &(values_[c1]), &size_, &(values_[c2]), &size_);
        }
        
        inline void swap_rows(int c1, int c2)
        {
            assert((c1 < size_) && (c2 < size_));
            if(c1==c2) return;
            fortran_int_t one=1;
            FORTRAN_ID(dswap)(&size_, &(values_[c1*size_]), &one, &(values_[c2*size_]), &one);
        }
        
        inline void swap_row_column(int c1, int c2)
        {
            swap_columns(c1,c2);
            swap_rows(c1,c2);
        }
         
        template<class mat> 
        matrix convert_from(const mat &M)
        { //convert dense or sparse matrices (or any other with  (i,j) and size1() to blas matrix.
            assert(M.size1()==M.size2());
            resize_nocopy(M.size1());
            for(int i=0;i<(int)M.size1();++i){
                for(int j=0;j<(int)M.size1();++j){
                    operator()(i,j)=M(i,j);
                }
            }
            return *this;
        }
      
        
        inline void right_multiply(const vector &v1, vector &v2) const
        { 
            if(size_==0) return;
            //perform v2[i]=M[ij]v1[j]
            //call the BLAS routine for matrix vector multiplication:
            assert((v1.size() == size_) && (v2.size() == size_));
            char trans='T';
            double alpha=1., beta=0.;        //no need to multiply a constant or add a vector
            fortran_int_t inc=1;
            FORTRAN_ID(dgemv)(&trans, &size_, &size_, &alpha, &values_[0], &size_, &v1.values()[0], &inc, &beta, &v2.values()[0], &inc);
        }
               
        vector right_multiply(const vector &v1) const
        {
            vector result(size_);
            right_multiply(v1,result);
            return result;
        }
    
        inline void left_multiply(const vector &v1, vector &v2) const
        { 
            if(size_==0) return;
            //perform v2[i]=v1[j]M[ji]
            //call the BLAS routine for matrix vector multiplication:
            assert((v1.size() == size_) && (v2.size() == size_));
            char trans='N';
            double alpha=1., beta=0.;       //no need to multiply a constant or add a vector
            fortran_int_t inc=1;
            FORTRAN_ID(dgemv)(&trans, &size_, &size_, &alpha, &values_[0], &size_, &v1.values()[0], &inc, &beta, &v2.values()[0], &inc);
        }
        
        vector left_multiply(const vector &v1) const
        {
            vector result(size_);
            left_multiply(v1,result);
            return result;
        }
        
        void clear(){
            values_.clear();
            values_.resize(size_*size_,0.);
        }
        
        //Mres=this*M2
        inline void matrix_right_multiply(const matrix &M2,  matrix &Mres) const
        {
            if(size_==0) return;
            assert((M2.size() == size_) && (Mres.size() == size_));
            //call the BLAS routine for matrix matrix multiplication:
            double one_double=1.;
            double zero_double=0.;
            char notrans='N';
            //Mres.clear(); //to satisfy valgrind on hreidar / acml
            FORTRAN_ID(dgemm)(&notrans, &notrans, &size_, &size_, &size_, &one_double, &M2.values_[0], &size_, &values_[0], &size_, &zero_double, &Mres.values_[0], &size_);
        }
        
        inline void matrix_right_multiply(const blas::general_matrix<double> &M2,  blas::general_matrix<double>&Mres) const
        {
            if(size_==0) return;
            //call the BLAS routine for matrix matrix multiplication:
            double one_double=1.;
            double zero_double=0.;
            char notrans='N';
            Mres.clear(); 
            fortran_int_t M2_size2=M2.size2();
            FORTRAN_ID(dgemm)(&notrans, &notrans, &M2_size2, &size_, &size_, 
                   &one_double, &(M2(0,0)), &(M2.size2()), &values_[0],
                   &size_, &zero_double, &(Mres(0,0)), &(Mres.size2()));
        }
        
        matrix matrix_right_multiply(const matrix &M2) const
        {
            matrix result(size_);
            matrix_right_multiply(M2, result);
            return result;
        }
        
        void set_to_identity()
        {
            clear();
            for(int i=0;i<size_;++i){
                operator()(i,i)=1.;
            }
        }
        
        inline double trace()const{
            double tr=0;
            for(int i=0;i<size_;++i) tr+=operator()(i,i);
            return tr;
        }
        
        inline void transpose(){
            for(int i=0;i<size_;++i){
                for(int j=0;j<i;++j){
                    std::swap(operator()(i,j), operator()(j,i));
                }
            }
        }
        
        inline double determinant() const
        {
            //the simple ones...
            if(size_==0) return 1;
            if(size_==1) return values_[0];
            if(size_==2) return values_[0]*values_[3]-values_[1]*values_[2];
            fortran_int_t info=0;
            fortran_int_t *ipiv = new fortran_int_t[size_];
            matrix identity(size_);
            identity.set_to_identity();
            matrix det_matrix(*this);
        
            //LU factorization
            FORTRAN_ID(dgesv)(&size_, &size_, &det_matrix.values_[0], &size_, ipiv, &identity.values_[0], &size_,&info);
            if(info < 0) {
                std::cout << "LAPACK ERROR IN APPLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
                std::cout << "INFO:" << info << std::endl;
            } else if(info > 0){
                //check dgesv page: that means that we haven an exactly singular
                //matrix and the det is therefore =0: 
                return 0.;
            }
            //compute the determinant:
            double det=1.;
            //CAREFUL when pivoting: fortran uses array indexing starting
            //from one. But we're in C here -> 'one off' error
            for(int i=0;i<size_;++i){
                if(ipiv[i]-1!=i){
                    det*=-det_matrix(i,i);
                }
                else{
                    det*=det_matrix(i,i);
                }
            }
            delete[] ipiv;
            return det;
        }
        bool operator==(const matrix &other) const 
        {
            if(size_!=other.size_) return false;
            std::vector<double> diff(size_*size_);
            transform(values_.begin(), values_.end(), other.values_.begin(), diff.begin(), std::minus<double>());
            std::vector<double>::iterator it= std::find_if(diff.begin(),diff.end(),deviates<double>);
            return (it == diff.end());
        }
        
        bool operator!=(const matrix &other)const 
        {
            return !(*this == other);
        }
                  
    
        matrix &operator *=(double lambda)
        {
            fortran_int_t inc=1;
            fortran_int_t total_size=size_*size_;
            dscal_(&total_size, &lambda, &values_[0], &inc);
            return *this;
        }
              
        matrix &operator+=(const matrix &other) 
        {
            assert(size_==other.size_);
            std::vector<double> temp(other.values_);
            transform(values_.begin(), values_.end(), temp.begin(), values_.begin(), std::plus<double>());
            return *this;
        }
        
        matrix operator+(const matrix &M2) const
        {
            matrix result(*this);
            result += M2;
            return result;
        }
        
        matrix &operator-=(const matrix &other) 
        {
            assert(size_==other.size_);
            std::vector<double> temp(other.values_);
            transform(values_.begin(), values_.end(), temp.begin(), values_.begin(), std::minus<double>());
            return *this;
        }
        
        matrix operator-(const matrix &M2) const
        {
            matrix result(*this);
            result -= M2;
            return result;
        }
           
        double max() const
        {
            return *std::max_element(values_.begin(), values_.end());        
        }
        
        double min() const
        {
            return *std::min_element(values_.begin(), values_.end());        
        }
        
         
        void eigenvalues_eigenvectors_symmetric( vector &eigenvalues, matrix &eigenvectors) const
        {
            if(size_==0) return;
            //EIGENVECTORS ARE STORED IN ROWS!
            //perform dsyev call (LAPACK)
            eigenvectors=*this;
            fortran_int_t lwork=-1;
            fortran_int_t info;
            double work_size;
            char jobs='V';
            char uplo='L';
            //get optimal size for work
            FORTRAN_ID(dsyev)(&jobs, &uplo, &size_, &eigenvectors.values_[0], &size_, &(eigenvalues.values()[0]),&work_size, &lwork, &info); 
            lwork=(int)work_size;
            double *work=new double[lwork];
            FORTRAN_ID(dsyev)(&jobs, &uplo, &size_, &eigenvectors.values_[0], &size_, &(eigenvalues.values()[0]),work, &lwork, &info); 
            delete[] work;
        }
          
        //compute M=D*M, where D is a diagonal matrix represented by the
        //vector.
        void multiply_diagonal_matrix(const blas::vector &diagonal_matrix)
        {
            fortran_int_t inc=1;
            for(int i=0;i<size_;++i){
                dscal_(&size_, &diagonal_matrix(i), &values_[i*size_], &inc);
            }
        }
           
        //same, but M=M*D. Slow because of much larger stride.
        void right_multiply_diagonal_matrix(const blas::vector &diagonal_matrix)
        {
            for(int i=0;i<size_;++i){
                dscal_(&size_, &diagonal_matrix(i), &values_[i], &size_);
            }
        }

        void save(alps::hdf5::archive &ar) const
        {
            using namespace alps;
            ar << make_pvp("", &values_.front(), std::vector<std::size_t>(2, size_));
        }

        void load(alps::hdf5::archive &ar)
        {
            using namespace alps;
            resize(ar.extent("")[0]);
            ar >> make_pvp("", &values_.front(), std::vector<std::size_t>(2, size_));
        }

    private:
        std::vector<double> values_;
        fortran_int_t size_;
  };
    
    inline vector operator*(const matrix &M, const vector &v1)
    {
        return M.right_multiply(v1);
    }
    
   
    inline vector operator*(const vector &v1, const matrix &M)
    {
        return M.left_multiply(v1);
    }
    
    inline matrix operator*(const matrix &M, const double lambda)
    {
        matrix M2=M;
        M2*=lambda;
        return M2;
    }
    
    inline matrix operator*(const double lambda, const matrix &M)
    {
        matrix M2=M;
        M2*=lambda;
        return M2;
    }
    
    inline matrix operator*(const matrix M, const matrix M2)
    {
        matrix Mres(M.size());
        M.matrix_right_multiply(M2, Mres);
        return Mres;
    }
    
    inline std::ostream &operator<<(std::ostream &os, const matrix &M)
    {
        os<<"[";
        for(int i=0;i<M.size();++i){
            os<<"  [ ";
            for(int j=0;j<M.size();++j){
                os<<M(i,j)<<" ";
            }
            os <<" ] "<< std::endl;
        }
        os<<"]"<<" ";
        return os;
    }
} //namespace

#ifdef UBLAS
inline complex_dense_matrix mult(const complex_dense_matrix &A, const blas::matrix &B){ //slow matrix multiplication real & complex.
    if((int)A.size1() !=(int)B.size() || (int)A.size2()!=(int)B.size()){ std::cerr<<"wrong matrix sizes in mult."<<std::endl; exit(1);}
    int size=A.size1();
    complex_dense_matrix D(B.size1(), B.size());
    complex_dense_matrix B_complex(B.size1(), B.size());
    B_complex.clear(); D.clear();
    for(int i=0;i<B.size();++i){ //copy matrix
        for(int j=0;j<B.size();++j){
            B_complex(i,j)=B(i,j);
        }
    }
    //blas call
    std::complex<double> one_complex=1.;
    std::complex<double> zero_complex=0.;
    char notrans='N';
    FORTRAN_ID(zgemm)(&notrans, &notrans, &size, &size, &size, &one_complex, (void*)&(A(0,0)), &size, (void*)&(B_complex(0,0)), &size,(void*) &zero_complex,(void*) &(D(0,0)), &size);
    return D;
}
inline complex_dense_matrix mult(const blas::matrix &A, const complex_dense_matrix &B){ //slow matrix multiplication real & complex.
    if((int)B.size1() !=A.size() || (int)B.size2()!=A.size()){ std::cerr<<"wrong matrix sizes in mult."<<std::endl; exit(1);}
    int size=B.size1();
    complex_dense_matrix D(B.size1(), B.size1());
    complex_dense_matrix A_complex(A.size(), A.size());
    D.clear(); A_complex.clear();
    for(int i=0;i<A.size();++i){ //copy matrix
        for(int j=0;j<A.size();++j){
            A_complex(i,j)=A(i,j);
        }
    }
    //blas call
    std::complex<double> one_complex=1.;
    std::complex<double> zero_complex=0.;
    char notrans='N';
    FORTRAN_ID(zgemm)(&notrans, &notrans, &size, &size, &size, &one_complex, (void*)&(A_complex(0,0)), &size, (void*)&(B(0,0)), &size,(void*) &zero_complex,(void*) &(D(0,0)), &size);
    return D;
}
inline complex_dense_matrix mult(const complex_dense_matrix &A, const complex_dense_matrix &B){ //slow matrix multiplication real & complex.
    if(B.size1() !=A.size1() || B.size2()!=A.size1() || A.size1() !=A.size2()){ std::cerr<<"wrong matrix sizes in mult."<<std::endl; exit(1);}
    int size=B.size1();
    complex_dense_matrix C(A.size1(), A.size1());
    complex_dense_matrix D(A.size1(), A.size1());
    C.clear(); D.clear();
    //blas call
    std::complex<double> one_complex=1.;
    std::complex<double> zero_complex=0.;
    char notrans='N';
    FORTRAN_ID(zgemm)(&notrans, &notrans, &size, &size, &size, &one_complex, (void*)&(A(0,0)), &size, (void*)&(B(0,0)), &size,(void*) &zero_complex,(void*) &(D(0,0)), &size);
    return D;
}
#endif //UBLAS
 
#endif
