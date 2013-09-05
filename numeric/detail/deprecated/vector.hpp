/*****************************************************************************
 *
 * ALPS DMFT Project - BLAS Compatibility headers
 *  Square Matrix Class
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
 
#ifndef BLAS_VECTOR
#define BLAS_VECTOR

#include "./blasheader.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/pointer.hpp>

inline double expfunc(double entry)
{
    return std::exp(entry);
}

namespace blas{
    
  //simple BLAS vector that uses BLAS calls for operations.
  class vector{
    public:
      vector(int size=0, double initial_value=0.)
      : values_(size, initial_value), size_(size)
      {
      }
      
      vector(const vector &rhs)
      : values_(rhs.values_), size_(rhs.size_)
      {
      }
      
      friend void swap(vector& x,vector& y)
      {
          std::swap(x.values_, y.values_);
          std::swap(x.size_, y.size_);
      }
      
      vector& operator=(const vector &rhs)
      {
        vector temp(rhs);
        swap(temp, *this);
        return *this;
      }
      
      inline double &operator()(const unsigned int i)
      {
        assert((i < size_));
        return values_[i];
      }
      
      inline const double &operator()(const unsigned int i) const 
      {
        assert((i < size_));
        return values_[i];
      }
    
      inline const int size() const
      {
        return size_;
      }
      
      inline double operator*(const vector &v2)
      {
          fortran_int_t inc=1;
          if (!size()) return 0.;
          return FORTRAN_ID(ddot)(&size_, &values_[0],&inc,&v2.values_[0],&inc);
      }
      
      vector & operator+=(const vector &rhs) 
      {
        assert(rhs.size() == size_);
        std::vector<double> temp = rhs.values();
        transform(values_.begin(), values_.end(), temp.begin(), values_.begin(), std::plus<double>());
        return *this;
      }
      
      vector & operator-=(const vector &rhs) 
      {
          assert(rhs.size() == size_);
          std::vector<double> temp = rhs.values();
          transform(values_.begin(), values_.end(), temp.begin(), values_.begin(), std::minus<double>());
          return *this;
      }
      
      const vector operator+(const vector &other) const {
          assert(other.size() == size_);
          vector result(*this);     
          result += other;           
          return result;              
      }
      
      const vector operator-(const vector &other) const {
          assert(other.size() == size_);
          vector result(*this);     
          result -= other;           
          return result;              
      }
      
      bool operator==(const vector &other) const {
          return (values_ == other.values());
      }
      
      bool operator!=(const vector &other) const {
          return !(*this == other);
      }
      
      inline double dot(const vector &v2)
      {   
        return (*this)*v2; 
      }
      
      //multiply vector by constant
      inline vector operator *=(double lambda)
      {
          fortran_int_t inc=1;
          if (size())
            FORTRAN_ID(dscal)(&size_, &lambda, &values_[0], &inc);
          return *this;
      }
      
      void clear(){
          values_.clear();
          values_.resize(size_,0.);
      }
      
      inline const std::vector<double> values() const
      {
          return values_; 
      }
      
      inline std::vector<double> &values() 
      {
          return values_; 
      }
      
      void exp( double c)
      {
          if (!size()) return;
#ifdef VECLIB
          //osx veclib exp
          fortran_int_t one=1;
          blas::dscal_(&size_,&c,&values_[0],&one);
          vecLib::vvexp(&values_[0], &values_[0], &size_); 
#else
#ifdef ACML
          //amd acml vector exp
          std::vector<double> scaled_values(size_);
          daxpy_(&size_, &c, &values_[0], &inc, &scaled_values[0], &inc);
          acml::vrda_exp(size_, &scaled_values[0], &values_[0]);
#else
#ifdef MKL
          //intel MKL vector exp
          std::vector<double> scaled_values(size_);
          fortran_int_t inc=1;
          daxpy_(&size_, &c, &values_[0], &inc, &scaled_values[0], &inc);
          mkl::vdExp(size_,  &scaled_values[0], &values_[0]);
#else
          //pedestrian method
          *this *= c;
          std::transform(values_.begin(), values_.end(), values_.begin(), expfunc);
#endif
#endif
#endif
      }
    
      void exp(double c, const vector &v)
      {
          vector temp(v);
          temp.exp(c);
          swap(temp, *this);
      }
      
      void resize(int new_size)
      {
          values_.resize(new_size);
          size_=new_size;
      }
      
      void insert(double value, unsigned int i)
      {
          assert((i <= size_));
          resize(size_+1);
          std::vector<double>::iterator it = values_.begin();
          values_.insert(it+i,value);
      }

      void save(alps::hdf5::archive & ar) const
      {
          using namespace alps;
          ar << make_pvp("", &values_.front(), std::vector<std::size_t>(1, size_));
      }

      void load(alps::hdf5::archive & ar)
      {
          using namespace alps;
          resize(ar.extent("")[0]);
          ar >> make_pvp("", &values_.front(), std::vector<std::size_t>(1, size_));
      }

    private:
      std::vector<double> values_;
      fortran_int_t size_; 
  };
    
  inline std::ostream &operator<<(std::ostream &os, const vector&M)
  {
    os<<"[ ";
    for(int i=0;i<M.size();++i){
      os<<M(i)<<" ";
    }
    os<<"]"<<std::endl;
    return os;
  }
} //namespace

#endif 
