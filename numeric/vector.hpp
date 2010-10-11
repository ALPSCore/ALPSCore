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
 
#ifndef BLAS_VECTOR
#define BLAS_VECTOR

#include "./blasheader.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <alps/numeric/scalar_product.hpp>

template<typename T>
inline T expfunc(T entry)
{
    return std::exp(entry);
}

namespace blas{
    
  //simple BLAS vector that uses BLAS calls for operations.
  template<typename T>
  class vector : public std::vector<T>
  {
    public:
      vector(unsigned int size=0, T initial_value=0.)
      : std::vector<T>(size, initial_value)
      {
      }
      
      friend void swap(vector<T>& x,vector<T>& y)
      {
          std::swap(x, y);
      }
      
      inline T &operator()(const unsigned int i)
      {
        assert((i < this->size()));
        return this->at(i);
      }
      
      inline const T &operator()(const unsigned int i) const 
      {
        assert((i < this->size()));
        return this->at(i);
      }
    
      vector<T> & operator+=(const vector<T> rhs) 
      {
        assert(rhs.size() == this->size());
        std::vector<T> temp(rhs);
        transform(this->begin(), this->end(), temp.begin(), this->begin(), std::plus<T>());
        return *this;
      }
      
      vector<T> & operator-=(const vector<T> rhs) 
      {
          assert(rhs.size() == this->size());
          std::vector<T> temp(rhs);
          transform(this->begin(), this->end(), temp.begin(), this->begin(), std::minus<T>());
          return *this;
      }
   
      void insert(T value, unsigned int i)
      {
          assert((i <= this->size()));
          resize(this->size()+1);
          this->std::vector<T>::insert(this->begin()+i,value);
      }
      
      void exp( double c)
      {
          if (!(this->size())) return;
#ifdef VECLIB
          //osx veclib exp
          int one=1;
          int s=this->size();
          blas::dscal_(&s, &c, &this->at(0),&one);
          vecLib::vvexp(&this->at(0), &this->at(0), &s); 
#else
#ifdef ACML
          //amd acml vector exp
          std::vector<double> scaled_values(size());
          int s=this->size();
          daxpy_(&s, &c, &(this->at(0)), &inc, &scaled_values[0], &inc);
          acml::vrda_exp(s, &scaled_values[0], &(this->at(0)));
#else
#ifdef MKL
          //intel MKL vector exp
          std::vector<double> scaled_values(size());
          int inc=1;
          int s=this->size();
          daxpy_(&s, &c, &(this->at(0)), &inc, &scaled_values[0], &inc);
          mkl::vdExp(s,  &scaled_values[0], &(this->at(0)));
#else
          //pedestrian method
          *this *= c;
          std::transform(this->begin(), this->end(), this->begin(), expfunc<T>);
#endif
#endif
#endif
      }
  };
    
    // return type of alps::numeric::scalar_product???
    template<typename T>
    inline T operator*(const vector<T> v1, const vector<T> v2)
    {
        return alps::numeric::scalar_product(v1,v2);
    }
    
    
    template<>
    inline double operator*(const vector<double> v1, const vector<double> v2)
    {
        int inc=1;
        int size=v1.size();
        if (v1.empty() || v2.empty()) return 0.;
        return ddot_(&size, &v1[0],&inc,&v2[0],&inc);
    }
    
    
    template<typename T>
    vector<T> operator+(const vector<T> v1, const vector<T> v2)  
    {
        assert(v1.size() == v2.size());
        vector<T> result(v1);     
        result += v2;           
        return result;              
    }
    
    template<typename T>
    vector<T> operator-(const vector<T> v1, const vector<T> v2)  
    {
        assert(v1.size() == v2.size());
        vector<T> result(v1);     
        result -= v2;           
        return result;              
    }  
    
    template<typename T>
    inline T dot(const vector<T> v1, const vector<T> v2)
    {   
        return v1*v2; 
    }
    
    //multiply vector by constant
    template<typename T>
    inline vector<T> operator *=(vector<T> &v, T lambda)
    {
        transform(v.begin(), v.end(), v.begin(), std::bind2nd(std::multiplies<T>(), lambda));
        return v;
    }
    
    
    template<>
    inline vector<double> operator *=(vector<double> &v, double lambda)
    {
        int inc=1;
        int size=v.size();
        if (!v.empty())
            dscal_(&size, &lambda, &v[0], &inc);
        return v;
    }
    
    template<typename T>
    void exp(T c, vector<T> &v)
    {
        vector<T> temp(v);
        temp.exp(c);
        swap(temp,v);
    }

  template<typename T>
  inline std::ostream &operator<<(std::ostream &os, const vector<T> &v)
  {
    os<<"[ ";
    for(unsigned int i=0;i<v.size()-1;++i){
      os<<v(i)<<", ";
    }
      os<< v(v.size()-1) << "]"<<std::endl;
    return os;
  }
} //namespace

#endif 
