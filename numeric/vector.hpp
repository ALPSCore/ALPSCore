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
  template<typename T>
  class vector : public std::vector<T>  
  {
    public:
      vector(std::size_t size=0, T initial_value=0.)
      : std::vector<T>(size, initial_value)
      {
      }
      
      friend void swap(vector<T>& x,vector<T>& y)
      {
          std::swap(x, y);
      }
      
      inline T &operator()(const std::size_t i)
      {
          assert((i < this->size()));
          return this->operator[](i);
      }
      
      inline const T &operator()(std::size_t i) const 
      {
          assert((i < this->size()));
          return this->operator[](i);
      }
    
      vector<T> & operator+=(const vector<T>& rhs) 
      {
          assert(rhs.size() == this->size());
          add_assign(this->begin(), this->end(), rhs.begin());
          return *this;
      }
      
      vector<T> & operator-=(const vector<T> rhs) 
      {
          assert(rhs.size() == this->size());
          minus_assign(this->begin(), this->end(), rhs.begin());
          return *this;
      }
      
      vector<T> & operator*=(const T lambda) 
      {
          multiplication_assign(this->begin(), this->end(), lambda);
          return *this;
      }
  };  
    
    template<typename T>
    void insert(vector<T> v, T value, std::size_t i)
    {
        assert((i <= v.size()));
        v.insert(v.begin()+i,value);
    }
    
    template <class InputIterator1, class InputIterator2>
    void add_assign(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2) 
    {
        std::transform(first1, last1, first2, first1, std::plus<typename std::iterator_traits<InputIterator2>::value_type >());
    }
    
    void add_assign(double* first1, double* last1, double* first2) 
    {
        int inc=1;
        int s=(last1-first1);
        double alpha=1.;
        daxpy_(&s, &alpha, first1, &inc, first2, &inc);
    }
    
    void add_assign(float* first1, float* last1, float* first2) 
    {
        int inc=1;
        int s=(last1-first1);
        float alpha=1.;
        saxpy_(&s, &alpha, first1, &inc, first2, &inc);
    }    
    
    template<typename T>
    vector<T> operator+(const vector<T> v1, const vector<T> v2)  
    {
        assert(v1.size() == v2.size());
        vector<T> result(v1);     
        result += v2;           
        return result;              
    }
    
    template <class InputIterator1, class InputIterator2>
    void minus_assign(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2) 
    {
        std::transform(first1, last1, first2, first1, std::minus<typename std::iterator_traits<InputIterator2>::value_type >());
    }
    
    void minus_assign(double* first1, double* last1, double* first2) 
    {
        int inc=1;
        int s=(last1-first1);
        double alpha=-1.;
        daxpy_(&s, &alpha, first2, &inc, first1, &inc);
    }
    
    void minus_assign(float* first1, float* last1, float* first2) 
    {
        int inc=1;
        int s=(last1-first1);
        float alpha=-1.;
        saxpy_(&s, &alpha, first2, &inc, first1, &inc);
    }
    
    template<typename T>
    vector<T> operator-(const vector<T> v1, const vector<T> v2)  
    {
        assert(v1.size() == v2.size());
        vector<T> result(v1);     
        result -= v2;           
        return result;              
    }  

    template <class ForwardIterator, typename T>
    void multiplication_assign(ForwardIterator start1, ForwardIterator end1, T lambda) 
    {
        std::transform(start1, end1, start1, std::bind2nd(std::multiplies<T>(), lambda));
    }

    void multiplication_assign(double* start1, double* end1, double lambda) 
    { 
        int inc=1;
        int size=(end1-start1); 
        if (size!=0)
            dscal_(&size, &lambda, start1, &inc);
    }
    
    void multiplication_assign(float* start1, float* end1, float lambda) 
    { 
        int inc=1;
        int size=(end1-start1); 
        if (size!=0)
            sscal_(&size, &lambda, start1, &inc);
    }
    
    template<typename T>
    inline T scalar_product(const vector<T> v1, const vector<T> v2)
    {   
        return alps::numeric::scalar_product(v1,v2);
    }
    
    template<>
    inline double scalar_product(const vector<double> v1, const vector<double> v2)
    {
        int inc=1;
        int size=v1.size();
        if (v1.empty() || v2.empty()) return 0.;
        return ddot_(&size, &v1[0],&inc,&v2[0],&inc);
    }
    
    template<>
    inline float scalar_product(const vector<float> v1, const vector<float> v2)
    {
        int inc=1;
        int size=v1.size();
        if (v1.empty() || v2.empty()) return 0.;
        return sdot_(&size, &v1[0],&inc,&v2[0],&inc);
    }
    
    template<typename T>
    inline vector<T> exp(T c, vector<T> v)
    {
        vector<T> result(v);
        v*=c;
        std::transform(v.begin(), v.end(), result.begin(), expfunc<T>);
        return result;
    }
    
    template<>
    inline vector<double> exp(double c, vector<double> v)
    {
        int s=v.size();
        vector<double> result(s);
        v*=c;
#ifdef VECLIB
        vecLib::vvexp(&result[0], &v[0], &s); 
#else
#ifdef ACML
        acml::vrda_exp(s, &v[0], &result[0]);
#else
#ifdef MKL
        mkl::vdExp(s,  &v[0], &result[0]);
#endif
#endif
#endif  
        return result;
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
