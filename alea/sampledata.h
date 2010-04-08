/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Ping Nang Ma <pingnang@itp.phys.ethz.ch>,
*                            Matthias Troyer <troyer@itp.phys.ethz.ch>,
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
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

/* $Id: nobinning.h 3520 2009-12-11 16:49:53Z gamperl $ */

#ifndef ALPS_SAMPLEDATA
#define ALPS_SAMPLEDATA

#include <alps/config.h>
#include <alps/numeric/vector_functions.hpp>
#include <alps/type_traits/element_type.hpp>

#ifdef ALPS_HAVE_PYTHON
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#endif

#include <boost/type_traits.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cassert>


namespace alps { 

  namespace alea {

    template<class T> 
    class sampledata 
    {
    public:
      typedef T                                     value_type;
      typedef typename alps::element_type<T>::type  element_type;
      typedef std::size_t                           size_type;
      typedef std::size_t                           index_type;
      typedef std::ptrdiff_t                        difference_type;
    
    private:
      value_type _mean;
      value_type _error;
    
    public:
      // constructors, assignment operator
#ifdef ALPS_HAVE_PYTHON
      sampledata(boost::python::object const & mean_nparray, boost::python::object const & error_nparray);
#endif
      sampledata(value_type mean =value_type(), value_type error =value_type())
        : _mean(mean)
        , _error(error) 
      {}
    
      inline sampledata<value_type>& operator= (sampledata<value_type> const rhs)
      {
        _mean  = rhs._mean;
        _error = rhs._error;
        return *this;
      }
    
      // call
      inline value_type mean()  const {  return _mean;  }
      inline value_type error() const {  return _error; }

#ifdef ALPS_HAVE_PYTHON
      boost::python::object mean_nparray() const;
      boost::python::object error_nparray() const;
#endif    
      // comparison
      inline bool operator==(sampledata const & rhs)
      {  return ((_mean == rhs._mean) && (_error == rhs._error));  }
    
    
      // intrinsic operations: ( += , -= , *= , /= )
      inline sampledata<value_type>& operator+=(sampledata<value_type> const & rhs)
      {
        using std::sqrt;
        using alps::numeric::sq;
        using alps::numeric::sqrt;
        using boost::numeric::operators::operator+;

        _error = sqrt(sq(_error)+sq(rhs._error));
        _mean  = _mean + rhs._mean;
        return *this;
      }
    
      inline sampledata<value_type>& operator+=(value_type const & rhs)
      {
        using boost::numeric::operators::operator+;
       
        _mean  = _mean + rhs;
        return *this;
      }
    
      inline sampledata<value_type>& operator-=(sampledata const & rhs)
      {
        using std::sqrt;
        using alps::numeric::sq;
        using alps::numeric::sqrt;
        using boost::numeric::operators::operator+;
        using boost::numeric::operators::operator-;
    
        _error = sqrt(sq(_error)+sq(rhs._error));
        _mean  = _mean - rhs._mean;
        return *this;
      }
    
      inline sampledata<value_type>& operator-=(value_type const & rhs)
      {
        using boost::numeric::operators::operator-;

        _mean  = _mean - rhs;
        return *this;
      }
    
      inline sampledata<value_type>& operator*=(sampledata<value_type> const & rhs)
      {
        using std::sqrt;
        using alps::numeric::sq;
        using alps::numeric::sqrt;
        using boost::numeric::operators::operator+;
        using boost::numeric::operators::operator*;
    
        _error =  sqrt(sq(rhs._mean)*sq(_error) + sq(_mean)*sq(rhs._error));
        _mean  = _mean * rhs._mean;
        return *this;
      }
    
      inline sampledata<value_type>& operator*=(value_type const & rhs)
      {
        using std::abs;
        using alps::numeric::abs;
        using boost::numeric::operators::operator*;
    
        _error = abs(_error * rhs);
        _mean  = _mean  * rhs;
        return *this;
      }
    
      inline sampledata<value_type>& operator/=(sampledata<value_type> const & rhs)
      {
        using std::sqrt;
        using alps::numeric::sq;
        using alps::numeric::sqrt;
        using boost::numeric::operators::operator+;
        using boost::numeric::operators::operator*;
        using boost::numeric::operators::operator/;
    
        _error = sqrt(sq(rhs._mean)*sq(_error) + sq(_mean)*sq(rhs._error));
        _error = _error / sq(rhs._mean);
        _mean  = _mean  /rhs._mean;
        return *this;
      }
    
      inline sampledata<value_type>& operator/=(value_type const & rhs)
      {
        using std::abs;
        using alps::numeric::abs;
        using boost::numeric::operators::operator/;
    
        _error = abs(_error / rhs);
        _mean  = _mean  / rhs;
        return *this;
      }
    
      // non- STL container support
      size_type size() const
      {
        assert(_mean.size() == _error.size());
        return (_mean.size());
      }

      void push_back(sampledata<element_type> const & rhs)
      {
        _mean.push_back(rhs.mean());
        _error.push_back(rhs.error());
      }

      void pop_back()
      {
        assert((_mean.size() != 0) && (_error.size() != 0));
        _mean.pop_back();
        _error.pop_back();
      }

      void clear()
      {
        _mean.clear();
        _error.clear();
      }

      void insert(index_type const & index, sampledata<element_type> const & value)
      {
        assert((_mean.size() > index) && (_error.size() > index));
        _mean.insert(_mean.begin()+index,value.mean());
        _error.insert(_error.begin()+index,value.error());
      }

      void erase(index_type const & index)
      {
        assert((_mean.size() > index) && (_error.size() > index));
        _mean.erase(_mean.begin()+index);
        _error.erase(_error.begin()+index);
      }

      sampledata<element_type> at(index_type const & index)
      {
        return sampledata<element_type>(_mean[index],_error[index]);
      }
    };
    
    
    // i/o operators
    template <class T>
    std::ostream& operator<< (std::ostream &out, sampledata<T> const& value)
    {
      out << value.mean() << " +/- " << value.error();
      return out;
    }
    
    template <class T>
    std::ostream& operator<< (std::ostream &out, sampledata<std::vector<T> > const & vec)
    {
      for (std::size_t index=0; index < vec.size(); ++index)  
      {
        out << vec.mean()[index] << " +/- " << vec.error()[index] << "\n";
      }
      return out;
    }
    
    
    // positivity operation, declaration of negativity and absolute declaration
    template <class T>
    inline sampledata<T>& operator+(sampledata<T>& rhs)  
    {  
      return rhs;  
    }
    
    template <class T>
    inline sampledata<T> operator-(sampledata<T> rhs)
    {
      using boost::numeric::operators::operator-;
      return sampledata<T>(-(rhs.mean()),rhs.error());
    }
    
    template <class T>
    inline sampledata<T> abs(sampledata<T> rhs)
    {
      using std::abs;
      using alps::numeric::abs;
      
      return sampledata<T>(abs(rhs.mean()),rhs.error());  
    }
    
    
    // ( + , - , * , / ) operators
    #define IMPLEMENT_OPERATION(OPERATOR_NAME,OPERATOR_ASSIGN) \
    template<class T> \
    inline sampledata<T> OPERATOR_NAME(sampledata<T> lhs, sampledata<T> const & rhs) \
    {  return lhs OPERATOR_ASSIGN rhs;  } \
    \
    template <class T> \
    inline sampledata<T> OPERATOR_NAME(sampledata<T> lhs, T const & rhs) \
    {  return lhs OPERATOR_ASSIGN rhs;  } \
    \
    template <class T> \
    inline sampledata<std::vector<T> > OPERATOR_NAME(sampledata<std::vector<T> > lhs, typename sampledata<std::vector<T> >::element_type const & rhs_elem) \
    { \
      std::vector<T> rhs(lhs.size(),rhs_elem); \
      return lhs OPERATOR_ASSIGN rhs; \
    }
    
    IMPLEMENT_OPERATION(operator+,+=)
    IMPLEMENT_OPERATION(operator-,-=)
    IMPLEMENT_OPERATION(operator*,*=)
    IMPLEMENT_OPERATION(operator/,/=)
    
    template <class T>
    inline sampledata<T> operator+(T const & lhs, sampledata<T> rhs)
    {  return rhs += lhs;  }
    
    template <class T>
    inline sampledata<T> operator-(T const & lhs, sampledata<T> rhs)
    { return -rhs + lhs;  }
    
    template <class T>
    inline sampledata<T> operator*(T const & lhs, sampledata<T> rhs)
    {  return rhs *= lhs;  }
    
    template <class T>
    inline sampledata<T> operator/(T const & lhs, sampledata<T> const & rhs)
    {
      using std::abs;
      using alps::numeric::abs;
      using boost::numeric::operators::operator*;
      using boost::numeric::operators::operator/;

      T inverse_mean = lhs/rhs.mean();
      return sampledata<T>(inverse_mean,abs(inverse_mean*rhs.error()/rhs.mean()));
    }

    #define IMPLEMENT_OPERATION2(OPERATOR_NAME,OPERATOR_ASSIGN) \
    template <class T> \
    inline sampledata<std::vector<T> > OPERATOR_NAME(typename sampledata<std::vector<T> >::element_type const & lhs_elem, sampledata<std::vector<T> > rhs) \
    { \
      std::vector<T> lhs(rhs.size(),lhs_elem); \
      return lhs OPERATOR_ASSIGN rhs; \
    }

    IMPLEMENT_OPERATION2(operator+,+)
    IMPLEMENT_OPERATION2(operator-,-)
    IMPLEMENT_OPERATION2(operator*,*)
    IMPLEMENT_OPERATION2(operator/,/)

    // pow, sq, sqrt, cb, cbrt, exp, log    
    template <class T>
    inline sampledata<T> pow(sampledata<T> rhs, typename sampledata<T>::element_type const & exponent)
    {
      if (exponent == 1.)
      {
        return rhs;
      }
      else
      {
        using std::pow;
        using std::abs;
        using alps::numeric::pow;
        using alps::numeric::abs;
        using boost::numeric::operators::operator-;
        using boost::numeric::operators::operator*;

        T dummy = pow(rhs.mean(),exponent-1.);
        return sampledata<T>(dummy*rhs.mean(),abs(exponent*dummy*rhs.error()));
      }
    }


    template<class T>
    inline sampledata<T> sq(sampledata<T> rhs)
    {
      using alps::numeric::sq;
      using std::abs;
      using alps::numeric::abs;
      using alps::numeric::operator*;
      using boost::numeric::operators::operator*;

      return sampledata<T>(sq(rhs.mean()),abs(2.*rhs.mean()*rhs.error()));
    }
    
    template<class T>
    sampledata<T> cb(sampledata<T> rhs)
    {
      using alps::numeric::sq;
      using std::abs;
      using alps::numeric::abs;
      using alps::numeric::operator*;
      using boost::numeric::operators::operator*;

      return sampledata<T>((sq(rhs.mean()))*rhs.mean(),abs(3.*(sq(rhs.mean()))*rhs.error()));
    }
    
    template<class T>
    sampledata<T> sqrt(sampledata<T> rhs)
    {
      using std::sqrt;
      using alps::numeric::sqrt;
      using std::abs;
      using alps::numeric::abs;
      using alps::numeric::operator*;
      using boost::numeric::operators::operator/;

      return sampledata<T>(sqrt(rhs.mean()),abs(rhs.error()/(2.*sqrt(rhs.mean()))));
    }
    
    template<class T>
    sampledata<T> cbrt(sampledata<T> rhs)
    {
      using alps::numeric::sq;
      using std::abs;
      using alps::numeric::abs;
      using std::pow;
      using alps::numeric::pow;
      using alps::numeric::operator*;
      using boost::numeric::operators::operator/;

      T dummy = pow(rhs.mean(),1./3);
      return sampledata<T>(dummy,abs(rhs.error()/(3.*sq(dummy))));
    }
    
    template<class T>
    sampledata<T> exp(sampledata<T> rhs)
    {
      using std::exp;
      using alps::numeric::exp;
      using boost::numeric::operators::operator*;

      T dummy = exp(rhs.mean());
      return sampledata<T>(dummy,dummy*rhs.error());
    }
    
    template<class T>
    sampledata<T> log(sampledata<T> rhs)
    {
      using std::log;
      using alps::numeric::log;
      using std::abs;
      using alps::numeric::abs;
      using boost::numeric::operators::operator/;

      return sampledata<T>(log(rhs.mean()),abs(rhs.error()/rhs.mean()));
    }

  
    // ( sin, ... , atanh ) operations
    template<class T>
    inline sampledata<T> sin(sampledata<T> rhs)
    {
      using std::abs;
      using alps::numeric::abs;
      using std::sin;
      using alps::numeric::sin;
      using std::cos;
      using alps::numeric::cos;
      using boost::numeric::operators::operator*;

      T derivative = cos(rhs.mean());
      return sampledata<T>(sin(rhs.mean()),abs(derivative*rhs.error()));
    }
    
    template<class T>
    sampledata<T> cos(sampledata<T> rhs)
    {
      using std::abs;
      using alps::numeric::abs;
      using std::sin;
      using alps::numeric::sin;
      using std::cos;
      using alps::numeric::cos;
      using boost::numeric::operators::operator-;
      using boost::numeric::operators::operator*;

      T derivative = -sin(rhs.mean());
      return sampledata<T>(cos(rhs.mean()),abs(derivative*rhs.error()));
    }
    
    template<class T>
    sampledata<T> tan(sampledata<T> rhs)
    {
      using std::abs;
      using alps::numeric::abs;
      using std::tan;
      using alps::numeric::tan;
      using std::cos;
      using alps::numeric::cos;
      using boost::numeric::operators::operator*;
      using alps::numeric::operator/;

      T derivative = 1./(cos(rhs.mean())*cos(rhs.mean()));
      return sampledata<T>(tan(rhs.mean()),abs(derivative*rhs.error()));
    }
    
    template<class T>
    sampledata<T> sinh(sampledata<T> rhs)
    {
      using std::abs;
      using alps::numeric::abs;
      using std::sinh;
      using alps::numeric::sinh;
      using std::cosh;
      using alps::numeric::cosh;
      using boost::numeric::operators::operator*;

      T derivative = cosh(rhs.mean());
      return sampledata<T>(sinh(rhs.mean()),abs(derivative*rhs.error()));
    }
    
    template<class T>
    sampledata<T> cosh(sampledata<T> rhs)
    {
      using std::abs;
      using alps::numeric::abs;
      using std::sinh;
      using alps::numeric::sinh;
      using std::cosh;
      using alps::numeric::cosh;
      using boost::numeric::operators::operator*;

      T derivative = sinh(rhs.mean());
      return sampledata<T>(cosh(rhs.mean()),abs(derivative*rhs.error()));
    }
    
    template<class T>
    sampledata<T> tanh(sampledata<T> rhs)
    {
      using std::abs;
      using alps::numeric::abs;
      using std::cosh;
      using alps::numeric::cosh;
      using std::tanh;
      using alps::numeric::tanh;
      using boost::numeric::operators::operator*;
      using alps::numeric::operator/;

      T derivative = 1./(cosh(rhs.mean())*cosh(rhs.mean()));
      return sampledata<T>(tanh(rhs.mean()),abs(derivative*rhs.error()));
    }
    
    template<class T>
    sampledata<T> asin(sampledata<T> rhs)
    {
      using std::abs;
      using alps::numeric::abs;
      using std::sqrt;
      using alps::numeric::sqrt;
      using std::asin;
      using alps::numeric::asin;
      using alps::numeric::operator-;
      using boost::numeric::operators::operator*;
      using alps::numeric::operator/;

      T derivative = 1./sqrt(1. - rhs.mean()*rhs.mean());
      return sampledata<T>(asin(rhs.mean()),abs(derivative*rhs.error()));
    }
    
    template<class T>
    sampledata<T> acos(sampledata<T> rhs)
    {
      using std::abs;
      using alps::numeric::abs;
      using std::sqrt;
      using alps::numeric::sqrt;
      using std::acos;
      using alps::numeric::acos;
      using alps::numeric::operator-;
      using boost::numeric::operators::operator-;
      using boost::numeric::operators::operator*;
      using alps::numeric::operator/;

      T derivative = -1./sqrt(1. - rhs.mean()*rhs.mean());
      return sampledata<T>(acos(rhs.mean()),abs(derivative*rhs.error()));
    }
    
    template<class T>
    sampledata<T> atan(sampledata<T> rhs)
    {
      using std::abs;
      using alps::numeric::abs;
      using std::atan;
      using alps::numeric::atan;
      using alps::numeric::operator+;
      using boost::numeric::operators::operator*;
      using alps::numeric::operator/;

      T derivative = 1./(1. + rhs.mean()*rhs.mean());
      return sampledata<T>(atan(rhs.mean()),abs(derivative*rhs.error()));
    }
    
    template<class T>
    sampledata<T> asinh(sampledata<T> rhs)
    {
      using std::abs;
      using alps::numeric::abs;
      using std::sqrt;
      using alps::numeric::sqrt;
      using boost::math::asinh;
      using alps::numeric::asinh;
      using alps::numeric::operator+;
      using boost::numeric::operators::operator*;
      using alps::numeric::operator/;

      T derivative = 1./sqrt(rhs.mean()*rhs.mean() + 1.);
      return sampledata<T>(asinh(rhs.mean()),abs(derivative*rhs.error()));
    }
    
    template<class T>
    sampledata<T> acosh(sampledata<T> rhs)
    {
      using std::abs;
      using alps::numeric::abs;
      using std::sqrt;
      using alps::numeric::sqrt;
      using boost::math::acosh;
      using alps::numeric::acosh;
      using alps::numeric::operator-;
      using boost::numeric::operators::operator*;
      using alps::numeric::operator/;

      T derivative = 1./sqrt(rhs.mean()*rhs.mean() - 1.);
      return sampledata<T>(acosh(rhs.mean()),abs(derivative*rhs.error()));
    }
    
    template<class T>
    sampledata<T> atanh(sampledata<T> rhs)
    {
      using std::abs;
      using alps::numeric::abs;
      using boost::math::atanh;
      using alps::numeric::atanh;
      using alps::numeric::operator-;
      using boost::numeric::operators::operator*;
      using alps::numeric::operator/;


      T derivative = 1./(1. - rhs.mean()*rhs.mean());
      return sampledata<T>(atanh(rhs.mean()),abs(derivative*rhs.error()));
    }


    // extended support for std::vector<sampledata<T> >    
    template<class T>
    inline std::vector<sampledata<T> > operator+(std::vector<sampledata<T> > const & lhs, std::vector<sampledata<T> > const & rhs)
    {
      return boost::numeric::operators::operator+ <sampledata<T>, sampledata<T> > (lhs,rhs);
    }

    template<class T>
    inline std::vector<sampledata<T> > operator+(std::vector<sampledata<T> > const & lhs, T const & rhs)
    {
      std::vector<T> rhs_vector;
      rhs_vector.resize(lhs.size());
      std::fill(rhs_vector.begin(),rhs_vector.end(),rhs);

      return boost::numeric::operators::operator+ <sampledata<T>, T> (lhs,rhs_vector);
    }

    template<class T>
    inline std::vector<sampledata<T> > operator+(T const & lhs, std::vector<sampledata<T> > const & rhs)
    {
      std::vector<T> lhs_vector;
      lhs_vector.resize(rhs.size());
      std::fill(lhs_vector.begin(),lhs_vector.end(),lhs);

      return boost::numeric::operators::operator+ <T, sampledata<T> > (lhs_vector,rhs);
    }

    template<class T>
    inline std::vector<sampledata<T> > operator+(std::vector<sampledata<T> > const & lhs, std::vector<T> const & rhs)
    {
      return boost::numeric::operators::operator+<sampledata<T>, T> (lhs,rhs);
    }

    template<class T>
    inline std::vector<sampledata<T> > operator+(std::vector<T> const & lhs, std::vector<sampledata<T> > const & rhs)
    {
      return boost::numeric::operators::operator+ <T, sampledata<T> > (lhs,rhs);
    }

    template<class T>
    inline std::vector<sampledata<T> > operator-(std::vector<sampledata<T> > const & lhs, std::vector<sampledata<T> > const & rhs)
    {
      return boost::numeric::operators::operator- <sampledata<T>, sampledata<T> > (lhs,rhs);
    }

    
    template<class T>
    inline std::vector<sampledata<T> > operator-(std::vector<sampledata<T> > const & lhs, T const & rhs)
    {
      std::vector<T> rhs_vector; 
      rhs_vector.resize(lhs.size());
      std::fill(rhs_vector.begin(),rhs_vector.end(),rhs);

      return boost::numeric::operators::operator- <sampledata<T>, T> (lhs,rhs_vector);
    }
    
    template<class T>
    inline std::vector<sampledata<T> > operator-(T const & lhs, std::vector<sampledata<T> > const & rhs)
    {
      std::vector<T> lhs_vector;
      lhs_vector.resize(rhs.size());
      std::fill(lhs_vector.begin(),lhs_vector.end(),lhs);

      return boost::numeric::operators::operator- <T, sampledata<T> > (lhs_vector,rhs);
    }

    template<class T>
    inline std::vector<sampledata<T> > operator-(std::vector<sampledata<T> > const & lhs, std::vector<T> const & rhs)
    {
      return boost::numeric::operators::operator- <sampledata<T>, T> (lhs,rhs);
    }
   
    template<class T>
    inline std::vector<sampledata<T> > operator-(std::vector<T> const & lhs, std::vector<sampledata<T> > const & rhs)
    {
      return boost::numeric::operators::operator- <T, sampledata<T> > (lhs,rhs);
    }
    
    template<class T>
    inline std::vector<sampledata<T> > operator*(std::vector<sampledata<T> > const & lhs, std::vector<sampledata<T> > const & rhs)
    {
      return boost::numeric::operators::operator* <sampledata<T>, sampledata<T> > (lhs,rhs);
    }
    
    template<class T>
    inline std::vector<sampledata<T> > operator*(std::vector<sampledata<T> > const & lhs, T const & rhs)
    {
      std::vector<T> rhs_vector;
      rhs_vector.resize(lhs.size());
      std::fill(rhs_vector.begin(),rhs_vector.end(),rhs);

      return boost::numeric::operators::operator* <sampledata<T>, T> (lhs,rhs_vector);
    }
    
    template<class T>
    inline std::vector<sampledata<T> > operator*(T const & lhs, std::vector<sampledata<T> > const & rhs)
    {
      std::vector<T> lhs_vector;
      lhs_vector.resize(rhs.size());
      std::fill(lhs_vector.begin(),lhs_vector.end(),lhs);

      return boost::numeric::operators::operator* <T, sampledata<T> > (lhs_vector,rhs);
    }
    
    template<class T>
    inline std::vector<sampledata<T> > operator*(std::vector<sampledata<T> > const & lhs, std::vector<T> const & rhs)
    {
      return boost::numeric::operators::operator* <sampledata<T>, T> (lhs,rhs);
    }

    template<class T>
    inline std::vector<sampledata<T> > operator*(std::vector<T> const & lhs, std::vector<sampledata<T> > const & rhs)
    {
      return boost::numeric::operators::operator* <T, sampledata<T> > (lhs,rhs);
    }

    template<class T>
    inline std::vector<sampledata<T> > operator/(std::vector<sampledata<T> > const & lhs, std::vector<sampledata<T> > const & rhs)
    {
      return boost::numeric::operators::operator/ <sampledata<T>, sampledata<T> > (lhs,rhs);
    }
    
    template<class T>
    inline std::vector<sampledata<T> > operator/(std::vector<sampledata<T> > const & lhs, T const & rhs)
    {
      std::vector<T> rhs_vector;
      rhs_vector.resize(lhs.size());
      std::fill(rhs_vector.begin(),rhs_vector.end(),rhs);

      return boost::numeric::operators::operator/ <sampledata<T>, T> (lhs,rhs_vector);
    }
    
    template<class T>
    inline std::vector<sampledata<T> > operator/(T const & lhs, std::vector<sampledata<T> > const & rhs)
    {
      std::vector<T> lhs_vector;
      lhs_vector.resize(rhs.size());
      std::fill(lhs_vector.begin(),lhs_vector.end(),lhs);

      return boost::numeric::operators::operator/ <T, sampledata<T> > (lhs_vector,rhs);
    }

    template<class T>
    inline std::vector<sampledata<T> > operator/(std::vector<sampledata<T> > const & lhs, std::vector<T> const & rhs)
    {
      return boost::numeric::operators::operator/ <sampledata<T>, T> (lhs,rhs);
    } 
   
    template<class T>
    inline std::vector<sampledata<T> > operator/(std::vector<T> const & lhs, std::vector<sampledata<T> > const & rhs)
    {
      return boost::numeric::operators::operator/ <T, sampledata<T> > (lhs,rhs);
    } 

    template<class T>
    inline std::vector<T> operator-(std::vector<T> const & rhs)
    {
      return rhs * (-1.);
    }



    template<class T>
    inline static std::vector<sampledata<T> > vec_pow(std::vector<sampledata<T> > const & rhs, T const & exponent)
    {
      std::vector<T> exponent_vec;
      exponent_vec.resize(rhs.size(),exponent);
      std::vector<sampledata<T> > res;
      res.reserve(rhs.size());
      std::transform(rhs.begin(),rhs.end(),exponent_vec.begin(),std::back_inserter(res),pow<T>);
      return res;
    }

    #define IMPLEMENT_VECTOR_OF_SAMPLEDATA_FUNCTION(NAME1,NAME2) \
    template<class T> \
    std::vector<sampledata<T> > NAME1(std::vector<sampledata<T> > const & rhs) \
    { \
      std::vector<sampledata<T> > res; \
      res.reserve(rhs.size()); \
      std::transform(rhs.begin(),rhs.end(),std::back_inserter(res),NAME2<T>); \
      return res; \
    }

    IMPLEMENT_VECTOR_OF_SAMPLEDATA_FUNCTION(vec_abs,abs)
    IMPLEMENT_VECTOR_OF_SAMPLEDATA_FUNCTION(vec_sq,sq)
    IMPLEMENT_VECTOR_OF_SAMPLEDATA_FUNCTION(vec_cb,cb)
    IMPLEMENT_VECTOR_OF_SAMPLEDATA_FUNCTION(vec_sqrt,sqrt)
    IMPLEMENT_VECTOR_OF_SAMPLEDATA_FUNCTION(vec_cbrt,cbrt)
    IMPLEMENT_VECTOR_OF_SAMPLEDATA_FUNCTION(vec_exp,exp)
    IMPLEMENT_VECTOR_OF_SAMPLEDATA_FUNCTION(vec_log,log)
    IMPLEMENT_VECTOR_OF_SAMPLEDATA_FUNCTION(vec_sin,sin)
    IMPLEMENT_VECTOR_OF_SAMPLEDATA_FUNCTION(vec_cos,cos)
    IMPLEMENT_VECTOR_OF_SAMPLEDATA_FUNCTION(vec_tan,tan)
    IMPLEMENT_VECTOR_OF_SAMPLEDATA_FUNCTION(vec_asin,asin)
    IMPLEMENT_VECTOR_OF_SAMPLEDATA_FUNCTION(vec_acos,acos)
    IMPLEMENT_VECTOR_OF_SAMPLEDATA_FUNCTION(vec_atan,atan)
    IMPLEMENT_VECTOR_OF_SAMPLEDATA_FUNCTION(vec_sinh,sinh)
    IMPLEMENT_VECTOR_OF_SAMPLEDATA_FUNCTION(vec_cosh,cosh)
    IMPLEMENT_VECTOR_OF_SAMPLEDATA_FUNCTION(vec_tanh,tanh)
    IMPLEMENT_VECTOR_OF_SAMPLEDATA_FUNCTION(vec_asinh,asinh)
    IMPLEMENT_VECTOR_OF_SAMPLEDATA_FUNCTION(vec_acosh,acosh)
    IMPLEMENT_VECTOR_OF_SAMPLEDATA_FUNCTION(vec_atanh,atanh)


    // std::vector<sampledata<T> >  interchanging with  sampledata<std::vector<T> >
    template <class T>
    std::vector<sampledata<T> > obtain_vector_of_sampledata_from_vector_with_error(sampledata<std::vector<T> > vec_with_error)
    {
      std::vector<sampledata<T> > res;
      for (std::size_t index=0; index < vec_with_error.size(); ++index)
      {
        res.push_back(vec_with_error.at(index));
      }
      return res;
    }

    template <class T>
    sampledata<std::vector<T> > obtain_vector_with_error_from_vector_of_sampledata(std::vector<sampledata<T> > vec_of_sampledata)
    {
      sampledata<std::vector<T> > res;
      for (std::size_t index=0; index < vec_of_sampledata.size(); ++index)
      {
        res.push_back(vec_of_sampledata[index]);
      }
      return res;
    }

  }
}

#endif

