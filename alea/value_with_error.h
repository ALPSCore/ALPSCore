/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Ping Nang Ma <pingnang@itp.phys.ethz.ch>,
*                            Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Bela Bauer <bauerb@itp.phys.ethz.ch>
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


#include <boost/math/special_functions.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cassert>


namespace alps { namespace alea {

template<class T> 
class value_with_error
{
public:
  typedef T        value_type;

  // constructors 
  value_with_error(value_type mean=value_type(), value_type error=value_type())
    : _mean(mean)
    , _error(error) {}


  // obtaining mean and error
  inline value_type mean() const {  return _mean;  }
  inline value_type error() const {  return _error; }


  // intrinsic operations 0: ( +:pos , -:neg , abs )
  inline value_with_error<value_type>& operator+()
  {
    return *this;
  }

  inline value_with_error<value_type> operator-()
  {
    return value_with_error(-_mean,_error);
  }

  inline static value_with_error<value_type> abs(value_with_error<value_type> const& rhs)
  {
    return value_with_error<value_type>(std::abs(rhs._mean),rhs._error);
  }

  inline value_with_error<value_type>& operator= (value_with_error<value_type> const rhs)
  {
    _mean  = rhs._mean;
    _error = rhs._error;
    return *this;
  }

  inline bool operator==(value_with_error const & rhs)
  {
    return ( (_mean == rhs._mean) && (_error == rhs._error) );
  }

  // intrinsic operations 1: ( += , -= , *= , /= )
  inline value_with_error& operator+=(value_with_error const & rhs)
  {
    _error = std::sqrt(_error*_error+rhs._error*rhs._error); 
    _mean  = _mean + rhs._mean;
    return *this;
  }  

  inline value_with_error& operator+=(value_type const & rhs)
  {
    _mean  = _mean + rhs;
    return *this;
  }

  inline value_with_error& operator-=(value_with_error const & rhs)
  {
    _error = std::sqrt(_error*_error+rhs._error*rhs._error);
    _mean  = _mean - rhs._mean;
    return *this; 
  }

  inline value_with_error& operator-=(value_type const & rhs)
  {
    _mean  = _mean - rhs;
    return *this;
  }

  inline value_with_error& operator*=(value_with_error const & rhs)
  {
    _error =  std::sqrt(rhs._mean*rhs._mean*_error*_error + _mean*_mean*rhs._error*rhs._error);
    _mean  = _mean * rhs._mean;
    return *this;
  }

  inline value_with_error& operator*=(value_type const & rhs)
  {
    _error = std::abs(_error * rhs);
    _mean  = _mean  * rhs;
    return *this;
  }

  inline value_with_error& operator/=(value_with_error const & rhs)
  {
    _error = std::sqrt(rhs._mean*rhs._mean*_error*_error + _mean*_mean*rhs._error*rhs._error);
    _error = _error / (rhs._mean*rhs._mean);
    _mean  = _mean  /rhs._mean;
    return *this;
  }

  inline value_with_error& operator/=(value_type const & rhs)
  {
    _error = std::abs(_error / rhs);
    _mean  = _mean  / rhs;
    return *this;
  }


private:
  value_type  _mean;
  value_type  _error;
};


template<class T>
class value_with_error_container    // this is not designed to be as a stl container  
{
public:
  // typedef
  typedef T                           value_type;
  typedef std::size_t                 size_type;
  typedef std::size_t                 index_type;
  typedef std::vector<T>              container_type;


  value_with_error_container(container_type _mean_container=container_type(), container_type _error_container=container_type()) {
    assert(_mean_container.size() == _error_container.size());
    for(index_type index=0; index < _mean_container.size(); ++index)
    {
      _mean_container.push_back(_mean_container[index]);
      _error_container.push_back(_error_container[index]);
    }
  }

  size_type size()
  {
    assert(_mean_container.size() == _error_container.size());
    return _mean_container.size();
  }

  void append(value_with_error<value_type> const &rhs)
  {
    _mean_container.push_back(rhs.mean());
    _error_container.push_back(rhs.error());   
  }

  void extend(value_with_error_container<value_type> const rhs)
  {
    for (typename container_type::const_iterator it_mean=rhs._mean_container.begin(), it_error=rhs._error_container.begin(); it_mean != rhs._mean_container.end(); ++it_mean,++it_error)
    {
      _mean_container.push_back(*it_mean);
      _error_container.push_back(*it_error);
    }
  }

  void fill(index_type index_begin, index_type index_end, value_with_error<value_type> const & rhs)
  {
    while (size() < index_begin)
    {
      append(value_with_error<value_type>());
    }

    index_type index = index_begin;
    while (index < index_end)
    {
      if (index < size()) 
      {
        _mean_container[index]  = rhs.mean();
        _error_container[index] = rhs.error();
      }
      else
      {
        append(rhs);
      }
      ++index;
    }
  }

  void insert(index_type index, value_with_error<value_type> const & rhs)
  {
    assert(index >= 0);
    if (index <= size())
    {
      _mean_container.insert(_mean_container.begin() + index, rhs.mean());
      _error_container.insert(_error_container.begin() + index, rhs.error());
    }
    else
    {
      fill(index,index+1,rhs);
    }
  }

  void push_back(value_with_error<value_type> const &rhs)
  {
    append(rhs);
  }

  void pop_back()
  {
    assert((_mean_container.size() != 0) && (_error_container.size() != 0));
    _mean_container.pop_back();
    _error_container.pop_back();
  }

  void erase(index_type index)
  {
    if (index < size())
    {
      _mean_container.erase(_mean_container.begin()+index);
      _error_container.erase(_error_container.begin()+index);
    }
  }

  void unfill(index_type index_begin, index_type index_end)
  {
    index_type index = index_end;
    while (index > index_begin)
    {
      --index;
      erase(index);
    } 
  }
  
  void clear()
  {
    _mean_container.clear();
    _error_container.clear();
  }

  value_with_error<value_type> getitem(index_type index) const
  {
    return value_with_error<value_type>(_mean_container[index],_error_container[index]);
  }

  value_with_error_container<value_type> getslice(index_type index_begin, index_type index_end) const
  {
    value_with_error_container<value_type> result;
    for (index_type index=index_begin; index < index_end; ++index)
    {
      result.append(getitem(index));
    } 
    return result;
  }

  void setitem(index_type index, value_with_error<value_type> const & value)
  {
    fill(index,index+1,value);
  }

  void setslice(index_type index_begin, index_type index_end, value_with_error<value_type> const & value)
  {
    fill(index_begin,index_end,value);
  }

  void delitem(index_type index)
  {
    erase(index);
  }

  void delslice(index_type index_begin, index_type index_end)
  {
    unfill(index_begin,index_end);
  }

  value_with_error<value_type> operator[] (index_type index)
  {
    return value_with_error<value_type>(_mean_container[index],_error_container[index]);
  }

  inline container_type mean_container()  const { return _mean_container; }
  inline container_type error_container() const { return _error_container; }

  inline value_type mean(index_type index) const  { return _mean_container[index]; }
  inline value_type error(index_type index) const { return _error_container[index]; }


private:
  container_type _mean_container;
  container_type _error_container;
};


/////////////////////////////////////////////////////////////////////////////////
// free functions dealing with value_with_error
// (other member functions are stored within the value_with_error<T> class itself)

template<class T>
inline value_with_error<T> operator+(value_with_error<T> lhs, value_with_error<T> const & rhs)
{
  return lhs += rhs;
}

template<class T>
inline value_with_error<T> operator+(value_with_error<T> lhs, T const & rhs)
{
  return lhs += rhs;
}

template<class T>
inline value_with_error<T> operator+(T const & lhs, value_with_error<T> rhs)
{
  return rhs += lhs;
}

template<class T>
inline value_with_error<T> operator-(value_with_error<T> lhs, value_with_error<T> const & rhs)
{
  return lhs -= rhs;
}

template<class T>
inline value_with_error<T> operator-(value_with_error<T> lhs, T const & rhs)
{
  return lhs -= rhs;
}

template<class T>
inline value_with_error<T> operator-(T const & lhs, value_with_error<T> rhs)
// *** pay special attention here...
{
  return -rhs + lhs;
}

template<class T>
inline value_with_error<T> operator*(value_with_error<T> lhs, value_with_error<T> const & rhs)
{
  return lhs *= rhs;
}

template<class T>
inline value_with_error<T> operator*(value_with_error<T> lhs, T const & rhs)
{
  return lhs *= rhs;
}

template<class T>
inline value_with_error<T> operator*(T const & lhs, value_with_error<T> rhs)
{
  return rhs *= lhs;
}

template<class T>
inline value_with_error<T> operator/(value_with_error<T> lhs, value_with_error<T> const & rhs)
{
  return lhs /= rhs;
}

template<class T>
inline value_with_error<T> operator/(value_with_error<T> lhs, T const & rhs)
{
  return lhs /= rhs;
}

template<class T>
inline value_with_error<T> operator/(T const & lhs, value_with_error<T> const & rhs)
// *** pay special attention here...
{
  T inverse_mean = lhs/rhs.mean();
  return value_with_error<T>(inverse_mean,std::abs(inverse_mean*rhs.error()/rhs.mean()));
}


// intrinsic operations 2: ( pow , sq , cb , sqrt , cbrt , exp , log )

template<class T>
inline value_with_error<T> value_with_error_pow(value_with_error<T> const & rhs, T const & index)
{
  if (index == 1.) 
  {   
    return rhs;
  }   
  else
  {   
    T dummy = std::pow(rhs.mean(),index-1.);
    return value_with_error<T>(dummy*rhs.mean(),std::abs(index*dummy*rhs.error()));
  }   
}

template<class T>
inline value_with_error<T> value_with_error_sq(value_with_error<T> const & rhs)
{
  return value_with_error<T>(rhs.mean()*rhs.mean(),std::abs(2.*rhs.mean()*rhs.error()));
}

template<class T>
inline value_with_error<T> value_with_error_cb(value_with_error<T> const & rhs)
{
  double mean_sq = rhs.mean()*rhs.mean();
  return value_with_error<T>(mean_sq*rhs.mean(),std::abs(3.*mean_sq*rhs.error()));
}

template<class T>
inline value_with_error<T> value_with_error_sqrt(value_with_error<T> const & rhs)
{
  T dummy = std::sqrt(rhs.mean());
  return value_with_error<T>(dummy,std::abs(rhs.error()/(2.*dummy)));
}

template<class T>
inline value_with_error<T> value_with_error_cbrt(value_with_error<T> const & rhs)
{
  T dummy = std::pow(rhs.mean(),1./3);
  return value_with_error<T>(dummy,std::abs(rhs.error()/(3.*dummy*dummy)));
}

template<class T>
inline static value_with_error<T> value_with_error_exp(value_with_error<T> const & rhs)
{
  T dummy = std::exp(rhs.mean());
  return value_with_error<T>(dummy,dummy*rhs.error());
}

template<class T>
inline static value_with_error<T> value_with_error_log(value_with_error<T> const & rhs)
{
  return value_with_error<T>(std::log(rhs.mean()),std::abs(rhs.error()/rhs.mean()));
}


// intrinsic operations 3: ( sin , cos , tan , asin , acos , atan , sinh , cosh , tanh , asinh , acosh, atanh)
template<class T>
inline value_with_error<T> value_with_error_sin(value_with_error<T> const & rhs)
{
  T derivative = std::cos(rhs.mean());
  return value_with_error<T>(std::sin(rhs.mean()),std::abs(derivative*rhs.error()));
}

template<class T>
inline static value_with_error<T> value_with_error_cos(value_with_error<T> const & rhs)
{
  T derivative = -std::sin(rhs.mean());
  return value_with_error<T>(std::cos(rhs.mean()),std::abs(derivative*rhs.error()));
}

template<class T>
inline static value_with_error<T> value_with_error_tan(value_with_error<T> const & rhs)
{
  T derivative = 1./(std::cos(rhs.mean())*std::cos(rhs.mean()));
  return value_with_error<T>(std::tan(rhs.mean()),std::abs(derivative*rhs.error()));
}

template<class T>
inline static value_with_error<T> value_with_error_sinh(value_with_error<T> const & rhs)
{
  T derivative = std::cosh(rhs.mean());
  return value_with_error<T>(std::sinh(rhs.mean()),std::abs(derivative*rhs.error()));
}

template<class T>
inline static value_with_error<T> value_with_error_cosh(value_with_error<T> const & rhs)
{
  T derivative = std::sinh(rhs.mean());
  return value_with_error<T>(std::cosh(rhs.mean()),std::abs(derivative*rhs.error()));
}

template<class T>
inline static value_with_error<T> value_with_error_tanh(value_with_error<T> const & rhs)
{
  T derivative = 1./(std::cosh(rhs.mean())*std::cosh(rhs.mean()));
  return value_with_error<T>(std::tanh(rhs.mean()),std::abs(derivative*rhs.error()));
}

template<class T>
inline static value_with_error<T> value_with_error_asin(value_with_error<T> const & rhs)
{
  T derivative = 1./std::sqrt(1. - rhs.mean()*rhs.mean());
  return value_with_error<T>(std::asin(rhs.mean()),std::abs(derivative*rhs.error()));
}

template<class T>
inline static value_with_error<T> value_with_error_acos(value_with_error<T> const & rhs)
{
  T derivative = -1./std::sqrt(1. - rhs.mean()*rhs.mean());
  return value_with_error<T>(std::acos(rhs.mean()),std::abs(derivative*rhs.error()));
}

template<class T>
inline static value_with_error<T> value_with_error_atan(value_with_error<T> const & rhs)
{
  T derivative = 1./(1. + rhs.mean()*rhs.mean());
  return value_with_error<T>(std::atan(rhs.mean()),std::abs(derivative*rhs.error()));
}

template<class T>
inline static value_with_error<T> value_with_error_asinh(value_with_error<T> const & rhs)
{
  T derivative = 1./std::sqrt(rhs.mean()*rhs.mean() + 1.);
  return value_with_error<T>(boost::math::asinh(rhs.mean()),std::abs(derivative*rhs.error()));
}

template<class T>
inline static value_with_error<T> value_with_error_acosh(value_with_error<T> const & rhs)
{
  T derivative = 1./std::sqrt(rhs.mean()*rhs.mean() - 1.);
  return value_with_error<T>(boost::math::acosh(rhs.mean()),std::abs(derivative*rhs.error()));
}

template<class T>
inline static value_with_error<T> value_with_error_atanh(value_with_error<T> const & rhs)
{
  T derivative = 1./(1. - rhs.mean()*rhs.mean());
  return value_with_error<T>(boost::math::atanh(rhs.mean()),std::abs(derivative*rhs.error()));
}




////////////////////////////////////////////////////////////////////////////////////////
// global functions dealing with value_with_error_container<T>

// intrinsic operations : ( + , - , * , / )

#define IMPLEMENT_VECTOR_OPERATION(FUNCTION, OPERATOR) \
template<class T> \
inline value_with_error_container<T> FUNCTION(value_with_error_container<T> lhs, value_with_error_container<T> rhs) \
{ \
  assert(lhs.size() == rhs.size()); \
  value_with_error_container<T> result; \
  for (std::size_t index=0; index < lhs.size(); ++index) \
  { \
    result.push_back(lhs[index] OPERATOR rhs[index]); \
  } \
  return result; \
} \
\
template<class T> \
inline value_with_error_container<T> FUNCTION(value_with_error_container<T> lhs, T rhs) \
{ \
  value_with_error_container<T> result; \
  for (std::size_t index=0; index < lhs.size(); ++index) \
  { \
    result.push_back(lhs[index] OPERATOR rhs); \
  } \
  return result; \
} \
\
template<class T> \
inline value_with_error_container<T> FUNCTION(T lhs, value_with_error_container<T> rhs) \
{ \
  value_with_error_container<T> result; \
  for (std::size_t index=0; index < rhs.size(); ++index) \
  { \
    result.push_back(lhs OPERATOR rhs[index]); \
  } \
  return result; \
} 

IMPLEMENT_VECTOR_OPERATION(operator+,+)
IMPLEMENT_VECTOR_OPERATION(operator-,-)
IMPLEMENT_VECTOR_OPERATION(operator*,*)
IMPLEMENT_VECTOR_OPERATION(operator/,/)



// intrinsic operations : ( +:pos , -:neg , abs )
template<class T>
inline value_with_error_container<T>& operator+(value_with_error_container<T> & rhs)
{
  return rhs;
}

template<class T>
inline value_with_error_container<T> operator-(value_with_error_container<T> const & rhs)
{
  return (rhs * (-1.));
}


// intrinsic operations : (abs , pow , sq , cb , sqrt, cbrt , exp , log ,  sin , cos , tan , asin , acos , atan , sinh , cosh , tanh , asinh , acosh, atanh)

template<class T>
inline value_with_error_container<T> value_with_error_container_abs(value_with_error_container<T> rhs)
{
  value_with_error_container<T> result;
  for (std::size_t index=0; index < rhs.size(); ++index)
  {
    result.push_back(value_with_error<T>::abs(rhs[index]));
  }
  return result;
}


template<class T>
inline value_with_error_container<T> value_with_error_container_pow(value_with_error_container<T> rhs, T exponent)
{
  value_with_error_container<T> result;
  for (std::size_t index=0; index < rhs.size(); ++index)
  {
    result.push_back(value_with_error_pow<T>(rhs[index],exponent));
  }
  return result;
}

#define IMPLEMENT_VECTOR_FUNCTION(NAME,NAME_CONTAINER) \
template<class T> \
inline value_with_error_container<T> NAME_CONTAINER(value_with_error_container<T> rhs) \
{ \
  value_with_error_container<T> result; \
  for (std::size_t index=0; index < rhs.size(); ++index) \
  { \
    result.push_back(NAME<T>(rhs[index])); \
  } \
  return result; \
}

IMPLEMENT_VECTOR_FUNCTION(value_with_error_sq,value_with_error_container_sq)
IMPLEMENT_VECTOR_FUNCTION(value_with_error_cb,value_with_error_container_cb)
IMPLEMENT_VECTOR_FUNCTION(value_with_error_sqrt,value_with_error_container_sqrt)
IMPLEMENT_VECTOR_FUNCTION(value_with_error_cbrt,value_with_error_container_cbrt)
IMPLEMENT_VECTOR_FUNCTION(value_with_error_exp,value_with_error_container_exp)
IMPLEMENT_VECTOR_FUNCTION(value_with_error_log,value_with_error_container_log)
IMPLEMENT_VECTOR_FUNCTION(value_with_error_sin,value_with_error_container_sin)
IMPLEMENT_VECTOR_FUNCTION(value_with_error_cos,value_with_error_container_cos)
IMPLEMENT_VECTOR_FUNCTION(value_with_error_tan,value_with_error_container_tan)
IMPLEMENT_VECTOR_FUNCTION(value_with_error_asin,value_with_error_container_asin)
IMPLEMENT_VECTOR_FUNCTION(value_with_error_acos,value_with_error_container_acos)
IMPLEMENT_VECTOR_FUNCTION(value_with_error_atan,value_with_error_container_atan)
IMPLEMENT_VECTOR_FUNCTION(value_with_error_sinh,value_with_error_container_sinh)
IMPLEMENT_VECTOR_FUNCTION(value_with_error_cosh,value_with_error_container_cosh)
IMPLEMENT_VECTOR_FUNCTION(value_with_error_tanh,value_with_error_container_tanh)
IMPLEMENT_VECTOR_FUNCTION(value_with_error_asinh,value_with_error_container_asinh)
IMPLEMENT_VECTOR_FUNCTION(value_with_error_acosh,value_with_error_container_acosh)
IMPLEMENT_VECTOR_FUNCTION(value_with_error_atanh,value_with_error_container_atanh)

}
}
