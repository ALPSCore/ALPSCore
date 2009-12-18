//  <value_with_error> class -- ver 2.0
//    -- value_with_error.py            (x) 
//    -- value_with_errorModule.cpp     (x)
//    -- value_with_errorModule.hpp     (o)
//
//  Written by Ping Nang MA (pingnang@itp.phys.ethz.ch) 
//    -- Institut fur Theoretische Physik, ETH Zurich 
//    -- 17/12/2009, 00:12 
//
//  Intended for ALPS 2.0 (http://alps.comp-phys.org) package 
//


#include <boost/math/special_functions.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>


namespace alps { namespace alea {

template<class T> 
class value_with_error
{
public:
  typedef T value_type;


  // constructors 
  value_with_error(value_type mean=value_type(), value_type error=value_type())
    : mean_(mean)
    , error_(error) {}


  // obtaining mean and error
  inline value_type mean() const {  return mean_;  }
  inline value_type error() const {  return error_; }


#ifdef TAMA__
  // for printing purpose...
  inline static boost::python::str print_as_str(value_with_error const & self)
  {
    return boost::python::str(boost::python::str(self.mean_) + " +/- " + boost::python::str(self.error_));
  }
#endif


  // intrinsic operations 0: ( +:pos , -:neg, abs )
  inline value_with_error& operator+()
  {
    return *this;
  }

  inline value_with_error operator-()
  {
    return value_with_error(-mean_,error_);
  }

  inline value_with_error abs()
  {
    return value_with_error(std::abs(mean_),error_);
  }


  // intrinsic operations 1: ( += , -= , *= , /= )
  inline value_with_error& operator+=(value_with_error const & rhs)
  {
    error_ =  std::sqrt(error_*error_+rhs.error_*rhs.error_); 
    mean_  += rhs.mean_;
    return *this;
  }  

  inline value_with_error& operator+=(value_type const & rhs)
  {
    mean_  += rhs;
    return *this;
  }

  inline value_with_error& operator-=(value_with_error const & rhs)
  {
    error_ =  std::sqrt(error_*error_+rhs.error_*rhs.error_);
    mean_  -= rhs.mean_;
    return *this; 
  }

  inline value_with_error& operator-=(value_type const & rhs)
  {
    mean_  -= rhs;
    return *this;
  }

  inline value_with_error& operator*=(value_with_error const & rhs)
  {
    error_ =  std::sqrt(rhs.mean_*rhs.mean_*error_*error_ + mean_*mean_*rhs.error_*rhs.error_);
    mean_  *= rhs.mean_;
    return *this;
  }

  inline value_with_error& operator*=(value_type const & rhs)
  {
    error_ *= rhs;
    mean_  *= rhs;
    return *this;
  }

  inline value_with_error& operator/=(value_with_error const & rhs)
  {
    error_ =  std::sqrt(rhs.mean_*rhs.mean_*error_*error_ + mean_*mean_*rhs.error_*rhs.error_);
    error_ /= (rhs.mean_*rhs.mean_);
    mean_  /= rhs.mean_;
    return *this;
  }

  inline value_with_error& operator/=(value_type const & rhs)
  {
    error_ /= rhs;
    mean_  /= rhs;
    return *this;
  }


  // intrinsic operations 2: ( pow , sq , cb , sqrt , cbrt , exp , log )
  inline value_with_error pow(value_type const & index)
  {
    if (index == 1.)
    {
      return *this;
    }
    else
    {
      value_type dummy = std::pow(mean_,index-1.);
      return value_with_error(dummy*mean_,std::abs(index*dummy*error_));
    }
  }

  inline value_with_error sq()
  {
    return value_with_error(mean_*mean_,std::abs(2.*mean_*error_));
  }

  inline value_with_error cb()
  {
    value_type mean_sq = mean_*mean_;
    return value_with_error(mean_sq*mean_,std::abs(3.*mean_sq*error_));
  }

  inline value_with_error sqrt()
  {
    value_type dummy = std::sqrt(mean_);
    return value_with_error(dummy,std::abs(error_/(2.*dummy)));
  }

  inline value_with_error cbrt()
  {
    value_type dummy = std::pow(mean_,1./3);
    return value_with_error(dummy,std::abs(error_/(3.*dummy*dummy)));
  }

  inline value_with_error exp()
  {
    value_type dummy = std::exp(mean_);
    return value_with_error(dummy,dummy*error_);
  }

  inline value_with_error log()
  {
    return value_with_error(std::log(mean_),std::abs(error_/mean_)); 
  }


  // intrinsic operations 3: ( sin , cos , tan , asin , acos , atan , sinh , cosh , tanh , asinh , acosh, atanh)
  inline value_with_error sin()
  {
    value_type derivative = std::cos(mean_);
    return value_with_error(std::sin(mean_),std::abs(derivative*error_));
  }

  inline value_with_error cos()
  {
    value_type derivative = -std::sin(mean_);
    return value_with_error(std::cos(mean_),std::abs(derivative*error_));
  }

  inline value_with_error tan()
  {
    value_type derivative = 1./(std::cos(mean_)*std::cos(mean_));
    return value_with_error(std::tan(mean_),std::abs(derivative*error_));
  }

  inline value_with_error sinh()
  {
    value_type derivative = std::cosh(mean_);
    return value_with_error(std::sinh(mean_),std::abs(derivative*error_));
  }

  inline value_with_error cosh()
  {
    value_type derivative = std::sinh(mean_);
    return value_with_error(std::cosh(mean_),std::abs(derivative*error_));
  }

  inline value_with_error tanh()
  {
    value_type derivative = 1./(std::cosh(mean_)*std::cosh(mean_));
    return value_with_error(std::tanh(mean_),std::abs(derivative*error_));
  }

  inline value_with_error asin()
  {
    value_type derivative = 1./std::sqrt(1. - mean_*mean_);
    return value_with_error(std::asin(mean_),std::abs(derivative*error_));
  }

  inline value_with_error acos()
  {
    value_type derivative = -1./std::sqrt(1. - mean_*mean_);
    return value_with_error(std::acos(mean_),std::abs(derivative*error_));
  }

  inline value_with_error atan()
  {
    value_type derivative = 1./(1. + mean_*mean_);
    return value_with_error(std::atan(mean_),std::abs(derivative*error_));
  }

  inline value_with_error asinh()
  {
    value_type derivative = 1./std::sqrt(mean_*mean_ + 1.);
    return value_with_error(boost::math::asinh(mean_),std::abs(derivative*error_));
  }

  inline value_with_error acosh()
  {
    value_type derivative = 1./std::sqrt(mean_*mean_ - 1.);
    return value_with_error(boost::math::acosh(mean_),std::abs(derivative*error_));
  }

  inline value_with_error atanh()
  {
    value_type derivative = 1./(1. - mean_*mean_);
    return value_with_error(boost::math::atanh(mean_),std::abs(derivative*error_));
  }



private:
  value_type  mean_;
  value_type  error_;
};


// further intrinsic operations: ( + , - , * , / )
inline value_with_error<double> operator+(value_with_error<double> lhs, value_with_error<double> const & rhs)
{
  return lhs += rhs;
}

inline value_with_error<double> operator+(value_with_error<double> lhs, value_with_error<double>::value_type const & rhs)
{
  return lhs += rhs;
}

inline value_with_error<double> operator+(value_with_error<double>::value_type const & lhs, value_with_error<double> rhs)
{
  return rhs += lhs;
}

inline value_with_error<double> operator-(value_with_error<double> lhs, value_with_error<double> const & rhs)
{
  return lhs -= rhs;
}

inline value_with_error<double> operator-(value_with_error<double> lhs, value_with_error<double>::value_type const & rhs)
{
  return lhs -= rhs;
}

inline value_with_error<double> operator-(value_with_error<double>::value_type const & lhs, value_with_error<double> rhs)
// *** pay special attention here...
{
  return -rhs + lhs;
}

inline value_with_error<double> operator*(value_with_error<double> lhs, value_with_error<double> const & rhs)
{
  return lhs *= rhs;
}

inline value_with_error<double> operator*(value_with_error<double> lhs, value_with_error<double>::value_type const & rhs)
{
  return lhs *= rhs;
}

inline value_with_error<double> operator*(value_with_error<double>::value_type const & lhs, value_with_error<double> rhs)
{
  return rhs *= lhs;
}

inline value_with_error<double> operator/(value_with_error<double> lhs, value_with_error<double> const & rhs)
{
  return lhs /= rhs;
}

inline value_with_error<double> operator/(value_with_error<double> lhs, value_with_error<double>::value_type const & rhs)
{
  return lhs /= rhs;
}

inline value_with_error<double> operator/(value_with_error<double>::value_type const & lhs, value_with_error<double> const & rhs)
// *** pay special attention here...
{
  double inverse_mean = lhs/rhs.mean();
  return value_with_error<double>(inverse_mean,std::abs(inverse_mean*rhs.error()/rhs.mean()));
}

}
}
