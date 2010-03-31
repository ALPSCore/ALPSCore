/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Matthias Troyer <troyer@comp-phys.org>,
*                            Beat Ammon <beat.ammon@bluewin.ch>,
*                            Andreas Laeuchli <laeuchli@comp-phys.org>,
*                            Synge Todo <wistaria@comp-phys.org>,
*                            Andreas Lange <alange@phys.ethz.ch>,
*                            Ping Nang Ma <pingnang@itp.phys.ethz.ch>
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

/* $Id: simpleobsdata.h 3545 2010-03-22 10:00:00Z tamama $ */


#ifndef ALPS_ALEA_BINNED_DATA_H
#define ALPS_ALEA_BINNED_DATA_H


#include <alps/config.h>
#include <alps/parser/parser.h>
#include <alps/alea/nan.h>
#include <alps/alea/simpleobservable.h>
#include <alps/numeric/vector_functions.hpp>
#include <alps/numeric/valarray_functions.hpp>
#include <alps/numeric/vector_valarray_conversion.hpp>
#include <alps/numeric/outer_product.hpp>
#include <alps/type_traits/change_value_type.hpp>
#include <alps/type_traits/average_type.hpp>
#include <alps/type_traits/covariance_type.hpp>
#include <alps/utility/numeric_cast.hpp>
#include <alps/utility/resize.hpp>

#include <boost/config.hpp>
#include <boost/functional.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/optional/optional.hpp>

#include <iostream>
#include <numeric>
#include <vector>


namespace alps { namespace alea {


using namespace boost::lambda;


template <class T>
class binned_data {
public:
  //typedefs
  typedef T                                                                         value_type;
  typedef typename change_value_type<T,double>::type                                time_type;
  typedef std::size_t                                                               size_type;
  typedef double                                                                    count_type;
  typedef typename average_type<T>::type                                            result_type;
  typedef typename change_value_type<T,int>::type                                   convergence_type;
  typedef typename change_value_type_replace_valarray<value_type,std::string>::type label_type;
  typedef typename covariance_type<T>::type                                         covariance_type;

  //get functions
  inline uint64_t                            count()                             const  { return is_bin_changed_ ? (bin_size()*bin_number() == 0 ? count_ : bin_size()*bin_number()) : count_;}
  inline uint64_t                            bin_size()                          const  { return binsize_;}
  inline uint64_t                            bin_number()                        const  { return values_.size(); }      
  inline const value_type&                   bin_value(uint64_t i)               const  { return values_[i];  }
  inline const result_type&                  mean()                              const;
  inline const result_type&                  error()                             const;
  inline const boost::optional<result_type>& variance()                          const;
  inline const boost::optional<time_type>&   tau()                               const;
  //covariance_type                            covariance(const binned_data<T>)  const;

  //set functions
  inline void set_bin_size(uint64_t binsize)        {   collect_bins((binsize-1)/binsize_+1);  binsize_=binsize;  }
  inline void set_bin_number(uint64_t bin_number)   {   collect_bins((values_.size()-1)/bin_number+1);  }

  //i/o operator
  template <class X>
  friend std::ostream& operator<< (std::ostream &out, const binned_data<X> obj);


private:
  mutable uint64_t count_;                         // total no of measurements
  mutable uint64_t binsize_;                       // no of measurements stored in a bin
  bool is_bin_changed_;                            // if yes, we have to recompute the statistics of the bins 
  mutable bool is_statistics_valid_;               // statistics like mean, error, tau...
  mutable bool is_jacknife_bins_filled_correctly_; // 
  bool is_nonlinear_operations_performed_;         // if yes, quantities like variance and tau are lost

  mutable result_type                  mean_;         // valid only if (is_statistics_valid_)
  mutable result_type                  error_;        // valid only if (is_statistics_valid_)
  mutable boost::optional<result_type> variance_opt_; // valid only if (is_statistics_valid_)
  mutable boost::optional<time_type>   tau_opt_;      // valid only if (is_statistics_valid_)

  mutable std::vector<value_type>  values_;           // bins
  mutable std::vector<result_type> jack_;             // jacknife bins: [0] -- all bins , [1] -- bin [1] being ommitted...


public:
  template <class X>
  friend class binned_data;

  binned_data();
  binned_data(AbstractSimpleObservable<value_type> const & obs);
  template <class X, class S>
  binned_data(binned_data<X> const & my_binned_data, S s);

  template <class X>
  binned_data const & operator=(binned_data<X> const & my_binned_data);

  template <class S>
  binned_data<typename element_type<T>::type> slice(S s) const  {  return binned_data<typename element_type<T>::type>(*this,s);  }


/*
  // unary operation: negation
  binned_data<T>& operator-();

  // operations with constant
  template <class X> binned_data<T>& operator+=(X);
  template <class X> binned_data<T>& operator-=(X);
  template <class X> binned_data<T>& operator*=(X);
  template <class X> binned_data<T>& operator/=(X);
  template<class X> void subtract_from(const X& x);
  template<class X> void divide(const X& x);
  
  // operations with another observable
  binned_data<T>& operator+=(const binned_data<T>&);
  binned_data<T>& operator-=(const binned_data<T>&);
  template <class X>
  binned_data<T>& operator*=(const binned_data<X>&);
  template <class X>
  binned_data<T>& operator/=(const binned_data<X>&);

  template <class OP> void transform(OP op);
*/


protected: 
  void collect_bins(uint64_t);
  void analyze() const;
  void fill_jack() const;
  void jackknife() const;

/*
  template <class X, class OP>
  void transform(const binned_data<X>& x, OP op, double factor=1.);

  template <class OP> 
  void transform_linear(OP op);
*/

};


// ### GET FUNCTIONS
template <class T>
const typename binned_data<T>::result_type& binned_data<T>::mean() const
{
  if (count() == 0) boost::throw_exception(NoMeasurementsError());
  analyze();  return mean_;
}

template <class T>
const typename binned_data<T>::result_type& binned_data<T>::error() const
{
  if (count() == 0) boost::throw_exception(NoMeasurementsError());
  analyze();   return error_;
}

template <class T>
const typename boost::optional<typename binned_data<T>::result_type>& binned_data<T>::variance() const
{
  if (count() == 0) boost::throw_exception(NoMeasurementsError());
  if (!variance_opt_)
    boost::throw_exception(std::logic_error("observable does not have variance"));
  analyze();  return variance_opt_;
}

template <class T>
inline
const typename boost::optional<typename binned_data<T>::time_type>& binned_data<T>::tau() const
{
  if (count() == 0) boost::throw_exception(NoMeasurementsError());
  if (!tau_opt_)
    boost::throw_exception(std::logic_error("observable does not have autocorrelation information"));
  analyze();  return tau_opt_;
}


// ### i/o
template <class T>
std::ostream& operator<< (std::ostream &out, const binned_data<T> obj)
{
  using std::operator<<;
  using alps::numeric::operator<<;

  out << "\ncount:\t"                << obj.count_
      << "\nbinsize:\t"              << obj.binsize_
      << "\nchanged:\t"              << obj.is_bin_changed_
      << "\nvalid:\t"                << obj.is_statistics_valid_
      << "\njack_valid:\t"           << obj.is_jacknife_bins_filled_correctly_
      << "\nnonlinear_operations:\t" << obj.is_nonlinear_operations_performed_
      << "\nmean:\t"                 << obj.mean_
      << "\nerror:\t"                << obj.error_;
  if (obj.variance_opt_)
    out << "\nvariance:\t"             << *(obj.variance_opt_);
  if (obj.tau_opt_)
    out  << "\ntau:\t"                  << *(obj.tau_opt_);
  out << "\nvalues:\t"               << obj.values_
      << "\njack:\t"                 << obj.jack_;

  return out;
}


// ### CONSTRUCTORS
template <class T>
binned_data<T>::binned_data()
  : count_(0)
  , binsize_(0)
  , is_bin_changed_(false)
  , is_statistics_valid_(true)
  , is_jacknife_bins_filled_correctly_(true)
  , is_nonlinear_operations_performed_(false)
  , mean_() 
  , error_() 
  , variance_opt_() 
  , tau_opt_() 
  , values_() 
  , jack_() 
{}


template <class T>
binned_data<T>::binned_data(AbstractSimpleObservable<T> const & obs)
 : count_(obs.count())
 , binsize_(obs.bin_size())
 , is_bin_changed_(false)
 , is_statistics_valid_(false)
 , is_jacknife_bins_filled_correctly_(false)
 , is_nonlinear_operations_performed_(false)
 , mean_()
 , error_()
 , variance_opt_()
 , tau_opt_()
 , values_()
 , jack_()
{
  if (count()) {
    assign(mean_,  obs.mean());
    assign(error_, obs.error());
    if (obs.has_variance())  {  assign(variance_opt_,obs.variance());  }
    if (obs.has_tau())       {  assign(tau_opt_,     obs.tau());  }

    for (uint64_t i = 0; i < obs.bin_number(); ++i)  {  values_.push_back(obs.bin_value(i));  }
  }
}


template <class T>
template <class X, class S>
binned_data<T>::binned_data(binned_data<X> const & my_binned_data, S s)
  : count_(my_binned_data.count_)
  , binsize_(my_binned_data.binsize_)
  , is_bin_changed_(my_binned_data.is_bin_changed_)
  , is_statistics_valid_(my_binned_data.is_statistics_valid_)
  , is_jacknife_bins_filled_correctly_(my_binned_data.is_jacknife_bins_filled_correctly_)
  , is_nonlinear_operations_performed_(my_binned_data.is_nonlinear_operations_performed_)
  , mean_()              
  , error_()
  , variance_opt_()  
  , tau_opt_()
  , values_()
  , jack_()
{
  mean_  = slice_value(my_binned_data.mean_, s);              
  error_ = slice_value(my_binned_data.error_, s);
  if (my_binned_data.variance_opt_)   {  variance_opt_ = slice_value(*(my_binned_data.variance_opt_), s);  }
  if (my_binned_data.tau_opt_)        {  tau_opt_      = slice_value(*(my_binned_data.tau_opt_), s);  }

  values_.reserve(my_binned_data.values_.size());
  std::transform(my_binned_data.values_.begin(), my_binned_data.values_.end(), std::back_inserter(values_), boost::bind2nd(slice_it<X>(),s));

  if (my_binned_data.is_jacknife_bins_filled_correctly_)
  {
    jack_.reserve(my_binned_data.jack_.size());
    std::transform(my_binned_data.jack_.begin(), my_binned_data.jack_.end(), std::back_inserter(jack_), boost::bind2nd(slice_it<X>(),s));
  }
}


template <class T>
template <class X>
binned_data<T> const & binned_data<T>::operator=(binned_data<X> const & my_binned_data)
{
  count_                             = my_binned_data.count_;
  binsize_                           = my_binned_data.binsize_;
  is_bin_changed_                    = my_binned_data.is_bin_changed_;
  is_statistics_valid_               = my_binned_data.is_statistics_valid_;
  is_jacknife_bins_filled_correctly_ = my_binned_data.is_jacknife_bins_filled_correctly_;
  is_nonlinear_operations_performed_ = my_binned_data.is_nonlinear_operations_performed_;
  assign(mean_,my_binned_data.mean_);
  assign(error_,my_binned_data.error_);
  assign(variance_opt_,my_binned_data.variance_opt_);
  assign(tau_opt_,my_binned_data.tau_opt_);
  values_                            = my_binned_data.values_;
  jack_                              = my_binned_data.jack_;

  return *this;
}


/*
template <class T> 
binned_data<T>& binned_data<T>::operator+=(const binned_data<T>& x)
{
  using std::sqrt;
  if(count() && x.count()) {
    mean_ += x.mean();
    
    // leave this right now...
    error_ *= error_;
    error_ += x.error()*x.error();
    error_ = sqrt(error_);
    // error_=sqrt(error_*error_+x.error()*x.error());
  }
  transform(x, _1+_2);
  return (*this);
}

template <class T>
binned_data<T>& binned_data<T>::operator-=(const binned_data<T>& x)
{
  using std::sqrt;
  if(count() && x.count()) {
    mean_ -= x.mean();
    error_ *= error_;
    error_ += x.error()*x.error();
    error_ = sqrt(error_);
    //error_=sqrt(error_*error_+x.error()*x.error());
  }
  transform(x,_1-_2);
  return (*this);
}

template <class T>
template<class X>
binned_data<T>& binned_data<T>::operator*=(const binned_data<X>& x)
{
  using std::sqrt;
  if(count() && x.count()) {
    error_=error()*error();
    error_*=x.mean()*x.mean();
    typename binned_data<X>::result_type tmp(x.error());
    tmp *=tmp;
    result_type tmp2(mean_);
    tmp2 *= tmp2*tmp;
    error_ += tmp2;
    error_ = sqrt(error_);
    mean_ *= x.mean();
    //error_=sqrt(error()*error()*x.mean()*x.mean()+mean()*mean()*x.error()*x.error());
  }
  transform(x,_1*_2,1./x.bin_size());
  return (*this);
}

template <class T>
template<class X>
binned_data<T>& binned_data<T>::operator/=(const binned_data<X>& x)
{
  using std::sqrt;
  if(count() && x.count()) {
    error_=error()*error();
    typename binned_data<X>::result_type m(x.mean());
    m *=m;
    typename binned_data<X>::result_type tmp(x.error());
    tmp *=m;
    tmp *=x.error()*m;
    error_ +=tmp;
    error_ /=m;
    error_ = sqrt(error_);
    mean_ /= x.mean();
    //error_ = sqrt((error()*error()+mean()*mean()*x.error()*x.error()/x.mean()/x.mean())/x.mean()/x.mean());
  }
  transform(x,_1/_2,x.bin_size());
  return (*this);
}

template <class T>
template <class X, class OP>
void binned_data<T>::transform(const binned_data<X>& x, OP op, double factor)
{
  if ((count()==0) || (x.count()==0))
    boost::throw_exception(std::runtime_error("both observables need measurements"));
    
  if(!is_jacknife_bins_filled_correctly_) fill_jack();
  if(!x.is_jacknife_bins_filled_correctly_) x.fill_jack();

  is_statistics_valid_ = false;
  is_nonlinear_operations_performed_ = true;
  is_bin_changed_ = true;
  
  tau_      = boost::detail::none_t;
  variance_ = boost::detail::none_t;
  
  
  for (uint64_t i = 0; i < bin_number(); ++i)
    values_[i] = op(values_[i], x.values_[i])*factor;
  for (uint64_t i = 0; i < jack_.size(); ++i)
    jack_[i] = op(jack_[i], x.jack_[i]);
  
     
}


template <class T> 
template <class OP>
void binned_data<T>::transform_linear(OP op)
{
  mean_ = op(mean_);
  std::transform(values_.begin(), values_.end(), values_.begin(), op);
  fill_jack();
  std::transform(jack_.begin(), jack_.end(), jack_.begin(), op);
}


// *** change as the one above...
//
template <class T> 
template <class OP>
void binned_data<T>::transform(OP op)
{
  is_statistics_valid_ = false;
  is_nonlinear_operations_performed_ = true;
  is_bin_changed_ = true;
  has_variance_ = false;
  has_tau_ = false;
  values2_.clear();
  has_minmax_ = false;
 
  // we should store bin means instead of bin sums in values...
  
  std::transform(values_.begin(), values_.end(), values_.begin(),_1/double(bin_size()));
  std::transform(values_.begin(), values_.end(), values_.begin(), op);
  std::transform(values_.begin(), values_.end(), values_.begin(),_1*double(bin_size()));
  fill_jack();
  std::transform(jack_.begin(), jack_.end(), jack_.begin(), op);
}
//
// ***
  
  
  
template <class T>
binned_data<T>& binned_data<T>::operator-()
{
  if (count()) {
    transform_linear(-_1);
  }
  return *this;
}

template <class T> template <class X>
binned_data<T>& binned_data<T>::operator+=(X x)
{
  if (count()) {
    transform_linear(_1 + x);
  }
  return *this;
}

template <class T> template <class X>
binned_data<T>& binned_data<T>::operator-=(X x)
{
  if(count()) {
    if (has_minmax_) {
      min_ -= x;
      max_ -= x;
    }
    transform_linear(_1-x);
    for (int i=0;i<values2_.size();++i)
      values2_[i] += -2.*values_[i]*x+x*x;
  }
  return (*this);
}

template <class T> template <class X>
void binned_data<T>::subtract_from(const X& x)
{
  if (count()) {
    if(has_minmax_) {
      min_ = x-max_;
      max_ = x-min_;
    }
    transform_linear(x-_1);
    for (int i=0;i<values2_.size();++i)
      values2_[i] += -2.*values_[i]*x+x*x;
  }
}

template <class T> template <class X>
binned_data<T>& binned_data<T>::operator*=(X x)
{
  if (count()) {
    error_ *= x;
    if(has_variance_)
      variance_ *= x*x;
    has_minmax_ = false;
    
    transform_linear(_1*x);
    std::transform(values2_.begin(),values2_.end(),values2_.begin(),_1*(x*x));
  }
  return (*this);
}

template <class T> template <class X>
binned_data<T>& binned_data<T>::operator/=(X x)
{
  if (count()) {
    error_ /= x;
    if(has_variance_)
      variance_ /= x*x;
    has_minmax_ = false;
    
    transform_linear(_1/x);
    std::transform(values2_.begin(),values2_.end(),values2_.begin(),_1/(x*x));
  }
  return (*this);
}

template <class T> template <class X>
void binned_data<T>::divide(const X& x)
{
  if (count()) {
    error_ = x *error_/mean_/mean_;
    has_minmax_ = false;
    has_variance_ = false;
    values2_.clear();
    has_tau_ = false;
    is_nonlinear_operations_performed_ = true;
    is_bin_changed_ = true;
    mean_ = x/mean_;
    std::transform(values_.begin(), values_.end(), values_.begin(), (x*bin_size()*bin_size())/_1);
    fill_jack();
    std::transform(jack_.begin(), jack_.end(), jack_.begin(), x/_1);
  }
}

*/


template <class T>
void binned_data<T>::collect_bins(uint64_t howmany)
{
  if (is_nonlinear_operations_performed_)
    boost::throw_exception(std::runtime_error("cannot change bins after nonlinear operations"));
  if (values_.empty() || howmany <= 1) return;

  // fill bins
  uint64_t newbins = values_.size() / howmany;
  for (uint64_t i = 0; i < newbins; ++i) {
    values_[i] = values_[howmany * i];
    for (uint64_t j = 1; j < howmany; ++j) {
      values_[i] += values_[howmany * i + j];
    }
  }
  values_.resize(newbins);
  binsize_ *= howmany;

  is_bin_changed_                    = true;
  is_jacknife_bins_filled_correctly_ = false;
  is_statistics_valid_               = false;
}


template <class T>
void binned_data<T>::analyze() const
{
  if (is_statistics_valid_) return;

  if (bin_number())
  {
    count_ = bin_size()*bin_number();

    // calculate mean and error
    jackknife();

    /*
     * Question (1): A) In <simpleobsdata.h>, we have has_variance_ and has_tau_ to be set false.
     *                  What about here?
     *               B) How to reset a boost optional variable?  -- can anyone have quick and good solution?
     */

  }
  is_statistics_valid_ = true;
}

  
template <class T>
void binned_data<T>::fill_jack() const
{
  // build jackknife data structure
  if (bin_number() && !is_jacknife_bins_filled_correctly_) {
    if (is_nonlinear_operations_performed_)
      boost::throw_exception(std::runtime_error("Cannot rebuild jackknife data structure after nonlinear operations"));

    jack_.clear();
    jack_.resize(bin_number() + 1);

    // Order-N initialization of jackknife data structure
    resize_same_as(jack_[0], bin_value(0));     
    for(uint64_t i = 0; i < bin_number(); ++i) 
      jack_[0] += alps::numeric_cast<result_type>(bin_value(i)) / count_type(bin_size());
    for(uint64_t i = 0; i < bin_number(); ++i) {
      resize_same_as(jack_[i+1], jack_[0]);
      result_type tmp(alps::numeric_cast<result_type>(bin_value(i)));
      tmp /= count_type(bin_size());
      jack_[i+1] = jack_[0]
          - tmp;
//        - (alps::numeric_cast<result_type>(bin_value(i)) / count_type(bin_size()));
      jack_[i+1] /= count_type(bin_number() - 1);
    }
    jack_[0] /= count_type(bin_number());
  }
  is_jacknife_bins_filled_correctly_ = true;
}


template <class T>
void binned_data<T>::jackknife() const
{
  fill_jack();

  if (jack_.size()) {
    // if any run is converged the errors will be OK
    result_type rav;
    resize_same_as(mean_, jack_[0]);  
    resize_same_as(error_, jack_[0]);  
    resize_same_as(rav, jack_[0]);  
    unsigned int k = jack_.size()-1;

    rav = 0;
    rav = std::accumulate(jack_.begin()+1, jack_.end(), rav);
    rav /= count_type(k);
    
    mean_ = jack_[0] - (rav - jack_[0]) * count_type(k - 1);

    // TODO check these equations
    error_ = 0.0;
    for (unsigned int i = 1; i < jack_.size(); ++i)
      error_ += (jack_[i] - rav) * (jack_[i] - rav);
      //error_ += jack_[i] * jack_[i];
    
    error_/=count_type(k);
    //error_-= rav * rav;
    //error_ = (error_ / count_type(k) - rav * rav);
    error_ *= count_type(k - 1);
    error_ = std::sqrt(error_);
  }
}


/*
template<class T>
typename binned_data<T>::covariance_type 
binned_data<T>::covariance(const binned_data<T> obs2) const
{
  fill_jack();
  obs2.fill_jack();
  if (jack_.size() && obs2.jack_.size()) {
    result_type rav1;
    result_type rav2;
    resize_same_as(rav1, jack_[0]);  
    resize_same_as(rav2, obs2.jack_[0]);  
    if (jack_.size() != obs2.jack_.size()) 
      boost::throw_exception(std::runtime_error("unequal number of bins in calculation of covariance matrix"));
    uint32_t k = jack_.size()-1;

    rav1 = 0;
    rav2 = 0;
    rav1 = std::accumulate(jack_.begin()+1, jack_.end(), rav1);
    rav2 = std::accumulate(obs2.jack_.begin()+1, obs2.jack_.end(), rav2);
    rav1 /= count_type(k);
    rav2 /= count_type(k);
    
    covariance_type cov = numeric::outer_product(jack_[1],obs2.jack_[1]);
    for (uint32_t i = 2; i < jack_.size(); ++i)
      cov += numeric::outer_product(jack_[i],obs2.jack_[i]);
    
    cov/=count_type(k);
    cov-= numeric::outer_product(rav1, rav2);
    cov *= count_type(k - 1);
    return cov;
  } else {
    boost::throw_exception(std::runtime_error ("no binning information available for calculation of covariances"));
    covariance_type dummy;
    return dummy;
  }
}

*/

} // end namespace alea
} // end namespace alps


// 1) get the operators +,-,*,/,.... and functions from simpleobseval.h
// 2) get std::vector arithmetic operations from boost::accumulator library...
// 3) define usual function (eg. sin, cos...) on std::vector in a seprate header...



#endif // ALPS_ALEA_BINNED_DATA_H
