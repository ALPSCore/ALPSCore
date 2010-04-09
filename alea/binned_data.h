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


#ifndef ALPS_ALEA_SAMPLEDATABINNING_H
#define ALPS_ALEA_SAMPLEDATABINNING_H


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

#ifdef ALPS_HAVE_PYTHON
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#endif

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
  inline std::size_t                         bin_number()                        const  { return values_.size(); }      
  inline const std::vector<value_type>&      bins()                              const  { return values_;  }
  inline const result_type                   mean()                              const;
  inline const result_type                   error()                             const;
  inline const boost::optional<result_type>& variance()                          const;
  inline const boost::optional<time_type>&   tau()                               const;
  covariance_type                            covariance(const binned_data<T>)  const;

  //set functions

  /* 
   * Question (1) :  in set_bin_size(...) , we update binsize_ even if we don't do anything ?
   */ 


  inline void set_bin_size(uint64_t binsize)        {   collect_bins((binsize-1)/binsize_+1);  binsize_=binsize;  }
  inline void set_bin_number(uint64_t bin_number)   {   collect_bins((values_.size()-1)/bin_number+1);  }

  //i/o operator
  template <class X>
  friend std::ostream& operator<< (std::ostream &out, const binned_data<X> obj);


private:
  mutable uint64_t count_;                         // total no of measurements
  mutable uint64_t binsize_;                       // no of measurements stored in a bin
  bool is_bin_changed_; // bins_have_changed                           // if yes, we have to recompute the statistics of the bins 
  mutable bool is_statistics_valid_;  // statistics_is_valid             // statistics like mean, error, tau...
  mutable bool is_jacknife_bins_filled_correctly_; // jacknife_bins_valid // 
  bool is_nonlinear_operations_performed_;  // performed_nonlinear_operations       // if yes, quantities like variance and tau are lost

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
  template <class X>
  binned_data(std::vector<X> const & bins);
#ifdef ALPS_HAVE_PYTHON
  binned_data(boost::python::object const & bins);
#endif
  binned_data(AbstractSimpleObservable<value_type> const & obs);
  template <class X, class S>
  binned_data(binned_data<X> const & my_binned_data, S s);

  template <class X>
  binned_data const & operator=(binned_data<X> const & my_binned_data);

  template <class S>
  binned_data<typename element_type<T>::type> slice(S s) const  {  return binned_data<typename element_type<T>::type>(*this,s);  }

  // comparison operators // what is that needed for?
  template <class X>
  bool operator==(binned_data<X> const & my_binned_data);

  // arithmetic operators
  
  /*
   * Question (2) : Why doesn't  template <class X>  appear in operator+=() and operator-=() <class X> ?
   */

  binned_data<T>& operator+=(binned_data<T> const & rhs);         
  binned_data<T>& operator-=(binned_data<T> const & rhs);
  template <class X>
  binned_data<T>& operator*=(binned_data<X> const & rhs);
  template <class X>
  binned_data<T>& operator/=(binned_data<X> const & rhs);

  template <class X> 
  binned_data<T>& operator+=(X const & rhs);
  template <class X> 
  binned_data<T>& operator-=(X const & rhs);
  template <class X> 
  binned_data<T>& operator*=(X const & rhs);
  template <class X> 
  binned_data<T>& operator/=(X const & rhs);

  //template<class X> void subtract_from(const X& x);
  //template<class X> void divide(const X& x);

  //binned_data<T>& operator-();


private: 
  void collect_bins(uint64_t);
  void analyze() const;
  void fill_jack() const;
  void jackknife() const;

  template <class OP>
  void transform_linear(OP op);
  template <class OP>
  void transform(OP op);
  template <class X, class OP>
  void transform(const binned_data<X>& x, OP op, double factor=1.);
};


// ### GET FUNCTIONS
template <class T>
const typename binned_data<T>::result_type binned_data<T>::mean() const
{
  if (count() == 0) 
    boost::throw_exception(NoMeasurementsError());
  analyze();  
  return mean_;
}

template <class T>
const typename binned_data<T>::result_type binned_data<T>::error() const
{
  if (count() == 0) 
    boost::throw_exception(NoMeasurementsError());
  analyze();   
  return error_;
}

template <class T>
const typename boost::optional<typename binned_data<T>::result_type>& binned_data<T>::variance() const
{
  if (count() == 0) boost::throw_exception(NoMeasurementsError());
  if (!variance_opt_)
    boost::throw_exception(std::logic_error("observable does not have variance"));
  analyze();  
  return variance_opt_;
}

template <class T>
inline
const typename boost::optional<typename binned_data<T>::time_type>& binned_data<T>::tau() const
{
  if (count() == 0) 
    boost::throw_exception(NoMeasurementsError());
  if (!tau_opt_)
    boost::throw_exception(std::logic_error("observable does not have autocorrelation information"));
  analyze();  
  return tau_opt_;
}


// ### i/o
template <class T>
std::ostream& operator<< (std::ostream &out, const binned_data<T> obj)
{
  // rename this to debug_print and write an operator<< that just prints mean +/ error (optionally convergence warning)
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
{}


template <class T>
template <class X>
binned_data<T>::binned_data(std::vector<X> const & timeseries_measurements, uint64_t const desired_bin_number)
  : count_(timeseries_measurements.size())
  , binsize_(1)
  , is_bin_changed_(false)
  , is_statistics_valid_(false)
  , is_jacknife_bins_filled_correctly_(false)
  , is_nonlinear_operations_performed_(false)
{
  if (count()) {
    values_ = timeseries_measurements;
    if (desired_bin_number >= 1)  {  set_bin_number(desired_bin_number);  }
  }
}


template <class T>
binned_data<T>::binned_data(AbstractSimpleObservable<T> const & obs)
 : count_(obs.count())
 , binsize_(obs.bin_size())
 , is_bin_changed_(false)
 , is_statistics_valid_(false)
 , is_jacknife_bins_filled_correctly_(false)
 , is_nonlinear_operations_performed_(false)
{
  if (count()) {
    mean_ = obs.mean();
    error_ = obs.error();
    if (obs.has_variance())
      variance_opt_ = obs.variance();
    if (obs.has_tau())
      tau_opt_ = obs.tau();

    for (std::size_t i = 0; i < obs.bin_number(); ++i)
      values_.push_back(obs.bin_value(i));
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
{
  mean_  = slice_value(my_binned_data.mean_, s);              
  error_ = slice_value(my_binned_data.error_, s);
  if (my_binned_data.variance_opt_) 
    variance_opt_ = slice_value(*(my_binned_data.variance_opt_), s);
  if (my_binned_data.tau_opt_)
    tau_opt_      = slice_value(*(my_binned_data.tau_opt_), s);

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
bool binned_data<T>::operator==(binned_data<X> const & my_binned_data)
{
  bool result = true;

  if (count_                             != my_binned_data.count_)                              {  result = false;  }
  if (binsize_                           != my_binned_data.binsize_)                            {  result = false;  }
  if (is_bin_changed_                    != my_binned_data.is_bin_changed_)                     {  result = false;  }
  if (is_statistics_valid_               != my_binned_data.is_statistics_valid_)                {  result = false;  }
  if (is_jacknife_bins_filled_correctly_ != my_binned_data.is_jacknife_bins_filled_correctly_)  {  result = false;  }
  if (is_nonlinear_operations_performed_ != my_binned_data.is_nonlinear_operations_performed_)  {  result = false;  }
  if (mean_                              != my_binned_data.mean_)                               {  result = false;  }
  if (error_                             != my_binned_data.error_)                              {  result = false;  }
  if (variance_opt_                      != my_binned_data.variance_opt_)                       {  result = false;  }
  if (tau_opt_                           != my_binned_data.tau_opt_)                            {  result = false;  }
  if (values_                            != my_binned_data.values_)                             {  result = false;  }
  if (jack_                              != my_binned_data.jack_)                               {  result = false;  }
  
  return result;
}


template <class T>
template <class OP>
void binned_data<T>::transform_linear(OP op)
{
  std::transform(values_.begin(), values_.end(), values_.begin(), op);
  if (is_jacknife_bins_filled_correctly_)
    std::transform(jack_.begin(), jack_.end(), jack_.begin(), op);
}


template <class T>
template <class OP>
void binned_data<T>::transform(OP op)
{
  is_statistics_valid_               = false;
  is_nonlinear_operations_performed_ = true;
  is_bin_changed_                    = true;

  variance_opt_ = boost::none_t();
  tau_opt_      = boost::none_t();

  using boost::lambda::_1;
  // CHANGE THE SEMANTICS OF THE BINS THROUGHOUT!
  // bins should contain the bin mean and not bin sum!
  // the constructor has to divide by bin size for now
  // be careful in I/O when reading old data files that contain the sum
  // will also fix Tama's problems
  std::transform(values_.begin(), values_.end(), values_.begin(), op);
  if (is_jacknife_bins_filled_correctly_)
    std::transform(jack_.begin(), jack_.end(), jack_.begin(), op);
}


template <class T>
template <class X, class OP>
void binned_data<T>::transform(const binned_data<X>& x, OP op)
{
  if ((count()==0) || (x.count()==0))
    boost::throw_exception(std::runtime_error("both observables need measurements"));
   
  is_statistics_valid_               = false;
  is_nonlinear_operations_performed_ = true;
  is_bin_changed_                    = true;

  variance_opt_ = boost::none_t();
  tau_opt_      = boost::none_t();
  
  for (uint64_t i = 0; i < bin_number(); ++i)
    values_[i] = op(values_[i], x.values_[i]);

  if (my_binned_data.is_jacknife_bins_filled_correctly_ && is_jacknife_bins_filled_correctly_)
    jack_[i] = op(jack_[i], x.jack_[i]);
}


template <class T>
binned_data<T>& binned_data<T>::operator+=(binned_data<T> const & rhs)
{
  using std::sqrt;
  using alps::numeric::sq;
  using alps::numeric::sqrt;
  using boost::numeric::operators::operator+;
  using boost::lambda::_1;
  using boost::lambda::_2;

  if(count() && rhs.count()) {
    error_ = sqrt(sq(error_)+sq(rhs.error_));
    mean_  = mean_ + rhs.mean_;

    transform(rhs, _1+_2);
  }

  return *this;
}


template <class T>
binned_data<T>& binned_data<T>::operator-=(binned_data<T> const & rhs)
{
  using std::sqrt;
  using alps::numeric::sq;
  using alps::numeric::sqrt;
  using boost::numeric::operators::operator+;
  using boost::numeric::operators::operator-;
  using boost::lambda::_1;
  using boost::lambda::_2;

  if (count() && rhs.count()) {
    error_ = sqrt(sq(error_)+sq(rhs.error_));
    mean_  = mean_ - rhs.mean_;

    transform(rhs,_1-_2);
  }

  return *this;
}


/*
 *
 * Question (3) : jack_ bins not updated correctly...
 *
 */


template <class T>
template<class X>
binned_data<T>& binned_data<T>::operator*=(binned_data<X> const & rhs)
{
  using std::sqrt;
  using alps::numeric::sq;
  using alps::numeric::sqrt;
  using boost::numeric::operators::operator+;
  using boost::numeric::operators::operator*;
  using boost::lambda::_1;
  using boost::lambda::_2;

  if(count() && rhs.count()) {
    error_ = sqrt(sq(rhs.mean_)*sq(error_) + sq(mean_)*sq(rhs.error_));
    mean_  = mean_ * rhs.mean_;

    transform(rhs,_1*_2);
  }

  return *this;
}

template <class T>
template<class X>
binned_data<T>& binned_data<T>::operator/=(binned_data<X> const & rhs)
{
  using std::sqrt;
  using alps::numeric::sq;
  using alps::numeric::sqrt;
  using boost::numeric::operators::operator+;
  using boost::numeric::operators::operator*;
  using boost::numeric::operators::operator/;
  using boost::lambda::_1;
  using boost::lambda::_2;

  if(count() && rhs.count()) {
    error_ = sqrt(sq(rhs.mean_)*sq(error_) + sq(mean_)*sq(rhs.error_));
    error_ = error_ / sq(rhs.mean_);
    mean_  = mean_  /rhs.mean_;

    transform(rhs,_1/_2);
  }
  return *this;
}


template <class T> 
template <class X>
binned_data<T>& binned_data<T>::operator+=(X const & rhs)
{
  using boost::numeric::operators::operator+;
  using boost::lambda::_1;
  if (count()) {
    mean_ = mean_ + rhs;
    transform_linear(_1+rhs);
  }
  return *this;
}

template <class T> 
template <class X>
binned_data<T>& binned_data<T>::operator-=(X const & rhs)
{
  using boost::numeric::operators::operator-;
  using boost::lambda::_1;
  if(count()) {
    mean_  = mean_ - rhs;
    transform_linear(_1-rhs);
  }
  return (*this);
}

template <class T> 
template <class X>
binned_data<T>& binned_data<T>::operator*=(X const & rhs)
{
  using std::abs;
  using alps::numeric::abs;
  using boost::numeric::operators::operator*;
  using boost::lambda::_1;
  if (count()) {
    error_ = abs(error_ * rhs);
    mean_  = mean_  * rhs;
    if(variance_opt_)
      variance_opt_ = *variance_opt_ * rhs * rhs;
    transform_linear(_1 * rhs);
  }
  return (*this);
}

template <class T> 
template <class X>
binned_data<T>& binned_data<T>::operator/=(X const & rhs)
{
  using std::abs;
  using alps::numeric::abs;
  using boost::numeric::operators::operator/;
  using boost::lambda::_1;
  if (count()) {
    error_ = abs(error_ / rhs);
    mean_  = mean_  / rhs;
    if(variance_opt_)
      variance_opt_ = *variance_opt_ /( rhs * rhs);
    transform_linear(_1/rhs);
  }
  return (*this);
}



/*
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


template <class T>
binned_data<T>& binned_data<T>::operator-()
{
  if (count()) {
    transform_linear(-_1);
  }
  return *this;
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

    variance_opt_ = boost::none_t();   // variance is lost after jacknife operation
    tau_opt_      = boost::none_t();   // tau is lost after jacknife operation
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
    for(uint64_t j = 0; j < bin_number(); ++j) { 
      jack_[0] += alps::numeric_cast<result_type>(values_[j]);     
    }  // to this point, jack_[0] = \sum_{j} x_j   (Note: x_j = (bin_value(j) / bin_size()))!!!

    for(uint64_t i = 0; i < bin_number(); ++i) {
      jack_[i+1] = jack_[0] - alps::numeric_cast<result_type>(values_[i]);
    }  // to this point, jack_[i+1] = \sum_{j != i} x_j   (Note: x_j = (bin_value(j) / bin_size()))!!!

    /*  
     *  Next, we want the following:
     *    a)  jack_[0]   =  <x>
     *    b)  jack_[i+1] =  <x_i>_{jacknife}
     */

    jack_[0] /= count_type(bin_number());      // up to this point, jack_[0] is the jacknife mean...
    for (uint64_t j = 0; j < bin_number(); ++j)  
      jack_[j+1] /= count_type(bin_number() - 1);

  }
  is_jacknife_bins_filled_correctly_ = true;
}


template <class T>
void binned_data<T>::jackknife() const
{
  fill_jack();

  if (jack_.size()) {
    // if any run is converged the errors will be OK
    resize_same_as(mean_, jack_[0]);  
    resize_same_as(error_, jack_[0]);  

    result_type unbiased_mean_;
    resize_same_as(unbiased_mean_, jack_[0]);  
    unbiased_mean_ = 0;
    unbiased_mean_ = std::accumulate(jack_.begin()+1, jack_.end(), unbiased_mean_);
    unbiased_mean_ /= count_type(bin_number());
    
    mean_ = jack_[0] - (unbiased_mean_-jack_[0])*(count_type(bin_number() - 1));   // evaluating biased mean...

    error_ = 0.;
    for (uint64_t i = 0; i < bin_number(); ++i)  {  error_ += (jack_[i+1] - unbiased_mean_)*(jack_[i+1] - unbiased_mean_);  }
    error_ /= count_type(bin_number());
    error_ *= count_type(bin_number() - 1);
    error_ = std::sqrt(error_);
  }
}


template<class T>
typename binned_data<T>::covariance_type 
binned_data<T>::covariance(const binned_data<T> obs2) const
{
  fill_jack();
  obs2.fill_jack();
  if (jack_.size() && obs2.jack_.size()) 
  {
    result_type unbiased_mean1_;
    result_type unbiased_mean2_;
    resize_same_as(unbiased_mean1_, jack_[0]);  
    resize_same_as(unbiased_mean2_, obs2.jack_[0]);  
    if (jack_.size() != obs2.jack_.size()) 
      boost::throw_exception(std::runtime_error("unequal number of bins in calculation of covariance matrix"));

    unbiased_mean1_ = 0;
    unbiased_mean2_ = 0;
    unbiased_mean1_ = std::accumulate(jack_.begin()+1, jack_.end(), unbiased_mean1_);
    unbiased_mean2_ = std::accumulate(obs2.jack_.begin()+1, obs2.jack_.end(), unbiased_mean2_);
    unbiased_mean1_ /= count_type(bin_number());
    unbiased_mean2_ /= count_type(obs2.bin_number());

    using alps::numeric::outer_product;

    covariance_type cov = outer_product(jack_[1],obs2.jack_[1]);
    for (uint32_t i = 1; i < bin_number(); ++i)
      cov += outer_product(jack_[i+1],obs2.jack_[i+1]);
    cov /= count_type(bin_number());
    cov -= outer_product(unbiased_mean1_, unbiased_mean2_);
    cov *= count_type(bin_number() - 1);

    return cov;
  } 
  else 
  {
    boost::throw_exception(std::runtime_error ("no binning information available for calculation of covariances"));
    covariance_type dummy;
    return dummy;
  }
}


#define IMPLEMENT_OPERATION(OPERATOR_NAME,OPERATOR_ASSIGN) \
template<class T> \
inline binned_data<T> OPERATOR_NAME(binned_data<T> lhs, binned_data<T> const & rhs) \
{  return lhs OPERATOR_ASSIGN rhs;  } \
\
template <class T> \
inline binned_data<T> OPERATOR_NAME(binned_data<T> lhs, T const & rhs) \
{  return lhs OPERATOR_ASSIGN rhs;  } \    \
template <class T> \
inline binned_data<std::vector<T> > OPERATOR_NAME(binned_data<std::vector<T> > lhs, typename binned_data<std::vector<T> >::element_type c
onst & rhs_elem) \
{ \
  std::vector<T> rhs(lhs.size(),rhs_elem); \
  return lhs OPERATOR_ASSIGN rhs; \
}

IMPLEMENT_OPERATION(operator+,+=)
IMPLEMENT_OPERATION(operator-,-=)
IMPLEMENT_OPERATION(operator*,*=)
IMPLEMENT_OPERATION(operator/,/=)

template <class T>
inline binned_data<T> operator+(T const & lhs, binned_data<T> rhs)
{  return rhs += lhs;  }
   
template <class T>
inline binned_data<T> operator-(T const & lhs, binned_data<T> rhs)
{ return -rhs + lhs;  }
   
template <class T>
inline binned_data<T> operator*(T const & lhs, binned_data<T> rhs)
{  return rhs *= lhs;  }
   
template <class T>
inline binned_data<T> operator/(T const & lhs, binned_data<T> const & rhs)
{
  using std::abs;
  using alps::numeric::abs;
  using boost::numeric::operators::operator*;
  using boost::numeric::operators::operator/;

  T inverse_mean = lhs/rhs.mean();

  transform(_1/_2);
  return ...
  //return binned_data<T>(inverse_mean,abs(inverse_mean*rhs.error()/rhs.mean()));
}

#define IMPLEMENT_OPERATION2(OPERATOR_NAME,OPERATOR_ASSIGN) \
template <class T> \
inline binned_data<std::vector<T> > OPERATOR_NAME(typename binned_data<std::vector<T> >::element_type const & lhs_elem, binned_data<std::vector<T> > rhs) \
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
inline binned_data<T> pow(binned_data<T> rhs, typename binned_data<T>::element_type const & exponent)
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

    transform(pow(_1,_2));
    return ... // *this;
    //return binned_data<T>(dummy*rhs.mean(),abs(exponent*dummy*rhs.error()));
  }
}

template<class T>
inline binned_data<T> sq(binned_data<T> rhs)
{
  using alps::numeric::sq;
  using std::abs;
  using alps::numeric::abs;
  using alps::numeric::operator*;
  using boost::numeric::operators::operator*;

  transform(sq(_1));
  return ... // *this  //binned_data<T>(sq(rhs.mean()),abs(2.*rhs.mean()*rhs.error()));
}
   
template<class T>
binned_data<T> cb(binned_data<T> rhs)
{
  using alps::numeric::sq;
  using std::abs;
  using alps::numeric::abs;
  using alps::numeric::operator*;
  using boost::numeric::operators::operator*;

  transform(cb(_1));
  return ... // *this  //binned_data<T>((sq(rhs.mean()))*rhs.mean(),abs(3.*(sq(rhs.mean()))*rhs.error()));
}
   
template<class T>
binned_data<T> sqrt(binned_data<T> rhs)
{
  using std::sqrt;
  using alps::numeric::sqrt;
  using std::abs;
  using alps::numeric::abs;
  using alps::numeric::operator*;
  using boost::numeric::operators::operator/;

  transform(sqrt(_1));
  return ... // *this  //binned_data<T>(sqrt(rhs.mean()),abs(rhs.error()/(2.*sqrt(rhs.mean()))));
}
   
template<class T>
binned_data<T> cbrt(binned_data<T> rhs)
{
  using alps::numeric::sq;
  using std::abs;
  using alps::numeric::abs;
  using std::pow;
  using alps::numeric::pow;
  using alps::numeric::operator*;
  using boost::numeric::operators::operator/;

  T dummy = pow(rhs.mean(),1./3);

  transform(cbrt(_1));
  return ... // *this  //binned_data<T>(dummy,abs(rhs.error()/(3.*sq(dummy))));
}
   
template<class T>
binned_data<T> exp(binned_data<T> rhs)
{
  using std::exp;
  using alps::numeric::exp;
  using boost::numeric::operators::operator*;

  T dummy = exp(rhs.mean());

  transform(exp(_1));
  return ... // *this  //binned_data<T>(dummy,dummy*rhs.error());
}
   
template<class T>
binned_data<T> log(binned_data<T> rhs)
{
  using std::log;
  using alps::numeric::log;
  using std::abs;
  using alps::numeric::abs;
  using boost::numeric::operators::operator/;

  transform(log(_1));
  return ... // *this  //binned_data<T>(log(rhs.mean()),abs(rhs.error()/rhs.mean()));
}

// ( sin, ... , atanh ) operations
template<class T>
inline binned_data<T> sin(binned_data<T> rhs)
{
  using std::abs;
  using alps::numeric::abs;
  using std::sin;
  using alps::numeric::sin;
  using std::cos;
  using alps::numeric::cos;
  using boost::numeric::operators::operator*;

  T derivative = cos(rhs.mean());

  transform(sin(_1));
  return ... // *this  //binned_data<T>(sin(rhs.mean()),abs(derivative*rhs.error()));
}
   
template<class T>
binned_data<T> cos(binned_data<T> rhs)
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

  transform(cos(_1));
  return ... // *this  //binned_data<T>(cos(rhs.mean()),abs(derivative*rhs.error()));
}
   
template<class T>
binned_data<T> tan(binned_data<T> rhs)
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

  transform(tan(_1));
  return ... // *this  //binned_data<T>(tan(rhs.mean()),abs(derivative*rhs.error()));
}
   
template<class T>
binned_data<T> sinh(binned_data<T> rhs)
{
  using std::abs;
  using alps::numeric::abs;
  using std::sinh;
  using alps::numeric::sinh;
  using std::cosh;
  using alps::numeric::cosh;
  using boost::numeric::operators::operator*;

  T derivative = cosh(rhs.mean());

  transform(sinh(_1));
  return ... // *this  //binned_data<T>(sinh(rhs.mean()),abs(derivative*rhs.error()));
}
   
template<class T>
binned_data<T> cosh(binned_data<T> rhs)
{
  using std::abs;
  using alps::numeric::abs;
  using std::sinh;
  using alps::numeric::sinh;
  using std::cosh;
  using alps::numeric::cosh;
  using boost::numeric::operators::operator*;

  T derivative = sinh(rhs.mean());


  transform(cosh(_1));
  return ... // *this  //binned_data<T>(cosh(rhs.mean()),abs(derivative*rhs.error()));
}
   
template<class T>
binned_data<T> tanh(binned_data<T> rhs)
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

  transform(tanh(_1));
  return ... // *this  //binned_data<T>(tanh(rhs.mean()),abs(derivative*rhs.error()));
}
   
template<class T>
binned_data<T> asin(binned_data<T> rhs)
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

  transform(asin(_1));
  return ... // *this  //binned_data<T>(asin(rhs.mean()),abs(derivative*rhs.error()));
}
   
template<class T>
binned_data<T> acos(binned_data<T> rhs)
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

  transform(acos(_1));
  return ... // *this  //binned_data<T>(acos(rhs.mean()),abs(derivative*rhs.error()));
}
   
template<class T>
binned_data<T> atan(binned_data<T> rhs)
{
  using std::abs;
  using alps::numeric::abs;
  using std::atan;
  using alps::numeric::atan;
  using alps::numeric::operator+;
  using boost::numeric::operators::operator*;
  using alps::numeric::operator/;

  T derivative = 1./(1. + rhs.mean()*rhs.mean());
  transform(atan(_1));

  return ... // *this  //binned_data<T>(atan(rhs.mean()),abs(derivative*rhs.error()));
}
   
template<class T>
binned_data<T> asinh(binned_data<T> rhs)
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
  transform(asinh(_1));
  return ... // *this  //binned_data<T>(asinh(rhs.mean()),abs(derivative*rhs.error()));
}
   
template<class T>
binned_data<T> acosh(binned_data<T> rhs)
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

  transform(acosh(_1));
  return ... // *this  //binned_data<T>(acosh(rhs.mean()),abs(derivative*rhs.error()));
}
   
template<class T>
binned_data<T> atanh(binned_data<T> rhs)
{
  using std::abs;
  using alps::numeric::abs;
  using boost::math::atanh;
  using alps::numeric::atanh;
  using alps::numeric::operator-;
  using boost::numeric::operators::operator*;
  using alps::numeric::operator/;


  T derivative = 1./(1. - rhs.mean()*rhs.mean());


  transform(atanh(_1));
  return ... // *this  //binned_data<T>(atanh(rhs.mean()),abs(derivative*rhs.error()));
}







} // end namespace alea
} // end namespace alps


#endif // ALPS_ALEA_SAMPLEDATABINNING_H
