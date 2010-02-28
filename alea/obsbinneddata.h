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

/* $Id: simpleobsdata.h 3545 2009-12-17 14:25:36Z gamperl $ */

//
// This is not completed yet...
//


#ifndef ALPS_ALEA_OBSDATABIN_H
#define ALPS_ALEA_OBSDATABIN_H

#include <alps/config.h>
#include <alps/parser/parser.h>
#include <alps/alea/nan.h>
#include <alps/alea/simpleobservable.h>
#include <alps/numeric/vector_functions.hpp>
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
  template <class X>
  friend class binned_data;    // significance? ... (4)

  typedef T                                              value_type;
  typedef typename change_value_type<T,double>::type     time_type;
  typedef std::size_t                                    size_type;
  typedef double                                         count_type;
  typedef typename average_type<T>::type                 result_type;
  typedef typename change_value_type<T,int>::type        convergence_type;
  typedef typename change_value_type_replace_valarray<value_type,std::string>::type label_type;
  typedef typename covariance_type<T>::type              covariance_type;

  // constructors
  binned_data();
  template <class U, class S>
  //binned_data(const binned_data<U>& x, S s);
  //binned_data(const AbstractSimpleObservable<value_type>& obs);



  //
  // ...(7) Shouldn't we replace uint64_t by std::size_t  ?
  //

  //uint64_t count() const { return changed_ ? (bin_size()*bin_number() == 0 ? count_ : bin_size()*bin_number()) : count_;}
  uint64_t count() const { return count_; }
  inline const result_type& mean() const;
  inline const result_type& error() const;
  //inline const convergence_type& converged_errors() const; // ... (8)
  inline const boost::optional<result_type>& variance() const;
  inline const boost::optional<time_type>& tau() const;

/*    
  covariance_type covariance(const binned_data<T>) const;

  uint64_t bin_size() const { return binsize_;}
  uint64_t bin_number() const { return values_.size()-discardedbins_;}
  const value_type& bin_value(uint64_t i) const {
    return values_[i+discardedbins_];
  }

  template <class S>
  binned_data<typename element_type<T>::type> slice(S s) const
  {
    return binned_data<typename element_type<T>::type>(*this,s);
  }
*/

/*
#ifdef ALPS_HAVE_HDF5
  void serialize(hdf5::oarchive & ar) const;
  void serialize(hdf5::iarchive & ar) const;
#endif
*/

/*
  inline void set_bin_size(uint64_t);
  inline void set_bin_number(uint64_t);
 
  // collect information from many data objects
  void collect_from(const std::vector<binned_data<T> >& runs);
*/

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


//protected: // ... (9) commented out in the debugging process
  void analyze() const;

/*
  void jackknife() const;
  void fill_jack() const;
*/

/*
  template <class X, class OP>
  void transform(const binned_data<X>& x, OP op, double factor=1.);

  template <class OP> 
  void transform_linear(OP op);
*/

//private: // ... (9) commented out in the debugging process 

  /// no of measurements
  mutable uint64_t count_;          

  /// size of bin
  mutable uint64_t binsize_;
  
  /// changed the bins -> mean and error has to be calculated again.
  bool changed_;
  
  /// calculated mean, varaince and error is correct.
  mutable bool valid_;
  
  /// jackknife bins are filled correctly.
  mutable bool jack_valid_;
  
  /// a nonlinear operation has been performed, tau and varaince are lost
  bool nonlinear_operations_; // nontrivial operations
  
    
  mutable result_type mean_;     // valid only if (valid_)
  mutable result_type error_;    // valid only if (valid_)
  mutable boost::optional<result_type> variance_opt_; // valid only if (valid_ && has_variance_)
  mutable boost::optional<time_type>   tau_opt_;        // valid only if (valid_ && has_tau_)


  
  /// bins
  mutable std::vector<value_type> values_;
  
  /// jackknife bins: [0] - all bins; [1] - [n] -- always 1 bin omitted.
  mutable std::vector<result_type> jack_;  
};

  


template <class T>
binned_data<T>::binned_data()
  : count_(0)
  , binsize_(0)
  , changed_(false)
  , valid_(true)
  , jack_valid_(true)
  , nonlinear_operations_(false)
  , mean_() 
  , error_() 
  , variance_opt_() 
  , tau_opt_() 
  , values_() 
  , jack_() 
{}



/*  
template <class T>
template <class U, class S>
inline
binned_data<T>::binned_data(const binned_data<U>& x, S s)
 : count_(x.count_),          
   binsize_(x.binsize_),
   changed_(x.changed_),
   valid_(x.valid_),
   jack_valid_(x.jack_valid_),
   nonlinear_operations_(x.nonlinear_operations_),
   mean_(slice_value(x.mean_, s)),
   error_(slice_value(x.error_, s)),

  
   /// ***
   /// has to change... Boost documentation...
   ///
   variance_(x.variance_ ? 
               boost::optional<result_type>(slice_value(*x.variance_, s)) : 
               boost::optional<result_type>()),
   tau_(has_tau_ ?   **COPY**
               slice_value(x.tau_, s) : time_type()),
   /// 
   /// ***
  
  
   values_(x.values_.size()),  
   jack_(x.jack_.size())
   
{
  values_.resize(x.values_.size());
  std::transform(x.values_.begin(), x.values_.end(), values_.begin(),
                 boost::bind2nd(slice_it(),s));
    if (jack_valid_) {
    jack_.resize(x.jack_.size());
    std::transform(x.jack_.begin(), x.jack_.end(), jack_.begin(),
                   boost::bind2nd(slice_it(),s));
  }
}


template <class T>
binned_data<T>::binned_data(const AbstractSimpleObservable<T>& obs)
 : count_(obs.count()),
   binsize_(obs.bin_size()),
   changed_(false),
   valid_(false),
   jack_valid_(false),
   nonlinear_operations_(false),
   mean_(), 
   error_(), 
   variance_(), 
   tau_(), 
   values_(), 
   jack_()
{
  if (count()) {
    mean_ = obs.mean();
   error_ = obs.error());
    if (obs.has_variance())
      variance_ = obs.variance();
    if (obs.has_tau())
      tau_ = obs.tau();

    // ** TODO: convert changes valarray to vector and leaves the rest untouched! **
    for (uint64_t i = 0; i < obs.bin_number(); ++i)
      values_.push_back(convert(obs.bin_value(i)));
    
    /// ***
    
  }
}
*/





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
    
  if(!jack_valid_) fill_jack();
  if(!x.jack_valid_) x.fill_jack();

  valid_ = false;
  nonlinear_operations_ = true;
  changed_ = true;
  
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
  valid_ = false;
  nonlinear_operations_ = true;
  changed_ = true;
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
//***
  
  
  
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
    nonlinear_operations_ = true;
    changed_ = true;
	mean_ = x/mean_;
    std::transform(values_.begin(), values_.end(), values_.begin(), (x*bin_size()*bin_size())/_1);
    fill_jack();
    std::transform(jack_.begin(), jack_.end(), jack_.begin(), x/_1);
  }
}

*/

  
  
/// Merges the bins...

/*
  
template <class T>
void binned_data<T>::collect_from(const std::vector<binned_data<T> >& runs)
{
  bool got_data = false;

  count_ = 0;

  changed_ = false;
  valid_ = false;
  jack_valid_ = false;
  nonlinear_operations_ = false;


  values_.clear();
  jack_.clear();

  // find smallest and largest bin sizes
  uint64_t minsize = std::numeric_limits<uint64_t>::max BOOST_PREVENT_MACRO_SUBSTITUTION ();
  uint64_t maxsize = 0;
  for (typename std::vector<binned_data<T> >::const_iterator
         r = runs.begin(); r != runs.end(); ++r) {
    if (r->count()) {
      if (r->bin_size() < minsize) minsize = r->bin_size();
      if (r->bin_size() > maxsize) maxsize = r->bin_size();
    }
  }

  binsize_ = maxsize;
  
  for (typename std::vector<binned_data<T> >::const_iterator
         r = runs.begin(); r != runs.end(); ++r) {
    if (r->count()) {
      if (!got_data) {
        // initialize
        jack_valid_ = true;

    
        nonlinear_operations_ = r->nonlinear_operations_;
        changed_ = r->changed_;
        mean_ = r->mean_;
        error_= r->error_;
        variance_=r->variance_;
        tau_=r->tau_;
        count_ = r->count();

        if (r->bin_size() == maxsize) {
          r->fill_jack();
          values_ = r->values_;
          jack_ = r->jack_;
        } else {
          binned_data<T> tmp(*r);
          tmp.set_bin_size(maxsize);
          tmp.fill_jack();
          values_ = tmp.values_;
          jack_ = tmp.jack_;
        }
        got_data=true;
      } else {
        // add
        jack_valid_ = false;

        nonlinear_operations_ = nonlinear_operations_ || r->nonlinear_operations_;
        changed_ = changed_ && r->changed_;
        mean_ = (double(count_)*mean_+double(r->count_)*r->mean_)
                / double(count_ + r->count_);
        using std::sqrt;

        error_ = sqrt(double(count_)*double(count_)*error_*error_
                      +double(r->count_)*double(r->count_)*r->error_*r->error_)
          / double(count_ + r->count_);
        if(variance_)
          variance_ = (double(count_)* *variance_+double(r->count_)* *(r->variance_))
            / double(count_ + r->count_);
        if(tau_)
          tau_ = (double(count_)* *tau_+double(r->count_)* *(r->tau_))
            / double(count_ + r->count_);

        count_ += r->count();

        if (r->bin_size() == maxsize) {
          std::copy(r->values_.begin(), r->values_.end(),
                    std::back_inserter(values_));
          std::copy(r->values2_.begin(), r->values2_.end(),
                    std::back_inserter(values2_));
        } else {
          binned_data<T> tmp(*r);
          tmp.set_bin_size(maxsize);
          std::copy(tmp.values_.begin(), tmp.values_.end(),
                    std::back_inserter(values_));
          std::copy(tmp.values2_.begin(), tmp.values2_.end(),
                    std::back_inserter(values2_));
        }
      }
    }
  }

  analyze();
}

*/


/*
#ifdef ALPS_HAVE_HDF5
template <typename T> void binned_data<T>::serialize(hdf5::oarchive & ar) const {
  ar << make_pvp("count", count_);
  if (valid_)
  { 
    ar << make_pvp("mean/value", mean_) << make_pvp("mean/error", error_);
    if (variance_)
      ar << make_pvp("variance/value", *variance_);
    if (tau_)
      ar << make_pvp("tau/value", *tau_);
    ar << make_pvp("timeseries/data", values_) << make_pvp("timeseries/data/@binningtype", "linear");
    if (jack_valid_)
      ar << make_pvp("jacknife/data", jack_) << make_pvp("jacknife/data/@binningtype", "linear");
  }
}

template <typename T> void binned_data<T>::serialize(hdf5::iarchive & ar) const {}
#endif
*/
  
/*  
template <class T>
void binned_data<T>::fill_jack() const
{
  // build jackknife data structure
  if (bin_number() && !jack_valid_) {
    if (nonlinear_operations_)
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
  jack_valid_ = true;
}
*/


template <class T>
void binned_data<T>::analyze() const
{
  return;

/*
  if (valid_) return;

  if (bin_number())
  {
    count_ = bin_size()*bin_number();

    // calculate mean and error
    jackknife();

  }
  valid_ = true;
*/
}


/*
template <class T>
void binned_data<T>::jackknife() const
{
  fill_jack();

  if (jack_.size()) {
    // if any run is converged the errors will be OK
    converged_errors_=any_converged_errors_;
    
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

/*
// add collect_bins again ... 
template <class T>
void binned_data<T>::collect_bins(uint64_t howmany)
{
  if (nonlinear_operations_)
    boost::throw_exception(std::runtime_error("cannot change bins after nonlinear operations"));
  if (values_.empty() || howmany <= 1) return;
    
  uint64_t newbins = values_.size() / howmany;
  
  // fill bins
  for (uint64_t i = 0; i < newbins; ++i) {
    values_[i] = values_[howmany * i];
    for (uint64_t j = 1; j < howmany; ++j) {
      values_[i] += values_[howmany * i + j];
    }
  }
  
  binsize_ *= howmany;
  discardedbins_ = (discardedmeas_ + binsize_ - 1) / binsize_;

  values_.resize(newbins);
  
  
  changed_ = true;
  jack_valid_ = false;
  valid_ = false;
}

template <class T>
void binned_data<T>::set_bin_size(uint64_t s)
{
  collect_bins((s-1)/binsize_+1);
  binsize_=s;
}

template <class T>
void binned_data<T>::set_bin_number(uint64_t binnum)
{
  collect_bins((values_.size()-1)/binnum+1);
}
*/


template <class T>
const typename binned_data<T>::result_type& binned_data<T>::mean() const
{
  if (count() == 0) boost::throw_exception(NoMeasurementsError());
  analyze();
  return mean_;
}

template <class T>
const typename binned_data<T>::result_type& binned_data<T>::error() const
{
  if (count() == 0) boost::throw_exception(NoMeasurementsError());
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


//  if (!has_variance_)
//    boost::throw_exception(std::logic_error("observable does not have variance"));
//  analyze();
//  return variance_;
}

template <class T>
inline
const typename boost::optional<typename binned_data<T>::time_type>& binned_data<T>::tau() const
{
  if (count() == 0) boost::throw_exception(NoMeasurementsError());
  if (!tau_opt_)
    boost::throw_exception(std::logic_error("observable does not have autocorrelation information"));
  analyze();
  return tau_opt_;

//  if (!has_tau_)
//    boost::throw_exception(std::logic_error("observable does not have autocorrelation information"));
//  analyze();
//  return tau_;
}




template <class T>
std::ostream& operator<< (std::ostream &out, binned_data<T> obj)
{
/*
  out << "\ncount:\t"                << obj.count_   
      << "\nbinsize:\t"              << obj.binsize_ 
      << "\nchanged:\t"              << obj.changed_
      << "\nvalid:\t"                << obj.valid_
      << "\njack_valid:\t"           << obj.jack_valid_
      << "\nnonlinear_operations:\t" << obj.nonlinear_operations_
      << "\nmean:\t"                 << obj.mean_
      << "\nerror:\t"                << obj.error_
      << "\nvariance:\t"             << *(obj.variance_opt_)
      << "\ntau:\t"                  << *(obj.tau_opt_) 
      << "\nvalues:\t"               << obj.values_
      << "\njack:\t"                 << obj.jack_;
*/
  out << obj.count_ << "\n";
  return out;
}








} // end namespace alea

} // end namespace alp


// 1) get the operators +,-,*,/,.... and functions from simpleobseval.h
// 2) get std::vector arithmetic operations from boost::accumulator library...
// 3) define usual function (eg. sin, cos...) on std::vector in a seprate header...



#endif // ALPS_ALEA_OBSDATABIN_H
