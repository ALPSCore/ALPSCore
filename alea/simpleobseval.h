/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2009 by Matthias Troyer <troyer@comp-phys.org>,
*                            Beat Ammon <ammon@ginnan.issp.u-tokyo.ac.jp>,
*                            Andreas Laeuchli <laeuchli@comp-phys.org>,
*                            Synge Todo <wistaria@comp-phys.org>
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

/* $Id$ */

#ifndef ALPS_ALEA_SIMPLEOBSEVAL_H
#define ALPS_ALEA_SIMPLEOBSEVAL_H

#include <alps/config.h>
#include <alps/alea/simpleobservable.h>
#include <alps/alea/simpleobsdata.h>
#include <alps/parser/parser.h>
#include <alps/math.hpp>

#include <algorithm>
#include <boost/functional.hpp>

#include <iostream>

#ifndef ALPS_WITHOUT_OSIRIS
# include <alps/osiris/std/vector.h>
#endif

namespace alps {

class RealObsevaluatorXMLHandler;
class RealVectorObsevaluatorXMLHandler;

//=======================================================================
// SimpleObservableEvaluator
//
// Observable class with variable autocorrelation analysis and binning
//-----------------------------------------------------------------------

template <class T>
class SimpleObservableEvaluator : public AbstractSimpleObservable<T>
{
  typedef AbstractSimpleObservable<T> super_type;
 public:
  template <class X>
  friend class SimpleObservableEvaluator;

  friend class RealObsevaluatorXMLHandler;
  friend class RealVectorObsevaluatorXMLHandler;

  typedef T value_type;
  typedef typename obs_value_traits<T>::time_type time_type;
  typedef typename AbstractSimpleObservable<T>::result_type result_type;
  typedef typename obs_value_traits<T>::convergence_type convergence_type;
  // typedef std::size_t count_type;
  // *** we may need more than 32 Bit
  typedef uint64_t count_type;
  typedef typename SimpleObservableData<T>::covariance_type covariance_type;

  typedef typename AbstractSimpleObservable<T>::label_type label_type;

  enum { version = obs_value_traits<T>::magic_id + (6 << 16) };
  uint32_t version_id() const { return version; }

  /** almost default constructor */
  SimpleObservableEvaluator(const std::string& n= "");
  SimpleObservableEvaluator(const char* n);

  /** copy constructor */
  SimpleObservableEvaluator(const SimpleObservableEvaluator& eval);

  /** constructor from an observable */
  SimpleObservableEvaluator(const Observable& obs, const std::string& n);
  SimpleObservableEvaluator(const Observable& obs);

  SimpleObservableEvaluator(const std::string& n, std::istream&, const XMLTag&);
  // /** needed for silcing: */
  // template<class S>
  // friend SimpleObservabelEvaluator<typename obs_value_slice<T,S>::value_type>

  /** assign an observable, replacing all observables in the class */
  const SimpleObservableEvaluator<T>& operator=(const SimpleObservableEvaluator<T>& eval);
  const SimpleObservableEvaluator<T>& operator=(const AbstractSimpleObservable<T>& obs);

  /** add an observable to the ones already in the class */
  SimpleObservableEvaluator<T>&
  operator<<(const AbstractSimpleObservable<T>& obs)
  { merge(obs); return *this; }

  void rename(const std::string& n) {
    Observable::rename(n);
    automatic_naming_ = false;
  }
  void rename(const std::string& n, bool a) {
    Observable::rename(n);
    automatic_naming_ = a;
  }
  ALPS_DUMMY_VOID reset(bool = false);

  bool has_tau() const { collect(); return all_.has_tau(); }
  bool has_variance() const { collect(); return all_.has_variance(); }
  bool has_minmax() const { return false; /*collect() ; return all_.has_minmax(); */} //min / max is no longer supported - too expensive.
  value_type max() const { collect(); return all_.max(); }
  value_type min() const { collect(); return all_.min(); }

  result_type value() const { collect(); return all_.mean(); }
  result_type mean() const { return value(); }
  result_type variance() const { collect(); return all_.variance(); }
  result_type error() const {collect(); return all_.error(); }
  convergence_type  converged_errors() const { collect(); return all_.converged_errors(); }
  time_type tau() const { collect(); return all_.tau(); };

  covariance_type covariance(SimpleObservableEvaluator& obs2) const {
    collect();
    obs2.collect();
    return all_.covariance(obs2.all_);
  }

  count_type bin_number() const { collect(); return all_.bin_number(); }
  const value_type& bin_value(count_type i) const { collect(); return all_.bin_value(i); }
  count_type bin_number2() const { collect(); return all_.bin_number2(); }
  const value_type& bin_value2(count_type i) const { collect(); return all_.bin_value2(i); }
  count_type bin_size() const { collect(); return all_.bin_size(); }
  count_type count() const { collect(); return all_.count(); }

  Observable* clone() const { return new SimpleObservableEvaluator<T>(*this); }

  void set_thermalization(uint32_t todiscard);
  uint32_t get_thermalization() const { collect(); return all_.get_thermalization(); }
  bool can_set_thermalization() const { collect(); return all_.can_set_thermalization(); }

  uint32_t number_of_runs() const;
  Observable* get_run(uint32_t) const;

  ALPS_DUMMY_VOID compact();

  // Transformations

  /// negate
  SimpleObservableEvaluator<T> operator-() const;

  /// add a constant
  template <class X> const SimpleObservableEvaluator<T>& operator+=(X);

  /// subtract a constant
  template <class X> const SimpleObservableEvaluator<T>& operator-=(X);

  /// multiply with a constant
  template <class X> const SimpleObservableEvaluator<T>& operator*=(X);

  /// divide by a constant
  template <class X> const SimpleObservableEvaluator<T>& operator/=(X);

  /// add another observable
  const SimpleObservableEvaluator<T>& operator+=(const SimpleObservableEvaluator<T>&);

  /// subtract another observable
  const SimpleObservableEvaluator<T>& operator-=(const SimpleObservableEvaluator<T>&);

  /// multiply by another observable
  template <class X>
  const SimpleObservableEvaluator<T>& operator*=(const SimpleObservableEvaluator<X>&);

  /// divide by another observable
  template <class X>
  const SimpleObservableEvaluator<T>& operator/=(const SimpleObservableEvaluator<X>&);

  ALPS_DUMMY_VOID output(std::ostream&) const;
  void output_scalar(std::ostream&) const;
  void output_vector(std::ostream&) const;

  template <class S>
  SimpleObservableEvaluator<typename obs_value_slice<T,S>::value_type>
    slice(const S& ,const std::string&) const;

  template <class S>
  SimpleObservableEvaluator<typename obs_value_slice<T,S>::value_type>
    slice(const S&) const;

  void operator<<(const SimpleObservableData<T>& obs);

  template <class OPV /* , class OPR */>
  const SimpleObservableEvaluator<T>& transform(OPV opv /* , OPR opr */, const std::string&);

#ifndef ALPS_WITHOUT_OSIRIS
  void extract_timeseries(ODump& dump) const;
  void save(ODump& dump) const;
  void load(IDump& dump);
#endif
  template<class X> void subtract_from(const X& x);
  template<class X> void divide(const X& x);

  std::string evaluation_method(Target t) const { return all_.evaluation_method(t);}

  void merge(const Observable&);
  bool can_merge() const {return true;}
  bool can_merge(const Observable&) const;
  Observable* convert_mergeable() const {return clone();}
  SimpleObservableEvaluator<value_type> make_evaluator() const { return *this;}

 private:
  typedef typename std::vector<SimpleObservableData<T> >::iterator iterator;
  typedef typename std::vector<SimpleObservableData<T> >::const_iterator const_iterator;
  void collect() const;


  mutable bool valid_;
  bool automatic_naming_; // true if explicit name was not given through
                          // constructor or rename() member function
  std::vector<SimpleObservableData<T> > runs_;
  mutable SimpleObservableData<T> all_;
};

typedef SimpleObservableEvaluator<double> RealObsevaluator;
typedef SimpleObservableEvaluator<int32_t> IntObsevaluator;
typedef SimpleObservableEvaluator<std::complex<double> > ComplexObsevaluator;
#ifdef ALPS_HAVE_VALARRAY
typedef SimpleObservableEvaluator<std::valarray<int32_t> > IntVectorObsevaluator;
typedef SimpleObservableEvaluator<std::valarray<double> > RealVectorObsevaluator;
#endif


template <class T> template <class OPV /* , class OPR */>
inline const SimpleObservableEvaluator<T>& SimpleObservableEvaluator<T>::transform(OPV opv, /* OPR opr, */ const std::string& n)
{
  collect();
  all_.transform(opv /* ,opr */);
  for (iterator r = runs_.begin(); r != runs_.end(); ++r)
    if (r->count()) r->transform(opv /* ,opr */);
  if (automatic_naming_) Observable::rename(n);
  return (*this);
}

template <class T>
inline SimpleObservableEvaluator<T> SimpleObservableEvaluator<T>::operator-() const
{
  collect();
  SimpleObservableEvaluator<T> tmp(*this);
  if (automatic_naming_) tmp.rename("-(" + super_type::name() + ")", true);
  tmp.all_.negate();
  for (iterator r = tmp.runs_.begin(); r != tmp.runs_.end(); ++r)
    if (r->count()) r->negate();
  return tmp;
}

template <class T> template <class X>
inline const SimpleObservableEvaluator<T>& SimpleObservableEvaluator<T>::operator+=(X x)
{
  collect();
  for (iterator r = runs_.begin(); r != runs_.end(); ++r)
    if (r->count()) *r += x;
  all_+=x;
  if (automatic_naming_)
    Observable::rename(super_type::name()+" + " + boost::lexical_cast<std::string,X>(x));
  return *this;
}

template <class T> template <class X>
inline void SimpleObservableEvaluator<T>::subtract_from(const X& x)
{
  collect();
  for (iterator r = runs_.begin(); r != runs_.end(); ++r)
    if (r->count()) r->subtract_from(x);
  all_.subtract_from(x);
  if (automatic_naming_)
    Observable::rename(boost::lexical_cast<std::string,X>(x) + " - " + super_type::name());
}

template <class T> template <class X>
inline const SimpleObservableEvaluator<T>& SimpleObservableEvaluator<T>::operator-=(X x)
{
  collect();
  for (iterator r = runs_.begin(); r != runs_.end(); ++r)
    if (r->count()) *r -= x;
  all_-=x;
  if (automatic_naming_)
    Observable::rename(super_type::name()+" - " + boost::lexical_cast<std::string,X>(x));
  return *this;
}

template <class T> template <class X>
inline const SimpleObservableEvaluator<T>& SimpleObservableEvaluator<T>::operator*=(X x)
{
  collect();
  for (iterator r = runs_.begin(); r != runs_.end(); ++r) {
    if (r->count()) *r *= x;
  }
  all_*=x;
  if (automatic_naming_)
    Observable::rename("("+super_type::name()+") * " + boost::lexical_cast<std::string,X>(x));
  return *this;
}

template <class T> template <class X>
inline const SimpleObservableEvaluator<T>& SimpleObservableEvaluator<T>::operator/=(X x)
{
  collect();
  for (iterator r = runs_.begin(); r != runs_.end(); ++r)
    if (r->count()) *r /= x;
  all_/=x;
  if (automatic_naming_)
    Observable::rename("("+super_type::name()+") / " + boost::lexical_cast<std::string,X>(x));
  return *this;
}

template <class T> template <class X>
inline void SimpleObservableEvaluator<T>::divide(const X& x)
{
  collect();
  for (iterator r = runs_.begin(); r != runs_.end(); ++r)
    if (r->count()) r->divide(x);
  all_.divide(x);
  if (automatic_naming_)
    Observable::rename(boost::lexical_cast<std::string,X>(x)+" / ("+super_type::name()+")");
}

#ifndef ALPS_WITHOUT_OSIRIS

template <class T>
inline void SimpleObservableEvaluator<T>::extract_timeseries(ODump& dump) const
{
  collect();
  all_.extract_timeseries(dump);
}

template <class T>
inline void SimpleObservableEvaluator<T>::save(ODump& dump) const
{
  AbstractSimpleObservable<T>::save(dump);
  dump << valid_ << runs_ << all_;
}

template <class T>
inline void SimpleObservableEvaluator<T>::load(IDump& dump)
{
  AbstractSimpleObservable<T>::load(dump);
  dump >> valid_ >> runs_ >> all_;
}
#endif

template <class T>
inline void SimpleObservableEvaluator<T>::collect() const
{
  if (!valid_) {
    all_.collect_from(runs_);
    valid_ = true;
  }
}

template <class T>
inline const SimpleObservableEvaluator<T>&  SimpleObservableEvaluator<T>::operator=(const SimpleObservableEvaluator<T>& eval)
{
  runs_ = eval.runs_;
  all_ = eval.all_;
  valid_ = eval.valid_;
  if (automatic_naming_ && super_type::name() == "") Observable::rename(eval.name());
  return *this;
}

template <class T>
inline const SimpleObservableEvaluator<T>&  SimpleObservableEvaluator<T>::operator=(const AbstractSimpleObservable<T>& obs)
{
  std::string oldname = super_type::name();
  bool a = automatic_naming_;
  SimpleObservableEvaluator<T> eval(obs);
  *this = eval;
  if (!a) rename(oldname);
  return *this;
}

template <class T>
inline void SimpleObservableEvaluator<T>::set_thermalization(uint32_t todiscard)
{
  std::for_each(runs_.begin(), runs_.end(), boost::bind2nd(boost::mem_fun_ref(&SimpleObservableData<T>::set_thermalization), todiscard));
  valid_ = false;
}

template <class T>
inline
void SimpleObservableEvaluator<T>::operator<<(const SimpleObservableData<T>& b)
{
  runs_.push_back(b);
  valid_ = false;
}

template <class T>
inline void SimpleObservableEvaluator<T>::merge(const Observable& o)
{
  if (automatic_naming_ && super_type::name()=="") Observable::rename(o.name());
  if (dynamic_cast<const RecordableObservable<T>*>(&o)!=0) {
    if(dynamic_cast<const RecordableObservable<T>&>(o).is_thermalized())
      (*this) <<
        SimpleObservableData<T>(dynamic_cast<const AbstractSimpleObservable<T>&>(o));
  } else {
    const SimpleObservableEvaluator<T>& eval =
      dynamic_cast<const SimpleObservableEvaluator<T>&>(o);
    if (automatic_naming_ && !eval.automatic_naming_) automatic_naming_ = false;
    for (unsigned int i = 0; i < eval.runs_.size(); ++i) (*this) << eval.runs_[i];
  }
}

template <class T>
inline uint32_t SimpleObservableEvaluator<T>::number_of_runs() const
{
  return runs_.size();
}

template <class T>
template <class S>
inline SimpleObservableEvaluator<typename obs_value_slice<T,S>::value_type>
SimpleObservableEvaluator<T>::slice(const S& sl, const std::string& n) const
{
  SimpleObservableEvaluator<typename obs_value_slice<T,S>::value_type>
     res(n.length()==0 ? super_type::name()+boost::lexical_cast<std::string,S>(sl) : n);

  for (typename std::vector<SimpleObservableData<T> >::const_iterator it=runs_.begin();
       it !=runs_.end();++it)
    res << it->slice(sl);
  return res;
}

template <class T>
template <class S>
inline SimpleObservableEvaluator<typename obs_value_slice<T,S>::value_type>
SimpleObservableEvaluator<T>::slice(const S& sl) const
{
  SimpleObservableEvaluator<typename obs_value_slice<T,S>::value_type>
     res(super_type::name()+boost::lexical_cast<std::string,S>(sl));

  for (typename std::vector<SimpleObservableData<T> >::const_iterator it=runs_.begin();
       it !=runs_.end();++it)
    res << it->slice(sl);
  return res;
}

template <class T>
inline Observable* SimpleObservableEvaluator<T>::get_run(uint32_t i) const
{
  SimpleObservableEvaluator<T>* res = new SimpleObservableEvaluator<T>(super_type::name());
  (*res) << runs_[i];
  return res;
}


template <class T>
inline ALPS_DUMMY_VOID SimpleObservableEvaluator<T>::reset(bool)
{
  runs_.clear();
  all_ = SimpleObservableData<T>();
  ALPS_RETURN_VOID
}

template <class T>
inline ALPS_DUMMY_VOID SimpleObservableEvaluator<T>::compact()
{
  collect();
  std::for_each(runs_.begin(), runs_.end(), boost::mem_fun_ref(&SimpleObservableData<T>::compact));
  all_.compact();
  ALPS_RETURN_VOID
}

template <class T>
ALPS_DUMMY_VOID SimpleObservableEvaluator<T>::output(std::ostream& out) const
{
  output_helper<obs_value_traits<T>::array_valued>::output(*this,out);
  ALPS_RETURN_VOID
}

template <class T>
void SimpleObservableEvaluator<T>::output_scalar(std::ostream& out) const
{
  out << super_type::name();
  if(count()==0)
    out << " no measurements.\n";
  else
  {
    out << ": " << std::setprecision(6) << alps::round<2>(mean()) << " +/- "
        << std::setprecision(3) << alps::round<2>(error());
    if(has_tau())
      out << std::setprecision(3) <<  "; tau = " << tau();
    if (converged_errors()==MAYBE_CONVERGED)
      out << " WARNING: check error convergence";
    if (converged_errors()==NOT_CONVERGED)
      out << " WARNING: ERRORS NOT CONVERGED!!!";
    if (error_underflow(mean(),error()))
      out << " Warning: potential error underflow. Errors might be smaller";
    out << std::setprecision(6) << std::endl;
  }
}
template <class T>
void SimpleObservableEvaluator<T>::output_vector(std::ostream& out) const
{
  out << super_type::name();
  if(count()==0)
    out << ": no measurements.\n";
  else {
    out << std::endl;
    result_type value_(mean());
    result_type error_(error());
    convergence_type conv_(converged_errors());
    time_type tau_;
    if (has_tau())
      obs_value_traits<value_type>::copy(tau_,tau());
    typename obs_value_traits<label_type>::slice_iterator it2=obs_value_traits<label_type>::slice_begin(super_type::label());
    for (typename obs_value_traits<result_type>::slice_iterator sit=
           obs_value_traits<result_type>::slice_begin(value_);
          sit!=obs_value_traits<result_type>::slice_end(value_);++sit,++it2)
    {
      std::string lab=obs_value_traits<label_type>::slice_value(super_type::label(),it2);
      if (lab=="")
        lab=obs_value_traits<result_type>::slice_name(value_,sit);
      out << "Entry[" << lab << "]: "
          << alps::round<2>(obs_value_traits<result_type>::slice_value(value_,sit)) << " +/- "
          << alps::round<2>(obs_value_traits<result_type>::slice_value(error_,sit));
      if(has_tau())
        out << "; tau = " << obs_value_traits<time_type>::slice_value(tau_,sit);
      if (obs_value_traits<convergence_type>::slice_value(conv_,sit)==MAYBE_CONVERGED)
        out << " WARNING: check error convergence";
      if (obs_value_traits<convergence_type>::slice_value(conv_,sit)==NOT_CONVERGED)
        out << " WARNING: ERRORS NOT CONVERGED!!!";
      if (error_underflow(obs_value_traits<result_type>::slice_value(value_,sit),
                          obs_value_traits<result_type>::slice_value(error_,sit)))
        out << " Warning: potential error underflow. Errors might be smaller";
      out << std::endl;
    }
  }
}

template <class T>
inline bool SimpleObservableEvaluator<T>::can_merge(const Observable& obs) const
{
  return dynamic_cast<const AbstractSimpleObservable<T>*>(&obs) != 0;
}

template <class T>
inline SimpleObservableEvaluator<T>::SimpleObservableEvaluator(const std::string& n)
  : AbstractSimpleObservable<T>(n), valid_(false), automatic_naming_(n=="") {}

template <class T>
inline SimpleObservableEvaluator<T>::SimpleObservableEvaluator(const char* n)
  : AbstractSimpleObservable<T>(std::string(n)), valid_(false),
    automatic_naming_(false) {
}

template <class T>
inline SimpleObservableEvaluator<T>::SimpleObservableEvaluator(const SimpleObservableEvaluator& eval)
  : AbstractSimpleObservable<T>(eval), valid_(eval.valid_),
    automatic_naming_(true), runs_(eval.runs_), all_(eval.all_) {}

template <class T>
inline SimpleObservableEvaluator<T>::SimpleObservableEvaluator(const Observable& b, const std::string& n)
  : AbstractSimpleObservable<T>(n,dynamic_cast<const AbstractSimpleObservable<T>&>(b).super_type::label()), valid_(false),
    automatic_naming_(n=="")
{
  merge(b);
}

template <class T>
inline SimpleObservableEvaluator<T>::SimpleObservableEvaluator(const Observable& b)
  : AbstractSimpleObservable<T>(dynamic_cast<const AbstractSimpleObservable<T>&>(b)), valid_(false),
    automatic_naming_(true)
{
  if (dynamic_cast<const AbstractSimpleObservable<T>*>(&b)==0)
    merge(b);
  else
    (*this) = dynamic_cast<const AbstractSimpleObservable<T>&>(b).make_evaluator();
}

template <class T>
inline SimpleObservableEvaluator<T>::SimpleObservableEvaluator(const std::string& n, std::istream& infile, const XMLTag& intag)
  : AbstractSimpleObservable<T>(n),
    valid_(true),
    automatic_naming_(false),
    all_(infile,intag,super_type::label_)
{}

template <class T>
inline const SimpleObservableEvaluator<T>& SimpleObservableEvaluator<T>::operator+=(const SimpleObservableEvaluator<T>& rh)
{
  if(runs_.size() != rh.runs_.size())
    boost::throw_exception(std::runtime_error("unequal number of runs in addition of observables"));
  collect();
  rh.collect();
  all_ += rh.all_;
  for (int i = 0; i < runs_.size(); ++i)
    if (runs_[i].count() || rh.runs_[i].count()) runs_[i] += rh.runs_[i];
  if (automatic_naming_) Observable::rename(super_type::name() + " + " + rh.name());
  return (*this);
}


template <class T>
inline const SimpleObservableEvaluator<T>& SimpleObservableEvaluator<T>::operator-=(const SimpleObservableEvaluator<T>& rh)
{
  if(runs_.size() != rh.runs_.size())
    boost::throw_exception(std::runtime_error("unequal number of runs in subtraction of observables"));
  collect();
  rh.collect();
  all_ -= rh.all_;
  for (int i = 0; i < runs_.size(); ++i)
    if (runs_[i].count() || rh.runs_[i].count()) runs_[i] -= rh.runs_[i];
  if (automatic_naming_) Observable::rename(super_type::name() + " - " + rh.name());
  return (*this);
}

template <class T> template <class X>
inline const SimpleObservableEvaluator<T>& SimpleObservableEvaluator<T>::operator*=(const SimpleObservableEvaluator<X>& rh)
{
  if(runs_.size() != rh.runs_.size())
    boost::throw_exception(std::runtime_error("unequal number of runs in multiplication of observables"));
  collect();
  rh.collect();
  all_ *= rh.all_;
  for (unsigned int i = 0; i < runs_.size(); ++i)
    if (runs_[i].count() || rh.runs_[i].count()) runs_[i] *= rh.runs_[i];
  if (automatic_naming_)
    Observable::rename("(" + super_type::name() + ") * (" + rh.name() + ")");
  return (*this);
}

template <class T> template <class X>
inline const SimpleObservableEvaluator<T>& SimpleObservableEvaluator<T>::operator/=(const SimpleObservableEvaluator<X>& rh)
{
  if(runs_.size() != rh.runs_.size())
    boost::throw_exception(std::runtime_error("unequal number of runs in division of observables"));
  collect();
  rh.collect();
  all_ /= rh.all_;
  for (unsigned int i = 0; i < runs_.size(); ++i)
    if (runs_[i].count() || rh.runs_[i].count()) runs_[i] /= rh.runs_[i];
  if (automatic_naming_)
    Observable::rename("(" + super_type::name() + ") / (" + rh.name() + ")");
  return (*this);
}

} // end namespace alps


//
// Basic Arithmetic operations with signature SimpleObservableEvaluator
// # SimpleObservableEvaluator
//

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

/// sum of two observables or of observable and number
template <class T, class Y>
inline alps::SimpleObservableEvaluator<T> operator+(const alps::SimpleObservableEvaluator<T>& x, const Y& y)
{
  alps::SimpleObservableEvaluator<T> tmp(x);
  tmp += y;
  return tmp;
}

/// difference of two observables or of observable and number
template <class T, class Y>
inline alps::SimpleObservableEvaluator<T> operator-(const alps::SimpleObservableEvaluator<T>& x, const Y& y)
{
  alps::SimpleObservableEvaluator<T> tmp(x);
  tmp -= y;
  return tmp;
}

/// product of two observables or of observable and number
template <class T, class Y>
inline alps::SimpleObservableEvaluator<T> operator*(const alps::SimpleObservableEvaluator<T>& x, const Y& y)
{
  alps::SimpleObservableEvaluator<T> tmp(x);
  tmp *= y;
  return tmp;
}

/// product of vector and scalar observable
template <class T>
inline alps::SimpleObservableEvaluator<std::valarray<T> > operator*(const alps::SimpleObservableEvaluator<std::valarray<T> >& x, const alps::SimpleObservableEvaluator<T>& y)
{
  alps::SimpleObservableEvaluator<T> tmp(x);
  tmp *= y;
  return tmp;
}

template <class T>
inline alps::SimpleObservableEvaluator<std::valarray<T> > operator*(const alps::SimpleObservableEvaluator<T>& y, const alps::SimpleObservableEvaluator<std::valarray<T> >& x)
{
  alps::SimpleObservableEvaluator<T> tmp(x);
  tmp *= y;
  return tmp;
}


/// ratio of two observables or of observable and number
template <class T, class Y>
inline alps::SimpleObservableEvaluator<T> operator/(const alps::SimpleObservableEvaluator<T>& x, const Y& y)
{
  alps::SimpleObservableEvaluator<T> tmp(x);
  tmp /= y;
  return tmp;
}

//
// Basic Arithmetic operations with signature T # SimpleObservableEvaluator
//

template <class T>
inline alps::SimpleObservableEvaluator<T> operator+(const T& x, const alps::SimpleObservableEvaluator<T>& y)
{
  return y + x;
}

template <class T>
inline alps::SimpleObservableEvaluator<T> operator-(const T& x, const alps::SimpleObservableEvaluator<T>& y)
{
  alps::SimpleObservableEvaluator<T> tmp(y);
  tmp.subtract_from(x);
  return tmp;
}

template <class T>
inline alps::SimpleObservableEvaluator<T> operator*(const T& x, const alps::SimpleObservableEvaluator<T>& y)
{
  return y * x;
}

template <class T>
inline alps::SimpleObservableEvaluator<T> operator/(const T& x, const alps::SimpleObservableEvaluator<T>& y)
{
  alps::SimpleObservableEvaluator<T> tmp(y);
  tmp.divide(x);
  return tmp;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

namespace alps {

#define OBSERVABLE_FUNCTION(F) \
namespace detail { \
template <class T> struct function_##F : public std::unary_function<T,T> \
{ T operator()(T x) const { using std:: F ; return F(x); } }; \
} \
template <class T> alps::SimpleObservableEvaluator<T> \
F(const alps::SimpleObservableEvaluator<T>& x) \
{ return alps::SimpleObservableEvaluator<T>(x).transform(alps::detail::function_##F<T>(), /* alps::detail::function_##F<typename alps::obs_value_traits<T>::result_type>(), */ #F"("+x.name()+")"); }

OBSERVABLE_FUNCTION(exp)
OBSERVABLE_FUNCTION(log)
OBSERVABLE_FUNCTION(sqrt)
OBSERVABLE_FUNCTION(sin)
OBSERVABLE_FUNCTION(cos)
OBSERVABLE_FUNCTION(tan)

#undef OBSERVABLE_FUNCTION

namespace detail {

template <class T> struct function_pow : public std::unary_function<T,T>
{
  function_pow(T p) : pow_(p) {}
  T operator()(T x) const { using std::pow; return pow(x, pow_); }
  T pow_;
};

}

template <class T>
alps::SimpleObservableEvaluator<T>
pow(const alps::SimpleObservableEvaluator<T>& x, double p)
{
  return alps::SimpleObservableEvaluator<T>(x).
    transform(alps::detail::function_pow<T>(T(p)),
              "pow(" + x.name() + "," + boost::lexical_cast<std::string>(p)
              + ")");
}

template <class T>
alps::SimpleObservableEvaluator<T>
pow(const alps::SimpleObservableEvaluator<T>& x, int p)
{
  return alps::SimpleObservableEvaluator<T>(x).
    transform(alps::detail::function_pow<T>(T(p)),
              "pow(" + x.name() + ", " + boost::lexical_cast<std::string>(p)
              + ")");
}

} // end namespace alps

#endif // ALPS_ALEA_SIMPLEOBSEVAL_H
