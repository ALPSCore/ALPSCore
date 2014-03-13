/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Matthias Troyer <troyer@comp-phys.org>,
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
#include <alps/osiris/std/vector.h>
#include <alps/utility/encode.hpp>
#include <alps/numeric/round.hpp>
#include <alps/numeric/is_nonzero.hpp>
#include <alps/type_traits/is_scalar.hpp>
#include <alps/type_traits/change_value_type.hpp>
#include <alps/type_traits/average_type.hpp>
#include <alps/alea/type_tag.hpp>

#include <boost/config.hpp>
#include <boost/functional.hpp>

#include <algorithm>
#include <iostream>

namespace alps {

class RealObsevaluatorXMLHandler;
class RealVectorObsevaluatorXMLHandler;

struct ObservableNamingHelper {
  template<typename T>
  static std::string generate(T const& t,
    typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0) {
    return boost::lexical_cast<std::string, T>(t);
  }
  
  template<typename T>
  static std::string generate(T const&,
    typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0) {
    return "(unnamed object)";
  }

  template<typename T>
  static std::string generate(std::valarray<T> const&) {
    return "(unnamed array)";
  }
};

//=======================================================================
// SimpleObservableEvaluator
//
// Observable class with variable autocorrelation analysis and binning
//-----------------------------------------------------------------------

template <class T>
class ALPS_TEMPL_DECL SimpleObservableEvaluator : public AbstractSimpleObservable<T>
{
  typedef AbstractSimpleObservable<T> super_type;
 public:
  template <class X>
  friend class SimpleObservableEvaluator;

  friend class RealObsevaluatorXMLHandler;
  friend class RealVectorObsevaluatorXMLHandler;

  typedef T value_type;
  typedef typename change_value_type<T,double>::type time_type;
  typedef typename AbstractSimpleObservable<T>::result_type result_type;
  typedef typename change_value_type<T,int>::type convergence_type;
  typedef typename AbstractSimpleObservable<T>::label_type label_type;
  // typedef std::size_t count_type;
  // *** we may need more than 32 Bit
  typedef uint64_t count_type;
  typedef typename SimpleObservableData<T>::covariance_type covariance_type;


  enum { version = type_tag<T>::value + (6 << 16) };
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

  bool has_tau() const { return all_.has_tau(); }
  bool has_variance() const { return all_.has_variance(); }

  result_type value() const { return all_.mean(); }
  result_type mean() const { return value(); }
  result_type variance() const { return all_.variance(); }
  result_type error() const {return all_.error(); }
  convergence_type  converged_errors() const { return all_.converged_errors(); }
  time_type tau() const { return all_.tau(); };

  covariance_type covariance(SimpleObservableEvaluator& obs2) const {
    return all_.covariance(obs2.all_);
  }

  count_type bin_number() const { return all_.bin_number(); }
  const value_type& bin_value(count_type i) const { return all_.bin_value(i); }
  count_type bin_number2() const { return all_.bin_number2(); }
  const value_type& bin_value2(count_type i) const { return all_.bin_value2(i); }
  count_type bin_size() const { return all_.bin_size(); }
  count_type count() const { return all_.count(); }

  const std::vector<value_type>& bins() const { return all_.bins();  }

  Observable* clone() const { return new SimpleObservableEvaluator<T>(*this); }

  uint32_t get_thermalization() const { return all_.get_thermalization(); }
  bool can_set_thermalization() const { return all_.can_set_thermalization(); }

  ALPS_DUMMY_VOID compact();

  // Transformations

  /// negate
  SimpleObservableEvaluator<T> operator-() const;

  /// add a constant
  template <class X> const SimpleObservableEvaluator<T>& operator+=(const X&);

  /// subtract a constant
  template <class X> const SimpleObservableEvaluator<T>& operator-=(const X&);

  /// multiply with a constant
  template <class X> const SimpleObservableEvaluator<T>& operator*=(const X&);

  /// divide by a constant
  template <class X> const SimpleObservableEvaluator<T>& operator/=(const X&);

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
  SimpleObservableEvaluator<typename element_type<T>::type>
    slice(const S& ,const std::string&) const;

  template <class S>
  SimpleObservableEvaluator<typename element_type<T>::type>
    slice(const S&) const;

  void operator<<(const SimpleObservableData<T>& obs);

  template <class OPV /* , class OPR */>
  const SimpleObservableEvaluator<T>& transform(OPV opv /* , OPR opr */, const std::string&);

  void extract_timeseries(ODump& dump) const;
  void save(ODump& dump) const;
  void load(IDump& dump);

  void save(hdf5::archive &) const;
  void load(hdf5::archive &);

  template<class X> void add_to(const X& x);
  template<class X> void subtract_from(const X& x);
  template<class X> void divide(const X& x);
  template<class X> void multiply_to(const X& x);

  std::string evaluation_method(Target t) const { return all_.evaluation_method(t);}

  void merge(const Observable&);
  bool can_merge() const {return true;}
  bool can_merge(const Observable&) const;
  Observable* convert_mergeable() const {return clone();}
  SimpleObservableEvaluator<value_type> make_evaluator() const { return *this;}

 private:
  typedef typename std::vector<SimpleObservableData<T> >::iterator iterator;
  typedef typename std::vector<SimpleObservableData<T> >::const_iterator const_iterator;

  bool automatic_naming_; // true if explicit name was not given through
                          // constructor or rename() member function
  mutable SimpleObservableData<T> all_;
};

typedef SimpleObservableEvaluator<double> RealObsevaluator;
typedef SimpleObservableEvaluator<int32_t> IntObsevaluator;
typedef SimpleObservableEvaluator<std::complex<double> > ComplexObsevaluator;
typedef SimpleObservableEvaluator<std::valarray<int32_t> > IntVectorObsevaluator;
typedef SimpleObservableEvaluator<std::valarray<double> > RealVectorObsevaluator;


template <class T> template <class OPV /* , class OPR */>
inline const SimpleObservableEvaluator<T>& SimpleObservableEvaluator<T>::transform(OPV opv, /* OPR opr, */ const std::string& n)
{
  all_.transform(opv /* ,opr */);
  if (automatic_naming_) Observable::rename(n);
  return (*this);
}

template <class T>
inline SimpleObservableEvaluator<T> SimpleObservableEvaluator<T>::operator-() const
{
  SimpleObservableEvaluator<T> tmp(*this);
  if (automatic_naming_) tmp.rename("-(" + super_type::name() + ")", true);
  tmp.all_.negate();
  return tmp;
}

template <class T> template <class X>
inline const SimpleObservableEvaluator<T>& SimpleObservableEvaluator<T>::operator+=(const X& x)
{
  all_+=x;
  if (automatic_naming_)
    Observable::rename(super_type::name()+" + " + ObservableNamingHelper::generate(x));
  return *this;
}

template <class T> template <class X>
inline void SimpleObservableEvaluator<T>::add_to(const X& x)
{
  std::string old_name = super_type::name();
  (*this) += x;
  if (automatic_naming_)
    Observable::rename(ObservableNamingHelper::generate(x) + " + " + old_name);
}

template <class T> template <class X>
inline void SimpleObservableEvaluator<T>::subtract_from(const X& x)
{
  all_.subtract_from(x);
  if (automatic_naming_)
    Observable::rename(ObservableNamingHelper::generate(x) + " - " + super_type::name());
}

template <class T> template <class X>
inline const SimpleObservableEvaluator<T>& SimpleObservableEvaluator<T>::operator-=(const X& x)
{
  all_-=x;
  if (automatic_naming_)
    Observable::rename(super_type::name()+" - " + ObservableNamingHelper::generate(x));
  return *this;
}

template <class T> template <class X>
inline void SimpleObservableEvaluator<T>::multiply_to(const X& x)
{
  std::string old_name = super_type::name();
  (*this) *= x;
  if (automatic_naming_)
    Observable::rename(ObservableNamingHelper::generate(x) + " * " + old_name);
}

template <class T> template <class X>
inline const SimpleObservableEvaluator<T>& SimpleObservableEvaluator<T>::operator*=(const X& x)
{
  all_*=x;
  if (automatic_naming_)
    Observable::rename("("+super_type::name()+") * " + ObservableNamingHelper::generate(x));
  return *this;
}

template <class T> template <class X>
inline const SimpleObservableEvaluator<T>& SimpleObservableEvaluator<T>::operator/=(const X& x)
{
  all_/=x;
  if (automatic_naming_)
    Observable::rename("("+super_type::name()+") / " + ObservableNamingHelper::generate(x));
  return *this;
}

template <class T> template <class X>
inline void SimpleObservableEvaluator<T>::divide(const X& x)
{
  all_.divide(x);
  if (automatic_naming_)
    Observable::rename(ObservableNamingHelper::generate(x) + " / (" + super_type::name() + ")");
}

template <class T>
inline void SimpleObservableEvaluator<T>::extract_timeseries(ODump& dump) const
{
  all_.extract_timeseries(dump);
}

template <class T>
inline void SimpleObservableEvaluator<T>::save(ODump& dump) const
{
  AbstractSimpleObservable<T>::save(dump);
  dump << all_;
}

template <class T>
inline void SimpleObservableEvaluator<T>::load(IDump& dump)
{
  AbstractSimpleObservable<T>::load(dump);
  if (dump.version() < 400 && dump.version() > 0) {
    bool valid;
    std::vector<SimpleObservableData<T> > runs;
    dump >> valid >> runs;
  }
  dump >> all_;
}

template <class T>
inline const SimpleObservableEvaluator<T>&  SimpleObservableEvaluator<T>::operator=(const SimpleObservableEvaluator<T>& eval)
{
  all_ = eval.all_;
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
inline
void SimpleObservableEvaluator<T>::operator<<(const SimpleObservableData<T>& b)
{
  all_ << b;
}

template <class T>
inline void SimpleObservableEvaluator<T>::merge(const Observable& o)
{
  if (automatic_naming_ && super_type::name()=="") Observable::rename(o.name());
  if (dynamic_cast<const RecordableObservable<T>*>(&o)!=0) {
    (*this) <<
        SimpleObservableData<T>(dynamic_cast<const AbstractSimpleObservable<T>&>(o));
  } else {
    const SimpleObservableEvaluator<T>& eval =
      dynamic_cast<const SimpleObservableEvaluator<T>&>(o);
    if (automatic_naming_ && !eval.automatic_naming_) automatic_naming_ = false;
    (*this) << eval.all_;
  }
}

template <class T>
template <class S>
inline SimpleObservableEvaluator<typename element_type<T>::type>
SimpleObservableEvaluator<T>::slice(const S& sl, const std::string& n) const
{
  SimpleObservableEvaluator<typename element_type<T>::type>
     res(n.length()==0 ? super_type::name()+boost::lexical_cast<std::string,S>(sl) : n);
  res << all_.slice(sl);
  return res;
}

template <class T>
template <class S>
inline SimpleObservableEvaluator<typename element_type<T>::type>
SimpleObservableEvaluator<T>::slice(const S& sl) const
{
  SimpleObservableEvaluator<typename element_type<T>::type>
     res(super_type::name()+boost::lexical_cast<std::string,S>(sl));
  res << all_.slice(sl);
  return res;
}

template <class T>
inline ALPS_DUMMY_VOID SimpleObservableEvaluator<T>::reset(bool)
{
  all_ = SimpleObservableData<T>();
  ALPS_RETURN_VOID
}

template <class T>
inline ALPS_DUMMY_VOID SimpleObservableEvaluator<T>::compact()
{
  all_.compact();
  ALPS_RETURN_VOID
}

template <class T>
ALPS_DUMMY_VOID SimpleObservableEvaluator<T>::output(std::ostream& out) const
{
  output_helper<typename is_scalar<T>::type>::output(*this,out);
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
    out << ": " << std::setprecision(6) << alps::numeric::round<2>(slice_value(mean(), 0)) << " +/- "
        << std::setprecision(3) << alps::numeric::round<2>(slice_value(error(), 0));
    if(has_tau())
      out << std::setprecision(3) <<  "; tau = " << (alps::numeric::is_nonzero<2>(slice_value(error(), 0)) ? slice_value(tau(), 0) : 0);
    if (alps::numeric::is_nonzero<2>(slice_value(error(), 0))) {
      if (slice_value(converged_errors(), 0)==MAYBE_CONVERGED)
        out << " WARNING: check error convergence";
      if (slice_value(converged_errors(), 0)==NOT_CONVERGED)
        out << " WARNING: ERRORS NOT CONVERGED!!!";
      if (error_underflow(slice_value(mean(), 0),slice_value(error(), 0)))
        out << " Warning: potential error underflow. Errors might be smaller";
    }
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
      assign(tau_,tau());
    typename alps::slice_index<label_type>::type it2=slices(this->label()).first;
    for (typename alps::slice_index<result_type>::type sit= slices(value_).first;
          sit!=slices(value_).second;++sit,++it2)
    {
      std::string lab=slice_value(this->label(),it2);
      if (lab=="")
        lab=slice_name(value_,sit);
      out << "Entry[" << lab << "]: "
          << alps::numeric::round<2>(slice_value(value_,sit)) << " +/- "
          << alps::numeric::round<2>(slice_value(error_,sit));
      if(has_tau())
        out << "; tau = " << (alps::numeric::is_nonzero<2>(slice_value(error_,sit)) ? slice_value(tau_,sit) : 0);
      if (alps::numeric::is_nonzero<2>(slice_value(error_,sit))) {
        if (slice_value(conv_,sit)==MAYBE_CONVERGED)
          out << " WARNING: check error convergence";
        if (slice_value(conv_,sit)==NOT_CONVERGED)
          out << " WARNING: ERRORS NOT CONVERGED!!!";
        if (error_underflow(slice_value(value_,sit),slice_value(error_,sit)))
          out << " Warning: potential error underflow. Errors might be smaller";
      }
      out << std::endl;
    }
  }
}

template <class T>
inline bool SimpleObservableEvaluator<T>::can_merge(const Observable& obs) const
{
  return dynamic_cast<const AbstractSimpleObservable<T> *>(&obs) != 0;
}

template <class T>
inline SimpleObservableEvaluator<T>::SimpleObservableEvaluator(const std::string& n)
  : AbstractSimpleObservable<T>(n), automatic_naming_(n=="") {}

template <class T>
inline SimpleObservableEvaluator<T>::SimpleObservableEvaluator(const char* n)
  : AbstractSimpleObservable<T>(std::string(n)), automatic_naming_(false) {
}

template <class T>
inline SimpleObservableEvaluator<T>::SimpleObservableEvaluator(const SimpleObservableEvaluator& eval)
  : AbstractSimpleObservable<T>(eval), automatic_naming_(true), all_(eval.all_) {}

template <class T>
inline SimpleObservableEvaluator<T>::SimpleObservableEvaluator(const Observable& b, const std::string& n)
  : AbstractSimpleObservable<T>(n,dynamic_cast<const AbstractSimpleObservable<T>&>(b).super_type::label()), automatic_naming_(n=="")
{
  merge(b);
}

template <class T>
inline SimpleObservableEvaluator<T>::SimpleObservableEvaluator(const Observable& b)
  : AbstractSimpleObservable<T>(dynamic_cast<const AbstractSimpleObservable<T>&>(b)), automatic_naming_(true)
{
  if (dynamic_cast<const AbstractSimpleObservable<T>*>(&b)==0)
    merge(b);
  else
    (*this) = dynamic_cast<const AbstractSimpleObservable<T>&>(b).make_evaluator();
}

template <class T>
inline SimpleObservableEvaluator<T>::SimpleObservableEvaluator(const std::string& n, std::istream& infile, const XMLTag& intag)
  : AbstractSimpleObservable<T>(n),
    automatic_naming_(false),
    all_(infile,intag,super_type::label_)
{}

template <class T>
inline const SimpleObservableEvaluator<T>& SimpleObservableEvaluator<T>::operator+=(const SimpleObservableEvaluator<T>& rh)
{
  all_ += rh.all_;
  if (automatic_naming_) Observable::rename(super_type::name() + " + " + rh.name());
  return (*this);
}


template <class T>
inline const SimpleObservableEvaluator<T>& SimpleObservableEvaluator<T>::operator-=(const SimpleObservableEvaluator<T>& rh)
{
  all_ -= rh.all_;
  if (automatic_naming_) Observable::rename(super_type::name() + " - " + rh.name());
  return (*this);
}

template <class T> template <class X>
inline const SimpleObservableEvaluator<T>& SimpleObservableEvaluator<T>::operator*=(const SimpleObservableEvaluator<X>& rh)
{
  all_ *= rh.all_;
  if (automatic_naming_)
    Observable::rename("(" + super_type::name() + ") * (" + rh.name() + ")");
  return (*this);
}

template <class T> template <class X>
inline const SimpleObservableEvaluator<T>& SimpleObservableEvaluator<T>::operator/=(const SimpleObservableEvaluator<X>& rh)
{
  all_ /= rh.all_;
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

/// sum of two observables
template <class T, class U>
inline alps::SimpleObservableEvaluator<T> operator+(alps::SimpleObservableEvaluator<T> const& x, const alps::SimpleObservableEvaluator<U>& y)
{
  alps::SimpleObservableEvaluator<T> res(x);
  res += y;
  return res;
}

/// sum of observable and number
template <class T, class Y>
inline alps::SimpleObservableEvaluator<T> operator+(alps::SimpleObservableEvaluator<T> const& x, const Y& y)
{
  alps::SimpleObservableEvaluator<T> res(x);
  res += y;
  return res;
}

/// sum of observable and number
template <class T, class Y>
inline alps::SimpleObservableEvaluator<T> operator+(const Y& y, alps::SimpleObservableEvaluator<T> const& x)
{
  alps::SimpleObservableEvaluator<T> res(x);
  res.add_to(y);
  return res;
}

/// difference of two observables (IBM AIX workaround)
template <class T, class U>
inline alps::SimpleObservableEvaluator<T> operator-(const alps::SimpleObservableEvaluator<T>& x, const alps::SimpleObservableEvaluator<U>& y)
{
  alps::SimpleObservableEvaluator<T> res(x);
  res -= y;
  return res;
}

/// difference of observable and number
template <class T, class Y>
inline alps::SimpleObservableEvaluator<T> operator-(alps::SimpleObservableEvaluator<T> const& x, const Y& y)
{
  alps::SimpleObservableEvaluator<T> res(x);
  res -= y;
  return res;
}

/// difference of observable and number
template <class T, class Y>
inline alps::SimpleObservableEvaluator<T> operator-(const Y& y, alps::SimpleObservableEvaluator<T> const& x)
{
  alps::SimpleObservableEvaluator<T> res(x);
  res.subtract_from(y);
  return res;
}


/// product of two observables (IBM AIX workaround)
template <class T, class U>
inline alps::SimpleObservableEvaluator<T> operator*(const alps::SimpleObservableEvaluator<T>& x, const alps::SimpleObservableEvaluator<U>& y)
{
  alps::SimpleObservableEvaluator<T> res(x);
  res *= y;
  return res;
}

/// product of observable and number
template <class T, class Y>
inline alps::SimpleObservableEvaluator<T> operator*(alps::SimpleObservableEvaluator<T> const& x, const Y& y)
{
  alps::SimpleObservableEvaluator<T> res(x);
  res *= y;
  return res;
}

/// product of observable and number
template <class T, class Y>
inline alps::SimpleObservableEvaluator<T> operator*(const Y& y, alps::SimpleObservableEvaluator<T> const& x)
{
  alps::SimpleObservableEvaluator<T> res(x);
  res.multiply_to(y);
  return res;
}

/// product of vector and scalar observable
template <class T>
inline alps::SimpleObservableEvaluator<std::valarray<T> > operator*(alps::SimpleObservableEvaluator<std::valarray<T> > const& x, const alps::SimpleObservableEvaluator<T>& y)
{
  alps::SimpleObservableEvaluator<T> res(x);
  res *= y;
  return res;
}

/// product of vector and scalar observable
template <class T>
inline alps::SimpleObservableEvaluator<std::valarray<T> > operator*(const alps::SimpleObservableEvaluator<T>& y, alps::SimpleObservableEvaluator<std::valarray<T> > const& x)
{
  alps::SimpleObservableEvaluator<T> res(x);
  res *= y;
  return res;
}


/// ratio of two observables (IBM AIX workaround)
template <class T, class U>
inline alps::SimpleObservableEvaluator<T> operator/(const alps::SimpleObservableEvaluator<T>& x, const alps::SimpleObservableEvaluator<U>& y)
{
  alps::SimpleObservableEvaluator<T> res(x);
  res /= y;
  return res;
}

/// ratio of observable and number
template <class T, class Y>
inline alps::SimpleObservableEvaluator<T> operator/(alps::SimpleObservableEvaluator<T> const& x, const Y& y)
{
  alps::SimpleObservableEvaluator<T> res(x);
  res /= y;
  return res;
}

/// ratio of number and observable
template <class T, class Y>
inline alps::SimpleObservableEvaluator<T> operator/(const Y& x, alps::SimpleObservableEvaluator<T> const& y)
{
  alps::SimpleObservableEvaluator<T> res(y);
  res.divide(x);
  return res;
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
{ return alps::SimpleObservableEvaluator<T>(x).transform(alps::detail::function_##F<T>(), #F"("+x.name()+")"); }

OBSERVABLE_FUNCTION(exp)
OBSERVABLE_FUNCTION(log)
OBSERVABLE_FUNCTION(sqrt)
OBSERVABLE_FUNCTION(sin)
OBSERVABLE_FUNCTION(cos)
OBSERVABLE_FUNCTION(tan)

#undef OBSERVABLE_FUNCTION

namespace detail {

template <class T> struct function_pow : public std::unary_function<T,double>
{
  function_pow(double p) : pow_(p) {}
  T operator()(T x) const { using std::pow; return pow(x, pow_); }
  double pow_;
};

}

template <class T>
alps::SimpleObservableEvaluator<T>
pow(const alps::SimpleObservableEvaluator<T>& x, double p)
{
  return alps::SimpleObservableEvaluator<T>(x).
    transform(alps::detail::function_pow<T>(p),
              "pow(" + x.name() + "," + ObservableNamingHelper::generate(p)
              + ")");
}

template <class T>
alps::SimpleObservableEvaluator<T>
pow(const alps::SimpleObservableEvaluator<T>& x, int p)
{
  return alps::SimpleObservableEvaluator<T>(x).
    transform(alps::detail::function_pow<T>(p),
              "pow(" + x.name() + ", " + ObservableNamingHelper::generate(p)
              + ")");
}

} // end namespace alps

#endif // ALPS_ALEA_SIMPLEOBSEVAL_H
