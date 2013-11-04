/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2009 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Beat Ammon <ammon@ginnan.issp.u-tokyo.ac.jp>,
*                            Andreas Laeuchli <laeuchli@itp.phys.ethz.ch>,
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

//=======================================================================
// This file defines the abstract base classes used for measurements,
// as well as classes for measurements of T observables, with and
// without a sign problem
//=======================================================================

#ifndef ALPS_ALEA_OBSERVABLE_H
#define ALPS_ALEA_OBSERVABLE_H

#include <alps/config.h>
#include <alps/alea/recordableobservable.h>
#include <alps/xml.h>

#ifdef ALPS_HAVE_HDF5_PARALLEL
#include <mpi.h>
#endif

#include <alps/hdf5.hpp>

#include <boost/filesystem/path.hpp>
#include <boost/type_traits.hpp>
#include <complex>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <utility>
#include <valarray>

#ifndef ALPS_WITHOUT_OSIRIS
# include <alps/osiris/dump.h>
#endif

namespace alps {

class ALPS_DECL ObservableSet;

enum Target {Mean, Error, Variance, Tau};
//=======================================================================
// Observable information error
//
//-----------------------------------------------------------------------

/// an error class if no measurement was performed

class NoMeasurementsError : public std::runtime_error {
public:
  NoMeasurementsError()
   : std::runtime_error("No measurements available.")
   { }
};




//=======================================================================
// Observable
//
// a general observable, abstract base class
// this base class is used to implement certain standard behavior,
// such as a name and the ability to reset the observable
//-----------------------------------------------------------------------

/// the base class for all observables

class ALPS_DECL Observable {
 public:
  friend class ObservableSet;
  typedef uint32_t version_type;
  /** standard constructors: just assign the name */
  Observable(const std::string& n);
  Observable(const Observable& o);

  /** dtor */
  virtual ~Observable();

  /** clones the observable */
  virtual Observable* clone() const;

  /** returns the name */
  const std::string& name() const;

  /// rename the observable
  virtual void rename(const std::string&);

  /** reset the observable */
  virtual ALPS_DUMMY_VOID reset(bool equilibrated=false);

  /** output the result */
  virtual ALPS_DUMMY_VOID output(std::ostream&) const;

  /** output the result */
  virtual void write_xml(oxstream& oxs, const boost::filesystem::path& fn_hdf5=boost::filesystem::path()) const;

#ifndef ALPS_WITHOUT_OSIRIS
  /// return a version ID uniquely identifying the class
  virtual uint32_t version_id() const;

  virtual void save(ODump& dump) const;
  virtual void load(IDump& dump);
#endif

  virtual void save(hdf5::archive &) const;
  virtual void load(hdf5::archive &);

  // Sign problem support

  /// is the observable signed?
  virtual bool is_signed() const;
  /// set the name of the observable containing the sign for this observable
  virtual void set_sign_name(const std::string& signname);
  /// set the observable containing the sign
  virtual void set_sign(const Observable& sign);
  /// clear any previosuly set sign observable
  virtual void clear_sign();
  /// get a reference to the sign observable
  virtual const Observable& sign() const;
  virtual const Observable& signed_observable() const;
  /// get the name of the observable containing the sign
  virtual const std::string sign_name() const;

  // Support for multiple runs

  /// get the number of runs which performed measurements for this observable
  virtual uint32_t number_of_runs() const;
  /// extract an observable from a specific run only
  virtual Observable* get_run(uint32_t) const;

  virtual void merge(const Observable&);
  /// can this observable be merged with one of the same type
  virtual bool can_merge() const;
  /// can this  observable be merged with one of the given type
  virtual bool can_merge(const Observable&) const;
  /// create a copy of the observable that can be merged
  virtual Observable* convert_mergeable() const;

  /// merge this observable with another or add measurement
  template <class T> inline void operator<<(const T& x);

  template <class T>
  void add(const T& x)
  {
    RecordableObservable<T>* obs = dynamic_cast<RecordableObservable<T>*>(this);
    if (obs==0)
      boost::throw_exception(std::runtime_error("Cannot add measurement to observable " + name()));
    obs->add(x);
  }

  template <class T,class S>
  void add(const T& x, S s)
  {
    RecordableObservable<T>* obs = dynamic_cast<RecordableObservable<T>*>(this);
    if (obs==0)
      boost::throw_exception(std::runtime_error("Cannot add measurement to observable " + name()));
    obs->add(x,s);
  }
private:
  void added_to_set() { in_observable_set_=true;}
  std::string name_; // the name
  bool in_observable_set_;
};

namespace detail {
template <bool F>
struct pick_add_merge {};

template<>
struct pick_add_merge<true> {
  template <class T> static void add_or_merge(Observable& obs, const T& x) { obs.merge(x);}
};

template<>
struct pick_add_merge<false> {
  template <class T> static void add_or_merge(Observable& obs, const T& x) { obs.add(x);}
};

}

template <class T>
void Observable::operator<<(const T& x)
{
    detail::pick_add_merge<boost::is_base_and_derived<Observable,T>::value>::add_or_merge(*this,x);
}

} // end namespace alps


#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

/// write an observable to a std::ostream
inline std::ostream& operator<<(std::ostream& out, const alps::Observable& m)
{ m.output(out); return out; }

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // ALPS_ALEA_OBSERVABLE_H
