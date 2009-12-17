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
#include <alps/alea/obsvalue.h>
#include <alps/alea/recordableobservable.h>
#include <alps/xml.h>

#ifdef ALPS_HAVE_HDF5_PARALLEL
#include <mpi.h>
#endif

#ifdef ALPS_HAVE_HDF5
#include <alps/hdf5.hpp>
#endif

#include <boost/filesystem/path.hpp>
#include <boost/type_traits.hpp>
#include <complex>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <utility>

#ifdef ALPS_HAVE_VALARRAY
# include <valarray>
#endif

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
  Observable(const char* n = "") : name_(n), in_observable_set_(false) {}
  /** standard constructors: just assign the name */
  Observable(const std::string& n) : name_(n), in_observable_set_(false) {}
  Observable(const Observable& o) : name_(o.name_), in_observable_set_(false) {}

  /** dtor */
  virtual ~Observable() {}

  /** clones the observable */
  virtual Observable* clone() const = 0;

  /** returns the name */
  const std::string& name() const;

  /// rename the observable
  virtual void rename(const std::string& newname);

  /** reset the observable */
  virtual ALPS_DUMMY_VOID reset(bool equilibrated=false) = 0;

  /** output the result */
  virtual ALPS_DUMMY_VOID output(std::ostream&) const = 0;

  /** output the result */
  virtual void write_xml(oxstream& oxs, const boost::filesystem::path& fn_hdf5=boost::filesystem::path()) const;
  virtual void write_hdf5(const boost::filesystem::path& fn_hdf, std::size_t realization=0, std::size_t clone=0) const;
  virtual void read_hdf5 (const boost::filesystem::path& fn_hdf, std::size_t realization=0, std::size_t clone=0);


#ifndef ALPS_WITHOUT_OSIRIS
  /// return a version ID uniquely identifying the class
  virtual uint32_t version_id() const =0;

  virtual void save(ODump& dump) const;
  virtual void load(IDump& dump);
#endif

#ifdef ALPS_HAVE_HDF5
	virtual void serialize(hdf5::oarchive &) const {};
	virtual void serialize(hdf5::iarchive &) {};
#endif

// Thermalization support

  /// can the thermalization be changed?
  virtual bool can_set_thermalization() const;
  /// set the number of measurements to be discarded for thermalization
  virtual void set_thermalization(uint32_t todiscard);
  /// get the number of measurements discarded for thermalization
  virtual uint32_t get_thermalization() const=0;

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


  /// compact the data, useful e.g. for time series when I just need th result
  virtual ALPS_DUMMY_VOID compact();

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
    if (dynamic_cast<RecordableObservable<T>*>(this)==0)
      boost::throw_exception(std::runtime_error("Cannot add measurement to observable " + name()));
    dynamic_cast<RecordableObservable<T> *>(this)->add(x);
  }

  template <class T,class S>
  void add(const T& x, S s)
  {
    if (dynamic_cast<RecordableObservable<T>*>(this)==0)
      boost::throw_exception(std::runtime_error("Cannot add measurement to observable " + name()));
    dynamic_cast<RecordableObservable<T> *>(this)->add(x,s);
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

//---------------------------------------------------------------------
// The output operator
//---------------------------------------------------------------------

/// write an observable to a std::ostream
inline std::ostream& operator<<(std::ostream& out, const alps::Observable& m)
{ m.output(out); return out; }

//
// OSIRIS support
//

#ifndef ALPS_WITHOUT_OSIRIS

inline alps::ODump& operator<<(alps::ODump& od, const alps::Observable& m)
{ m.save(od); return od; }

inline alps::IDump& operator>>(alps::IDump& id, alps::Observable& m)
{ m.load(id); return id; }

#endif // !ALPS_WITHOUT_OSIRIS

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // ALPS_ALEA_OBSERVABLE_H
