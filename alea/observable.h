/***************************************************************************
* PALM++/alea library
*
* alea/observable.h     Monte Carlo observable class
*
* $Id$
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Beat Ammon <ammon@ginnan.issp.u-tokyo.ac.jp>,
*                            Andreas Laeuchli <laeuchli@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
**************************************************************************/

//=======================================================================
// This file defines the abstract base classes used for measurements,
// as well as classes for measurements of T observables, with and
// without a sign problem
//=======================================================================

#ifndef ALPS_ALEA_OBSERVABLE_H
#define ALPS_ALEA_OBSERVABLE_H

#include <alps/config.h>
#include <alps/alea/obsvalue.h>
#include <alps/xml.h>

#include <boost/filesystem/path.hpp>

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
# include <alps/osiris.h>
#endif

namespace alps {

class ObservableSet;

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

class Observable
{
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
  virtual ALPS_DUMMY_VOID reset(bool forthermalization=false) = 0;
  
  /** output the result */
  virtual ALPS_DUMMY_VOID output(std::ostream&) const = 0;

  /** output the result */
  virtual void write_xml(std::ostream&,const boost::filesystem::path& fn_hdf5=boost::filesystem::path()) const;
  virtual void write_xml(oxstream& oxs, const boost::filesystem::path& fn_hdf5=boost::filesystem::path()) const;

#ifndef ALPS_WITHOUT_OSIRIS
  /// return a version ID uniquely identifying the class
  virtual uint32_t version_id() const =0;

  virtual void save(ODump& dump) const;
  virtual void load(IDump& dump);
#endif

/**@name Thermalization support */
//@{
  /// can the thermalization be changed?
  virtual bool can_set_thermalization() const;
  /// set the number of measurements to be discarded for thermalization
  virtual void set_thermalization(uint32_t todiscard);
  /// get the number of measurements discarded for thermalization
  virtual uint32_t get_thermalization() const=0;
//@}

/**@name Sign problem support */
//@{
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
  /// get the name of the observable containing the sign
  virtual const std::string sign_name() const;
//@}
    
/**@name Support for multiple runs */
//@{
  /// get the number of runs which performed measurements for this observable
  virtual uint32_t number_of_runs() const;
  /// extract an observable from a specific run only
  virtual Observable* get_run(uint32_t) const;
//@}

/// compact the data, useful e.g. for time series when I just need th result
  virtual ALPS_DUMMY_VOID compact();

protected:
  virtual void merge(const Observable&);
  /// can this observable be merged with one of the same type  
  virtual bool can_merge() const; 
  /// can this  observable be merged with one of the given type
  virtual bool can_merge(const Observable&) const;
  /// create a copy of the observable that can be merged
  virtual Observable* convert_mergeable() const;
  /// merge this observable with another
  void operator<<(const Observable& o) { merge(o);}

private:
  void added_to_set() { in_observable_set_=true;}
  std::string name_; // the name
  bool in_observable_set_;
};

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
