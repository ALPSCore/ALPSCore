/***************************************************************************
* ALPS++/alea library
*
* alps/alea/observableset.h     Monte Carlo measurements class
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

#ifndef ALPS_ALEA_OBSERVABLESET_H
#define ALPS_ALEA_OBSERVABLESET_H

#include <alps/config.h>
#include <alps/alea/observable.h>
#include <alps/parser/parser.h>

#ifndef BOOST_NO_VOID_RETURNS
# include <boost/functional.hpp>
#else
# include <boost/functional_void.hpp>
#endif
#include <boost/filesystem/path.hpp>
#include <map>

#ifndef ALPS_WITHOUT_OSIRIS
# include <alps/osiris.h>
#endif

namespace alps {

namespace detail {

inline void deleteit(Observable& obs)
{
  delete &obs;
}

class AbstractObservableFactory {
public:
  virtual Observable* make(const std::string& name) const =0;
};

template <class T>
class ObservableFactory : public AbstractObservableFactory
{
public:
  Observable* make(const std::string& name ) const { return new T(name);}
;
};

class Factory : public std::map<uint32_t,detail::AbstractObservableFactory*>
{
public:
  Factory();
  ~Factory();
};
} // end namespace detail


/** A class to collect the various measurements performed in a simulation
    It is implemented as a map, with std::string as key type */
    
class ObservableSet: public std::map<std::string,Observable*>
{
  typedef std::map<std::string,Observable*> base_type; 
  static detail::Factory factory_;
 public:
  /// register a class derived from observable to allow automatic loading
  template <class T>
  static void register_type()
  {
#ifndef ALPS_NO_DELETE
    if (factory_.find(T::version)!=factory_.end())
      delete factory_[T::version] ;
#endif
    factory_[T::version] = new detail::ObservableFactory<T>();
  }

  /// the default constructor
  ObservableSet() {};
  /// sign problem support requires a non-trivial copy constructor
  ObservableSet(const ObservableSet& m);
  /// a non-trivial destructor
  virtual ~ObservableSet();
  
  /// sign problem support requires non-trivial assignment
  const ObservableSet& operator=(const ObservableSet& m);
  
  /** merge two observable set.
      If observables with identical names exist in both sets, a merger of the
      observables is attempted. In case of failure an exception is thrown.
      @throws std::bad_cast if merging fails
  */
  const ObservableSet& operator<<(const ObservableSet& obs);
  
  /** merge an observable into the set.
      If an observables with identical names exists, a merger of the
      observables is attempted. In case of failure an exception is thrown.
      @throws std::bad_cast if merging fails
  */
  const ObservableSet& operator<<(const Observable& obs);

  /** add an observable to the set.
      The ObservableSet will delete the object at the end.
      If an observable with the same name exists, it is replaced. This is
      different behavior than operator<<.
  */
  void addObservable(Observable *obs);
  
  /// remove an observable with a given name
  void removeObservable(const std::string& name);
    
  /** get an observable with the given name
      @throws throws a std::runtime_error if no observable exists with the given name
  */
  Observable&       operator[](const std::string& name);
  /** get an observable with the given name
      @throws throws a std::runtime_error if no observable exists with the given name
  */
  const Observable& operator[](const std::string& name) const;
  /// check if an observable with the given name exists
  bool has(const std::string& name) const;
  
  /** reset all observables
      @param flag a value of true means that reset is called after thermalization
                  and information about thermalization should be kept.
  */
  void reset(bool flag=false);

  /// apply a unary function to all observables
  template <class F>
  void do_for_all(F f) const
    {
      for(base_type::const_iterator it=base_type::begin();
	  it!=base_type::end(); ++it)
	{
          if(it->second)
	    f(*(it->second));
	}
    }

  /// apply a unary function to all observables
  template <class F>
  void do_for_all(F f)
    {
      for(base_type::iterator it=base_type::begin();
	  it!=base_type::end();
	  ++it)
	{
          if(it->second)
	    f(*(it->second));
	}
    }
    
      /** get an observable with the given name and type
      @@throws throws a std::runtime_error if no observable exists with the given name
  */

  template <class T>
  T& get(const std::string& name)
    {
      base_type::iterator it=base_type::find(name);
      if (it==base_type::end())
        boost::throw_exception(std::out_of_range("No Observable found with the name: "+name));
      T* retval=dynamic_cast<T*>(((*it).second));
      if (retval==0) 
        boost::throw_exception(std::runtime_error("No Observable found with the right type and name: "+name));
      return *retval;
    }

  /** get an observable with the given name and type
      @@throws throws a std::runtime_error if no observable exists with the given name
  */

  template <class T>
  const T& get(const std::string& name) const
    {
      base_type::const_iterator it=base_type::find(name);
      if (it==base_type::end())
        boost::throw_exception(std::out_of_range("No Observable found with the name: "+name));
      const T* retval=dynamic_cast<const T*>(((*it).second));
      if (retval==0) 
        boost::throw_exception(std::runtime_error("No Observable found with the right type and name: "+name));
      return *retval;
    }

  /// can the thermalization information be set for all observables?
  bool can_set_thermalization_all() const;
  
  /// can the thermalization information be set for any observable?
  bool can_set_thermalization_any() const;
  
  /// set the thermalization information for all observables where it is possible
  void set_thermalization(uint32_t todiscard);
  
  /// get the minimum number of thermalization steps for all observables
  uint32_t get_thermalization() const;
  
  /** the number of runs from which the observables were collected.
      Care must be taken that if some observables did not occur in all sets the
      numbering is not consistent and problems can result. 
  */
  uint32_t number_of_runs() const;

  /** the number of runs from which the observables were collected.
      Care must be taken that if some observables did not occur in all sets the
      numbering is not consistent and problems can result. 
  */
  ObservableSet get_run(uint32_t) const;

#ifndef ALPS_WITHOUT_OSIRIS
  virtual void save(ODump& dump) const;
  virtual void load(IDump& dump);
#endif

  // sign support
  void update_signs();
  
  /// compact the observables to save space, discarding e.g. time series information
  void compact();

  void write_xml(std::ostream& xml, const boost::filesystem::path& =boost::filesystem::path()) const;

  void read_xml(std::istream& infile, const XMLTag& tag);

private:
  typedef std::multimap<std::string,std::string> signmap;	
  signmap signs_;
};



} // end namespace alps


#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

/// output all observables in an ObservableSet
inline std::ostream& operator<<(std::ostream& out,const alps::ObservableSet& obs)
{
#ifndef BOOST_NO_VOID_RETURNS
  obs.do_for_all(boost::bind2nd(boost::mem_fun_ref(&alps::Observable::output),out));
#else
  obs.do_for_all(boost::bind2nd_void(boost::mem_fun_ref(&alps::Observable::output),out));
#endif
  return out;
}

#ifndef ALPS_WITHOUT_OSIRIS

inline alps::ODump& operator<<(alps::ODump& od, const alps::ObservableSet& m)
{ m.save(od); return od; }

inline alps::IDump& operator>>(alps::IDump& id, alps::ObservableSet& m)
{ m.load(id); return id; }

#endif // !ALPS_WITHOUT_OSIRIS

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // ALPS_ALEA_OBSERVABLESET_H
