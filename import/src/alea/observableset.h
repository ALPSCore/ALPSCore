/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_ALEA_OBSERVABLESET_H
#define ALPS_ALEA_OBSERVABLESET_H

// for MSVC
#if defined(_MSC_VER)
# pragma warning(disable:4275)
#endif


#include <alps/config.h>
#include <alps/hdf5.hpp>
#include <alps/alea/observable.h>
#include <alps/alea/observablefactory.h>
#include <alps/osiris/archivedump.h>
#include <alps/parser/parser.h>
#include <alps/xml.h>
#include <alps/alea.h>

#include <boost/functional.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/serialization/split_member.hpp>
#include <map>

namespace alps {

class ALPS_DECL ObservableSet: public std::map<std::string,Observable*>
{
  typedef std::map<std::string,Observable*> base_type;
  static ObservableFactory factory_;

 public:
  template <class T>
  static void register_observable() { factory_.register_observable<T>();}

  /// the default constructor
  ObservableSet() {};
  /// sign problem support requires a non-trivial copy constructor
  ObservableSet(const ObservableSet& m);
  /// a non-trivial destructor
  virtual ~ObservableSet();

  /// sign problem support requires non-trivial assignment
  ObservableSet& operator=(const ObservableSet& m);

  /** merge two observable set.
      If observables with identical names exist in both sets, a merger of the
      observables is attempted. In case of failure an exception is thrown.
      @throws std::bad_cast if merging fails
  */
  ObservableSet& operator<<(const ObservableSet& obs);

  /** merge an observable into the set.
      If an observables with identical names exists, a merger of the
      observables is attempted. In case of failure an exception is thrown.
      @throws std::bad_cast if merging fails
  */
  ObservableSet& operator<<(const Observable& obs);
  ObservableSet& operator<<(const boost::shared_ptr<Observable>& obs)
  {
    return (*this) << (*obs);
  }

  /** add an observable to the set.
      The ObservableSet will delete the object at the end.
      If an observable with the same name exists, it is replaced. This is
      different behavior than operator<<.
  */
  void addObservable(Observable *obs);
  void addObservable(const Observable& obs);

  /// remove an observable with a given name
  void removeObservable(const std::string& name);

  /** get an observable with the given name
      @throws throws a std::runtime_error if no observable exists with the given name
  */
  Observable& operator[](const std::string& name);
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
  void reset(bool =true /* deprecated flag */);

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
      @throws throws a std::runtime_error if no observable exists with the given name
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
      @throws throws a std::runtime_error if no observable exists with the given name
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

virtual void save(hdf5::archive &) const;
virtual void load(hdf5::archive &);

  BOOST_SERIALIZATION_SPLIT_MEMBER()

  /// support for Boost serialization
  template<class Archive>
  void save(Archive& ar, const unsigned int) const {
    archive_odump<Archive> dump(ar);
    save(dump);
  }

  /// support for Boost serialization
  template<class Archive>
  void load(Archive& ar, const unsigned int) {
    archive_idump<Archive> dump(ar);
    load(dump);
  }

  // sign support
  void update_signs();
  void set_sign(const std::string&);

  /// compact the observables to save space, discarding e.g. time series information
  void compact() {} // deprecated

  void write_xml(oxstream& oxs, const boost::filesystem::path& =boost::filesystem::path()) const;
  void write_xml_with_id(oxstream& oxs, int id,
    const boost::filesystem::path& = boost::filesystem::path()) const;

  void read_xml(std::istream& infile, const XMLTag& tag);

  void write_hdf5(boost::filesystem::path const &, std::size_t realization=0, std::size_t clone=0) const;
  void read_hdf5(boost::filesystem::path const &, std::size_t realization=0, std::size_t clone=0);

  void clear();

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
  obs.do_for_all(boost::bind2nd(boost::mem_fun_ref(&alps::Observable::output),out));
  return out;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // ALPS_ALEA_OBSERVABLESET_H
