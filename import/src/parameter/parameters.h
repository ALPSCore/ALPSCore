/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2009 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_PARAMETER_PARAMETERS_H
#define ALPS_PARAMETER_PARAMETERS_H

// for MSVC
#if defined(_MSC_VER)
# pragma warning(disable:4251)
#endif

#include "parameter.h"
#include <alps/osiris/dump.h>
#include <alps/parser/parser.h>
#include <alps/xml.h>
#include <boost/foreach.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/throw_exception.hpp>
#include <list>
#include <map>
#include <stdexcept>
#include <string>

#include <alps/hdf5.hpp>

/// \file parameters.h
/// \brief classes to store simulation parameters

namespace alps {

/// \brief a class storing a set of parameters
///
/// the class acts like an associative array but at the same time remembers the order in which elements were added
class ALPS_DECL Parameters
{
public:
  /// the key (parameter name) is a string
  typedef std::string                     key_type;
  /// the parameter value is a String Value, able to store any type in a text representation
  typedef StringValue                     value_type;
  /// the name-value pair is stored as a Parameter
  typedef Parameter                       parameter_type;

  /// the type of container used internally to store the sequential order
  typedef std::list<parameter_type>       list_type;
  /// an integral type to store the number oif elements
  typedef list_type::size_type            size_type;

  /// the pointer type
  typedef parameter_type *            pointer_type;
  /// the const pointer type
  typedef const parameter_type *      const_pointer_type;
  /// the reference type
  typedef parameter_type &            reference_type;
  /// the const reference type
  typedef const parameter_type &      const_reference_type;
  /// \brief the iterator type
  ///
  /// iteration goes in the order of insertion into the class, not alphabetically like in a std::map
  typedef list_type::iterator         iterator;
  /// \brief the const iterator type
  ///
  /// iteration goes in the order of insertion into the class, not alphabetically like in a std::map
  typedef list_type::const_iterator   const_iterator;

  /// the type of container used internally to implment the associative array access
  typedef std::map<key_type, iterator>    map_type;

  /// an empty container of parameters
  Parameters() {}
  /// parameters read from a text file
  Parameters(std::istream& is) { parse(is); }
  /// paramaters read from a hdf5 file
  Parameters(alps::hdf5::archive & ar) {
    std::string context = ar.get_context();
    ar.set_context("/parameters");
    load(ar);
    ar.set_context(context);
  }

  /// copy constructor
  Parameters(Parameters const& params) : list_(params.list_), map_() {
    for (iterator itr = list_.begin(); itr != list_.end(); ++itr) map_[itr->key()] = itr;
  }

  /// assignment operator
  Parameters& operator=(Parameters const& rhs) {
    list_ = rhs.list_;
    map_.clear();
    for (iterator itr = list_.begin(); itr != list_.end(); ++itr) map_[itr->key()] = itr;
    return *this;
  }

  /// read parameters from a text file
  void parse(std::istream& is, bool replace_env = true);

  /// replace '${FOO}' in each parameter with the the content of environment variable FOO
  void replace_envvar();

  /// erase all parameters
  void clear() { list_.clear(); map_.clear(); }
  /// the number of parameters
  size_type size() const { return list_.size(); }

  /// returns true if size == 0
  bool empty() const { return map_.empty();}

  /// does a parameter with the given name exist?
  bool defined(const key_type& k) const { return (map_.find(k) != map_.end());}

  /// accessing parameters by key (name)
  value_type& operator[](const key_type& k) {
    if (defined(k)) {
      return map_.find(k)->second->value();
    } else {
      push_back(k, value_type());
      return list_.rbegin()->value();
    }
  }

  /// accessing parameters by key (name)
  const value_type& operator[](const key_type& k) const {
    if (!defined(k))
      boost::throw_exception(std::runtime_error("parameter " + k + " not defined"));
    return map_.find(k)->second->value();
  }

   /// \brief erase a parameter with a specific key (this takes O(N) time)
  /// \param k the parameter key (name)
  void erase(key_type const& k) {
    map_type::iterator itr = map_.find(k);
    if (itr != map_.end()) {
      list_.erase(itr->second);
      map_.erase(itr);
    } else {
      std::cerr<<"key not found!"<<std::endl;
    }
  }

  /// \brief returns the value or a default
  /// \param k the key (name) of the parameter
  /// \param v the default value
  /// \return if a parameter with the given name \a k exists, its value is returned, otherwise the default v
  value_type value_or_default(const key_type& k, const value_type& v) const {
    return defined(k) ? (*this)[k] : v;
  }

  /// \brief returns the value or a default
  /// \param k the key (name) of the parameter
  /// \param v the default value
  /// \return if a parameter with the given name \a k exists, its value is returned, otherwise the default v
  value_type required_value(const key_type& k) const {
    if (!defined(k))
      boost::throw_exception(std::runtime_error("parameter " + k + " not defined"));
    return map_.find(k)->second->value();
  }

  /// an iterator pointing to the beginning of the parameters
  iterator begin() { return list_.begin(); }
  /// a const iterator pointing to the beginning of the parameters
  const_iterator begin() const { return list_.begin(); }
  /// an iterator pointing past the of the parameters
  iterator end() { return list_.end(); }
  /// a const iterator pointing past the of the parameters
  const_iterator end() const { return list_.end(); }

  /// \brief appends a new parameter to the container
  /// \param p the parameter
  /// \param allow_overwrite indicates whether existing parameters may be overwritten
  /// \throw a std::runtime_error if the parameter key is empty or if it exists already and \a allow_overwrite is false
  void push_back(const parameter_type& p, bool allow_overwrite = false);

  /// \brief appends a new parameter to the container
  /// \param k the parameter key (name)
  /// \param v the parameter value
  /// \param allow_overwrite indicates whether existing parameters may be overwritten
  /// \throw a std::runtime_error if the parameter key \a k is empty or if it exists already and \a allow_overwrite is false
  void push_back(const key_type& k, const value_type& v, bool allow_overwrite = false) {
    push_back(Parameter(k, v), allow_overwrite);
  }

  /// \brief set a parameter value, overwriting any existing value
  Parameters& operator<<(const parameter_type& p) {
    (*this)[p.key()] = p.value();
    return *this;
  }

  /// \brief set parameter values, overwriting any existing value
  Parameters& operator<<(const Parameters& params);

  /// \brief set parameter values, without overwriting existing value
  void copy_undefined(const Parameters& p);

  /// read from an XML file, using the ALPS XML parser
  void read_xml(XMLTag tag, std::istream& xml,bool ignore_duplicates=false);
  /// extract the contents from the first <PARAMETERS> element in the XML stream
  void extract_from_xml(std::istream& xml);
  
  BOOST_SERIALIZATION_SPLIT_MEMBER()

  /// support for Boost serialization
  template<class Archive>
  void save(Archive& ar, const unsigned int) const {
    ar & list_;
  }

  /// support for Boost serialization
  template<class Archive>
  void load(Archive& ar, const unsigned int) {
    ar & list_;
    for (iterator itr = list_.begin(); itr != list_.end(); ++itr) map_[itr->key()] = itr;
  }

  void save(hdf5::archive &) const;
  void load(hdf5::archive &);

private:
  list_type list_;
  map_type map_;
};

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

/// write parameters in text-form to a std::ostream
ALPS_DECL std::ostream& operator<<(std::ostream& os, const alps::Parameters& p);

/// parse parameters in text-form from a std::istream
inline std::istream& operator>>(std::istream& is, alps::Parameters& p)
{
  p.parse(is);
  return is;
}

/// ALPS serialization of parameters
inline alps::ODump& operator<<(alps::ODump& od, const alps::Parameters& params) {
  od << uint32_t(params.size());
  BOOST_FOREACH(alps::Parameter const& p, params) od << p;
  return od;
}

/// ALPS de-serialization of parameters
inline alps::IDump& operator>>(alps::IDump& id, alps::Parameters& p)
{
  p.clear();
  uint32_t n(id);
  for (std::size_t i = 0; i < n; ++i) {
    Parameter m;
    id >> m;
    p.push_back(m);
  }
  return id;
}


/// \brief XML output of parameters
///
/// follows the schema on http://xml.comp-phys.org/
inline alps::oxstream& operator<<(alps::oxstream& oxs,
                                  const alps::Parameters& parameters)
{
  oxs << alps::start_tag("PARAMETERS");
  alps::Parameters::const_iterator p_end = parameters.end();
  for (alps::Parameters::const_iterator p = parameters.begin(); p != p_end;
       ++p) oxs << *p;
  oxs << alps::end_tag("PARAMETERS");
  return oxs;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // ALPS_PARAMETER_PARAMETERS_H
