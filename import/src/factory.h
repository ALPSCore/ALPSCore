/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2005 by Matthias Troyer <troyer@itp.phys.ethz.ch>
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

/// \file factory.h
/// \brief object factories
/// 
/// This header contains an implementation of an object factory

#ifndef ALPS_FACTORY_H
#define ALPS_FACTORY_H

#include <alps/config.h>
#include <boost/shared_ptr.hpp>
#include <boost/throw_exception.hpp>
#include <map>
#include <stdexcept>

namespace alps {

namespace detail {

/// a class to construct objects derived from a given type
/// \param BASE the base class for the objects created
template <class BASE>
class abstract_creator {
public:
/// the type of the abse class
  typedef BASE base_type;
  virtual ~abstract_creator() {}
  /// a virtual function to create an object derived from \c base_type
  virtual base_type* create() const =0;
};

/// a class to default-onstruct an object
/// \param T the type of object to be constructed
/// \param BASE the base class of the object
template <class BASE, class T>
class creator : public abstract_creator<BASE>
{
public:
  /// the type of the abse class
  typedef BASE base_type;
  virtual ~creator() {}
  /// create and default-construct an object of type T
  virtual base_type* create() const { return new T();}
};

}


/// a factory class
/// \param KEY the key type used to identify concrete derived types
/// \param BASE the type of the base class
/// The factory can create instances of default-constructible types derived from the type \a BASE, 
/// where the concrete derived type is specified by a key of type \c KEY. 
/// Each concrete type needs to be associated with a key bey calling the \c register function.

template <class KEY,class BASE>
class factory
{
public:
  /// the type of the base class from which all objects created by the factory are derived
  typedef BASE base_type;
  
  /// the type of the key used to identify derived classes
  typedef KEY key_type;
  
  /// there is only a default constructor
  factory() {}
  virtual ~factory() {}

  /// \brief register a type
  ///
  /// a new derived type is registered by passing the type as template parameter and the key as argument. 
  //// A second call with the same key will override the registration done by the previous call.
  /// \param k the key associated with the type
  /// \returns \c true if a type was already associated with the key
  template <class T>
  bool register_type(key_type k) {
    bool isnew=(creators_.find(k)==creators_.end());
    creators_[k] = pointer_type(new detail::creator<BASE,T>());
    return isnew;
  }

  /// \brief unregister a type
  ///
  /// the registration information for the key given is deleted
  /// \param k the key to be deleted
  /// \returns \c true if there was a type associated with the key.
  bool unregister_type(key_type k) 
  {
    iterator it = creators_.find(k);
    if (it == creators_.end()) return false;
    creators_.erase(it);
    return true;
  }

  /// \brief create an object
  ///
  /// attempts to create an object of the type previously associated with the key.
  /// \param k the key referring to the object type
  /// \returns a pointer to a new object of the type registered with the key
  /// \throws \c std::runtime_error if no type was associated with the key
  base_type* create(key_type k) const
  {
    const_iterator it = creators_.find(k);
    if (it == creators_.end() || it->second == 0)
      boost::throw_exception(std::runtime_error("Type not registered in alps::factory::create"));
    return it->second->create();
  }

private:
  typedef boost::shared_ptr<detail::abstract_creator<base_type> > pointer_type;
  typedef std::map<key_type,pointer_type> map_type;
  typedef typename map_type::iterator iterator;
  typedef typename map_type::const_iterator const_iterator;
  map_type creators_;
};

} // end namespace alps

#endif // ALPS_ALEA_OBSERVABLESET_H
