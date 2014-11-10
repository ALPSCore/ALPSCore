/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

/// \file factory.hpp
/// \brief object factories
/// 
/// This header contains an implementation of an object factory

#ifndef ALPS_UTILITY_FACTORY_HPP
#define ALPS_UTILITY_FACTORY_HPP

#include <boost/shared_ptr.hpp>
#include <boost/throw_exception.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/construct.hpp>
#include <boost/function.hpp>

#include <map>
#include <stdexcept>

namespace alps {

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

  /// \brief register a type
  ///
  /// a new derived type is registered by passing the type as template parameter and the key as argument. 
  //// A second call with the same key will override the registration done by the previous call.
  /// \param k the key associated with the type
  /// \returns \c true if a type was already associated with the key
  template <class T>
  bool register_type(key_type k) {
    bool isnew=(creators_.find(k)==creators_.end());
    creators_[k] =  function_type(boost::lambda::bind(
                boost::lambda::constructor<boost::shared_ptr<base_type> >(), 
                boost::lambda::bind(lambda::new_ptr<T>()
              ));
    return isnew;
  }

  /// \brief unregister a type
  ///
  /// the registration information for the key given is deleted
  /// \param k the key to be deleted
  /// \returns \c true if there was a type associated with the key.
  bool unregister_type(key_type k) 
  {
    typename map_type::iterator it = creators_.find(k);
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
  base_type* operator()(key_type k) const
  {
    typename map_type::const_iterator it = creators_.find(k);
    if (it == creators_.end())
      boost::throw_exception(std::runtime_error("Type not registered in alps::factory::create"));
    return it->second();
  }

private:
  typedef boost::function<boost::shared_ptr<base_type>(void) > function_type;
  typedef std::map<key_type,function_type> map_type;
  map_type creators_;
};

} // end namespace alps

#endif // ALPS_ALEA_OBSERVABLESET_H
