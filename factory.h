/***************************************************************************
* ALPS++/alea library
*
* alps/alea/factory.h     a factory class
*
* $Id$
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*
* This software is part of the ALPS library, published under the 
* ALPS Library License; you can use, redistribute it and/or modify 
* it under the terms of the License, either version 1 or (at your option) 
* any later version.
*
* You should have received a copy of the ALPS Library License along with 
* the ALPS Library; see the file License.txt. If not, the license is also 
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
**************************************************************************/

#ifndef ALPS_FACTORY_H
#define ALPS_FACTORY_H

#include <alps/config.h>
#include <boost/shared_ptr.hpp>
#include <boost/throw_exception.hpp>
#include <map>
#include <stdexcept>

namespace alps {

namespace detail {

template <class BASE>
class abstract_creator {
public:
  typedef BASE base_type;
  virtual base_type* create() const =0;
};

template <class BASE, class T>
class creator : public abstract_creator<BASE>
{
public:
  typedef BASE base_type;
  base_type* create() const { return new T();}
};

}

template <class KEY,class BASE>
class factory
{
public:
  typedef BASE base_type;
  typedef KEY key_type;
  typedef boost::shared_ptr<detail::abstract_creator<base_type> > pointer_type;

  factory() {}

  template <class T>
  bool register_type(key_type k) {
    bool isnew=(creators_.find(k)==creators_.end());
    creators_[k] = pointer_type(new detail::creator<BASE,T>());
    return isnew;
  }

  template <class T>
  bool unregister_type(key_type k) 
  {
    iterator it = creators_.find(k);
    if (it == creators_.end()) return false;
    creators_.erase(it);
    return true;
  }

  base_type* create(key_type k) const
  {
    const_iterator it = creators_.find(k);
    if (it == creators_.end() || it->second == 0)
      boost::throw_exception(std::runtime_error("Type not registered in alps::factory::create"));
    return it->second->create();
  }

private:
  typedef std::map<key_type,pointer_type> map_type;
  typedef typename map_type::iterator iterator;
  typedef typename map_type::const_iterator const_iterator;
  map_type creators_;
};

} // end namespace alps

#endif // ALPS_ALEA_OBSERVABLESET_H
