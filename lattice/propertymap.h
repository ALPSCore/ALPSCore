/***************************************************************************
* PALM++/lattice library
*
* lattice/propertymap.h    some useful property maps for graphs
*
* $Id$
*
* Copyright (C) 2003-2003 by Matthias Troyer <troyer@comp-phys.org>
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

#ifndef PALM_LATTICE_PROPERTYMAP_H
#define PALM_LATTICE_PROPERTYMAP_H

#include <boost/property_map.hpp>
#include <boost/graph/properties.hpp>

namespace alps {

template <class V> // singleton_property_map and constant_property_map
class singleton_property_map {
public:
  typedef std::size_t key_type;
  typedef V value_type;
  typedef V& reference;
  typedef boost::lvalue_property_map_tag category;

  singleton_property_map(V v=V()) : v_(v) {}

  operator V () const { return v_;}

  const singleton_property_map<V>& operator=(const V& v) { v_=v; return *this;}

  template <class T>
  V& operator[](T ) { return v_;}

  template <class T>
  const V& operator[](T ) const { return v_;}
private:
  V v_;
};


template <class Container, class Graph, class Property>
struct container_property_map
{
  typedef typename Container::value_type value_type;
  typedef typename Container::reference reference;
  typedef typename Container::const_reference const_reference;
  typedef typename Container::iterator iterator;
  typedef typename Container::const_iterator const_iterator;
  typedef typename boost::property_map<Graph,Property>::const_type 
                     index_map_type;
  typedef boost::iterator_property_map<iterator, index_map_type,
              value_type, reference> type;
  typedef boost::iterator_property_map<const_iterator, index_map_type,
              value_type, const_reference> const_type;
};

template <class Container, class Graph, class Property>
typename container_property_map<Container,Graph,Property>::type
make_container_property_map(Container& c, const Graph& g, Property p)
{
  return typename container_property_map<Container,Graph,Property>::type
    (c.begin(),boost::get(p,g));
}

template <class Container, class Graph, class Property>
typename container_property_map<Container,Graph,Property>::const_type
make_const_container_property_map(Container& c, const Graph& g, Property p)
{
  return typename container_property_map<Container,Graph,Property>::type
    (c.begin(),boost::get(p,g));
}

} // end namespace alps

#endif

