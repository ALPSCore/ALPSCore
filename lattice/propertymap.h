/***************************************************************************
* PALM++/lattice library
*
* lattice/propertymap.h    some useful property maps for graphs
*
* $Id$
*
* Copyright (C) 2003-2003 by Matthias Troyer <troyer@comp-phys.org>
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

