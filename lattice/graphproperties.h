/***************************************************************************
* ALPS++/lattice library
*
* lattice/graphproperties.h    the graph property types
*
* $Id$
*
* Copyright (C) 2001-2003 by Matthias Troyer <troyer@comp-phys.org>
*                            Synge Todo <wistaria@comp-phys.org>
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

#ifndef ALPS_LATTICE_GRAPH_PROPERTIES_H
#define ALPS_LATTICE_GRAPH_PROPERTIES_H

#include <alps/config.h>
#include <alps/lattice/propertymap.h>
#include <boost/limits.hpp>
//#include <boost/cstdint.hpp>
#include <boost/pending/property.hpp>
#include <boost/graph/adjacency_list.hpp>

#include <string>
#include <vector>

namespace alps {

struct vertex_type_t { typedef boost::vertex_property_tag kind; };
struct original_vertex_type_t { typedef boost::vertex_property_tag kind; };
typedef vertex_type_t site_type_t;
typedef original_vertex_type_t original_site_type_t;
struct coordinate_t { typedef boost::vertex_property_tag kind; };
struct parity_t { typedef boost::vertex_property_tag kind; };

struct edge_type_t { typedef boost::edge_property_tag kind; };
struct original_edge_type_t { typedef boost::edge_property_tag kind; };
typedef edge_type_t bond_type_t;
typedef original_edge_type_t original_bond_type_t;
struct source_offset_t { typedef boost::edge_property_tag kind; };
struct target_offset_t { typedef boost::edge_property_tag kind; };
struct boundary_crossing_t { typedef boost::edge_property_tag kind; };
struct graph_name_t { typedef boost::graph_property_tag kind; };
struct dimension_t { typedef boost::graph_property_tag kind; };

using boost::vertex_index_t;
using boost::edge_index_t;

namespace detail {

// helper functions to probe for graph properties

template <class T, class DEFAULT=int>
struct existing_property {
  static const bool result=true;
  typedef T type;
};

template <class DEFAULT>
struct existing_property<boost::detail::error_property_not_found,DEFAULT> {
  static const bool result=false;
  typedef DEFAULT type;
};

template <bool FLAG, class T1, class T2>
struct choose {
  typedef T1 type;
};

template <class T1, class T2>
struct choose<false,T1,T2> {
  typedef T2 type;
};

} // end namespace detail



template <class Property, class Graph, class Default=int>
struct has_property {
  static const bool edge_property=false;
  static const bool vertex_property=false;
  static const bool graph_property=false;
  static const bool any_property=false;
  typedef Default vertex_property_type;
  typedef Default edge_property_type;
  typedef Default graph_property_type;
  typedef Default property_type;
  typedef property_type type;
};

template <class s1, class s2, class s3, class VP, class EP, class GP, class s4, class P, class D>
struct has_property<P, boost::adjacency_list<s1,s2,s3,VP,EP,GP,s4>, D>
{
  typedef boost::adjacency_list<s1,s2,s3,VP,EP,GP,s4> Graph;
  static const bool edge_property = detail::existing_property<
    typename boost::property_value<EP,P>::type,D>::result;
  static const bool vertex_property = detail::existing_property<
    typename boost::property_value<VP,P>::type,D>::result;
  static const bool graph_property = detail::existing_property<
    typename boost::property_value<GP,P>::type,D>::result;
  static const bool any_property =
    edge_property || vertex_property || graph_property;
  typedef typename detail::existing_property<
    typename boost::property_value<EP,P>::type,D>::type edge_property_type;
  typedef typename detail::existing_property<
    typename boost::property_value<VP,P>::type,D>::type vertex_property_type;
  typedef typename detail::existing_property<
    typename boost::property_value<GP,P>::type,D>::type graph_property_type;
  typedef typename detail::choose<edge_property,edge_property_type,
    typename detail::choose<vertex_property,vertex_property_type,
    graph_property_type>::type>::type property_type;
  typedef property_type type;
};

template <class s1, class s2, class s3, class VP, class EP, class GP, class s4, class P, class D>
struct has_property<P, const boost::adjacency_list<s1,s2,s3,VP,EP,GP,s4>, D>
{
  typedef boost::adjacency_list<s1,s2,s3,VP,EP,GP,s4>  Graph;
  static const bool edge_property = detail::existing_property<
    typename boost::property_value<EP,P>::type,D>::result;
  static const bool vertex_property = detail::existing_property<
    typename boost::property_value<VP,P>::type,D>::result;
  static const bool graph_property = detail::existing_property<
    typename boost::property_value<GP,P>::type,D>::result;
  static const bool any_property =
    edge_property || vertex_property || graph_property;
  typedef typename detail::existing_property<
    typename boost::property_value<EP,P>::type,D>::type edge_property_type;
  typedef typename detail::existing_property<
    typename boost::property_value<VP,P>::type,D>::type vertex_property_type;
  typedef typename detail::existing_property<
    typename boost::property_value<GP,P>::type,D>::type graph_property_type;
  typedef typename detail::choose<edge_property,edge_property_type,
    typename detail::choose<vertex_property,vertex_property_type,
    graph_property_type>::type>::type property_type;
  typedef property_type type;
};

template <class P, class G, class Default>
struct property_map
{
  typedef 
    typename detail::choose<has_property<P,G>::graph_property,
      typename has_property<P,G>::graph_property_type&,
      typename detail::choose<has_property<P,G>::any_property,
        typename boost::property_map<G,P>::type,
        singleton_property_map<Default> 
      >::type
    >::type type;
};

template <class P, class G, class Default>
struct property_map<P, const G, Default>
{
  typedef 
    typename detail::choose<has_property<P,G>::graph_property,
      const typename has_property<P,G>::graph_property_type&,
      typename detail::choose<has_property<P,G>::any_property,
        typename boost::property_map<G,P>::const_type,
        singleton_property_map<Default> 
      >::type
    >::type type;
};

namespace detail {

template <bool F>
struct put_get_helper
{
  template <class P, class G, class V>
  static singleton_property_map<V> get (P, const G&, const V& v)
  { return singleton_property_map<V>(v);}

  template <class P, class G>
  static typename property_map<P,G,int>::type get_property (P p, G& g) 
  { return boost::get(p,g);}
};

template <>
struct put_get_helper<true>
{
  template <class P, class G, class T>
  static typename property_map<P,G,int>::type get (P p, G& g, const T&) {
    return put_get_helper<has_property<P,G>::graph_property>::get_property(p,g);
  }

  template <class P, class G>
  static typename property_map<P,G,int>::type get_property (P p, G& g) 
  { return boost::get_property(g,p);}
};

} // end namespace detail

template <class P, class G, class V>
inline typename property_map<P,G,V>::type
get_or_default(P p, G& g, const V& v=V())
{
  return detail::put_get_helper<has_property<P,G>::any_property>::get(p,g,v);
}

template <class SRC, class DST, class PROPERTY, bool has_property>
struct copy_property_helper
{
  template <class SRCREF, class DSTREF>
  static void copy(const SRC&, const SRCREF&, DST&, const DSTREF&) {}
};

template <class SRC, class DST, class PROPERTY>
struct copy_property_helper<SRC,DST,PROPERTY,true>
{
  template <class SRCREF, class DSTREF>
  static void copy(const SRC& s, const SRCREF& sr, DST& d, const DSTREF& dr) {
    boost::put(PROPERTY(),d,dr, boost::get(PROPERTY(),s,sr));
  }
};

// the default graph class

namespace detail {
  typedef std::vector<double> coordinate_type;
  typedef int type_type;
}

struct boundary_crossing {
  typedef unsigned int dimension_type;
  typedef int direction_type;

  boundary_crossing() : bc(0) {}
  operator bool() const { return bc!=0;}
  
  direction_type crosses(dimension_type d) const 
  { 
    return (bc&(1<<2*d)) ? +1 : ( (bc&(2<<2*d)) ? -1 : 0);
  }
  
  const boundary_crossing& set_crossing(dimension_type d, direction_type dir) 
  { 
    bc &= ~(3<<2*d);
    bc |= (dir>0 ? (1<<2*d) : (dir <0? (2<<2*d) : 0));
    return *this;
  }
  
  const  boundary_crossing& invert() 
  {
    integer_type rest=bc;
    int dim=0;
    while (rest) {
      invert(dim);
      dim++;
      rest >>=2;
    }
    return *this;
  }
  
private:  
  typedef uint8_t integer_type;
  integer_type bc;
  const  boundary_crossing& invert(dimension_type d) {
    integer_type mask = 3<<2*d;
    if (bc&mask)
      bc^=mask;
    return *this;
  }
};


typedef boost::adjacency_list<boost::vecS,boost::vecS,boost::undirectedS,
			      // vertex property
                              boost::property<coordinate_t,detail::coordinate_type,
			        boost::property<parity_t,int8_t,
				   boost::property<vertex_type_t,int > > >,
			      // edge property
                              boost::property<edge_type_t,int,
			        boost::property<boost::edge_index_t,int,
				  boost::property<boundary_crossing_t,boundary_crossing> > >,
			      // graph property
                              boost::property<dimension_t,std::size_t,
			        boost::property<graph_name_t,std::string > >
			      , boost::vecS> coordinate_graph_type;

} // end namespace alps

#endif // ALPS_LATTICE_GRAPH_PROPERTIES_H
