/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2005 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_LATTICE_PARITY_H
#define ALPS_LATTICE_PARITY_H

#include <alps/config.h>
#include <alps/math.hpp>
#include <alps/lattice/graphproperties.h>

#include <boost/graph/undirected_dfs.hpp>
#include <boost/graph/visitors.hpp>
#include <boost/throw_exception.hpp>
#include <boost/vector_property_map.hpp>
#include <stdexcept>

namespace alps {

template<class Parity, class Graph>
struct parity_traits;

template<class Graph>
struct parity_traits<parity_t, Graph> {
  typedef typename has_property<parity_t, Graph>::type value_type;
  BOOST_STATIC_CONSTANT(value_type, white = 0);
  BOOST_STATIC_CONSTANT(value_type, black = 1);
  BOOST_STATIC_CONSTANT(value_type, undefined = 2);
};

namespace detail {

template<class Graph, class Parity, class PropertyMap>
class ParityVisitor : public boost::dfs_visitor<>
{
public:
  typedef typename boost::graph_traits<Graph>::vertex_descriptor
    vertex_descriptor;
  typedef typename boost::graph_traits<Graph>::edge_descriptor
    edge_descriptor;

  ParityVisitor(PropertyMap& map, bool* check) :
    p_(parity_traits<Parity, Graph>::white), map_(map), check_(check)
  { *check_ = true; }

  void discover_vertex(vertex_descriptor s, const Graph&)
  {
    flip();
    map_[s] = p_;
  }
  void back_edge(edge_descriptor e, const Graph& g) { check(e, g); }
  void finish_vertex(vertex_descriptor, const Graph&) { flip(); }

protected:
  void flip()
  {
    p_ = is_equal(p_, parity_traits<Parity, Graph>::white) ?
      parity_traits<Parity, Graph>::black :
      parity_traits<Parity, Graph>::white;
  }
  void check(edge_descriptor e, const Graph& g) {
    if (is_equal(map_[boost::source(e, g)], map_[boost::target(e, g)])) 
      *check_ = false;
  }

private:
  typename parity_traits<Parity, Graph>::value_type p_;
  PropertyMap map_;
  bool* check_;
};

} // end namespace detail

template <class Graph, class Parity, class Map>
bool set_parity(const Graph& g, Parity, Map map)
{
  typedef typename detail::ParityVisitor<Graph, Parity, Map> visitor_type;

  bool check = true;

  std::vector<boost::default_color_type> vcolor_map(boost::num_vertices(g));
  std::vector<boost::default_color_type> ecolor_map(boost::num_edges(g));
  boost::undirected_dfs(g, visitor_type(map, &check),
    boost::make_iterator_property_map(vcolor_map.begin(),
      boost::get(vertex_index_t(), g)),
    boost::make_iterator_property_map(ecolor_map.begin(),
      boost::get(edge_index_t(), g)));

  if (!check) {
    typename boost::graph_traits<Graph>::vertex_iterator vi, vi_end;
    for (boost::tie(vi, vi_end) = boost::vertices(g); vi != vi_end; ++vi)
      map[*vi] = parity_traits<Parity, Graph>::undefined;
  }
  return check;
}

namespace detail {

template<bool HasParity>
struct helper
{
  template<class Graph, class Parity>
  static bool set_parity(Graph&, Parity) { return false; }
};

template<>
struct helper<true>
{
  template<class Graph, class Parity>
  static bool set_parity(Graph& g, Parity)
  {
    typename property_map<Parity, Graph,
      typename has_property<Parity, Graph>::type>::type
      map = boost::get(Parity(), g);
    return alps::set_parity(g, Parity(), map);
  }
};

} // end namespace detail


template<class Graph, class Parity>
inline bool set_parity(Graph& g, Parity)
{
  return detail::helper<has_property<Parity, Graph>::vertex_property>
    ::set_parity(g, Parity());
}

template<class Graph>
inline bool set_parity(Graph& g)
{
  return set_parity(g, parity_t());
}

} // end namespace alps

#endif // ALPS_LATTICE_PARITY_H
