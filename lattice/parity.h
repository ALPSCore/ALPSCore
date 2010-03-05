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

#ifndef ALPS_LATTICE_PARITY_H
#define ALPS_LATTICE_PARITY_H

#include <alps/config.h>
#include <alps/lattice/graphproperties.h>
#include <boost/classic_spirit.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/undirected_dfs.hpp>
#include <boost/graph/visitors.hpp>
#include <boost/throw_exception.hpp>
#include <boost/version.hpp>
#if BOOST_VERSION < 104000
# include <boost/vector_property_map.hpp>
#else
# include <boost/property_map/vector_property_map.hpp>
#endif
#include <boost/detail/workaround.hpp>
#include <stdexcept>

namespace alps {

template<class Parity, class Graph>
struct parity_traits;

template<class Graph>
struct parity_traits<parity_t, Graph> {
  typedef typename has_property<parity_t, Graph>::type value_type;
#if BOOST_WORKAROUND(__IBMCPP__, <= 700)
  enum {white, black, undefined };
#else
  BOOST_STATIC_CONSTANT(value_type, white = 0);
  BOOST_STATIC_CONSTANT(value_type, black = 1);
  BOOST_STATIC_CONSTANT(value_type, undefined = 2);
#endif
};

#if !BOOST_WORKAROUND(__IBMCPP__, <= 800) && !defined(BOOST_NO_INCLASS_MEMBER_INITIALIZATION)
template<class Graph>
const typename parity_traits<parity_t, Graph>::value_type
  parity_traits<parity_t, Graph>::white;
template<class Graph>
const typename parity_traits<parity_t, Graph>::value_type
  parity_traits<parity_t, Graph>::black;
template<class Graph>
const typename parity_traits<parity_t, Graph>::value_type
  parity_traits<parity_t, Graph>::undefined;
#endif


namespace detail {

template<class Graph, class Parity, class PropertyMap>
class ParityVisitor : public boost::dfs_visitor<>
{
public:
  typedef typename boost::graph_traits<Graph>::vertex_descriptor
    vertex_descriptor;
  typedef typename boost::graph_traits<Graph>::edge_descriptor
    edge_descriptor;

  ParityVisitor(PropertyMap& map, bool* bipartite) :
    map_(map), bipartite_(bipartite) {
    *bipartite_ = true;
  }

  void initialize_vertex(vertex_descriptor s, Graph const&) {
    map_[s] = parity_traits<Parity, Graph>::undefined;
  }

  void start_vertex(vertex_descriptor s, Graph const&) {
    map_[s] = parity_traits<Parity, Graph>::black;
  }

  void tree_edge(edge_descriptor e, Graph const& g) {
    map_[target(e, g)] =
      (map_[source(e, g)] == parity_traits<Parity, Graph>::black ?
       parity_traits<Parity, Graph>::white :
       parity_traits<Parity, Graph>::black);
  }

  void back_edge(edge_descriptor e, const Graph& g) const {
    if (map_[source(e, g)] == map_[target(e, g)])
      *bipartite_ = false;
  }

private:
  PropertyMap map_;
  bool* bipartite_;
};


template<typename Graph>
struct backbone_edges {
  typedef typename boost::graph_traits<Graph>::edge_descriptor edge_descriptor;
  backbone_edges() {}
  backbone_edges(Graph const& g, std::set<int> const& bb) :
    graph(&g), backbone(&bb) {}
  bool operator()(edge_descriptor e) const {
    return backbone->find(get(edge_type_t(), *graph, e)) != backbone->end();
  }
  const Graph* graph;
  const std::set<int>* backbone;
};


template<typename Graph, typename Parity, bool HasParity>
struct parity_helper
{
  template<typename G>
  static bool set_parity(G const&) { return false; }
};

template<typename Graph, typename Parity>
struct parity_helper<Graph, Parity, true>
{
  typedef typename property_map<Parity, Graph,
    typename has_property<Parity, Graph>::type>::type map_type;

  template<typename G>
  static bool set_parity(G& g)
  {
    typedef typename boost::graph_traits<G>::vertex_iterator vertex_iterator;
    typedef ParityVisitor<G, Parity, map_type> visitor_type;

    bool bipartite = true;
    map_type map = boost::get(Parity(), g);

    std::vector<boost::default_color_type> vcolor_map(num_vertices(g));
    std::vector<boost::default_color_type> ecolor_map(num_edges(g));
    undirected_dfs(g, visitor_type(map, &bipartite),
      boost::make_iterator_property_map(vcolor_map.begin(),
        get(vertex_index_t(), g)),
      boost::make_iterator_property_map(ecolor_map.begin(),
        get(edge_index_t(), g)));

    if (!bipartite) {
      vertex_iterator vi, vi_end;
      for (boost::tie(vi, vi_end) = vertices(g); vi != vi_end; ++vi)
        map[*vi] = parity_traits<Parity, Graph>::undefined;
    }
    return bipartite;
  }
};

} // end namespace detail


template<typename Graph, typename Parity>
bool set_parity(Graph& g, alps::Parameters const& p, Parity) {
  using namespace boost::spirit;
  typedef detail::parity_helper<Graph, Parity,
    has_property<Parity, Graph>::vertex_property> parity_helper;

  if (p.defined("BACKBONE_TYPES")) {
    std::vector<int> t;
    rule<> r = *space_p >> *(int_p[push_back_a(t)] >> *space_p);
    if (!parse(p["BACKBONE_TYPES"].c_str(), r).full)
      boost::throw_exception(
        std::invalid_argument("parsing BACKBONE_TYPES failed"));
    std::set<int> v(t.begin(), t.end());
    boost::filtered_graph<Graph, detail::backbone_edges<Graph> >
      fg(g, detail::backbone_edges<Graph>(g, v));
    return parity_helper::set_parity(fg);
  } else {
    return parity_helper::set_parity(g);
  }
}

template<typename Graph>
bool set_parity(Graph& g, alps::Parameters const& p) {
  return set_parity(g, p, parity_t());
}

template<typename Graph>
bool set_parity(Graph& g) {
  return set_parity(g, Parameters(), parity_t());
}

} // end namespace alps

#endif // ALPS_LATTICE_PARITY_H
