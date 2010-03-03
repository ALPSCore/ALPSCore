/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2000-2009 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_LATTICE_GRAPHHELPER_H
#define ALPS_LATTICE_GRAPHHELPER_H

#include <alps/lattice/latticelibrary.h>
#include <alps/lattice/disorder.h>
#include <alps/lattice/graph_traits.h>
#include <alps/lattice/parity.h>
#include <alps/lattice/propertymap.h>
#include <alps/multi_array.hpp>
#include <boost/version.hpp>
#if BOOST_VERSION < 104000
# include <boost/vector_property_map.hpp>
#else
# include <boost/property_map/vector_property_map.hpp>
#endif

namespace alps {

namespace detail {

template <bool F>
struct graph_dimension_helper {
  template <class G>
  static std::size_t dimension(const G&) { return 0; }
};

template <>
struct graph_dimension_helper<true> {
  template <class G>
  static  std::size_t dimension(const G& g)
  { return boost::get_property(g,dimension_t()); }
};

}

// helper functions

template <class G>
void throw_if_xyz_defined(const Parameters& p, const G& graph)
{
  // check whether x, y, or z is set
  unsigned int dim = detail::graph_dimension_helper<
    has_property<dimension_t, G>::graph_property>::dimension(graph);
  if ((dim >= 1 && p.defined("x")) ||
      (dim >= 2 && p.defined("y")) ||
      (dim >= 3 && p.defined("z")))
    boost::throw_exception(std::runtime_error(
      "x, y or z is predefined as parameter and used as coordinate"));
}

template <class G>
Parameters coordinate_as_parameter(const G& graph,
  const typename boost::graph_traits<G>::edge_descriptor& edge)
{
  Parameters parms;
  unsigned int dim = detail::graph_dimension_helper<
    has_property<dimension_t, G>::graph_property>::dimension(graph);
  switch (dim) {
  case 3 :
    parms["z"] = 0.5 * (boost::get(coordinate_t(), graph,
                                   boost::source(edge, graph))[2] +
                        boost::get(coordinate_t(), graph,
                                   boost::target(edge, graph))[2]);
    // continue
  case 2 :
    parms["y"] = 0.5 * (boost::get(coordinate_t(), graph,
                                   boost::source(edge, graph))[1] +
                        boost::get(coordinate_t(), graph,
                                   boost::target(edge, graph))[1]);
    // continue
  case 1 :
    parms["x"] = 0.5 * (boost::get(coordinate_t(), graph,
                                   boost::source(edge, graph))[0] +
                        boost::get(coordinate_t(), graph,
                                   boost::target(edge, graph))[0]);
  default :
    break;
  }
  return parms;
}

template <class G>
Parameters coordinate_as_parameter(const G& graph,
  const typename boost::graph_traits<G>::vertex_descriptor& vertex)
{
  Parameters parms;
  unsigned int dim = detail::graph_dimension_helper<
    has_property<dimension_t, G>::graph_property>::dimension(graph);
  switch (dim) {
  case 3 :
    parms["z"] = boost::get(coordinate_t(), graph, vertex)[2];
    // continue
  case 2 :
    parms["y"] = boost::get(coordinate_t(), graph, vertex)[1];
    // continue
  case 1 :
    parms["x"] = boost::get(coordinate_t(), graph, vertex)[0];
  default :
    break;
  }
  return parms;
}


template<typename G>
std::vector<std::string> site_labels(G const& g, int precision = 0)
{
  std::vector<std::string> label;
  typename alps::graph_traits<G>::vertex_iterator itr, itr_end;
  for (boost::tie(itr, itr_end) = vertices(g); itr != itr_end; ++itr)
    label.push_back(coordinate_to_string(get(coordinate_t(), g, *itr), precision));
  return label;
}

template<typename G>
std::vector<std::string> bond_labels(G const& g, int precision = 0)
{
  std::vector<std::string> label;
  typename alps::graph_traits<G>::edge_iterator itr, itr_end;
  for (boost::tie(itr, itr_end) = edges(g); itr != itr_end; ++itr)
    label.push_back(
      coordinate_to_string(get(coordinate_t(), g, source(*itr, g)), precision) + " -- " +
      coordinate_to_string(get(coordinate_t(), g, target(*itr, g)), precision));
  return label;
}


template <class G=coordinate_graph_type>
class graph_helper : public LatticeLibrary
{
public:
  typedef G graph_type;
  typedef lattice_graph<hypercubic_lattice<coordinate_lattice<simple_lattice<GraphUnitCell> > >,graph_type> lattice_type;

  typedef typename graph_traits<graph_type>::vertex_iterator vertex_iterator;
  typedef typename graph_traits<graph_type>::edge_iterator edge_iterator;
  typedef typename graph_traits<graph_type>::out_edge_iterator out_edge_iterator;
  typedef typename graph_traits<graph_type>::in_edge_iterator in_edge_iterator;
  typedef typename graph_traits<graph_type>::edge_descriptor edge_descriptor;
  typedef typename graph_traits<graph_type>::vertex_descriptor vertex_descriptor;
  typedef typename graph_traits<graph_type>::vertices_size_type vertices_size_type;
  typedef typename graph_traits<graph_type>::edges_size_type edges_size_type;
  typedef typename graph_traits<graph_type>::degree_size_type degree_size_type;
  typedef typename graph_traits<graph_type>::adjacency_iterator adjacency_iterator;

  typedef typename graph_traits<graph_type>::site_iterator site_iterator;
  typedef typename graph_traits<graph_type>::bond_iterator bond_iterator;
  typedef typename graph_traits<graph_type>::neighbor_bond_iterator neighbor_bond_iterator;
  typedef typename graph_traits<graph_type>::bond_descriptor bond_descriptor;
  typedef typename graph_traits<graph_type>::site_descriptor site_descriptor;
  typedef typename graph_traits<graph_type>::sites_size_type sites_size_type;
  typedef typename graph_traits<graph_type>::bonds_size_type bonds_size_type;
  typedef typename graph_traits<graph_type>::neighbors_size_type neighbors_size_type;
  typedef typename graph_traits<graph_type>::neighbor_iterator neighbor_iterator;

  typedef typename lattice_traits<lattice_type>::unit_cell_type unit_cell_type;
  typedef typename lattice_traits<lattice_type>::cell_descriptor cell_descriptor;
  typedef typename lattice_traits<lattice_type>::offset_type offset_type;
  typedef typename lattice_traits<lattice_type>::vector_type vector_type;
  typedef typename lattice_traits<lattice_type>::size_type size_type;
  typedef typename lattice_traits<lattice_type>::cell_iterator cell_iterator;
  typedef typename lattice_traits<lattice_type>::momentum_iterator momentum_iterator;
  typedef typename lattice_traits<lattice_type>::basis_vector_iterator basis_vector_iterator;
  typedef typename lattice_traits<lattice_type>::boundary_crossing_type boundary_crossing_type;

  typedef typename property_map<edge_type_t,graph_type,type_type>::const_type edge_type_map_type;
  typedef edge_type_map_type bond_type_map_type;
  typedef typename property_map<vertex_type_t,graph_type,type_type>::const_type vertex_type_map_type;
  typedef vertex_type_map_type site_type_map_type;
  typedef boost::vector_property_map<type_type> inhomogeneous_vertex_type_map_type;
  typedef inhomogeneous_vertex_type_map_type inhomogeneous_site_type_map_type;
  typedef boost::vector_property_map<type_type,typename property_map<edge_index_t,graph_type,type_type>::const_type> inhomogeneous_edge_type_map_type;
  typedef inhomogeneous_edge_type_map_type inhomogeneous_bond_type_map_type;

 graph_helper(std::istream& in, const Parameters& p)
   : LatticeLibrary(in),
         to_delete_(false),
     g_(make_graph(p)),
     is_bipartite_(set_parity(graph(), p)),
     parity_map_(get_or_default(parity_t(),const_graph(),0.)),
     edge_type_map_(get_or_default(edge_type_t(),const_graph(),0.)),
     vertex_type_map_(get_or_default(vertex_type_t(),const_graph(),0.)),
     coordinate_map_(get_or_default(coordinate_t(),const_graph(),0)),
     bond_vector_map_(get_or_default(bond_vector_t(),const_graph(),0)),
     bond_vector_relative_map_(get_or_default(bond_vector_relative_t(),const_graph(),0)),
     inhomogeneous_vertex_type_map_(),
     inhomogeneous_edge_type_map_(get_or_default(edge_index_t(),const_graph(),0)),
     distances_calculated_(false)
  {
    d_.disorder_vertices(graph(),inhomogeneous_vertex_type_map_);
    d_.disorder_edges(graph(),inhomogeneous_edge_type_map_);
  }


  graph_helper(const alps::Parameters& p)
   : LatticeLibrary(p),
         to_delete_(false),
     g_(make_graph(p)),
     is_bipartite_(set_parity(graph(), p)),
     parity_map_(get_or_default(parity_t(),const_graph(),0.)),
     edge_type_map_(get_or_default(edge_type_t(),const_graph(),0)),
     vertex_type_map_(get_or_default(vertex_type_t(),const_graph(),0)),
     coordinate_map_(get_or_default(coordinate_t(),const_graph(),std::vector<double>())),
     inhomogeneous_vertex_type_map_(),
     inhomogeneous_edge_type_map_(get_or_default(edge_index_t(),const_graph(),0)),
     distances_calculated_(false)
  {
    d_.disorder_vertices(graph(),inhomogeneous_vertex_type_map_);
    d_.disorder_edges(graph(),inhomogeneous_edge_type_map_);
  }

  ~graph_helper() { if (to_delete_) delete g_; }

  graph_type& graph() { return *g_; }
  const graph_type& graph() const { return *g_; }

  template<class H>
  typename graph_traits<H>::graph_type& graph(H& g) const
  { return detail::graph_wrap(g); }
  template<class H>
  const typename graph_traits<H>::graph_type& graph(const H& g) const
  { return detail::graph_wrap(g); }

  lattice_type& lattice() { return l_; }
  const lattice_type& lattice() const { return l_; }

  vertices_size_type num_vertices() const
  { return detail::num_vertices_wrap(graph()); }
  edges_size_type num_edges() const
  { return detail::num_edges_wrap(graph()); }
  std::pair<vertex_iterator, vertex_iterator> vertices() const
  { return detail::vertices_wrap(graph());}
  std::pair<edge_iterator, edge_iterator> edges() const
  { return detail::edges_wrap(graph()); }
  degree_size_type out_degree (const vertex_descriptor& v) const
  { return detail::out_degree_wrap(v, graph()); }
  degree_size_type in_degree (const vertex_descriptor& v) const
  { return detail::in_degree_wrap(v, graph()); }
  degree_size_type degree(const vertex_descriptor& v) const
  { return detail::degree_wrap(v, graph()); }
  std::pair<out_edge_iterator, out_edge_iterator>
  out_edges(const vertex_descriptor& v) const
  { return detail::out_edges_wrap(v, graph()); }
  std::pair<in_edge_iterator, in_edge_iterator>
  in_edges(const vertex_descriptor& v) const
  { return detail::in_edges_wrap(v, graph()); }
  std::pair<adjacency_iterator, adjacency_iterator>
  adjacent_vertices(const vertex_descriptor& v) const
  { return detail::adjacent_vertices_wrap(v, graph()); }
  vertex_descriptor vertex(vertices_size_type i) const
  { return detail::vertex_wrap(i, graph()); }
  site_descriptor source(const bond_descriptor& b) const
  { return detail::source_wrap(b, graph()); }
  site_descriptor target(const bond_descriptor& b) const
  { return detail::target_wrap(b, graph()); }

  template<class H>
  typename graph_traits<H>::vertices_size_type num_vertices(const H& g) const
  { return detail::num_vertices_wrap(g); }
  template<class H>
  typename graph_traits<H>::edges_size_type num_edges(const H& g) const
  { return detail::num_edges_wrap(g); }
  template<class H>
  std::pair<typename graph_traits<H>::vertex_iterator,
            typename graph_traits<H>::vertex_iterator>
  vertices(const H& g) const
  { return detail::vertices_wrap(g);}
  template<class H>
  std::pair<typename graph_traits<H>::edge_iterator,
            typename graph_traits<H>::edge_iterator>
  edges(const H& g) const
  { return detail::edges_wrap(g); }
  template<class H>
  typename graph_traits<H>::degree_size_type
  out_degree(const typename graph_traits<H>::vertex_descriptor& v,
             const H& g) const
  { return detail::out_degree_wrap(v, g); }
  template<class H>
  typename graph_traits<H>::degree_size_type
  in_degree(const typename graph_traits<H>::vertex_descriptor& v,
            const H& g) const
  { return detail::in_degree_wrap(v, g); }
  template<class H>
  typename graph_traits<H>::degree_size_type
  degree(const typename graph_traits<H>::vertex_descriptor& v,
         const H& g) const
  { return detail::degree_wrap(v, g); }
  template<class H>
  std::pair<typename graph_traits<H>::out_edge_iterator,
            typename graph_traits<H>::out_edge_iterator>
  out_edges(const typename graph_traits<H>::vertex_descriptor& v,
            const H& g) const
  { return detail::out_edges_wrap(v, g); }
  template<class H>
  std::pair<typename graph_traits<H>::in_edge_iterator,
            typename graph_traits<H>::in_edge_iterator>
  in_edges(const typename graph_traits<H>::vertex_descriptor& v,
           const H& g) const
  { return detail::in_edges_wrap(v, g); }
  template<class H>
  std::pair<typename graph_traits<H>::adjacency_iterator,
            typename graph_traits<H>::adjacency_iterator>
  adjacent_vertices(const typename graph_traits<H>::vertex_descriptor& v,
                    const H& g) const
  { return detail::adjacent_vertices_wrap(v, g); }
  template<class H>
  typename graph_traits<H>::vertex_descriptor
  vertex(typename graph_traits<H>::vertices_size_type i, const H& g) const
  { return detail::vertex_wrap(i, g); }
  template<class H>
  typename graph_traits<H>::site_descriptor
  source(const typename graph_traits<H>::bond_descriptor& b, const H& g) const
  { return detail::source_wrap(b, g); }
  template<class H>
  typename graph_traits<H>::site_descriptor
  target(const typename graph_traits<H>::bond_descriptor& b, const H& g) const
  { return detail::target_wrap(b, g); }

  sites_size_type num_sites() const { return detail::num_sites_wrap(graph()); }
  bonds_size_type num_bonds() const { return detail::num_bonds_wrap(graph()); }
  std::pair<site_iterator, site_iterator>
  sites() const { return detail::sites_wrap(graph()); }
  site_descriptor site(sites_size_type i) const
  { return detail::site_wrap(i, graph()); }
  std::pair<bond_iterator, bond_iterator>
  bonds() const { return detail::bonds_wrap(graph()); }
  bond_descriptor bond(bonds_size_type i) const
  { return detail::bond_wrap(i, graph()); }
  neighbors_size_type num_neighbors (const site_descriptor& v) const
  { return detail::num_neighbors_wrap(v, graph()); }
  std::pair<neighbor_bond_iterator, neighbor_bond_iterator>
  neighbor_bonds(const site_descriptor& v) const
  { return detail::neighbor_bonds_wrap(v, graph()); }
  std::pair<neighbor_iterator, neighbor_iterator>
  neighbors(const site_descriptor& v) const
  { return detail::neighbors_wrap(v, graph()); }
  site_descriptor
  neighbor(const site_descriptor& v, neighbors_size_type i) const
  { return detail::neighbor_wrap(v, i, graph()); }

  template<class H>
  typename graph_traits<H>::sites_size_type num_sites(const H& g) const
  { return detail::num_sites_wrap(g); }
  template<class H>
  typename graph_traits<H>::bonds_size_type num_bonds(const H& g) const
  { return detail::num_bonds_wrap(g); }
  template<class H>
  std::pair<typename graph_traits<H>::site_iterator,
            typename graph_traits<H>::site_iterator>
  sites(const H& g) const { return detail::sites_wrap(g); }
  template<class H>
  typename graph_traits<H>::site_descriptor
  site(typename graph_traits<H>::sites_size_type i, const H& g) const
  { return detail::site_wrap(i, g); }
  template<class H>
  std::pair<typename graph_traits<H>::bond_iterator,
            typename graph_traits<H>::bond_iterator>
  bonds(const H& g) const { return detail::bonds_wrap(g); }
  template<class H>
  typename graph_traits<H>::bond_descriptor
  bond(typename graph_traits<H>::bonds_size_type i, const H& g) const
  { return detail::bond_wrap(i, g); }
  template<class H>
  typename graph_traits<H>::neighbors_size_type
  num_neighbors(const typename graph_traits<H>::site_descriptor& v,
                const H& g) const
  { return detail::num_neighbors_wrap(v, g); }
  template<class H>
  std::pair<typename graph_traits<H>::neighbor_bond_iterator,
            typename graph_traits<H>::neighbor_bond_iterator>
  neighbor_bonds(const typename graph_traits<H>::site_descriptor& v,
                 const H& g) const
  { return detail::neighbor_bonds_wrap(v, g); }
  template<class H>
  std::pair<typename graph_traits<H>::neighbor_iterator,
            typename graph_traits<H>::neighbor_iterator>
  neighbors(const typename graph_traits<H>::site_descriptor& v,
            const H& g) const
  { return detail::neighbors_wrap(v, g); }
  template<class H>
  typename graph_traits<H>::site_descriptor
  neighbor(const typename graph_traits<H>::site_descriptor& v,
           typename graph_traits<H>::neighbors_size_type i, const H& g) const
  { return detail::neighbor_wrap(v, i, g); }

  double parity(const site_descriptor& v) const
  { return parity_map_[v]==0 ? 1. :  parity_map_[v]==1 ? -1. : 0.; }
  bool is_bipartite() const { return is_bipartite_; }

  vertex_type_map_type vertex_type_map() const { return vertex_type_map_; }
  site_type_map_type site_type_map() const { return vertex_type_map_; }
  edge_type_map_type edge_type_map() const { return edge_type_map_; }
  bond_type_map_type bond_type_map() const { return edge_type_map_; }

  type_type vertex_type(const vertex_descriptor& v) const
  { return vertex_type_map_[v]; }
  type_type site_type(const site_descriptor& s) const
  { return vertex_type_map_[s]; }
  type_type edge_type(const edge_descriptor& e) const
  { return edge_type_map_[e]; }
  type_type bond_type(const bond_descriptor& b) const
  { return edge_type_map_[b]; }

  //
  // inhomogeneous
  //

  bool inhomogeneous() const { return d_.inhomogeneous(); }
  bool inhomogeneous_vertices() const { return d_.inhomogeneous_vertices(); }
  bool inhomogeneous_sites() const { return d_.inhomogeneous_sites(); }
  bool inhomogeneous_edges() const { return d_.inhomogeneous_edges(); }
  bool inhomogeneous_bonds() const { return d_.inhomogeneous_bonds(); }

  inhomogeneous_vertex_type_map_type inhomogeneous_vertex_type_map() const
  { return inhomogeneous_vertex_type_map_; }
  inhomogeneous_site_type_map_type inhomogeneous_site_type_map() const
  { return inhomogeneous_vertex_type_map_; }
  inhomogeneous_edge_type_map_type inhomogeneous_edge_type_map() const
  { return inhomogeneous_edge_type_map_; }
  inhomogeneous_bond_type_map_type inhomogeneous_bond_type_map() const
  { return inhomogeneous_edge_type_map_; }

  type_type inhomogeneous_vertex_type(const vertex_descriptor& v) const
  {
    return d_.inhomogeneous_vertices() ?
      inhomogeneous_vertex_type_map_[v] : vertex_type_map_[v];
  }
  type_type inhomogeneous_site_type(const site_descriptor& s) const
  { return inhomogeneous_vertex_type(s); }
  type_type inhomogeneous_edge_type(const edge_descriptor& e) const
  {
    return d_.inhomogeneous_edges() ?
      inhomogeneous_edge_type_map_[e] : edge_type_map_[e];
  }
  type_type inhomogeneous_bond_type(const bond_descriptor& b) const
  { return inhomogeneous_edge_type(b); }

  const vector_type& coordinate(const site_descriptor& s) const { return coordinate_map_[s];}
  std::string coordinate_string(const site_descriptor& s, int precision = 0) const {
    return coordinate_to_string(coordinate(s), precision);
  }
  const vector_type& bond_vector(const bond_descriptor& b) const { return bond_vector_map_[b];}
  const vector_type& bond_vector_relative(const bond_descriptor& b) const { return bond_vector_relative_map_[b];}
  std::size_t dimension() const { return detail::graph_dimension_helper<has_property<dimension_t,G>::graph_property>::dimension(graph());}
  std::pair<momentum_iterator,momentum_iterator> momenta() const { return alps::momenta(lattice());}

  void throw_if_xyz_defined(const Parameters& p,
                            const vertex_descriptor&) const
  { alps::throw_if_xyz_defined(p, graph()); }

  void throw_if_xyz_defined(const Parameters& p,
                            const edge_descriptor&) const
  { alps::throw_if_xyz_defined(p, graph()); }

  Parameters coordinate_as_parameter(const edge_descriptor& e) const
  { return alps::coordinate_as_parameter(graph(), e); }

  Parameters coordinate_as_parameter(const vertex_descriptor& v) const
  { return alps::coordinate_as_parameter(graph(), v); }

  size_type volume() const { return alps::volume(lattice());}
  const unit_cell_type& unit_cell() const { return alps::unit_cell(lattice());}
  cell_descriptor cell(const offset_type& o) const { return alps::cell(o,lattice());}
  std::pair<cell_iterator,cell_iterator> cells() const { return alps::cells(lattice());}
  const offset_type& offset(const cell_descriptor& c) const { return alps::offset(c,lattice());}
  bool on_lattice(const offset_type& o) const { return alps::on_lattice(o,lattice());}
  std::pair<bool,boundary_crossing_type> shift(offset_type& o, const offset_type& s) const { return alps::shift(o,s,lattice());}

  size_type cell_index(const cell_descriptor& c) const
  { return alps::index(c, lattice()); }
  size_type vertex_index(const vertex_descriptor& v) const
  { return boost::get(vertex_index_t(), graph(), v); }
  size_type edge_index(const edge_descriptor& e) const
  { return boost::get(edge_index_t(), graph(), e); }

  size_type index(const cell_descriptor& c) const { return cell_index(c); }
  size_type index(const vertex_descriptor& v) const { return vertex_index(v); }
  size_type index(const edge_descriptor& e) const { return edge_index(e); }

  std::pair<basis_vector_iterator,basis_vector_iterator> basis_vectors() const { return alps::basis_vectors(lattice());}
  std::pair<basis_vector_iterator,basis_vector_iterator> reciprocal_basis_vectors() const { return alps::reciprocal_basis_vectors(lattice());}
  vector_type origin(const cell_descriptor& c) const { return alps::origin(c,lattice());}
  vector_type coordinate(const cell_descriptor& c, const vector_type& p) const { return alps::coordinate(c,p,lattice());}
  vector_type momentum(const vector_type& m) const { return alps::momentum(m,lattice());}

  size_type num_distances() const { return have_lattice_ && !inhomogeneous() ? l_.num_distances() : num_sites()*num_sites(); }

  std::vector<unsigned int> distance_multiplicities() const
  {
    if (have_lattice_ && !inhomogeneous())
      return l_.distance_multiplicities();
    std::vector<unsigned int> m(num_distances(),1u);
    return m;
  }

  std::vector<std::string> momenta_labels(int precision = 0) const
  {
      return l_.momenta_labels(precision);
  }

  std::vector<std::string> distance_labels(int precision = 0) const
  {
    if (have_lattice_ && !inhomogeneous())
      return l_.distance_labels(precision);
    std::vector<std::string> label(num_distances());
    for (vertex_iterator it1=vertices().first; it1 != vertices().second;++it1)
      for (vertex_iterator it2=vertices().first; it2 != vertices().second;++it2)
        if (have_lattice_)
          label[int(*it1)*num_vertices()+int(*it2)] =
            alps::coordinate_to_string(coordinate(*it1), precision)+" -- " +
            alps::coordinate_to_string(coordinate(*it2), precision);
        else
          label[int(*it1)*num_vertices()+int(*it2)] =
            boost::lexical_cast<std::string>(int(*it1)) + " -- " +
            boost::lexical_cast<std::string>(int(*it2));
    return label;
  }

  std::vector<std::string> site_labels(int precision = 0) const {
    return alps::site_labels(graph(), precision);
  }

  std::vector<std::string> bond_labels(int precision = 0) const {
    return alps::bond_labels(graph(), precision);
  }

  size_type distance(vertex_descriptor x, vertex_descriptor y) const
  {
    if (inhomogeneous() ||!have_lattice_)
      return size_type(x)*num_sites()+size_type(y);
    if (!distances_calculated_)
      calculate_distances();
    return distance_lookup_[int(x)][int(y)];
  }

  void calculate_distances() const
  {
    distance_lookup_.resize(boost::extents[num_sites()][num_sites()]);
    for (vertex_iterator it1=vertices().first; it1 != vertices().second;++it1)
      for (vertex_iterator it2=vertices().first; it2 != vertices().second;++it2)
        distance_lookup_[int(*it1)][int(*it2)]=(have_lattice_ && !inhomogeneous() ? l_.distance(*it1,*it2) : (*it1)*num_sites()+(*it2));
    distances_calculated_=true;
  }

  std::vector<std::pair<std::complex<double>,std::vector<std::size_t> > > translations(const vector_type& k) const
  {
    if (have_lattice_&& !inhomogeneous())
      return l_.translations(k);
    else
      return std::vector<std::pair<std::complex<double>,std::vector<std::size_t> > >();
  }

  std::vector<int> translation_directions() const
  {
    if (have_lattice_ && !inhomogeneous())
      return l_.translation_directions();
    else
      return std::vector<int>();
  }

  std::vector<vector_type> translation_momenta() const
  {
    if (have_lattice_ && !inhomogeneous())
      return l_.translation_momenta();
    else
      return std::vector<vector_type>();
  }


private:
  graph_helper(const graph_helper& o)
   : LatticeLibrary(o),
     to_delete_(o.to_delete_),
     g_(to_delete_ ? new graph_type(*o.g_) : o.g_),
     is_bipartite_(set_parity(graph())),
     parity_map_(get_or_default(parity_t(),const_graph(),0.)),
     edge_type_map_(get_or_default(edge_type_t(),const_graph(),0)),
     vertex_type_map_(get_or_default(vertex_type_t(),const_graph(),0)),
     coordinate_map_(get_or_default(coordinate_t(),const_graph(),std::vector<double>())),
     inhomogeneous_vertex_type_map_(),
     inhomogeneous_edge_type_map_(get_or_default(edge_index_t(),const_graph(),0)),
     distances_calculated_(false)
  {
    d_.disorder_vertices(graph(),inhomogeneous_vertex_type_map_);
    d_.disorder_edges(graph(),inhomogeneous_edge_type_map_);
  }

  const graph_helper& operator=(const graph_helper&) {return *this;}
    graph_type* make_graph(const Parameters& p);
  const graph_type& const_graph() const { return *g_;}

  lattice_type l_;
  bool to_delete_;
  InhomogeneityDescriptor d_;
  graph_type* g_;
  bool is_bipartite_;
  typename property_map<parity_t,graph_type,double>::const_type parity_map_;
  edge_type_map_type edge_type_map_;
  typename property_map<edge_index_t,graph_type,unsigned int>::const_type edge_index_map_;
  vertex_type_map_type vertex_type_map_;
  typename property_map<coordinate_t,graph_type,vector_type>::const_type coordinate_map_;
  typename property_map<bond_vector_t,graph_type,vector_type>::const_type bond_vector_map_;
  typename property_map<bond_vector_relative_t,graph_type,vector_type>::const_type bond_vector_relative_map_;
  inhomogeneous_vertex_type_map_type inhomogeneous_vertex_type_map_;
  inhomogeneous_edge_type_map_type inhomogeneous_edge_type_map_;
  bool have_lattice_;
  mutable bool distances_calculated_;
  mutable boost::multi_array<size_type,2> distance_lookup_;
};


template <class G>
G* graph_helper<G>::make_graph(const Parameters& parms)
{
  std::string name;
  bool have_graph = parms.defined("GRAPH");
  bool have_lattice = parms.defined("LATTICE");
  graph_type* g;

  if (have_lattice && have_graph)
    boost::throw_exception(std::runtime_error("both GRAPH and LATTICE were specified"));
  if (have_graph)
    name = static_cast<std::string>(parms["GRAPH"]);
  if (have_lattice)
    name = static_cast<std::string>(parms["LATTICE"]);
  if (have_lattice && has_lattice(name)) {
    LatticeGraphDescriptor desc(lattice_descriptor(name));
    desc.set_parameters(parms);
    l_ = lattice_type(desc);
    d_ = desc.inhomogeneity();
    g = &(l_.graph());
    to_delete_=false;
    have_lattice_=true;
  }
  else if ((have_lattice || have_graph) && has_graph(name)) {
    g = new graph_type();
    get_graph(*g,name);
    to_delete_=true;
    have_lattice_=false;
  }
  else if (parms.defined("UNITCELL")) {
    // check for unit cell
    name = static_cast<std::string>(parms["UNITCELL"]);
    LatticeGraphDescriptor desc(name,unitcells_);
    desc.set_parameters(parms);
    l_ = lattice_type(desc);
    d_ = desc.inhomogeneity();
    g = &(l_.graph());
    to_delete_=false;
    have_lattice_=true;
  }
  else{
    boost::throw_exception(std::runtime_error("could not find graph/lattice specified in parameters"));
    g=NULL;
  }
  return g;
}

} // end namespace

#endif
