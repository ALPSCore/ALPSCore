/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>
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

#ifndef ALPS_SCHEDULER_LATTICEPLUGIN_H
#define ALPS_SCHEDULER_LATTICEPLUGIN_H


#include <alps/lattice/latticelibrary.h>

namespace alps {

template <class G=graph_factory<>::graph_type>
class LatticeFactory
{
public:
  typedef G graph_type;
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
  
  LatticeFactory(const alps::Parameters& p)
   : factory_(p), 
     graph_(factory_.graph()), 
     is_bipartite_(set_parity(graph())),
     parity_map_(get_or_default(parity_t(),const_graph(),0.)),
     bond_type_map_(get_or_default(bond_type_t(),const_graph(),0.))
  {
  }
   
  graph_type& graph() { return graph_;}
  const graph_type& graph() const { return graph_;}
  
  sites_size_type num_sites() const { return alps::num_sites(graph());}
  bonds_size_type num_bonds() const { return alps::num_bonds(graph());}
  std::pair<site_iterator,site_iterator> sites() const { return alps::sites(graph());}
  std::pair<bond_iterator,bond_iterator> bonds() const { return alps::bonds(graph());}
  bond_descriptor bond(bonds_size_type i) const { return *(bonds().first+i);}
  neighbors_size_type num_neighbors (const site_descriptor& v) const { return alps::num_neighbors(v,graph());}
  std::pair<neighbor_bond_iterator,neighbor_bond_iterator> neighbor_bonds (const site_descriptor& v) const 
    { return alps::neighbor_bonds(v,graph());}
  std::pair<neighbor_iterator,neighbor_iterator> neighbors (const site_descriptor& v) const 
    { return alps::neighbors(v,graph());}
  site_descriptor neighbor (const site_descriptor& v, neighbors_size_type i) const { return alps::neighbor(v,i,graph());} 
  site_descriptor site(sites_size_type i) const { return alps::site(i,graph());}
  site_descriptor source(const bond_descriptor& b) const { return alps::source_impl(b,graph());}  
  site_descriptor target(const bond_descriptor& b) const { return alps::target_impl(b,graph());}  
  
  vertices_size_type num_vertices() const { return num_vertices(graph());}
  edges_size_type num_edges() const { return num_edges(graph());}
  std::pair<vertex_iterator,vertex_iterator> vertices() const { return vertices(graph());}
  std::pair<edge_iterator,edge_iterator> edges() const { return edges(graph());}
  degree_size_type out_degree (const vertex_descriptor& v) const { return out_degree(v,graph());}
  degree_size_type in_degree (const vertex_descriptor& v) const { return in_degree(v,graph());}
  degree_size_type degree (const vertex_descriptor& v) const { return degree(v,graph());}
  out_edge_iterator out_edges (const vertex_descriptor& v) const { return out_edges(v,graph());}
  in_edge_iterator in_edges (const vertex_descriptor& v) const { return in_edges(v,graph());}
  std::pair<adjacency_iterator,adjacency_iterator> adjacent_vertices (const site_descriptor& v) const 
  { return adjacent_vertices(v,graph());}
  vertex_descriptor vertex(vertices_size_type i) const { return vertex(i,graph());}
  double parity(const site_descriptor& v) const { return parity_map_[v]==0 ? 1. :  parity_map_[v]==1 ? -1. : 0.;}
  bool is_bipartite() const { return is_bipartite_;}
  int bond_type(const bond_descriptor& b) const { return bond_type_map_[b];}
private:
   const graph_type& const_graph() const { return graph_;}
   graph_factory<G> factory_;
   graph_type& graph_;
   bool is_bipartite_;
   typename property_map<parity_t,graph_type,double>::const_type parity_map_;
   typename property_map<bond_type_t,graph_type,int>::const_type bond_type_map_;
};

} // end namespace

#endif
