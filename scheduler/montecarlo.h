/***************************************************************************
* ALPS++/scheduler library
*
* scheduler/montecarlo.h
*
* $Id$
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_SCHEDULER_MONTECARLO_H
#define ALPS_SCHEDULER_MONTECARLO_H

//=======================================================================
// This file all include files for Monte Carlo simulations
//=======================================================================

#include <alps/scheduler/scheduler.h>
#include <alps/scheduler/worker.h>
#include <alps/scheduler/task.h>
#include <alps/lattice.h>
#include <alps/alea.h>
#include <alps/osiris.h>
#include <boost/smart_ptr.hpp>

namespace alps {
namespace scheduler {

class MCRun : public Worker
{
public:
  MCRun(const ProcessList&,const alps::Parameters&,int);

  void save_worker(ODump&) const;
  void load_worker(IDump&);
  virtual void save(ODump&) const;
  virtual void load(IDump&);

  void write_xml(const boost::filesystem::path& name, const boost::filesystem::path& osirisname="") const;
  const ObservableSet& get_measurements() const { return measurements;}
  ObservableSet get_compacted_measurements() const;

  std::string work_phase();
  void run();
  virtual bool is_thermalized() const;
  bool handle_message(const Process& runmaster,int tag);
protected:
  ObservableSet measurements;
};

class DummyMCRun : public MCRun
{
public:
  DummyMCRun(const ProcessList& w,const alps::Parameters& p,int n);
  DummyMCRun();
  void dostep();
  double work_done() const;
};

class MCSimulation : public Task
{	
public:
  MCSimulation(const ProcessList& w,const boost::filesystem::path& p) : Task(w,p) { construct();}	
  ObservableSet get_measurements(bool compact=false) const;
  MCSimulation& operator<<(const Observable& obs);
private:
  std::string worker_tag() const;
  void write_xml_header(std::ostream&) const;
  void write_xml_trailer(std::ostream&) const;
  void write_xml_body(std::ostream&, const boost::filesystem::path&) const;
  virtual void handle_tag(std::istream&, const XMLTag&);
  ObservableSet measurements;
};

template <class G=graph_factory<>::graph_type>
class LatticeMCRun : public MCRun
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
  
  LatticeMCRun(const ProcessList& w,const alps::Parameters& p,int n)
   : MCRun(w,p,n), factory_(parms), graph_(factory_.graph()) {}
   
  graph_type& graph() { return graph_;}
  const graph_type& graph() const { return graph_;}
  
  sites_size_type num_sites() const { return alps::num_sites(graph());}
  bonds_size_type num_bonds() const { return alps::num_bonds(graph());}
  std::pair<site_iterator,site_iterator> sites() const { return alps::sites(graph());}
  std::pair<bond_iterator,bond_iterator> bonds() const { return alps::bonds(graph());}
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
private:
   graph_factory<G> factory_;
   graph_type& graph_;
};

template <class WORKER>
class SimpleMCFactory : public SimpleFactory<WORKER,MCSimulation>
{
public:
  SimpleMCFactory() {}
};

} // end namespace
} // end namespace

#endif
