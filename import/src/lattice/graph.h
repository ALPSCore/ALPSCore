/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_LATTICE_GRAPH_H
#define ALPS_LATTICE_GRAPH_H

#include <alps/lattice/dimensional_traits.h>
#include <alps/parser/parser.h>
#include <alps/parser/xmlstream.h>
#include <alps/lattice/graphproperties.h>
#include <alps/lattice/graph_traits.h>
#include <alps/lattice/coordinate_traits.h>
#include <alps/lattice/propertymap.h>
#include <alps/utilities/vectorio.hpp>

#include <boost/pending/property.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/lexical_cast.hpp>

#include <iostream>

namespace alps {

// XML I/O for graphs

template <class GRAPH>
inline void write_graph_xml(oxstream& out, const GRAPH& g, const std::string& n="")
{
  typedef GRAPH graph_type;
  typedef const GRAPH const_graph_type;
  typedef typename boost::graph_traits<graph_type>::vertex_iterator vertex_iterator;
  typedef typename boost::graph_traits<graph_type>::edge_iterator edge_iterator;

  typename boost::property_map<graph_type,boost::vertex_index_t>::const_type
    vertexindex = boost::get(vertex_index_t(),g);

  typename property_map<vertex_type_t,graph_type,int>::const_type
    vertextype = get_or_default(vertex_type_t(),g,0);

  typename property_map<edge_type_t,graph_type,int>::const_type
    edgetype = get_or_default(edge_type_t(),g,0);

  typename property_map<edge_index_t,graph_type,int>::const_type
    edgeindex = get_or_default(edge_index_t(),g,0);

  typename property_map<coordinate_t,graph_type,std::vector<double> >::const_type
    vertexcoordinate = get_or_default(coordinate_t(),g,std::vector<double>());

  typename property_map<edge_vector_t,graph_type,coordinate_type>::const_type
    edgevector = get_or_default(edge_vector_t(),g,coordinate_type(0));

  out << start_tag("GRAPH");

  std::string name(n);
  if (name=="")
    name=get_or_default(graph_name_t(),g,std::string());
  if(name!="")
    out << attribute("name", name);

  uint32_t dim = get_or_default(dimension_t(),g,uint32_t(0));
  if (dim>0)
    out << attribute("dimension", dim);

  out << attribute("vertices", boost::num_vertices(g))
      << attribute("edges", boost::num_edges(g));

  for (vertex_iterator it=boost::vertices(g).first;
                       it!=boost::vertices(g).second;++it) {
    out << start_tag("VERTEX") << attribute("id", vertexindex[*it]+1);
    if (has_property<vertex_type_t,graph_type>::vertex_property)
      out << attribute("type", vertextype[*it]);
    if (has_property<coordinate_t,graph_type>::vertex_property)
      if(alps::coordinates(vertexcoordinate[*it]).first !=
         alps::coordinates(vertexcoordinate[*it]).second) {
        out << no_linebreak
            << start_tag("COORDINATE")
            << write_vector(vertexcoordinate[*it])
            << end_tag("COORDINATE");
      }
    out << end_tag("VERTEX");
  }
  typedef typename boost::graph_traits<graph_type>::edge_iterator edge_iterator;
  for (edge_iterator it=boost::edges(g).first;
                     it!=boost::edges(g).second;++it) {
    out << start_tag("EDGE") << attribute("source", vertexindex[boost::source(*it,g)]+1)
        << attribute("target", vertexindex[boost::target(*it,g)]+1);
    if (has_property<boost::edge_index_t,graph_type>::edge_property)
      out << attribute("id", edgeindex[*it]+1);
    if (has_property<edge_type_t,graph_type>::edge_property)
      out << attribute("type", edgetype[*it]);
    if (has_property<edge_vector_t,graph_type>::edge_property &&
        alps::dimension(edgevector[*it]))
      out << attribute("vector", write_vector(edgevector[*it]));
    out << end_tag("EDGE");
  }
  out << end_tag("GRAPH");
}

template <class GRAPH>
inline std::string read_graph_xml(std::istream& in, GRAPH& g)
{
  XMLTag tag=parse_tag(in);
  if(tag.name!="GRAPH")
    boost::throw_exception(std::runtime_error("did not get <GRAPH> tag\n"));
  return read_graph_xml(tag,in,g);
}

template <class GRAPH>
inline std::string read_graph_xml(const XMLTag& intag, std::istream& p, GRAPH& g)
{
  typedef GRAPH graph_type;

  //typename boost::property_map<graph_type,vertex_index_t>::const_type
  //vertexindex = boost::get(vertex_index_t(),g);

  typename property_map<vertex_type_t,graph_type,int>::type
    vertextype = get_or_default(vertex_type_t(),g,0);

  typename property_map<edge_type_t,graph_type,int>::type
    edgetype = get_or_default(edge_type_t(),g,0);

  typename property_map<edge_index_t,graph_type,int>::type
    edgeindex = get_or_default(edge_index_t(),g,0);

  typename property_map<coordinate_t,graph_type,std::vector<double> >::type
    vertexcoordinate = get_or_default(coordinate_t(),g,std::vector<double>());

  typedef typename boost::graph_traits<graph_type>::vertex_iterator vertex_iterator;

  XMLTag tag(intag);
  bool fixed_nvertices=false;
  uint32_t vertex_number=0;
  uint32_t num_edges=0;
  std::string name;
  if (tag.attributes["vertices"]!="") {
    uint32_t nvert=boost::lexical_cast<uint32_t,std::string>(tag.attributes["vertices"]);
    g = graph_type(nvert); // graph type needs to have a constructor taking # vertices as argument
    fixed_nvertices=true;
    for (vertex_iterator it = boost::vertices(g).first ; it !=  boost::vertices(g).second ; ++it)
      vertextype[*it]=0;
  }
  get_or_default(dimension_t(),g,uint32_t(0))  = (tag.attributes["dimension"]=="" ? 0 :
    boost::lexical_cast<uint32_t, std::string>(tag.attributes["dimension"]));
  name = tag.attributes["name"];
  get_or_default(graph_name_t(),g,std::string()) = name;

  if (tag.type !=XMLTag::SINGLE)
  while(true) {
    tag=parse_tag(p);
    if(tag.name=="/GRAPH")
      break;
    else if (tag.name=="VERTEX") {
      int id=-1;
      type_type t=0;
      coordinate_type coord;
      t = tag.attributes["type"]=="" ? boost::lexical_cast<type_type,int>(0) : boost::lexical_cast<type_type,std::string>(tag.attributes["type"]);
      id = tag.attributes["id"]=="" ? vertex_number++
           : boost::lexical_cast<int,std::string>(tag.attributes["id"])-1;
      if (id>=int(boost::num_vertices(g))) {
        if (fixed_nvertices)
          boost::throw_exception(std::runtime_error("too many vertices in <GRAPH>"));
        int oldsize=boost::num_vertices(g);
        for (int i=oldsize;i<=id;++i)
          vertextype[boost::add_vertex(g)]=0;
      }

      if (tag.type!=XMLTag::SINGLE) {
              tag = parse_tag(p);
        if(tag.name=="COORDINATE") {
          if (tag.type!=XMLTag::SINGLE) {
            read_vector_resize(parse_content(p),coord);
            tag = parse_tag(p);
            if (tag.name!="/COORDINATE")
              boost::throw_exception(std::runtime_error("closing </COORDINATE> tag missing"));
          }
          tag=parse_tag(p);
        }
              if (tag.name!="/VERTEX")
          boost::throw_exception(std::runtime_error("closing </VERTEX> tag missing"));
      }

      typename boost::graph_traits<graph_type>::vertex_iterator vit = boost::vertices(g).first+id;
      vertextype[*vit]=t;
      vertexcoordinate[*vit]=coord;
    }
    else if (tag.name=="EDGE") {
      uint32_t source, target;
      type_type t = boost::lexical_cast<type_type,int>(0);
      num_edges++;

      source=boost::lexical_cast<uint32_t,std::string>(tag.attributes["source"]);
      target=boost::lexical_cast<uint32_t,std::string>(tag.attributes["target"]);
      if(tag.attributes["type"]!="")
        t=boost::lexical_cast<type_type,std::string>(tag.attributes["type"]);
      // ignoring id
      if (tag.type!=XMLTag::SINGLE)  {
        tag = parse_tag(p);
        if (tag.name!="/EDGE")
          boost::throw_exception(std::runtime_error("Nonempty <EDGE> tag in <GRAPH>"));
      }

      typename boost::graph_traits<graph_type>::edge_descriptor edge =
               boost::add_edge(source-1,target-1,g).first;
      edgetype[edge]=t;
      edgeindex[edge]=num_edges-1;
    }
    else
      boost::throw_exception(std::runtime_error("illegal element <" + tag.name + "> in <GRAPH>"));
  }
  return name;
}

template <class PROPERTY, class SRC, class SRCREF, class DST, class DSTREF>
inline void copy_property(PROPERTY, const SRC& s, const SRCREF& sr, DST& d, DSTREF& dr)
{
  typedef SRC source_graph_type;
  typedef DST destination_graph_type;
  typedef PROPERTY property_type;
  copy_property_helper<source_graph_type,destination_graph_type,property_type,
    (has_property<property_type,source_graph_type>::any_property &&
    has_property<property_type,destination_graph_type>::any_property)
    >::copy(s,sr,d,dr);
}

template <class PROPERTY, class SRC, class DST>
inline void copy_property(PROPERTY, const SRC& s, DST& d)
{
  typedef SRC source_graph_type;
  typedef DST destination_graph_type;
  typedef PROPERTY property_type;
  copy_property_helper<source_graph_type,destination_graph_type,property_type,
    (has_property<property_type,source_graph_type>::graph_property &&
    has_property<property_type,destination_graph_type>::graph_property)
    >::copy(s,d);
}

template <class SRC, class DST>
inline void copy_graph(const SRC& src, DST& dst)
{
  typedef SRC source_graph_type;
  typedef DST destination_graph_type;

  typedef typename boost::graph_traits<source_graph_type>::vertex_iterator vertex_iterator;
  typedef typename boost::graph_traits<source_graph_type>::edge_iterator edge_iterator;
  typedef typename boost::graph_traits<destination_graph_type>::vertex_descriptor vertex_descriptor;
  typedef typename boost::graph_traits<destination_graph_type>::edge_descriptor edge_descriptor;

//  typename property_map<vertex_index_t,destination_graph_type,int>::type
//    vertexindex = get_or_default(vertex_index_t(),dst,0);

  typename property_map<edge_index_t,destination_graph_type,int>::type
    edgeindex = get_or_default(edge_index_t(),dst,0);

  int i=0;
  for (vertex_iterator it=boost::vertices(src).first;
                       it!=boost::vertices(src).second;++it,++i)
  {
    vertex_descriptor v=boost::add_vertex(dst);
    copy_property(vertex_type_t(),src,*it,dst,v);
    copy_property(coordinate_t(),src,*it,dst,v);
    copy_property(parity_t(),src,*it,dst,v);
//    vertexindex[v]=i;
  }

  i=0;
  for (edge_iterator it=boost::edges(src).first;
                     it!=boost::edges(src).second;++it,++i)
  {
    edge_descriptor v=boost::add_edge(boost::source(*it,src), boost::target(*it,src),dst).first;
    copy_property(edge_type_t(),src,*it,dst,v);
    copy_property(boundary_crossing_t(),src,*it,dst,v);
    copy_property(bond_vector_t(),src,*it,dst,v);
    copy_property(bond_vector_relative_t(),src,*it,dst,v);
    edgeindex[v]=i;
  }
//  copy_property(dimension_t(),src,dst);
//  copy_property(graph_name_t(),src,dst);

  get_or_default(dimension_t(),dst,uint32_t(0)) = 
    static_cast<uint32_t>(get_or_default(dimension_t(),src,uint32_t(0)));
  get_or_default(graph_name_t(),dst,std::string()) = 
    static_cast<std::string>(get_or_default(graph_name_t(),src,std::string()));
}

template <class G>
inline int constant_degree(const G& g)
{
  typename boost::graph_traits<G>::vertex_iterator it,end;
  boost::tie(it,end)=boost::vertices(g);
  int degree=0;
  if (it!=end)
  {
    degree=boost::out_degree(*it,g);
    ++it;
  }
  for (; it!=end;++it)
  {
    if (boost::out_degree(*it,g)!=degree)
      return -1;
  }
  return degree;
}

template <class G>
inline std::size_t maximum_edge_type(const G& g)
{
  if(!has_property<edge_type_t,G>::edge_property)
    return 0;

  typename property_map<edge_type_t, const G, int>::type
  edge_type_map = get_or_default(edge_type_t(),g,0);

  typename boost::graph_traits<G>::edge_iterator it,end;
  boost::tie(it,end)=boost::edges(g);

  std::size_t num=0;
  for (; it!=end;++it)
    num = std::max BOOST_PREVENT_MACRO_SUBSTITUTION (num,static_cast<std::size_t>(edge_type_map[*it]));

  return num;
}

template <class G>
inline std::size_t maximum_vertex_type(const G& g)
{
  if(!has_property<vertex_type_t,G>::vertex_property)
    return 0;

  typename boost::graph_traits<G>::vertex_iterator it,end;

  typename property_map<vertex_type_t, const G, unsigned int>::type
  vertex_type_map = get_or_default(vertex_type_t(),g,0u);

  std::size_t num=0;
  for (boost::tie(it,end)=boost::vertices(g); it!=end;++it)
    num = std::max BOOST_PREVENT_MACRO_SUBSTITUTION (num,static_cast<std::size_t>(vertex_type_map[*it]));

  return num;
}

} // end namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace boost {
#endif

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6>
inline std::ostream& operator<<(std::ostream& os,
  const boost::adjacency_list<T0, T1, T2, T3, T4, T5, T6>& g)
{
  alps::oxstream oxs(os);
  alps::write_graph_xml(oxs, g);
  return os;
}

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6>
inline alps::oxstream& operator<<(alps::oxstream& oxs,
  const boost::adjacency_list<T0, T1, T2, T3, T4, T5, T6>& g)
{
  alps::write_graph_xml(oxs, g);
  return oxs;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace boost
#endif

#endif // ALPS_LATTICE_GRAPH_H
