/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2004 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_LATTICE_GRAPH_H
#define ALPS_LATTICE_GRAPH_H

#include <alps/parser/parser.h>
#include <alps/parser/xmlstream.h>
#include <alps/lattice/graphproperties.h>
#include <alps/lattice/graph_traits.h>
#include <alps/lattice/coordinate_traits.h>
#include <alps/lattice/propertymap.h>
#include <alps/vectorio.h>

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
    
  typename property_map<graph_name_t,graph_type,std::string>::const_type
    graphname = get_or_default(graph_name_t(),g,std::string());

  typename property_map<dimension_t,graph_type,uint32_t>::const_type
    graphdimension = get_or_default(dimension_t(),g,uint32_t(0));
    
  out << start_tag("GRAPH");

  std::string name(n);
  if (name=="")
    name=graphname;
  if(name!="")
    out << attribute("name", name);
  
  uint32_t dim = graphdimension;
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
            << vector_writer(vertexcoordinate[*it])
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
    
  typename property_map<graph_name_t,graph_type,std::string>::type
    graphname = get_or_default(graph_name_t(),g,std::string());

  typename property_map<dimension_t,graph_type,uint32_t>::type
    graphdimension = get_or_default(dimension_t(),g,uint32_t(0));

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
  graphdimension = (tag.attributes["dimension"]=="" ? 0 :
    boost::lexical_cast<uint32_t, std::string>(tag.attributes["dimension"]));
  graphname = name = tag.attributes["name"];
  
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
      if (id>=boost::num_vertices(g)) {
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
    edgeindex[v]=i;
  }
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
inline int32_t maximum_edge_type(const G& g)
{
  if(!has_property<edge_type_t,G>::edge_property)
    return 0;
    
  typename property_map<edge_type_t, const G, int>::type 
  edge_type_map = get_or_default(edge_type_t(),g,0);

  typename boost::graph_traits<G>::edge_iterator it,end;
  boost::tie(it,end)=boost::edges(g);
  
  int num=0;
  for (; it!=end;++it)
    num = std::max(num,edge_type_map[*it]);

  return num;
}

template <class G>
inline unsigned int maximum_vertex_type(const G& g)
{
  if(!has_property<vertex_type_t,G>::vertex_property)
    return 0;
    
  typename boost::graph_traits<G>::vertex_iterator it,end;
  
  typename property_map<vertex_type_t, const G, unsigned int>::type 
  vertex_type_map = get_or_default(vertex_type_t(),g,0u);

  unsigned int num=0;
  for (boost::tie(it,end)=boost::vertices(g); it!=end;++it)
    num = std::max(num,static_cast<unsigned int>(vertex_type_map[*it]));

  return num;
}

} // end namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
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
} // end namespace alps
#endif

#endif // ALPS_LATTICE_GRAPH_H
