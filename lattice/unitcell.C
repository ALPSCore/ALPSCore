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

#include <alps/config.h>

#ifndef ALPS_WITHOUT_XML

#include <alps/lattice/unitcell.h>
#include <alps/utility/vectorio.hpp>
#include <alps/expression.h>
#include <boost/lexical_cast.hpp>

namespace alps {

GraphUnitCell::GraphUnitCell() : dim_(0) {}

GraphUnitCell::GraphUnitCell(const EmptyUnitCell& e) : 
  dim_(alps::dimension(e)) {}

GraphUnitCell::GraphUnitCell(const XMLTag& intag, std::istream& p)
{
  XMLTag tag(intag);
  bool fixed_nvertices=false;
  uint32_t vertex_number=0;
  name_ = tag.attributes["name"];
  if (!tag.attributes.defined("dimension"))
    boost::throw_exception(std::runtime_error("Dimension attribute is missing in <UNITCELL>"));
  dim_ = boost::lexical_cast<uint32_t,std::string>(tag.attributes["dimension"]);
  if (dim_<1)
    boost::throw_exception(std::runtime_error("Illegal dimension "+tag.attributes["dimension"]+ " in unit cell"));
  if (tag.attributes["vertices"]!="") {
      uint32_t nvert=boost::lexical_cast<uint32_t,std::string>(tag.attributes["vertices"]);
      graph_ = graph_type(nvert);
      fixed_nvertices=true;
  }
  
  if (tag.type !=XMLTag::SINGLE)
  while(true) {
    tag=parse_tag(p);
    if(tag.name=="/UNITCELL")
      break;
    else if (tag.name=="VERTEX") {
      coordinate_type coord;
      type_type t=tag.attributes["type"]=="" ? boost::lexical_cast<type_type,int>(0) : boost::lexical_cast<type_type,std::string>(tag.attributes["type"]);
      int id=tag.attributes["id"]=="" ? -1 : boost::lexical_cast<int,std::string>(tag.attributes["id"])-1;
      if (id==-1)
        id=vertex_number++;
      if (id>=(int)(boost::num_vertices(graph_))) {
        if (fixed_nvertices)
          boost::throw_exception(std::runtime_error("too many vertices"));
        for (int i=boost::num_vertices(graph_);i<=id;++i)
          boost::put(vertex_type_t(),graph_,boost::add_vertex(graph_),0);
      }

      if (tag.type!=XMLTag::SINGLE) {
              tag = parse_tag(p);
        if(tag.name=="COORDINATE") {
          if (tag.type!=XMLTag::SINGLE) {
            std::vector<std::string> v;
            read_vector(parse_content(p),v,dimension());
            for (unsigned int i=0;i<v.size();++i)
              coord.push_back(alps::evaluate<double>(v[i]));
            //read_vector(parse_content(p),coord,dimension());
            tag = parse_tag(p);
            if (tag.name!="/COORDINATE")
              boost::throw_exception(std::runtime_error("closing </COORDINATE> tag missing"));
          }
          tag=parse_tag(p);
        }
              if (tag.name!="/VERTEX")
          boost::throw_exception(std::runtime_error("closing </VERTEX> tag missing"));
      }

      boost::graph_traits<graph_type>::vertex_iterator vit = boost::vertices(graph_).first+id;
      boost::put(vertex_type_t(),graph_,*vit,t);
      boost::put(coordinate_t(),graph_,*vit,coord);
            
    }
    else if (tag.name=="EDGE") {
      uint32_t source=0, target=0;
      offset_type source_offset(dimension()), target_offset(dimension());
      
      bool got_source=false;
      bool got_target=false;
      
      type_type t =tag.attributes["type"]=="" ? boost::lexical_cast<type_type>(0) 
        : boost::lexical_cast<type_type>(tag.attributes["type"]);
      
      if (tag.type!=XMLTag::SINGLE) while (true) {
        tag = parse_tag(p);
        if (tag.name=="/EDGE")
          break;
        if (tag.name=="SOURCE") {
          if(got_source)
            boost::throw_exception(std::runtime_error("Got two <SOURCE> tags in <EDGE>"));
          got_source=true;
          source=boost::lexical_cast<uint32_t,std::string>(tag.attributes["vertex"])-1;
          if(tag.attributes["offset"].size())
            read_vector(tag.attributes["offset"],source_offset);
          if (tag.type!=XMLTag::SINGLE) {
            tag = parse_tag(p);
            if (tag.name !="/SOURCE")
               boost::throw_exception(std::runtime_error("</SOURCE> tag missing"));
          }
        }
        else if(tag.name=="TARGET") {
          if(got_target)
            boost::throw_exception(std::runtime_error("Got two <TARGET> tags in <EDGE>"));
          got_target=true;
          target=boost::lexical_cast<uint32_t,std::string>(tag.attributes["vertex"])-1;
          if(tag.attributes["offset"].size())
            read_vector(tag.attributes["offset"],target_offset);
          if (tag.type!=XMLTag::SINGLE)
          {
            tag = parse_tag(p);
            if (tag.name !="/TARGET")
               boost::throw_exception(std::runtime_error("</TARGET> tag missing"));
          }
        }
      }
      if (!got_source || !got_target)
        boost::throw_exception(std::runtime_error("did not get <SOURCE> and <TARGET> in <EDGE>"));

      boost::graph_traits<graph_type>::edge_descriptor edge = boost::add_edge(source,target,graph_).first;
      boost::put(edge_type_t(),graph_,edge,t);
      boost::put(source_offset_t(),graph_,edge,source_offset);
      boost::put(target_offset_t(),graph_,edge,target_offset);
    }
    else
      boost::throw_exception(std::runtime_error("encountered illegal tag <" + tag.name + "> in UNITCELL"));
  }
  update_bond_vectors();
}

void GraphUnitCell::update_bond_vectors()
{
  // calculate bond vectors
  for (graph_type::edge_iterator it = boost::edges(graph_).first; it !=boost::edges(graph_).second ; ++it) {
    offset_type source_offset=boost::get(source_offset_t(),graph_,*it);
    offset_type target_offset=boost::get(target_offset_t(),graph_,*it);
    coordinate_type source_coordinate=boost::get(coordinate_t(),graph_,boost::source(*it,graph_));
    coordinate_type target_coordinate=boost::get(coordinate_t(),graph_,boost::target(*it,graph_));
    coordinate_type bond_coordinate(dimension());
    std::pair<coordinate_type::const_iterator,coordinate_type::const_iterator> scit=alps::coordinates(source_coordinate);
    std::pair<coordinate_type::const_iterator,coordinate_type::const_iterator> tcit=alps::coordinates(target_coordinate);
    std::pair<offset_type::const_iterator,offset_type::const_iterator> soit=alps::coordinates(source_offset);
    std::pair<offset_type::const_iterator,offset_type::const_iterator> toit=alps::coordinates(target_offset);
    std::pair<coordinate_type::iterator,coordinate_type::iterator> bit=alps::coordinates(bond_coordinate);
    while (bit.first != bit.second) {
      if (scit.first !=scit.second) *(bit.first) -= *(scit.first++);
      if (tcit.first !=tcit.second) *(bit.first) += *(tcit.first++);
      if (soit.first !=soit.second) *(bit.first) -= *(soit.first++);
      if (toit.first !=toit.second) *(bit.first) += *(toit.first++);
      ++bit.first;
    }
    if (scit.first != scit.second || tcit.first != tcit.second || soit.first != soit.second || toit.first != toit.second)
      boost::throw_exception(std::logic_error("Iterator range errors in constructing unit cell"));
    boost::put(bond_vector_t(),graph_,*it,bond_coordinate);
  }
}

GraphUnitCell::GraphUnitCell(const std::string& name, std::size_t dim) :
  graph_(), dim_(dim), name_(name) {}

const GraphUnitCell& GraphUnitCell::operator=(const EmptyUnitCell& e)
{
  if (dim_==0) dim_=alps::dimension(e);
  return *this;
}

void GraphUnitCell::write_xml(oxstream& xml) const
{
  xml << start_tag("UNITCELL");
  if (name()!="")
    xml << attribute("name", name());
  xml << attribute("dimension", dimension());
  xml << attribute("vertices", boost::num_vertices(graph_));
  
  typedef boost::graph_traits<graph_type>::vertex_iterator vertex_iterator;
  typedef boost::graph_traits<graph_type>::edge_iterator edge_iterator;
  for (vertex_iterator it=boost::vertices(graph_).first;
                       it!=boost::vertices(graph_).second;++it) {
    xml << start_tag("VERTEX") 
        << attribute("id", boost::get(vertex_index_t(),graph_,*it)+1) 
        << attribute("type", boost::get(vertex_type_t(),graph_,*it));
    if (alps::dimension(boost::get(coordinate_t(),graph_,*it)))
      xml << attribute("coordinate", write_vector(boost::get(coordinate_t(),graph_,*it)));
    xml << end_tag("VERTEX");
  }
  typedef boost::graph_traits<graph_type>::edge_iterator edge_iterator;
  for (edge_iterator it=boost::edges(graph_).first;
                     it!=boost::edges(graph_).second;++it) {
    xml << start_tag("EDGE")
        << attribute("type", boost::get(edge_type_t(),graph_,*it))
        << attribute("vector", write_vector(boost::get(bond_vector_t(),graph_,*it)))
        << no_linebreak;
    xml << start_tag("SOURCE")
        << attribute("vertex",boost::source(*it,graph_)+1);
    if (boost::get(source_offset_t(),graph_,*it).size())
      xml << attribute("offset", write_vector(boost::get(source_offset_t(),graph_,*it)));
    xml << end_tag("SOURCE");
    xml << start_tag("TARGET")
        << attribute("vertex", boost::target(*it,graph_)+1);
    if (boost::get(target_offset_t(),graph_,*it).size())
      xml << attribute("offset", write_vector(boost::get(target_offset_t(),graph_,*it)));
    xml << end_tag("TARGET");
    xml << end_tag("EDGE");
  }
  xml << end_tag("UNITCELL");
}

std::size_t GraphUnitCell::add_vertex(int type, const coordinate_type& coord)
{
  boost::graph_traits<graph_type>::vertex_descriptor
    vd = boost::add_vertex(graph_);
  boost::put(vertex_type_t(), graph_, vd, type);
  boost::put(coordinate_t(), graph_, vd, coord);
  return boost::num_vertices(graph_);
}

std::size_t GraphUnitCell::add_edge(int type,
                                    uint32_t si, const offset_type& so,
                                    uint32_t ti, const offset_type& to)
{
  boost::graph_traits<graph_type>::edge_descriptor
    ed = boost::add_edge(*(boost::vertices(graph_).first + si - 1),
                         *(boost::vertices(graph_).first + ti - 1),
                         graph_).first;
  boost::put(edge_type_t(), graph_, ed, type);
  boost::put(source_offset_t(), graph_, ed, so);
  boost::put(target_offset_t(), graph_, ed, to);
  update_bond_vectors();
  return boost::num_edges(graph_);
}

} // end namespace alps

#endif
