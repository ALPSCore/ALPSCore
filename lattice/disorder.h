/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2004 by Matthias Troyer <troyer@comp-phys.orgh>,
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

#ifndef ALPS_LATTICE_DISORDER_H
#define ALPS_LATTICE_DISORDER_H

#include <alps/parser/parser.h>
#include <alps/lattice/unitcell.h>
#include <alps/lattice/graph.h>
#include <alps/parser/xmlstream.h>

namespace alps {
namespace detail {
class BasicVertexReference {
public:
  typedef GraphUnitCell::offset_type offset_type;
  BasicVertexReference() {}
  BasicVertexReference(XMLTag);
  const offset_type& cell_offset() const { return cell_;}
  const offset_type& offset() const { return offset_;}
  unsigned int vertex() const { return vertex_;}
private:
  offset_type cell_;
  offset_type offset_;
  unsigned int vertex_;
};

class VertexReference : public BasicVertexReference {
public:
  VertexReference(XMLTag, std::istream&);
  type_type new_type() const { return new_type_;}
private:
  type_type new_type_;
};


class EdgeReference {
public:
  EdgeReference(XMLTag, std::istream&);
  const BasicVertexReference& source() const { return source_;}
  const BasicVertexReference& target() const { return target_;}
  type_type new_type() const { return new_type_;}
private:
  BasicVertexReference source_;
  BasicVertexReference target_;
  type_type new_type_;
};

}

class DisorderDescriptor
{
public:
  DisorderDescriptor() : disorder_all_vertices_(false), disorder_all_edges_(false) {}
  DisorderDescriptor(XMLTag&, std::istream&);
  
  void write_xml(oxstream&) const;
  
  bool disordered_vertices() const { return disorder_all_vertices_ || !disordered_vertices_.empty();}
  bool disordered_edges() const { return disorder_all_edges_ || !disordered_edges_.empty();}
  bool disordered_sites() const { return disordered_vertices();}
  bool disordered_bonds() const { return disordered_edges();}
  bool disordered() const { return disordered_edges() || disordered_vertices();}
  
  template <class G, class M>
  void disorder_edges(G& g, M& m) const {
    if (!changed_edges_.empty()) 
      boost::throw_exception(
	    std::runtime_error("Changed edges not yet implemented. Please contact troyer@comp-phys.org"));
    if (disorder_all_edges_)
	  alps::disorder_edges(g,m);
	else if(!disordered_edges_.empty())
      boost::throw_exception(
	    std::runtime_error("Disordering special edge types not yet implemented. Please contact troyer@comp-phys.org"));
  }

  template <class G, class M>
  void disorder_vertices(G& g, M& m) const {
    if (!changed_vertices_.empty()) 
      boost::throw_exception(
	    std::runtime_error("Changed vertices not yet implemented. Please contact troyer@comp-phys.org"));
    if (disorder_all_vertices_)
	  alps::disorder_vertices(g,m);
	else if(!disordered_vertices_.empty())
      boost::throw_exception(
	    std::runtime_error("Disordering special vertex types not yet implemented. Please contact troyer@comp-phys.org"));
  }
  
  template <class G>
  void disorder_vertices(G& g) const {
    typename property_map<vertex_type_t,G,unsigned int>::type o=get_or_default(vertex_type_t(),g,0);
    disorder_vertices(g,o);
  }
  
  template <class G>
  void disorder_edges(G& g) const {
    typename property_map<edge_type_t,G,unsigned int>::type o=get_or_default(edge_type_t(),g,0);
    disorder_edges(g,o);
  }

  template <class G> void disorder_sites(G& g) const { disorder_vertices(g);}
  template <class G, class M> void disorder_sites(G& g, M& m) const {disorder_vertices(g,m);} 
  template <class G> void disorder_bonds(G& g) const { disorder_edges(g);}
  template <class G, class M> void disorder_bonds(G& g, M& m) const {disorder_edges(g,m);} 
  
private:
  std::vector<detail::VertexReference> changed_vertices_;
  std::vector<detail::EdgeReference> changed_edges_;
  bool disorder_all_vertices_;
  bool disorder_all_edges_;
  std::vector<type_type> disordered_vertices_;
  std::vector<type_type> disordered_edges_;
};


namespace detail {
template <class IT, class MAP, class T>
T disorder_it(IT start, IT end, MAP& type, T i=0)
{
  for (; start!=end;++start)
    type[*start]=i++;
  return i;
}

template <class IT, class MAP>
unsigned int disorder_it(IT start, IT end, MAP& type)
{
  return disorder_it(start,end,type,0u);
}

}

template <class G, class MAP>
void disorder_vertices(G& g, MAP& type)
{
  detail::disorder_it(boost::vertices(g).first,boost::vertices(g).second,type); 
}

template <class G, class MAP>
void disorder_edges(G& g, MAP& type)
{
  detail::disorder_it(boost::edges(g).first,boost::edges(g).second,type); 
}

template <class G, class MAP>
void disorder_bonds(G& g, MAP& type)
{
  disorder_edges(g,type);
}

template <class G, class MAP>
void disorder_sites(G& g, MAP& t)
{
  disorder_vertices(g,t);
}

} // end namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline alps::oxstream& operator<< (alps::oxstream& out, const alps::DisorderDescriptor& l)
  {
    l.write_xml(out);
    return out;
  }

inline std::ostream& operator<< (std::ostream& out, const alps::DisorderDescriptor& l)
  {
    alps::oxstream xml(out);
    xml << l;
    return out;
  }

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace detail {
#endif

alps::oxstream& operator<< (alps::oxstream&, const alps::detail::BasicVertexReference&);
alps::oxstream& operator<< (alps::oxstream&, const alps::detail::VertexReference&);
alps::oxstream& operator<< (alps::oxstream&, const alps::detail::EdgeReference&);


#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace detail
} // end namespace alps
#endif

#endif // ALPS_LATTICE_DISORDER_H
