/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2006 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_LATTICE_LATTICEGRAPH_H
#define ALPS_LATTICE_LATTICEGRAPH_H

#include <alps/config.h>
#include <alps/parameter.h>
#include <alps/lattice/disorder.h>
#include <alps/parser/parser.h>
#include <alps/lattice/graph.h>
#include <alps/lattice/lattice.h>
#include <alps/lattice/coordinatelattice.h>
#include <alps/lattice/hypercubic.h>
#include <alps/utility/vectorio.hpp>

namespace alps {

template<class L, class G> class lattice_graph;

template <class LATTICE, class GRAPH>
inline void make_graph_from_lattice(GRAPH& g,const LATTICE& l, 
    DepletionDescriptor depl_desc = DepletionDescriptor ())
{
  typedef GRAPH graph_type;
  typedef LATTICE lattice_type;
  typedef typename lattice_traits<lattice_type>::unit_cell_type unit_cell_type;
  typedef typename unit_cell_type::graph_type unit_graph_type;
  typedef typename boost::graph_traits<unit_graph_type>::vertex_iterator unitcell_vertexiterator;
  typedef typename boost::graph_traits<graph_type>::vertex_iterator cell_vertexiterator;
  typedef typename boost::graph_traits<unit_graph_type>::edge_iterator edge_iterator;
  typedef typename boost::graph_traits<graph_type>::edge_descriptor edge_descriptor;
  typedef typename lattice_traits<lattice_type>::cell_iterator cell_iterator;
  typedef typename lattice_traits<lattice_type>::offset_type offset_type;
  typedef typename lattice_traits<lattice_type>::vector_type vector_type;
  typedef typename lattice_traits<lattice_type>::size_type size_type;
  typedef typename lattice_traits<lattice_type>::boundary_crossing_type boundary_crossing_type;
  typedef typename lattice_traits<lattice_type>::basis_vector_iterator basis_vector_iterator;

  Depletion depletion(depl_desc,volume(l) * num_vertices(graph::graph(unit_cell(l))));
  
  int num  = depletion.num_sites();
  
  const unit_graph_type& ug(graph::graph(unit_cell(l)));
  uint32_t unit_cell_vertices = num_vertices(ug);
  
  

  typename property_map<vertex_type_t,graph_type,int>::type
    vertextype = get_or_default(vertex_type_t(),g,0);

  typename property_map<edge_type_t,graph_type,int>::type
    edgetype = get_or_default(edge_type_t(),g,0);

  typename property_map<boundary_crossing_t,graph_type,boundary_crossing>::type
    edgeboundary = get_or_default(boundary_crossing_t(),g,boundary_crossing());

  typename property_map<edge_index_t,graph_type,int>::type
    edgeindex = get_or_default(edge_index_t(),g,0);

  typename property_map<coordinate_t,graph_type,std::vector<double> >::type
    vertexcoordinate = get_or_default(coordinate_t(),g,std::vector<double>());

  typename property_map<dimension_t,graph_type,uint32_t>::type
    graphdimension = get_or_default(dimension_t(),g,uint32_t(0));

  typename property_map<bond_vector_t,graph_type,std::vector<double> >::type
    bondvector = get_or_default(bond_vector_t(),g,std::vector<double>());

  typename property_map<bond_vector_relative_t,graph_type,std::vector<double> >::type
    bondvectorrelative = get_or_default(bond_vector_relative_t(),g,std::vector<double>());
  
  for (int i=0;i<num;++i)
    boost::add_vertex(g);

  // set vertex types
  cell_vertexiterator vit=boost::vertices(g).first;
  unitcell_vertexiterator uvit, uvend;
  cell_iterator cit,cend; 
  edge_iterator first_edge, last_edge;
  size_type edge_index=0;
  
  prevent_optimization(); 
  int original_vertex_number=0;
  for ( boost::tie(cit,cend)=cells(l); cit != cend ; ++cit)
  {
    // vertex properties
    for ( boost::tie(uvit,uvend)=boost::vertices(ug); uvit!=uvend;++uvit,++original_vertex_number)
    {
      // verrtex is not depleted
      if (depletion.exists(original_vertex_number)) {
        // vertex kind
        vertextype[*vit]=boost::get(vertex_type_t(),ug,*uvit);
        // vertex coordinate
        vertexcoordinate[*vit] = alps::coordinate(*cit,boost::get(coordinate_t(),ug,*uvit),l);
        ++vit;
      }
    }

    // edge properties
    for (boost::tie(first_edge,last_edge)=boost::edges(ug);
         first_edge!=last_edge;++first_edge)   {
      // calculate cell offsets and check if these cells are on the lattice
      offset_type off_source(alps::offset(*cit,l));
      offset_type off_target(off_source);
      
      std::pair<bool,boundary_crossing_type> source_cross, target_cross;
      source_cross = alps::shift(off_source,boost::get(source_offset_t(),ug,*first_edge),l);
      target_cross = alps::shift(off_target,boost::get(target_offset_t(),ug,*first_edge),l);
      if(source_cross.first && target_cross.first) {
        // calculate vertex index from cell index and unit cell vertex index
        int source_index=alps::index(alps::cell(off_source,l),l)*unit_cell_vertices
                  +boost::get(boost::vertex_index_t(),ug,boost::source(*first_edge,ug));
        int target_index=alps::index(alps::cell(off_target,l),l)*unit_cell_vertices
                  +boost::get(boost::vertex_index_t(),ug,boost::target(*first_edge,ug));

        if(source_index!=target_index && depletion.exists(source_index) && depletion.exists(target_index)) {
          source_index = depletion.mapped_site(source_index);
          target_index = depletion.mapped_site(target_index);
          edge_descriptor edge=boost::add_edge(source_index,target_index,g).first;
          
          // store bond kind and index
          edgeindex[edge]=edge_index++;
          edgetype[edge]=boost::get(edge_type_t(),ug,*first_edge);
          
          // store boundary crossing
          if (source_cross.second && target_cross.second)
            // std::cout << "Offsets:\n";
            // << write_vector(boost::get(source_offset_t(),ug,*first_edge))
            // << " - " << write_vector(boost::get(target_offset_t(),ug,*first_edge)) << "\n";
            boost::throw_exception(
              std::logic_error("ALPS++::lattice: Cannot calculate boundary crossing if neither vertex is in the original cell"));
          boundary_crossing_type bt( source_cross.second ? source_cross.second.invert() : target_cross.second);
          edgeboundary[edge]=bt;
          
          // store edge vector
          vector_type bondvector_relative = boost::get(bond_vector_t(),ug,*first_edge);
          vector_type bondvector_absolute(alps::dimension(bondvector_relative));
          // perform basis transformation to lattice basis
          basis_vector_iterator first, last;
          boost::tie(first,last) = basis_vectors(l);
          for (typename vector_type::iterator rit = coordinates(bondvector_relative).first; first!=last; ++first, ++rit)
            for (unsigned int i=0;i<alps::dimension(*first);++i)
              bondvector_absolute[i] += *rit * (*first)[i];
          bondvector[edge]=bondvector_absolute;
          // divide relative bond vector by lattice extent
          for (unsigned int i=0; i<dimension(l) && i<bondvector_relative.size() ; ++i)
            bondvector_relative[i]/=alps::extent(l,i);
          bondvectorrelative[edge]=bondvector_relative;
        }
      }
    }
  }
  graphdimension=alps::dimension(l);
}

template <class LATTICE, class GRAPH> 
class lattice_graph : public LATTICE
{
public:
  typedef LATTICE super_type;
  typedef LATTICE base_type;
  typedef typename lattice_traits<base_type>::unit_cell_type unit_cell_type;
  typedef typename lattice_traits<base_type>::offset_type offset_type;
  typedef typename lattice_traits<base_type>::extent_type extent_type;
  typedef typename lattice_traits<base_type>::vector_type vector_type;
  typedef typename lattice_traits<base_type>::basis_vector_iterator basis_vector_iterator;
  typedef typename lattice_traits<base_type>::cell_iterator cell_iterator;
  typedef typename lattice_traits<base_type>::boundary_crossing_type boundary_crossing_type;
  typedef typename lattice_traits<base_type>::size_type size_type;
  typedef GRAPH graph_type;

  typedef typename boost::graph_traits<graph_type>::vertex_iterator vertex_iterator;
  typedef typename boost::graph_traits<graph_type>::vertex_descriptor vertex_descriptor;
  typedef typename boost::graph_traits<graph_type>::edge_iterator edge_iterator;

  lattice_graph() {}
  template <class L2>
  lattice_graph(const L2&);

  const graph_type& graph() const { return graph_;}
  graph_type& graph() { return graph_;}

  template<class H>
  typename H::graph_type& graph(H& g) const
  { return detail::graph_wrap(g); }
  template<class H>
  const typename H::graph_type& graph(const H& g) const
  { return detail::graph_wrap(g); }

  std::vector<std::string> distance_labels(int precision = 0) const
  {
    typename property_map<coordinate_t,graph_type,vector_type>::const_type
      coordinate_map = get_or_default(coordinate_t(),graph(),0);
    std::vector<std::string> label(num_distances());
    for (vertex_iterator it1 = vertices(graph()).first;
         it1 != vertices(graph()).second; ++it1) {
      for (vertex_iterator it2 = vertices(graph()).first;
           it2 != vertices(graph()).second; ++it2) {
        std::size_t d=distance(*it1,*it2);
        if (label[d].empty())
          label[d] =
            alps::coordinate_to_string(coordinate_map[*it1], precision) + " -- " + 
            alps::coordinate_to_string(coordinate_map[*it2], precision);
      }
    }
    return label;
  }

  std::vector<unsigned int> distance_multiplicities() const
  {
    std::vector<unsigned int> mult(num_distances());
    for (vertex_iterator it1=boost::vertices(graph()).first; it1 != boost::vertices(graph()).second;++it1)
      for (vertex_iterator it2=boost::vertices(graph()).first; it2 != boost::vertices(graph()).second;++it2)
        mult[distance(*it1,*it2)]++;
    return mult;
  }


  size_type num_distances() const
  {
    size_type vertices_in_cell = num_vertices(graph(super_type::unit_cell()));
    return super_type::num_distances()*vertices_in_cell*vertices_in_cell;
  }
  
 size_type distance(vertex_descriptor x, vertex_descriptor y) const
  {
    int vertices_in_cell = num_vertices(graph(super_type::unit_cell()));
    int cell_num_x = int(x) / vertices_in_cell;
    int cell_num_y = int(y) / vertices_in_cell;
    int vert_num_x = int(x) % vertices_in_cell;
    int vert_num_y = int(y) % vertices_in_cell;
    offset_type offset_x = alps::offset(super_type::cell(cell_num_x),*this);
    offset_type offset_y = alps::offset(super_type::cell(cell_num_y),*this);
    return super_type::distance(offset_x,offset_y)*vertices_in_cell*vertices_in_cell
            + vert_num_x*vertices_in_cell+vert_num_y;
  }

  std::vector<std::pair<std::complex<double>,std::vector<std::size_t> > > translations(const vector_type& k) const
  {
    typedef std::vector<std::pair<std::complex<double>, 
      std::vector<std::size_t> > > translation_type;
    translation_type trans = super_type::translations(k);
    translation_type graph_trans;

    int vertices_in_cell = num_vertices(alps::graph::graph(super_type::unit_cell()));
    for (typename translation_type::const_iterator it=trans.begin();
         it!=trans.end(); ++it) {
      std::vector<std::size_t> shifted_vertices;
      for (typename std::vector<std::size_t>::const_iterator
             sit = it->second.begin(); sit != it->second.end(); ++sit)
        for (int i=0; i<vertices_in_cell; ++i)
          shifted_vertices.push_back(*sit*vertices_in_cell+i);
      graph_trans.push_back(std::make_pair(it->first,shifted_vertices));
    }
    return graph_trans;
  }

private:
  GRAPH graph_;        
};

template <class LATTICE, class GRAPH>
template <class L2>
inline lattice_graph<LATTICE,GRAPH>::lattice_graph(const L2& d)
 : LATTICE(d)
{
  make_graph_from_lattice(graph_,*this,d.depletion());
}

template <class L, class G>
struct lattice_traits<lattice_graph<L,G> >
{
  typedef typename lattice_graph<L,G>::unit_cell_type unit_cell_type;
  typedef typename lattice_graph<L,G>::cell_descriptor cell_descriptor;
  typedef typename lattice_graph<L,G>::offset_type offset_type;
  typedef typename lattice_graph<L,G>::extent_type extent_type;
  typedef typename lattice_graph<L,G>::basis_vector_iterator basis_vector_iterator;
  typedef typename lattice_graph<L,G>::cell_iterator cell_iterator;
  typedef typename lattice_graph<L,G>::momentum_iterator momentum_iterator;
  typedef typename lattice_graph<L,G>::size_type size_type;
  typedef typename lattice_graph<L,G>::vector_type vector_type;
  typedef typename lattice_graph<L,G>::boundary_crossing_type boundary_crossing_type;
};

template <class L, class G>
struct graph_traits<lattice_graph<L,G> >
{
  typedef G graph_type;
};

template<class L, class G>
std::size_t dimension(const lattice_graph<L,G>& l) { return l.dimension(); }



namespace graph {

template<class L, class G>
typename lattice_graph<L, G>::graph_type&
graph(lattice_graph<L, G>& l) { return l.graph(); }


template<class L, class G>
const typename lattice_graph<L, G>::graph_type&
graph(const lattice_graph<L, G>& l) { return l.graph(); }

} // end namespace graph


} // end namespace alps

#endif // ALPS_LATTICE_LATTICEGRAPH_H
