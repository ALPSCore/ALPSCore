/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_LATTICE_COORDINATE_GRAPH_H
#define ALPS_LATTICE_COORDINATE_GRAPH_H

#include <alps/lattice/propertymap.h>
#include <alps/lattice/boundary.h>
#include <alps/lattice/graphproperties.h>
#include <boost/pending/property.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <string>

namespace alps {
// the default graph class

typedef boost::adjacency_list<boost::vecS,boost::vecS,boost::undirectedS,
                              // vertex property
                              boost::property<coordinate_t,coordinate_type,
                                 boost::property<parity_t,int8_t,
                                   boost::property<vertex_type_t,type_type> >  >,
                              // edge property
                              boost::property<edge_type_t,type_type,
                                boost::property<boost::edge_index_t,unsigned int,
                                   boost::property<boundary_crossing_t,boundary_crossing,
                                    boost::property<bond_vector_t,coordinate_type
#if !BOOST_WORKAROUND(__IBMCPP__, <= 700)
                                       , boost::property<bond_vector_relative_t,coordinate_type>
#endif
                              > > > >,
                              // graph property
                              boost::property<dimension_t,std::size_t,
                                boost::property<graph_name_t,std::string > >
                              , boost::vecS> coordinate_graph_type;

typedef boost::adjacency_list<boost::vecS,boost::vecS,boost::undirectedS,
                              // vertex property
                              boost::no_property,
                              // edge property
                              boost::property<boost::edge_index_t,unsigned int>,
                              // graph property
                              boost::property<dimension_t,std::size_t,
                                boost::property<graph_name_t,std::string > >
                              , boost::vecS> minimal_graph_type;

} // end namespace alps

#endif //ALPS_LATTICE_COORDINATE_GRAPH_H
