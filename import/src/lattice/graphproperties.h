/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_LATTICE_GRAPH_PROPERTIES_H
#define ALPS_LATTICE_GRAPH_PROPERTIES_H

#include <vector>
#include <boost/graph/properties.hpp>

namespace alps {

struct vertex_type_t { typedef boost::vertex_property_tag kind; };
typedef vertex_type_t site_type_t;
struct coordinate_t { typedef boost::vertex_property_tag kind; };
struct parity_t { typedef boost::vertex_property_tag kind; };

struct edge_type_t { typedef boost::edge_property_tag kind; };
typedef edge_type_t bond_type_t;
struct source_offset_t { typedef boost::edge_property_tag kind; };
struct target_offset_t { typedef boost::edge_property_tag kind; };
struct boundary_crossing_t { typedef boost::edge_property_tag kind; };
struct edge_vector_t { typedef boost::edge_property_tag kind; };
typedef edge_vector_t bond_vector_t;
struct edge_vector_relative_t { typedef boost::edge_property_tag kind; };
typedef edge_vector_relative_t bond_vector_relative_t;

struct graph_name_t { typedef boost::graph_property_tag kind; };
struct dimension_t { typedef boost::graph_property_tag kind; };

using boost::vertex_index_t;
typedef vertex_index_t site_index_t;
using boost::edge_index_t;
typedef edge_index_t bond_index_t;

typedef std::vector<double> coordinate_type;
typedef std::vector<int> offset_type;
typedef std::vector<int> distance_type;
typedef unsigned int type_type;

} // end namespace alps

#endif // ALPS_LATTICE_GRAPH_PROPERTIES_H
