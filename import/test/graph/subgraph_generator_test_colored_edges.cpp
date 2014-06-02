/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

//#define USE_COMPRESSED_EMBEDDING2

#include <alps/graph/subgraph_generator.hpp>
#include <alps/graph/utils.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <iostream>



enum { test_graph_size = 7 };

template <typename Graph>
void subgraph_generator_test(unsigned int order_ )
{
    std::ifstream in("../../lib/xml/lattices.xml");
    alps::Parameters parm;
    parm["LATTICE"] = "coupled ladders";
    parm["L"] = 2*order_+1;
    alps::graph_helper<> alps_lattice(in,parm);

    typedef alps::coordinate_graph_type lattice_graph_type;
    lattice_graph_type& lattice_graph = alps_lattice.graph();
    std::vector<unsigned int> edge_type_map(3,0);
    edge_type_map[0] = 0;
    edge_type_map[1] = 1;
    edge_type_map[2] = 0;
    alps::graph::remap_edge_types(lattice_graph,edge_type_map);


    typedef alps::graph::subgraph_generator<Graph,lattice_graph_type> graph_gen_type;
    graph_gen_type graph_gen(lattice_graph,2*order_*order_);

    typename graph_gen_type::iterator it,end;
    boost::tie(it,end) = graph_gen.generate_up_to_n_edges(order_);
    std::cout<<std::distance(it,end)<<std::endl;
}

int main()
{
    typedef boost::adjacency_list<boost::vecS, boost::vecS,boost::undirectedS, boost::no_property, boost::property<alps::edge_type_t,alps::type_type> > graph_type;
    subgraph_generator_test<graph_type>(test_graph_size);
    return 0;
}
