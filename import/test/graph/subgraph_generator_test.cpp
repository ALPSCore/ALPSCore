/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

//#define USE_COMPRESSED_EMBEDDING2

#include <alps/graph/subgraph_generator.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <iostream>



enum { test_graph_size = 10 };

template <typename Graph>
void subgraph_generator_test(unsigned int order_ )
{
    std::ifstream in("../../lib/xml/lattices.xml");
    alps::Parameters parm;
    parm["LATTICE"] = "square lattice";
    parm["L"] = 2*order_+1;
    alps::graph_helper<> alps_lattice(in,parm);

    typedef alps::coordinate_graph_type lattice_graph_type;
    lattice_graph_type lattice_graph = alps_lattice.graph();


    typedef alps::graph::subgraph_generator<Graph,lattice_graph_type> graph_gen_type;
    graph_gen_type graph_gen(lattice_graph,2*order_*order_);

    typename graph_gen_type::iterator it,end;
    boost::tie(it,end) = graph_gen.generate_up_to_n_edges(order_);
    std::cout<<std::distance(it,end)<<std::endl;
}

int main()
{
    typedef boost::adjacency_list<boost::vecS, boost::vecS,boost::undirectedS> graph_type;
    subgraph_generator_test<graph_type>(test_graph_size);
    return 0;
}
