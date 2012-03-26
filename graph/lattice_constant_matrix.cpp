#include <alps/graph/lattice_constant.hpp>
#include <alps/lattice/lattice.h>
#include <boost/graph/adjacency_list.hpp>
#include <alps/numeric/detail/general_matrix.hpp>
#include <iostream>


int main() {
    using boost::get;
    using boost::put;
    using alps::graph::canonical_properties;
    using alps::graph::canonical_properties_type;
    using alps::graph::graph_label;
    using alps::graph::label;


    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph_type;

    typedef alps::graph_helper<>::vertex_iterator vertex_iterator;
    typedef alps::graph_helper<>::edge_iterator edge_iterator;
    
    alps::Parameters parm;
	unsigned int side_length = 40;
	
    std::ifstream in("../../lib/xml/lattices.xml");
    parm["LATTICE"] = "square lattice";
    parm["L"] = side_length;

    parm["L"] = side_length;
    alps::graph_helper<> lattice(in,parm);
	
	graph_type lattice_graph(num_vertices(lattice.graph()));
	boost::graph_traits<alps::graph_helper<>::graph_type>::edge_iterator it, et;
	for(boost::tie(it, et) = edges(lattice.graph()); it != et; ++it)
    {
        boost::graph_traits<graph_type>::edge_descriptor  e = add_edge(source(*it, lattice.graph()), target(*it, lattice.graph()), lattice_graph).first;
    }

    typedef unsigned int contrib_type;
    typedef std::vector< std::pair<graph_label<graph_type>::type, std::vector<contrib_type> > > input_type;
    typedef blas::general_matrix<contrib_type> output_type;
    std::vector<boost::tuple<graph_type, input_type, output_type> > test_graphs;

    //
    // 1---0
    //
    {
        graph_type g;
        add_edge(0,1,g);
        canonical_properties_type<graph_type>::type gp = canonical_properties(g);

        input_type in;
        {
            // Orbit partition: (0 1) -> [0] (1)
            unsigned int breaking_vertex = 0;
            std::vector<contrib_type> part_contrib(2);
            part_contrib[0] = 2; // c[0 -> 0]
            part_contrib[1] = 3; // c[0 -> 1]
            in.push_back(std::make_pair(get<label>(canonical_properties(g,breaking_vertex)), part_contrib));
        }

        output_type out(2,2);
        out(0,0) = 2*2*2;
        out(0,1) = 1*3;
        out(1,0) = 1*3;
        out(1,1) = 0;
        test_graphs.push_back(boost::make_tuple(g,in,out));
    }

    //
    // 1---0---2
    //     |
    //     3
    //

    {
        graph_type g;
        add_edge(0,1,g);
        add_edge(0,2,g);
        add_edge(0,3,g);

        input_type in;
        {
            // Orbit partition: (0)(1 2 3) -> [0] (1 2 3)
            unsigned int breaking_vertex = 0;
            std::vector<contrib_type> part_contrib(2);
            part_contrib[0] = 2;
            part_contrib[1] = 3;
            in.push_back(std::make_pair(get<label>(canonical_properties(g,breaking_vertex)), part_contrib));
        }

        {
            // Orbit partition: (0)(1 2 3) -> (0) [1] (2 3)
            unsigned int breaking_vertex = 1;
            std::vector<contrib_type> part_contrib(3);
            part_contrib[0] = 5;
            part_contrib[1] = 7;
            part_contrib[2] = 11;
            in.push_back(std::make_pair(get<label>(canonical_properties(g,breaking_vertex)), part_contrib));
        }

        output_type out(3,3);
        out(0,0) = 4*2 + 4*5;
        out(0,1) = 3*3 + 1*7;
        out(1,0) = 3*3 + 1*7;
        out(1,1) = 2*11;
        out(0,2) = 1*11;
        out(2,0) = 1*11;
        test_graphs.push_back(boost::make_tuple(g,in,out));
    }

    int success = 0;
    for(std::vector<boost::tuple<graph_type, input_type, output_type> >::iterator it= test_graphs.begin(); it != test_graphs.end(); ++it)
    {
        output_type lc = alps::graph::lattice_constant(
			  get<1>(*it)
            , get<0>(*it)
			, lattice_graph
			, lattice.lattice()
			, alps::cell(std::vector<int>(2,side_length/2),lattice.lattice()) //side_length * side_length / 2 + side_length / 2 - 1
		);
        output_type ref = get<2>(*it);
        if ( lc != ref )
        {
            std::cerr<<"ERROR: lattice constant does not match!"<<std::endl;
            std::cerr<<"Graph:"<<std::distance(test_graphs.begin(),it)<<" Calculated: "<<lc<<"\tReference: "<<ref<<std::endl<<std::endl;
            success = -1;
        }
    }
    return success;
}
