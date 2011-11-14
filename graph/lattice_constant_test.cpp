#include <alps/graph/lattice_constant.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <iostream>

int main() {
    using boost::get;
    using alps::graph::canonical_properties;

    typedef unsigned int lc_type;
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph_type;
    typedef alps::graph_helper<>::vertex_iterator vertex_iterator;
    typedef alps::graph_helper<>::edge_iterator edge_iterator;

    alps::Parameters parm;
	unsigned int side_length = 20;
	
    parm["LATTICE"] = "square lattice";
    parm["L"] = side_length;
    alps::graph_helper<> lattice(parm);
	
	graph_type lattice_graph(num_vertices(lattice.graph()));
	boost::graph_traits<alps::graph_helper<>::graph_type>::edge_iterator it, et;
	for(boost::tie(it, et) = edges(lattice.graph()); it != et; ++it)
		add_edge(source(*it, lattice.graph()), target(*it, lattice.graph()), lattice_graph);

    std::vector<std::pair<graph_type,lc_type> > g;

//    //
//    //  0---1
//    //  |   |
//    //  2---3
//    //
//    g.push_back(std::make_pair(graph_type(), 1));
//    add_edge(0, 1, g.back().first);
//    add_edge(1, 3, g.back().first);
//    add_edge(3, 2, g.back().first);
//    add_edge(2, 0, g.back().first);
//
//    //
//    //  1---0---2
//    //
//    g.push_back(std::make_pair(graph_type(),6));
//    add_edge(0, 1,g.back().first);
//    add_edge(0, 2,g.back().first);
//    
//    //
//    //  3---1---0---2
//    //
//    g.push_back(std::make_pair(graph_type(),18));
//    add_edge(0, 1,g.back().first);
//    add_edge(0, 2,g.back().first);
//    add_edge(1, 3,g.back().first);
//
//    //
//    //     3
//    //     |
//    // 1---0---2
//    //     |
//    //     4
//    //
//    g.push_back(std::make_pair(graph_type(),1));
//    add_edge(0, 1,g.back().first);
//    add_edge(0, 2,g.back().first);
//    add_edge(0, 3,g.back().first);
//    add_edge(0, 4,g.back().first);
//
//    //
//    //   2       5
//    //    \     /
//    //     0---1
//    //    /     \
//    //   3       4
//    //
//    g.push_back(std::make_pair(graph_type(),18));
//    add_edge(0, 1,g.back().first);
//    add_edge(0, 2,g.back().first);
//    add_edge(0, 3,g.back().first);
//    add_edge(1, 4,g.back().first);
//    add_edge(1, 5,g.back().first);
//
    //
    //           8
    //           |
    //           4
    //           |
    //   6---2---0---1---5
    //           |
    //           3
    //           |
    //           7
    //
    g.push_back(std::make_pair(graph_type(),47));
    add_edge(0, 1,g.back().first);
    add_edge(0, 2,g.back().first);
    add_edge(0, 3,g.back().first);
    add_edge(0, 4,g.back().first);
    add_edge(1, 5,g.back().first);
    add_edge(2, 6,g.back().first);
    add_edge(3, 7,g.back().first);
    add_edge(4, 8,g.back().first);

    int success = 0;
    for(std::vector<std::pair<graph_type,lc_type> >::iterator it= g.begin(); it != g.end(); ++it)
    {
        lc_type lc = alps::graph::lattice_constant(
			  it->first
			, lattice_graph
			, lattice.lattice()
			, lattice.graph()
			, std::vector<boost::graph_traits<graph_type>::vertex_descriptor>(1,side_length*side_length/2+side_length/2 - 1)
		);
        if ( lc != it->second)
        {
            std::cerr<<"ERROR: lattice constant does not match!"<<std::endl;
            std::cerr<<"Calculated: "<<lc<<"\tReference: "<<it->second<<std::endl<<std::endl;
            success = -1;
        }
    }
    return success;
}
