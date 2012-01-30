#include <alps/graph/lattice_constant.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <iostream>

int main() {
    using boost::get;
    using boost::put;
    using alps::graph::canonical_properties;

    typedef unsigned int lc_type;
    
    typedef boost::property<alps::edge_type_t,alps::type_type> edge_props;

    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,boost::no_property,edge_props> graph_type;
    typedef alps::graph_helper<>::vertex_iterator vertex_iterator;
    typedef alps::graph_helper<>::edge_iterator edge_iterator;

    alps::Parameters parm;
	unsigned int side_length = 40;
	
    parm["LATTICE"] = "coupled ladders";
    parm["L"] = side_length;

    parm["L"] = side_length;
    alps::graph_helper<> lattice(parm);
	
	graph_type lattice_graph(num_vertices(lattice.graph()));
	boost::graph_traits<alps::graph_helper<>::graph_type>::edge_iterator it, et;
	for(boost::tie(it, et) = edges(lattice.graph()); it != et; ++it)
    {
//        std::cout<<source(*it, lattice.graph())<<" - "<<target(*it, lattice.graph())<<" t:";
//        std::cout<<get(alps::edge_type_t(),lattice.graph(),*it)<<std::endl;
        boost::graph_traits<graph_type>::edge_descriptor  e = add_edge(source(*it, lattice.graph()), target(*it, lattice.graph()), lattice_graph).first;
        put(alps::edge_type_t(), lattice_graph, e, get(alps::edge_type_t(),lattice.graph(),*it) );
    }
    
    std::vector<std::pair<graph_type,lc_type> > g;
    //
    //  0...1
    //  |   |
    //  2...3
    //
    g.push_back(std::make_pair(graph_type(), 1));
    boost::graph_traits<graph_type>::edge_descriptor e = add_edge(0, 1, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 0);
    e = add_edge(1, 3, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 1);
    e = add_edge(3, 2, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 0);
    e = add_edge(2, 0, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 1);
    
    //
    //  0...1
    //  .   |
    //  2___3
    //
    g.push_back(std::make_pair(graph_type(), 0));
    e = add_edge(0, 1, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 0);
    e = add_edge(1, 3, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 1);
    e = add_edge(3, 2, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 1);
    e = add_edge(2, 0, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 0);

    //
    //  1___0___2
    //
    g.push_back(std::make_pair(graph_type(),6));
    e = add_edge(0, 1,g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 0);
    e = add_edge(0, 2,g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 0);

    //
    //  1...0___2
    //
    g.push_back(std::make_pair(graph_type(),6));
    e = add_edge(0, 1,g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 0);
    e = add_edge(0, 2,g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 1);

    /*
    parm["LATTICE"] = "anisotropic square lattice";
    parm["L"] = side_length;
    alps::graph_helper<> lattice(parm);
	
	graph_type lattice_graph(num_vertices(lattice.graph()));
	boost::graph_traits<alps::graph_helper<>::graph_type>::edge_iterator it, et;
	for(boost::tie(it, et) = edges(lattice.graph()); it != et; ++it)
    {
//        std::cout<<source(*it, lattice.graph())<<" - "<<target(*it, lattice.graph())<<" t:";
//        std::cout<<get(alps::edge_type_t(),lattice.graph(),*it)<<std::endl;
        boost::graph_traits<graph_type>::edge_descriptor  e = add_edge(source(*it, lattice.graph()), target(*it, lattice.graph()), lattice_graph).first;
        put(alps::edge_type_t(), lattice_graph, e, get(alps::edge_type_t(),lattice.graph(),*it) );
    }

    std::vector<std::pair<graph_type,lc_type> > g;
    //
    //  0...1
    //  |   |
    //  2...3
    //
    g.push_back(std::make_pair(graph_type(), 1));
    boost::graph_traits<graph_type>::edge_descriptor e = add_edge(0, 1, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 0);
    e = add_edge(1, 3, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 1);
    e = add_edge(3, 2, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 0);
    e = add_edge(2, 0, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 1);
    
    //
    //  0...1
    //  .   |
    //  2___3
    //
    g.push_back(std::make_pair(graph_type(), 0));
    e = add_edge(0, 1, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 0);
    e = add_edge(1, 3, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 1);
    e = add_edge(3, 2, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 1);
    e = add_edge(2, 0, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 0);



    //
    //  1___0___2
    //
    g.push_back(std::make_pair(graph_type(),1));
    e = add_edge(0, 1,g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 0);
    e = add_edge(0, 2,g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 0);
    
    //
    //  1...0___2
    //
    g.push_back(std::make_pair(graph_type(),4));
    e = add_edge(0, 1,g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 0);
    e = add_edge(0, 2,g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 1);
    */

//    
//    //
//    //  3---1---0---2
//    //
//    g.push_back(std::make_pair(graph_type(),18));
//    add_edge(0, 1,g.back().first);
//    add_edge(0, 2,g.back().first);
//    add_edge(1, 3,g.back().first);
//
////    //
////    //     3
////    //     |
////    // 1---0---2
////    //     |
////    //     4
////    //
////    g.push_back(std::make_pair(graph_type(),1));
////    add_edge(0, 1,g.back().first);
////    add_edge(0, 2,g.back().first);
////    add_edge(0, 3,g.back().first);
////    add_edge(0, 4,g.back().first);
////
////    //
////    //   2       5
////    //    \     /
////    //     0---1
////    //    /     \ 
////    //   3       4
////    //
////    g.push_back(std::make_pair(graph_type(),18));
////    add_edge(0, 1,g.back().first);
////    add_edge(0, 2,g.back().first);
////    add_edge(0, 3,g.back().first);
////    add_edge(1, 4,g.back().first);
////    add_edge(1, 5,g.back().first);
////
////    //
////    //           8
////    //           |
////    //           4
////    //           |
////    //   6---2---0---1---5
////    //           |
////    //           3
////    //           |
////    //           7
////    //
////    g.push_back(std::make_pair(graph_type(),47));
////    add_edge(0, 1,g.back().first);
////    add_edge(0, 2,g.back().first);
////    add_edge(0, 3,g.back().first);
////    add_edge(0, 4,g.back().first);
////    add_edge(1, 5,g.back().first);
////    add_edge(2, 6,g.back().first);
////    add_edge(3, 7,g.back().first);
////    add_edge(4, 8,g.back().first);
////
////	// graph no. 32340
////	// 14	15	1	64272
////    g.push_back(std::make_pair(graph_type(15),64272));
////    add_edge(13, 14,g.back().first);
////    add_edge(12, 14,g.back().first);
////    add_edge( 9, 14,g.back().first);
////    add_edge( 7, 14,g.back().first);
////    add_edge( 8, 13,g.back().first);
////    add_edge( 1, 13,g.back().first);
////    add_edge( 6, 12,g.back().first);
////    add_edge( 0, 12,g.back().first);
////    add_edge(10,  9,g.back().first);
////    add_edge( 2,  7,g.back().first);
////    add_edge(11,  8,g.back().first);
////    add_edge( 3,  6,g.back().first);
////    add_edge( 4, 10,g.back().first);
////    add_edge( 5, 11,g.back().first);
//
    int success = 0;
    for(std::vector<std::pair<graph_type,lc_type> >::iterator it= g.begin(); it != g.end(); ++it)
    {
        lc_type lc = alps::graph::lattice_constant(
			  it->first
			, lattice_graph
			, lattice.lattice()
			, side_length * side_length / 2 + side_length / 2 - 1
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
