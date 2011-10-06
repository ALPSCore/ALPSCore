#include <alps/graph/lattice_constant.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <iostream>

int main() {
    using boost::get;
    using alps::graph::canonical_properties;

    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph_type;
    graph_type g;

    add_edge(0, 1,g);
    add_edge(0, 2,g);
    add_edge(0, 3,g);
    add_edge(2, 1,g);

//    alps::graph::graph_label<graph_type>::type label(get<1>(canonical_properties(g)));
    
//    std::cout << label << std::endl;
    return 0;
}
