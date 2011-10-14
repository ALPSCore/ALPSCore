// TODO: remove
#include <iostream>


#include <alps/graph/lattice_constant.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <iostream>

template <typename Graph>
Graph build_square_lattice(unsigned int side_length)
{
    Graph g;
    for(unsigned int j=1; j < side_length; ++j)
        add_edge(j,j-1,g);
    
    for(unsigned int i=1; i < side_length; ++i)
    {
        add_edge(i*side_length,(i-1)*side_length,g);
        for(unsigned int j=1; j < side_length; ++j)
        {
            add_edge(i*side_length+j,(i-1)*side_length+j,g);
            add_edge(i*side_length+j,i*side_length+j-1,g);
        }
    }
    return g;
}

int main() {
    using boost::get;
    using alps::graph::canonical_properties;

    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph_type;
    typedef unsigned int lc_type;

    unsigned int side_length = 3;
//    unsigned int side_length = 10;
    graph_type lattice = build_square_lattice<graph_type>(side_length);

    std::vector<std::pair<graph_type,lc_type> > g;

    //
    //  0---1
    //  |   |
    //  2---3
    //
    g.push_back(std::make_pair(graph_type(),4));
    add_edge(0, 1,g.back().first);
    add_edge(1, 3,g.back().first);
    add_edge(3, 2,g.back().first);
    add_edge(2, 0,g.back().first);
/*
    //
    //  1---0---2
    //
    g.push_back(std::make_pair(graph_type(),6));
    add_edge(0, 1,g.back().first);
    add_edge(0, 2,g.back().first);

    //
    //     3
    //     |
    // 1---0---2
    //     |
    //     4
    //
    g.push_back(std::make_pair(graph_type(),1));
    add_edge(0, 1,g.back().first);
    add_edge(0, 2,g.back().first);
    add_edge(0, 3,g.back().first);
    add_edge(0, 4,g.back().first);

    //
    //   2       5
    //    \     /
    //     0---1
    //    /     \
    //   3       4
    //
    g.push_back(std::make_pair(graph_type(),18));
    add_edge(0, 1,g.back().first);
    add_edge(0, 2,g.back().first);
    add_edge(0, 3,g.back().first);
    add_edge(1, 4,g.back().first);
    add_edge(1, 5,g.back().first);
*/
    int success = 0;
    for(std::vector<std::pair<graph_type,lc_type> >::iterator it=g.begin(); it != g.end(); ++it)
    {
        lc_type lc = alps::graph::lattice_constant(it->first, lattice, std::vector<boost::graph_traits<graph_type>::vertex_descriptor>(1,side_length*side_length/2+side_length/2 - 1));
        if ( lc != it->second)
        {
            std::cout<<"ERROR: lattice constant does not match!"<<std::endl;
            std::cout<<"Calculated: "<<lc<<"\tReference: "<<it->second<<std::endl<<std::endl;
            success = -1;
        }
    }
    return success;
}
