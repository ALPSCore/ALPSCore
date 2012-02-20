//#define USE_COMPRESSED_EMBEDDING2

#include <alps/graph/subgraph_generator.hpp>
#include <alps/lattice/graph_helper.h>
#include <boost/graph/adjacency_list.hpp>
#include <iostream>

#include <boost/timer/timer.hpp>


enum { test_graph_size = 10 };

template <typename Graph>
void subgraph_generator_test2(unsigned int order_ )
{
    alps::Parameters parm;
    std::ifstream in("../../lib/xml/lattices.xml");
    parm["LATTICE"] = "square lattice";
    parm["L"] = 2*order_+1;
    alps::graph_helper<> alps_lattice(in,parm);

    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> lattice_graph_type;
    boost::graph_traits<alps::graph_helper<>::graph_type>::edge_iterator eit, eend;
	lattice_graph_type lattice_graph(num_vertices(alps_lattice.graph()));
    for(boost::tie(eit,eend) = edges(alps_lattice.graph()); eit != eend; ++eit)
        add_edge( static_cast<unsigned int>(source(*eit,alps_lattice.graph()))
                , static_cast<unsigned int>(target(*eit,alps_lattice.graph()))
                , lattice_graph
                );
    
    typedef alps::graph::subgraph_generator<Graph,lattice_graph_type> graph_gen_type;
    graph_gen_type graph_gen(lattice_graph,2*order_*order_);
    
    typename graph_gen_type::iterator it,end;
    boost::timer::auto_cpu_timer t;
    boost::tie(it,end) = graph_gen.generate_up_to_n_edges(order_);
    std::cout<<std::distance(it,end)<<std::endl;
}

int main()
{
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::property<alps::vertex_type_t,alps::type_type>, boost::property<alps::edge_type_t,alps::type_type>  > graph_type;
    subgraph_generator_test2<graph_type>(test_graph_size);
    return 0;
}
