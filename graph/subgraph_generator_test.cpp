//#define USE_COMPRESSED_EMBEDDING2

#include <alps/graph/subgraph_generator.hpp>
#include <alps/lattice/graph_helper.h>
#include <boost/graph/adjacency_list.hpp>
#include <iostream>

#include <boost/timer/timer.hpp>


enum { test_graph_size = 10 };

template <typename Graph>
void subgraph_generator_test(unsigned int order_ )
{
    Graph lattice_;

    alps::Parameters parm;
    parm["LATTICE"] = "square lattice";
    parm["L"] = 2*order_+1;
    alps::graph_helper<> alps_lattice(parm);

    typedef alps::graph_traits<alps::graph_helper<>::lattice_type >::graph_type lattice_graph_type;
    boost::graph_traits<lattice_graph_type>::vertex_iterator vit,vend;
    boost::graph_traits<lattice_graph_type>::edge_iterator eit,eend;
    for(boost::tie(vit,vend) = vertices(alps_lattice.graph()); vit != vend; ++vit)
        add_vertex(lattice_);
    for(boost::tie(eit,eend) = edges(alps_lattice.graph()); eit != eend; ++eit)
        add_edge( static_cast<unsigned int>(source(*eit,alps_lattice.graph()))
                , static_cast<unsigned int>(target(*eit,alps_lattice.graph()))
                , lattice_
                );
    
    typedef alps::graph::subgraph_generator<Graph,Graph> graph_gen_type;
    graph_gen_type graph_gen(lattice_,2*order_*order_);
    
    typename graph_gen_type::iterator it,end;
    boost::timer::auto_cpu_timer t;
    boost::tie(it,end) = graph_gen.generate_up_to_n_edges(order_);
    std::cout<<std::distance(it,end)<<std::endl;
}

int main()
{
    typedef boost::adjacency_list<boost::vecS, boost::vecS,boost::undirectedS> graph_type;
    subgraph_generator_test<graph_type>(test_graph_size);
    return 0;
}
