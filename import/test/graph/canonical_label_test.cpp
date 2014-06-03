/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/graph/canonical_properties.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <iostream>


using boost::get;
using alps::graph::canonical_properties;


bool colored_edges_test() {
    std::cout << "colored_edges_test()" << std::endl;
    typedef boost::property<alps::edge_type_t,unsigned int> edge_props;
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, edge_props> graph_type;
    typedef boost::graph_traits<graph_type>::edge_descriptor edge_descriptor;
    typedef boost::property_map<graph_type,alps::edge_type_t>::type edge_color_map_type;

    graph_type g(4);
    {
        edge_color_map_type edge_color = get(alps::edge_type_t(),g);
        edge_descriptor e;
        e = add_edge(0, 1, g).first;
        edge_color[e] = 0;
        e = add_edge(1, 2, g).first;
        edge_color[e] = 0;
        e = add_edge(2, 3, g).first;
        edge_color[e] = 0;
        e = add_edge(0, 3, g).first;
        edge_color[e] = 0;
    }

    graph_type h(g);
    {
        edge_descriptor e = edge(2,3,h).first;
        get(alps::edge_type_t(),h)[e] = 1;
    }

    graph_type i(g);
    {
        edge_descriptor e = edge(2,3,i).first;
        get(alps::edge_type_t(),i)[e] = 2;
    }

    graph_type j(h);
    {
        edge_descriptor e = edge(2,3,j).first;
        get(alps::edge_type_t(),j)[e] = 1;
    }

    graph_type k(g);
    {
        edge_color_map_type edge_color = get(alps::edge_type_t(),k);
        edge_descriptor e;
        e = edge( 0, 1, k).first;
        edge_color[e] = 1;
        e = edge( 1, 2, k).first;
        edge_color[e] = 1;
        e = edge( 2, 3, k).first;
        edge_color[e] = 1;
        e = edge( 0, 3, k).first;
        edge_color[e] = 1;
    }

    alps::graph::graph_label<graph_type>::type label_g(get<1>(canonical_properties(g)));
    alps::graph::graph_label<graph_type>::type label_h(get<1>(canonical_properties(h)));
    alps::graph::graph_label<graph_type>::type label_i(get<1>(canonical_properties(i)));
    alps::graph::graph_label<graph_type>::type label_j(get<1>(canonical_properties(j)));
    alps::graph::graph_label<graph_type>::type label_k(get<1>(canonical_properties(k)));

    std::cout<<label_g<<std::endl;
    std::cout<<label_h<<std::endl;
    std::cout<<label_i<<std::endl;
    std::cout<<label_j<<std::endl;
    std::cout<<label_k<<std::endl;

    std::cout<<std::boolalpha<<(label_g == label_h)<<std::endl;
    std::cout<<std::boolalpha<<(label_h == label_i)<<std::endl;
    std::cout<<std::boolalpha<<(label_h == label_j)<<std::endl;
    std::cout<<std::boolalpha<<(label_g == label_k)<<std::endl;

    return true;
}

void colored_edges_test2() {
    std::cout << "colored_edges_test2()" << std::endl;
    typedef boost::property<alps::edge_type_t,unsigned int> edge_props;
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, edge_props> graph_type;
    typedef boost::graph_traits<graph_type>::edge_descriptor edge_descriptor;
    typedef boost::property_map<graph_type,alps::edge_type_t>::type edge_color_map_type;

    graph_type g;
    edge_descriptor e;
    e = add_edge(0,1,g).first;
    boost::put(alps::edge_type_t(),g,e,0);
    e = add_edge(0,2,g).first;
    boost::put(alps::edge_type_t(),g,e,0);

    graph_type h;
    e = add_edge(0,1,h).first;
    boost::put(alps::edge_type_t(),h,e,1);
    e = add_edge(0,2,h).first;
    boost::put(alps::edge_type_t(),h,e,1);

    graph_type i;
    e = add_edge(0,1,i).first;
    boost::put(alps::edge_type_t(),i,e,3);
    e = add_edge(0,2,i).first;
    boost::put(alps::edge_type_t(),i,e,3);

    graph_type j;
    e = add_edge(0,1,j).first;
    boost::put(alps::edge_type_t(),j,e,1);
    e = add_edge(0,2,j).first;
    boost::put(alps::edge_type_t(),j,e,3);

    graph_type k;
    e = add_edge(0,1,k).first;
    boost::put(alps::edge_type_t(),k,e,0);
    e = add_edge(0,2,k).first;
    boost::put(alps::edge_type_t(),k,e,1);


    alps::graph::graph_label<graph_type>::type label_g(get<1>(canonical_properties(g)));
    alps::graph::graph_label<graph_type>::type label_h(get<1>(canonical_properties(h)));
    alps::graph::graph_label<graph_type>::type label_i(get<1>(canonical_properties(i)));
    alps::graph::graph_label<graph_type>::type label_j(get<1>(canonical_properties(j)));
    alps::graph::graph_label<graph_type>::type label_k(get<1>(canonical_properties(k)));

    std::cout<<label_g<<std::endl;
    std::cout<<label_h<<std::endl;
    std::cout<<label_i<<std::endl;
    std::cout<<label_j<<std::endl;
    std::cout<<label_k<<std::endl;

    std::cout<<std::boolalpha<<(label_g == label_h)<<std::endl;
    std::cout<<std::boolalpha<<(label_h == label_i)<<std::endl;
    std::cout<<std::boolalpha<<(label_h == label_j)<<std::endl;
    std::cout<<std::boolalpha<<(label_g == label_k)<<std::endl;
}

bool simple_test() {
    std::cout << "simple_label_test()" << std::endl;
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph_type;
    graph_type g;
    add_edge(0, 1,g);
    add_edge(0, 2,g);
    add_edge(0, 3,g);
    add_edge(2, 1,g);

    alps::graph::graph_label<graph_type>::type label(get<1>(canonical_properties(g)));

    std::cout << label << std::endl;
    return true;
}

int main() {
    simple_test();
    colored_edges_test();
    colored_edges_test2();
    return 0;
}
