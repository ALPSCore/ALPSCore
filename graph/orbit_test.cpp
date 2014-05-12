/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2013 by Andreas Hehn <hehn@phys.ethz.ch>                          *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <alps/graph/canonical_properties.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <iostream>

using alps::graph::canonical_properties;

typedef boost::property<alps::edge_type_t,unsigned int> edge_props;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, edge_props> graph_type;
typedef boost::graph_traits<graph_type>::edge_descriptor edge_descriptor;
typedef boost::property_map<graph_type,alps::edge_type_t>::type edge_color_map_type;

template <typename Partition>
void dump_partition(std::ostream& os, Partition const& partition)
{
    os << "(";
    for (typename Partition::const_iterator it = partition.begin(); it != partition.end(); ++it) {
        os << "(";
        for (typename Partition::value_type::const_iterator jt = it->begin(); jt != it->end(); ++jt)
            os << (jt == it->begin() ? "" : " ") << *jt;
        os << ")";
    }
    os << ")";
}

void orbit_test1()
{
    // 0---3
    // |   *
    // 1***2
    graph_type g(4);
    {
        edge_color_map_type edge_color = get(alps::edge_type_t(),g);
        edge_descriptor e;
        e = add_edge(0, 1, g).first;
        edge_color[e] = 0;
        e = add_edge(1, 2, g).first;
        edge_color[e] = 1;
        e = add_edge(2, 3, g).first;
        edge_color[e] = 1;
        e = add_edge(0, 3, g).first;
        edge_color[e] = 0;
    }

    alps::graph::canonical_properties_type<graph_type>::type gp(canonical_properties(g));

    alps::graph::color_partition<graph_type>::type color_symmetry;
    color_symmetry[0] = 0;
    color_symmetry[1] = 0;

    alps::graph::canonical_properties_type<graph_type>::type gp_with_sym(canonical_properties(g,color_symmetry));

    dump_partition(std::cout, get<alps::graph::partition>(gp));
    std::cout << std::endl;
    dump_partition(std::cout, get<alps::graph::partition>(gp_with_sym));
    std::cout << std::endl;
}

void orbit_test2()
{
    //    2
    //  /  .
    // 0++++1
    //  .  /
    //   3
    graph_type g(4);
    {
        edge_color_map_type edge_color = get(alps::edge_type_t(),g);
        edge_descriptor e;
        e = add_edge(0, 1, g).first;
        edge_color[e] = 0;
        e = add_edge(0, 2, g).first;
        edge_color[e] = 1;
        e = add_edge(0, 3, g).first;
        edge_color[e] = 2;
        e = add_edge(1, 2, g).first;
        edge_color[e] = 2;
        e = add_edge(1, 3, g).first;
        edge_color[e] = 1;
    }

    alps::graph::canonical_properties_type<graph_type>::type gp(canonical_properties(g));

    alps::graph::color_partition<graph_type>::type color_symmetry;
    color_symmetry[0] = 0;
    color_symmetry[1] = 0;
    color_symmetry[2] = 0;

    alps::graph::canonical_properties_type<graph_type>::type gp_with_sym(canonical_properties(g,color_symmetry));

    dump_partition(std::cout, get<alps::graph::partition>(gp));
    std::cout << std::endl;
    dump_partition(std::cout, get<alps::graph::partition>(gp_with_sym));
    std::cout << std::endl;
}

void orbit_test3()
{
    //    2+++4
    //  /    /
    // 0++++1
    //  . 
    //   3
    graph_type g(4);
    {
        edge_color_map_type edge_color = get(alps::edge_type_t(),g);
        edge_descriptor e;
        e = add_edge(0, 1, g).first;
        edge_color[e] = 0;
        e = add_edge(0, 2, g).first;
        edge_color[e] = 1;
        e = add_edge(0, 3, g).first;
        edge_color[e] = 2;
        e = add_edge(1, 4, g).first;
        edge_color[e] = 1;
        e = add_edge(2, 4, g).first;
        edge_color[e] = 0;
    }

    alps::graph::canonical_properties_type<graph_type>::type gp(canonical_properties(g));

    alps::graph::color_partition<graph_type>::type color_symmetry;
    color_symmetry[0] = 0;
    color_symmetry[1] = 0;
    color_symmetry[2] = 0;

    alps::graph::canonical_properties_type<graph_type>::type gp_with_sym(canonical_properties(g,color_symmetry));

    dump_partition(std::cout, get<alps::graph::partition>(gp));
    std::cout << std::endl;
    dump_partition(std::cout, get<alps::graph::partition>(gp_with_sym));
    std::cout << std::endl;
}
int main() {
    orbit_test1();
    orbit_test2();
    orbit_test3();
}
