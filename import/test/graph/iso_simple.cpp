/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
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

#include <alps/graph/subgraphs.hpp>
#include <alps/graph/canonical_properties.hpp>

#include <boost/graph/adjacency_list.hpp>

#include <iostream>
#include <iterator>

typedef boost::adjacency_list<
  boost::vecS, boost::vecS, boost::undirectedS
> graph_type;

typedef boost::adjacency_list<
      boost::vecS
    , boost::vecS
    , boost::undirectedS
    , boost::property<alps::vertex_type_t, alps::type_type>
    , boost::property<alps::edge_type_t, alps::type_type>
> colored_graph_type;

using namespace alps::graph;

// ostream operator for partition_type
template<typename Stream> Stream & operator<< (Stream & os, partition_type<colored_graph_type>::type const & pi) {
    os << "(";
    for (partition_type<colored_graph_type>::type::const_iterator it = pi.begin(); it != pi.end(); ++it) {
        os << "(";
        for (partition_type<colored_graph_type>::type::value_type::const_iterator jt = it->begin(); jt != it->end(); ++jt)
            os << (jt == it->begin() ? "" : " ") << *jt;
        os << ")";
    }
    os << ")";
    return os;
}

int main() {
    {
        enum { A, B, C, D, N };

        colored_graph_type g(N), h(N);
        
        /*
            A - B       A   B
            | / |  vs.  | X |
            C - D       C - D
        */

        add_edge(A, B, g);
        add_edge(A, C, g);
        add_edge(B, C, g);
        add_edge(B, D, g);
        add_edge(C, D, g);

        add_edge(A, C, h);
        add_edge(A, D, h);
        add_edge(B, C, h);
        add_edge(B, D, h);
        add_edge(C, D, h);

        boost::property_map<colored_graph_type, alps::vertex_type_t>::type g_vertex_name = get(alps::vertex_type_t(), g);
        g_vertex_name[A] = 0;
        g_vertex_name[B] = 1;
        g_vertex_name[C] = 1;
        g_vertex_name[D] = 0;

        boost::property_map<colored_graph_type, alps::vertex_type_t>::type h_vertex_name = get(alps::vertex_type_t(), h);
        h_vertex_name[A] = 0;
        h_vertex_name[B] = 0;
        h_vertex_name[C] = 1;
        h_vertex_name[D] = 1;
        
        boost::graph_traits<colored_graph_type>::edge_iterator it, end;

        boost::property_map<colored_graph_type, alps::edge_type_t>::type g_edge_name = get(alps::edge_type_t(), g);
        for (boost::tie(it, end) = edges(g); it != end; ++it)
            g_edge_name[*it] = (source(*it, g) == B && target(*it, g) == C) ? 1 : 0;

        boost::property_map<colored_graph_type, alps::edge_type_t>::type h_edge_name = get(alps::edge_type_t(), h);
        for (boost::tie(it, end) = edges(h); it != end; ++it)
            h_edge_name[*it] = (source(*it, h) == C && target(*it, h) == D) ? 1 : 0;

        std::vector<boost::graph_traits<colored_graph_type>::vertex_descriptor> g_ordering, h_ordering;
        graph_label<colored_graph_type>::type g_label, h_label;
        partition_type<colored_graph_type>::type g_orbit, h_orbit;
        boost::tie(g_ordering, g_label, g_orbit) = canonical_properties(g);
        boost::tie(h_ordering, h_label, h_orbit) = canonical_properties(h);

        for (std::vector<boost::graph_traits<colored_graph_type>::vertex_descriptor>::const_iterator it = g_ordering.begin(); it != g_ordering.end(); ++it)
            std::cout << (it != g_ordering.begin() ? " " : "(") << *it;
        std::cout << ")" << std::endl;
        for (std::vector<boost::graph_traits<colored_graph_type>::vertex_descriptor>::const_iterator it = h_ordering.begin(); it != h_ordering.end(); ++it)
            std::cout << (it != h_ordering.begin() ? " " : "(") << *it;
        std::cout << ")" << std::endl;

        std::cout << g_orbit << std::endl;
        std::cout << h_orbit << std::endl;
    
        std::cout << g_label << std::endl;
        std::cout << h_label << std::endl;
    }

    std::cout << std::endl;

    {
        enum {A, B, C, D, E, F, G, H, I, N};

        graph_type g(N);
        
        /*
            A - B - C
            |   |   |
            D - E - F
            |   |   |
            G - H - I
        */

        add_edge(A, B, g);
        add_edge(B, C, g);
        
        add_edge(A, D, g);
        add_edge(B, E, g);
        add_edge(C, F, g);
        
        add_edge(D, E, g);
        add_edge(E, F, g);
        
        add_edge(D, G, g);
        add_edge(E, H, g);
        add_edge(F, I, g);
        
        add_edge(G, H, g);
        add_edge(H, I, g);

        std::vector<boost::graph_traits<graph_type>::vertex_descriptor> g_ordering;
        graph_label<graph_type>::type g_label;
        partition_type<graph_type>::type g_orbit;
        boost::tie(g_ordering, g_label, g_orbit) = canonical_properties(g);
        
        for (std::vector<boost::graph_traits<graph_type>::vertex_descriptor>::const_iterator it = g_ordering.begin(); it != g_ordering.end(); ++it)
            std::cout << (it != g_ordering.begin() ? " " : "(") << *it;
        std::cout << ")" << std::endl;

        std::cout << g_orbit << std::endl;
        std::cout << g_label << std::endl;
    }

    std::cout << std::endl;

    {
        enum { A, B, C, D, N };

        graph_type g(N), h(N);
        
        /*
            A - B       A   B
            | / |  vs.  | X |
            C - D       C - D
        */
        
        add_edge(A, B, g);
        add_edge(A, C, g);
        add_edge(B, C, g);
        add_edge(B, D, g);
        add_edge(C, D, g);

        add_edge(A, C, h);
        add_edge(A, D, h);
        add_edge(B, C, h);
        add_edge(B, D, h);
        add_edge(C, D, h);

        std::vector<boost::graph_traits<graph_type>::vertex_descriptor> g_ordering, h_ordering;
        graph_label<graph_type>::type g_label, h_label;
        partition_type<graph_type>::type g_orbit, h_orbit;
        boost::tie(g_ordering, g_label, g_orbit) = canonical_properties(g);
        boost::tie(h_ordering, h_label, h_orbit) = canonical_properties(h);

        for (std::vector<boost::graph_traits<graph_type>::vertex_descriptor>::const_iterator it = g_ordering.begin(); it != g_ordering.end(); ++it)
            std::cout << (it != g_ordering.begin() ? " " : "(") << *it;
        std::cout << ")" << std::endl;
        for (std::vector<boost::graph_traits<graph_type>::vertex_descriptor>::const_iterator it = h_ordering.begin(); it != h_ordering.end(); ++it)
            std::cout << (it != h_ordering.begin() ? " " : "(") << *it;
        std::cout << ")" << std::endl;

        std::cout << g_orbit << std::endl;
        std::cout << h_orbit << std::endl;
        
        std::cout << g_label << std::endl;
        std::cout << h_label << std::endl;
    }

    std::cout << std::endl;

    {
        /*
                 F
                 |
              E--A--B
                / \
               D   C

                vs.

                  D
                  |
               A--B--C--E--F

                vs.

                   D
                   |
                A--B--C--F
                   |
                   E
        */
        enum { A, B, C, D, E, F, N};

        graph_type g(N), h(N), i(N);

        add_edge(A,B,g);
        add_edge(A,C,g);
        add_edge(A,D,g);
        add_edge(A,E,g);
        add_edge(A,F,g);

        add_edge(A,B,h);
        add_edge(B,C,h);
        add_edge(B,D,h);
        add_edge(C,E,h);
        add_edge(E,F,h);

        add_edge(A,B,i);
        add_edge(B,C,i);
        add_edge(B,D,i);
        add_edge(B,E,i);
        add_edge(C,F,i);
        
        std::vector<boost::graph_traits<graph_type>::vertex_descriptor> g_ordering, h_ordering, i_ordering;
        graph_label<graph_type>::type g_label, h_label, i_label;
        partition_type<graph_type>::type g_orbit, h_orbit, i_orbit;
        boost::tie(g_ordering, g_label, g_orbit) = canonical_properties(g);
        boost::tie(h_ordering, h_label, h_orbit) = canonical_properties(h);
        boost::tie(i_ordering, i_label, i_orbit) = canonical_properties(i);

        for (std::vector<boost::graph_traits<graph_type>::vertex_descriptor>::const_iterator it = g_ordering.begin(); it != g_ordering.end(); ++it)
            std::cout << (it != g_ordering.begin() ? " " : "(") << *it;
        std::cout << ")" << std::endl;
        for (std::vector<boost::graph_traits<graph_type>::vertex_descriptor>::const_iterator it = h_ordering.begin(); it != h_ordering.end(); ++it)
            std::cout << (it != h_ordering.begin() ? " " : "(") << *it;
        std::cout << ")" << std::endl;
        for (std::vector<boost::graph_traits<graph_type>::vertex_descriptor>::const_iterator it = i_ordering.begin(); it != i_ordering.end(); ++it)
            std::cout << (it != i_ordering.begin() ? " " : "(") << *it;
        std::cout << ")" << std::endl;

        std::cout << g_orbit << std::endl;
        std::cout << h_orbit << std::endl;
        std::cout << i_orbit << std::endl;
        
        std::cout << g_label << std::endl;
        std::cout << h_label << std::endl;
        std::cout << i_label << std::endl;
    }

    std::cout << std::endl;

    {
        enum { A, B, C, N };

        graph_type g(N);
        std::set<boost::dynamic_bitset<> > g_sub;
        
        /*
            A - B
            | /
            C
        */
        
        add_edge(A, B, g);
        add_edge(A, C, g);
        add_edge(B, C, g);
        
        subgraphs(g_sub, g);

        for (std::set<boost::dynamic_bitset<> >::const_iterator it = g_sub.begin(); it != g_sub.end(); ++it)
            std::cout << *it << std::endl;
    }

    {
    //#v = 10 #e = 14 Missed automorphism!
    //(0100000100010011001001001010011000001010000000010000000)
    //   0 1 2 3 4 5 6 7 8 9
    // 0 .
    // 1 . .
    // 2 . . .
    // 3 . . . .
    // 4 . . . x .
    // 5 . . . . . .
    // 6 . . . x x . .
    // 7 x x x . . . . .
    // 8 . . x . . x x . .
    // 9 . x . x x . . . x .

        graph_type g;
        add_edge(4,3,g);
        add_edge(6,3,g);
        add_edge(6,4,g);
        add_edge(7,0,g);
        add_edge(7,1,g);
        add_edge(7,2,g);
        add_edge(8,2,g);
        add_edge(8,5,g);
        add_edge(8,6,g);
        add_edge(9,1,g);
        add_edge(9,3,g);
        add_edge(9,4,g);
        add_edge(9,8,g);

        std::cout << get<alps::graph::label>(canonical_properties(g)) << std::endl;

        std::cout << get<alps::graph::partition>(canonical_properties(g)) << std::endl;
    }
    return EXIT_SUCCESS;
}
