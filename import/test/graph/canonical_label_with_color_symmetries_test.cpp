/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/graph/canonical_properties.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <iostream>

typedef boost::property<alps::edge_type_t,unsigned int> edge_props;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, edge_props> graph_type;
typedef boost::graph_traits<graph_type>::edge_descriptor edge_descriptor;
typedef boost::graph_traits<graph_type>::edge_iterator edge_iterator;
typedef boost::property_map<graph_type,alps::edge_type_t>::type edge_color_map_type;

bool colored_edges_with_color_symmetry_test1() {
    std::cout << "colored_edges_with_color_symmetry_test1()" << std::endl;
    typedef alps::graph::graph_label<graph_type>::type label_type;
    using alps::graph::canonical_properties;

    // g: O---O+++O+++O
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
    }

    // h: O+++O---O---O
    graph_type h(g);
    {
        unsigned int const map[] = {1,0};
        edge_iterator it,end;
        for(boost::tie(it,end) = edges(h); it != end; ++it)
            get(alps::edge_type_t(),h)[*it] = map[get(alps::edge_type_t(),h)[*it]];
    }

    // i: O---O+++O---O
    graph_type i(g);
    {
        edge_descriptor e = edge(2,3,i).first;
        get(alps::edge_type_t(),i)[e] = 0;
    }

    // j: O+++O---O---O
    graph_type j(i);
    {
        unsigned int const map[] = {1,0};
        edge_iterator it,end;
        for(boost::tie(it,end) = edges(j); it != end; ++it)
            get(alps::edge_type_t(),j)[*it] = map[get(alps::edge_type_t(),j)[*it]];
    }


    alps::graph::color_partition<graph_type>::type color_symmetry;
    // makepair(color,color_partition)
    color_symmetry.insert(std::make_pair(0,0));
    color_symmetry.insert(std::make_pair(1,0));


    label_type lg(get<1>(canonical_properties(g)));
    label_type lh(get<1>(canonical_properties(h)));
    label_type li(get<1>(canonical_properties(i)));
    label_type lj(get<1>(canonical_properties(j)));


    label_type lg_with_sym(get<1>(canonical_properties(g,color_symmetry)));
    label_type lh_with_sym(get<1>(canonical_properties(h,color_symmetry)));
    label_type li_with_sym(get<1>(canonical_properties(i,color_symmetry)));
    label_type lj_with_sym(get<1>(canonical_properties(j,color_symmetry)));

    std::cout << lg << std::endl;
    std::cout << lh << std::endl;
    std::cout << li << std::endl;
    std::cout << lj << std::endl;
    std::cout << lg_with_sym << std::endl;
    std::cout << lh_with_sym << std::endl;
    std::cout << li_with_sym << std::endl;
    std::cout << lj_with_sym << std::endl;

    // False statements
    std::cout << std::boolalpha
        << (lg == lh) << (lg == li) << (lg == lj)
        << (lh == li) << (lh == lj)
        << (li == lj) << std::endl;

    // True statements
    std::cout << std::boolalpha
        << (lg_with_sym == lh_with_sym)
        << (li_with_sym == lj_with_sym) << std::endl;

    // False statements
    std::cout << std::boolalpha
        << (lg_with_sym == li_with_sym) << std::endl;

    return true;
}

bool colored_edges_with_color_symmetry_test2() {
    std::cout << "colored_edges_with_color_symmetry_test2()" << std::endl;
    typedef alps::graph::graph_label<graph_type>::type label_type;
    using alps::graph::canonical_properties;

    // g:
    //       4              c0 ---
    //       +              c1 +++
    //   1---0+++2          c2 ===
    //   \\ / 
    //     3
    graph_type g(5);
    {
        edge_color_map_type edge_color = get(alps::edge_type_t(),g);
        edge_descriptor e;
        e = add_edge(0,1,g).first;
        edge_color[e] = 0;
        e = add_edge(0,2,g).first;
        edge_color[e] = 1;
        e = add_edge(0,3,g).first;
        edge_color[e] = 0;
        e = add_edge(0,4,g).first;
        edge_color[e] = 1;
        e = add_edge(1,3,g).first;
        edge_color[e] = 2;
    }

    // h:
    //       4              c0 ---
    //       +              c1 +++
    //   1===0+++2          c2 ===
    //    \ // 
    //     3
    graph_type h(g);
    {
        unsigned int const map[] = {2,1,0};
        edge_iterator it,end;
        for(boost::tie(it,end) = edges(h); it != end; ++it)
            get(alps::edge_type_t(),h)[*it] = map[get(alps::edge_type_t(),h)[*it]];
    }


    // i:
    //       4              c0 ---
    //       ||             c1 +++
    //   1---0===2          c2 ===
    //    + / 
    //     3
    graph_type i(g);
    {
        unsigned int const map[] = {0,2,1};
        edge_iterator it,end;
        for(boost::tie(it,end) = edges(i); it != end; ++it)
            get(alps::edge_type_t(),i)[*it] = map[get(alps::edge_type_t(),i)[*it]];
    }


    // color   grp0  grp1
    // colors (0,2)  (1)
    alps::graph::color_partition<graph_type>::type color_symmetry;
    // makepair(color,color_partition)
    color_symmetry.insert(std::make_pair(0,0));
    color_symmetry.insert(std::make_pair(1,1));
    color_symmetry.insert(std::make_pair(2,0));

    label_type lg(get<1>(canonical_properties(g)));
    label_type lh(get<1>(canonical_properties(h)));
    label_type li(get<1>(canonical_properties(i)));

    label_type lg_with_sym(get<1>(canonical_properties(g,color_symmetry)));
    label_type lh_with_sym(get<1>(canonical_properties(h,color_symmetry)));
    label_type li_with_sym(get<1>(canonical_properties(i,color_symmetry)));

    std::cout << lg << std::endl;
    std::cout << lh << std::endl;
    std::cout << li << std::endl;
    std::cout << lg_with_sym << std::endl;
    std::cout << lh_with_sym << std::endl;
    std::cout << li_with_sym << std::endl;

    // False statements
    std::cout << std::boolalpha
        << (lg == lh) << (lg == li)
        << (lh == li) << std::endl;

    // True statements
    std::cout << std::boolalpha
        << (lg_with_sym == lh_with_sym) << std::endl;

    // False statements
    std::cout << std::boolalpha
        << (lg_with_sym == li_with_sym) << std::endl;

    return true;
}

bool colored_edges_with_color_symmetry_test3() {
    std::cout << "colored_edges_with_color_symmetry_test3()" << std::endl;
    typedef alps::graph::graph_label<graph_type>::type label_type;
    using alps::graph::canonical_properties;

    // g: O---O---O---O
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
    }

    // h: O+++O+++O+++O
    graph_type h(g);
    {
        unsigned int const map[] = {1,0};
        edge_iterator it,end;
        for(boost::tie(it,end) = edges(h); it != end; ++it)
            get(alps::edge_type_t(),h)[*it] = map[get(alps::edge_type_t(),h)[*it]];
    }

    alps::graph::color_partition<graph_type>::type color_symmetry;
    // makepair(color,color_partition)
    color_symmetry.insert(std::make_pair(0,0));
    color_symmetry.insert(std::make_pair(1,0));

    label_type lg(get<1>(canonical_properties(g)));
    label_type lh(get<1>(canonical_properties(h)));

    label_type lg_with_sym(get<1>(canonical_properties(g,color_symmetry)));
    label_type lh_with_sym(get<1>(canonical_properties(h,color_symmetry)));

    std::cout << lg << std::endl;
    std::cout << lh << std::endl;
    std::cout << lg_with_sym << std::endl;
    std::cout << lh_with_sym << std::endl;

    // False statements
    std::cout << std::boolalpha
        << (lg == lh) << std::endl;

    // True statements
    std::cout << std::boolalpha
        << (lg_with_sym == lh_with_sym) << std::endl;
    return true;
}

void colored_edges_with_color_symmetry_test4() {
    std::cout << "colored_edges_with_color_symmetry_test4()" << std::endl;
    typedef alps::graph::graph_label<graph_type>::type label_type;
    using alps::graph::canonical_properties;

    // g:
    //   2++++4     c0 +++
    //  /    /      c1 ---
    // 0++++1       c2 ...
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

    // h:
    //        2++++4     c0 +++
    //       /    /      c1 ---
    //      0++++1       c2 ...
    //     /
    //    3
    graph_type h(g);
    {
        unsigned int const map[] = {0,1,0};
        edge_iterator it,end;
        for(boost::tie(it,end) = edges(h); it != end; ++it)
            get(alps::edge_type_t(),h)[*it] = map[get(alps::edge_type_t(),h)[*it]];
    }

    // i:
    //        2++++4     c0 +++
    //       /    /      c1 ---
    // 3++++0++++1       c2 ...
    //
    graph_type i(g);
    {
        unsigned int const map[] = {0,1,1};
        edge_iterator it,end;
        for(boost::tie(it,end) = edges(i); it != end; ++it)
            get(alps::edge_type_t(),i)[*it] = map[get(alps::edge_type_t(),i)[*it]];
    }
    // color   grp0
    // colors (0,1,2)
    alps::graph::color_partition<graph_type>::type color_symmetry;
    // makepair(color,color_partition)
    color_symmetry.insert(std::make_pair(0,0));
    color_symmetry.insert(std::make_pair(1,0));
    color_symmetry.insert(std::make_pair(2,0));

    label_type lg(get<1>(canonical_properties(g)));
    label_type lh(get<1>(canonical_properties(h)));
    label_type li(get<1>(canonical_properties(i)));

    label_type lg_with_sym(get<1>(canonical_properties(g,color_symmetry)));
    label_type lh_with_sym(get<1>(canonical_properties(h,color_symmetry)));
    label_type li_with_sym(get<1>(canonical_properties(i,color_symmetry)));

    std::cout << lg << std::endl;
    std::cout << lh << std::endl;
    std::cout << li << std::endl;
    std::cout << lg_with_sym << std::endl;
    std::cout << lh_with_sym << std::endl;
    std::cout << li_with_sym << std::endl;

    // False statements
    std::cout << std::boolalpha
        << (lg == lh) << (lg == li)
        << (lh == li) << std::endl;

    // True statements
    std::cout << std::boolalpha
        << (lh_with_sym == li_with_sym) << std::endl;

    // False statements
    std::cout << std::boolalpha
        << (lg_with_sym == lh_with_sym) << std::endl;
}

int main() {
    colored_edges_with_color_symmetry_test1();
    colored_edges_with_color_symmetry_test2();
    colored_edges_with_color_symmetry_test3();
    colored_edges_with_color_symmetry_test4();
    return 0;
}
