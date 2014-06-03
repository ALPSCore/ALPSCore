/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef NISY_HPP
#define NISY_HPP

#include "nisy_impl.hpp"
#include "util.hpp"

#include <iostream>

#include <boost/mpl/if.hpp>

template <
    class graph_type 
  , class vertex_color_type = void
  , class edge_color_type = void
> class nisy {
  public:
    typedef typename std::list<typename boost::graph_traits<graph_type>::vertex_descriptor> cell_type;
    typedef typename std::list<cell_type> partition_type;
    typedef typename partition_type::iterator partition_iterator_type;
    typedef typename detail::canonical_label_type<vertex_color_type, edge_color_type>::type canonical_label_type;
    typedef typename detail::canonical_ordering_iterator<partition_type> canonical_ordering_iterator;
    nisy(
        graph_type const & graph
    ) {
      BOOST_STATIC_ASSERT(
        (boost::is_void<vertex_color_type>::value)
      ); 
      BOOST_STATIC_ASSERT(
        (boost::is_void<edge_color_type>::value)
      ); 
      impl = new detail::nisy_derived<
          graph_type
        , vertex_color_type
        , edge_color_type
        , boost::no_property
        , boost::no_property
      >(
          detail::no_coloring_tag()
        , graph
      );
    }
    template <
        class color_property_map_type
    > nisy(
        graph_type const & graph
      , color_property_map_type color_property
    ) {
    
    
    
    
    
    
/*      BOOST_STATIC_ASSERT((
           (boost::is_void<vertex_color_type>::value == false && boost::is_void<vertex_color_type>::value == true)
        || (boost::is_void<vertex_color_type>::value == true && boost::is_void<vertex_color_type>::value == false)
      ));
 */
 
 
 
      impl = new detail::nisy_derived<
          graph_type
        , vertex_color_type
        , edge_color_type
        , typename boost::mpl::if_<
              boost::is_void<vertex_color_type>
            , boost::no_property
            , color_property_map_type
          >::type
        , typename boost::mpl::if_<
              boost::is_void<edge_color_type>
            , boost::no_property
            , color_property_map_type
          >::type
      >(
          typename boost::mpl::if_<
              boost::is_void<vertex_color_type>
            , detail::edge_coloring_tag
            , detail::vertex_coloring_tag
          >::type()
        , graph
        , color_property
      );
    }
    template <
        class vertex_color_property_map_type
      , class edge_color_property_map_type
    > nisy(
        graph_type const & graph
      , vertex_color_property_map_type vertex_property
      , edge_color_property_map_type edge_property
    ) {
      BOOST_STATIC_ASSERT(
        (boost::is_void<vertex_color_type>::value == false)
      );
      BOOST_STATIC_ASSERT(
        (boost::is_void<edge_color_type>::value == false)
      );
      impl = new detail::nisy_derived<
          graph_type
        , vertex_color_type
        , edge_color_type
        , vertex_color_property_map_type
        , edge_color_property_map_type
      >(
          detail::vertex_edge_coloring_tag()
        , graph
        , vertex_property
        , edge_property
      );
    }
    virtual ~nisy() {
      delete impl;
    }
    inline void invalidate() {
      impl->invalidate(); 
    }
    inline std::pair<canonical_ordering_iterator, canonical_ordering_iterator> get_canonical_ordering() const {
      return impl->get_canonical_ordering(); 
    }
    inline canonical_label_type const & get_canonical_label() const {
      return impl->get_canonical_label(); 
    }
    inline partition_type const & get_orbit_partition() const { 
      return impl->get_orbit_partition(); 
    }
    template<
        class graph_type1
    > inline bool operator==(
        nisy<graph_type1, vertex_color_type, edge_color_type> const & T
    ) const {
      if (num_vertices(*(T.impl)) != num_vertices(*impl) || num_edges(*(T.impl)) != num_edges(*impl))
        return false;
      return T.get_canonical_label() == get_canonical_label();
    }
    template<
        class graph_type1
    > inline bool operator!=(
        nisy<graph_type1, vertex_color_type, edge_color_type> const & T
    ) const {
      return !operator==(T);
    }
  private:
    mutable detail::nisy_base<graph_type, vertex_color_type, edge_color_type> *impl;
};

template<
    class graph_type1
  , class graph_type2
  , class vertex_color_type1
  , class vertex_color_type2
  , class edge_color_type1
  , class edge_color_type2
> inline std::map<
      typename boost::graph_traits<graph_type1>::vertex_descriptor
    , typename boost::graph_traits<graph_type2>::vertex_descriptor
  > isomorphism(
      nisy<graph_type1, vertex_color_type1, edge_color_type1> const & T1
    , nisy<graph_type2, vertex_color_type2, edge_color_type2> const & T2
  ) 
{
  if (T1 != T2)
    boost::throw_exception(std::runtime_error("The passed Graphes are not isomorph."));
  std::map<
      typename boost::graph_traits<graph_type1>::vertex_descriptor
    , typename boost::graph_traits<graph_type2>::vertex_descriptor
  > isomorphism;
  typename nisy<graph_type1, vertex_color_type1, edge_color_type1>::canonical_ordering_iterator it1, end1;
  typename nisy<graph_type2, vertex_color_type2, edge_color_type2>::canonical_ordering_iterator it2, end2;
  boost::tie(it1, end1) = T1.get_canonical_ordering();
  boost::tie(it2, end2) = T2.get_canonical_ordering();
  for (; it1 != end1 && it2 != end2; ++it1, ++it2)
    isomorphism.insert(std::make_pair(*it1, *it2));
  return isomorphism;
}

#endif // NISY_HPP
