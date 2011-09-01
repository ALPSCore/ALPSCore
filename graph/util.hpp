/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2010 by Lukas Gamper
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

#ifndef UTIL_HPP
#define UTIL_HPP

#include <vector>
#include <deque>
#include <map>
#include <stdexcept>
#include <stack>

#include <boost/graph/graph_traits.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/throw_exception.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/operators.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>
#include <boost/assert.hpp>  
#include <boost/mpl/logical.hpp>
#include <boost/mpl/if.hpp>

namespace alps {
namespace graph {
namespace detail {
  struct embedding_moving_tag {};
  struct embedding_fixed_tag {};
  template<bool is_glued> struct embedding_is_fixed {
    BOOST_STATIC_ASSERT(is_glued && !is_glued); 
  };
  template<> struct embedding_is_fixed<true> {
    typedef embedding_fixed_tag type;
  };
  template<> struct embedding_is_fixed<false> {
    typedef embedding_moving_tag type;
  };
  struct no_coloring_tag {};
  struct vertex_coloring_tag {};
  struct edge_coloring_tag {};
  struct vertex_edge_coloring_tag
    : public vertex_coloring_tag
    , public edge_coloring_tag
  {};
  template<bool has_vertex_coloring, bool has_edge_coloring> struct coloring_tag {
    BOOST_STATIC_ASSERT(has_vertex_coloring && !has_vertex_coloring);
  };
  template<> struct coloring_tag<true, true> {
    typedef no_coloring_tag type;
  };
  template<> struct coloring_tag<false, true> {
    typedef vertex_coloring_tag type;
  };
  template<> struct coloring_tag<true, false> {
    typedef edge_coloring_tag type;
  };
  template<> struct coloring_tag<false, false> {
    typedef vertex_edge_coloring_tag type;
  };
  template<
      class vertex_color_type
    , class edge_color_type
    , class tag
  > struct canonical_label_type_impl {
    BOOST_STATIC_ASSERT(sizeof(tag) == 0);
  };
  template<
     class vertex_color_type
    , class edge_color_type
 > struct canonical_label_type_impl<
     vertex_color_type
   , edge_color_type
   , no_coloring_tag
  > {
    typedef boost::dynamic_bitset<> type; // nauty adjacency matrix 
  };
  template<
      class vertex_color_type
    , class edge_color_type
  > struct canonical_label_type_impl<
      vertex_color_type
    , edge_color_type
    , vertex_coloring_tag
  > {
    typedef boost::tuple<
        boost::dynamic_bitset<> // nauty adjacency matrix 
      , boost::dynamic_bitset<> // vertex color matrix
      , std::vector<vertex_color_type> // vertex color list
    > type;
  };
  template<
      class vertex_color_type
    , class edge_color_type
  > struct canonical_label_type_impl<
      vertex_color_type
    , edge_color_type
    , edge_coloring_tag
  > {
    typedef boost::tuple<
        boost::dynamic_bitset<> // nauty adjacency matrix 
      , boost::dynamic_bitset<> // edge color matrix
      , std::vector<edge_color_type> // edge color list
    > type;
  };
  template<
      class vertex_color_type
    , class edge_color_type
  > struct canonical_label_type_impl<
      vertex_color_type
    , edge_color_type
    , vertex_edge_coloring_tag
  > {
    typedef boost::tuple<
        boost::dynamic_bitset<> // nauty adjacency matrix 
      , boost::dynamic_bitset<> // vertex color matrix
      , std::vector<vertex_color_type> // vertex color list
      , boost::dynamic_bitset<> // edge color matrix
      , std::vector<edge_color_type> // edge color list
    > type;
  };
  template<
      class vertex_color_type
    , class edge_color_type
  > struct canonical_label_type {
    typedef typename canonical_label_type_impl<
        vertex_color_type
      , edge_color_type
      , typename coloring_tag<
            boost::is_void<vertex_color_type>::value
          , boost::is_void<edge_color_type>::value
        >::type
    >::type type;
  };
  template <class partition_type> class canonical_ordering_iterator
    : public boost::forward_iterator_helper<
        canonical_ordering_iterator<partition_type>
      , typename partition_type::value_type::value_type
      , std::ptrdiff_t
      , typename partition_type::value_type::value_type *
      , typename partition_type::value_type::value_type &
    >
  {
    public:
      canonical_ordering_iterator() {}
      canonical_ordering_iterator(
          typename partition_type::iterator const & it
      )
        : it_(it)
      {}
      typename partition_type::value_type::value_type operator*() const {
        return it_->front();
      }
      void operator++() {
        ++it_;
      }
      bool operator==(
        canonical_ordering_iterator<partition_type> const & T
      ) const {
        return it_ == T.it_;
      }  
    private:
      typename partition_type::iterator it_;
  };
} // namespace detail
} // namespace graph
} // namespace alps

#endif // UTIL_HPP
