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

#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP

#include "util.hpp"

template<
    class property_map_type
  , class subgraph_type
  , class graph_type
  , class subgraph_prop0_type = boost::no_property
  , class graph_prop0_type = boost::no_property
  , class subgraph_prop1_type = boost::no_property
  , class graph_prop1_type = boost::no_property
  , class subgraph_prop2_type = boost::no_property
  , class graph_prop2_type = boost::no_property
> class embedding_iterator
  : public boost::forward_iterator_helper<
      embedding_iterator<
          property_map_type
        , subgraph_type
        , graph_type
        , subgraph_prop0_type
        , graph_prop0_type
        , subgraph_prop1_type
        , graph_prop1_type
        , subgraph_prop2_type
        , graph_prop2_type
      >
    , property_map_type
    , std::ptrdiff_t
    , property_map_type *
    , property_map_type &
  >
{
  public:
    typedef typename boost::graph_traits<subgraph_type>::vertex_descriptor subgraph_vertex_descriptor;  
    typedef typename boost::graph_traits<graph_type>::vertex_descriptor graph_vertex_descriptor;  
    typedef typename boost::graph_traits<subgraph_type>::edge_descriptor subgraph_edge_descriptor;  
    typedef typename boost::graph_traits<graph_type>::edge_descriptor graph_edge_descriptor;  
    typedef typename boost::graph_traits<subgraph_type>::adjacency_iterator subgraph_adjacency_iterator;  
    typedef typename boost::graph_traits<graph_type>::adjacency_iterator graph_adjacency_iterator;  
    typedef typename boost::graph_traits<subgraph_type>::vertex_iterator subgraph_vertex_iterator;  
    typedef typename boost::graph_traits<graph_type>::vertex_iterator graph_vertex_iterator;  
    typedef typename boost::graph_traits<subgraph_type>::edge_iterator subgraph_edge_iterator;  
    typedef typename boost::graph_traits<graph_type>::edge_iterator graph_edge_iterator;  
  private:
    typedef typename boost::mpl::and_<
        boost::is_same<
            subgraph_vertex_descriptor
          , subgraph_prop0_type
        >
      , boost::is_same<
            graph_vertex_descriptor
          , graph_prop0_type
        >
    >::type fixed_tag;
    typedef embedding_iterator<
        property_map_type
      , subgraph_type
      , graph_type
      , subgraph_prop0_type
      , graph_prop0_type
      , subgraph_prop1_type
      , graph_prop1_type
      , subgraph_prop2_type
      , graph_prop2_type
    > base_type;    
  public:  
    typedef typename boost::mpl::if_<
        fixed_tag
      , subgraph_prop1_type
      , subgraph_prop0_type
    >::type subgraph_vertex_coloring_type;
    typedef typename boost::mpl::if_<
        fixed_tag
      , graph_prop1_type
      , graph_prop0_type
    >::type graph_vertex_coloring_type;
    typedef typename boost::mpl::if_<
        fixed_tag
      , subgraph_prop1_type
      , subgraph_prop2_type
    >::type subgraph_edge_coloring_type;
    typedef typename boost::mpl::if_<
        fixed_tag
      , graph_prop1_type
      , graph_prop2_type
    >::type graph_edge_coloring_type;
    embedding_iterator()
      : closed_(true)
      , subgraph_(NULL)
      , graph_(NULL) 
      {}
    embedding_iterator(
        property_map_type & property_map
      , subgraph_type const & subgraph
      , graph_type const & graph
      , subgraph_prop0_type const & subgraph_prop0
      , graph_prop0_type const & graph_prop0
      , subgraph_prop1_type const & subgraph_prop1
      , graph_prop1_type const & graph_prop1
      , subgraph_prop2_type const & subgraph_prop2
      , graph_prop2_type const & graph_prop2
    )
      : fixed_graph_vertex_(fixed_tag::value)
      , property_map_(&property_map)
      , subgraph_(&subgraph)
      , graph_(&graph)
    {
      construct(
          property_map
        , subgraph
        , graph
        , subgraph_prop0
        , graph_prop0
        , subgraph_prop1
        , graph_prop1
        , subgraph_prop2
        , graph_prop2
        , typename detail::embedding_is_fixed<fixed_tag::value>::type()
      );
    }
    property_map_type const & operator*() const {
      return *property_map_;
    }
    void operator++() {
      next();
    }
    bool operator==(base_type const & T) const {
      return T.closed_ ? closed_ : false;
    }
    base_type & operator=(base_type const & T) {
      closed_ = T.closed_;
      fixed_graph_vertex_ = T.fixed_graph_vertex_;
      trace_ = T.trace_;
      visited_ = T.visited_;
      occupied_ = T.occupied_;
      placed_ = T.placed_;
      property_map_ = T.property_map_;
      subgraph_ = T.subgraph_;
      graph_ = T.graph_;
      subgraph_vertex_properties_ = T.subgraph_vertex_properties_;
      graph_vertex_properties_ = T.graph_vertex_properties_;
      subgraph_edge_properties_ = T.subgraph_edge_properties_;
      graph_edge_properties_ = T.graph_edge_properties_;
      vertex_it_ = T.vertex_it_;
      vertex_end_ = T.vertex_end_;
      return *this;
    }
  private:
    typedef boost::tuple<
        subgraph_vertex_descriptor
      , subgraph_vertex_descriptor
      , graph_adjacency_iterator
      , graph_adjacency_iterator
    > search_node;
    void construct(
        property_map_type & property_map
      , subgraph_type const & subgraph
      , graph_type const & graph
      , subgraph_vertex_coloring_type const & subgraph_vertex_properties
      , graph_vertex_coloring_type const & graph_vertex_properties
      , subgraph_edge_coloring_type const & subgraph_edge_properties
      , graph_edge_coloring_type const & graph_edge_properties
      , subgraph_prop2_type
      , graph_prop2_type
      , detail::embedding_moving_tag const &
    ) {
      BOOST_STATIC_ASSERT((boost::is_same<
          typename detail::coloring_tag<
              boost::is_same<subgraph_vertex_coloring_type, boost::no_property>::value
            , boost::is_same<subgraph_edge_coloring_type, boost::no_property>::value
          >::type
        , typename detail::coloring_tag<
              boost::is_same<graph_vertex_coloring_type, boost::no_property>::value
            , boost::is_same<graph_edge_coloring_type, boost::no_property>::value
          >::type
      >::value)); 
      closed_ = true;
      subgraph_vertex_properties_ = subgraph_vertex_properties;
      graph_vertex_properties_ = graph_vertex_properties;
      subgraph_edge_properties_ = subgraph_edge_properties;
      graph_edge_properties_ = graph_edge_properties;
      for (boost::tie(vertex_it_, vertex_end_) = vertices(*graph_); closed_ && vertex_it_ != vertex_end_; ++vertex_it_) {
        initialize(*(vertices(subgraph).first), *vertex_it_);
        has_canonical_order();
      }
    }
    void construct(
        property_map_type & property_map
      , subgraph_type const & subgraph
      , graph_type const & graph
      , subgraph_vertex_descriptor subgraph_vertex
      , graph_vertex_descriptor graph_vertex
      , subgraph_vertex_coloring_type const & subgraph_vertex_properties
      , graph_vertex_coloring_type const & graph_vertex_properties
      , subgraph_edge_coloring_type const & subgraph_edge_properties
      , graph_edge_coloring_type const & graph_edge_properties
      , detail::embedding_fixed_tag const &
    ) {
      BOOST_STATIC_ASSERT((boost::is_same<
          typename detail::coloring_tag<
              boost::is_same<subgraph_vertex_coloring_type, boost::no_property>::value
            , boost::is_same<subgraph_edge_coloring_type, boost::no_property>::value
          >::type
        , typename detail::coloring_tag<
              boost::is_same<graph_vertex_coloring_type, boost::no_property>::value
            , boost::is_same<graph_edge_coloring_type, boost::no_property>::value
          >::type
      >::value));
      fixed_graph_vertex_ = true;
      property_map_ = &property_map;
      subgraph_ = &subgraph;
      graph_ = &graph;
      subgraph_vertex_properties_ = subgraph_vertex_properties;
      graph_vertex_properties_ = graph_vertex_properties;
      subgraph_edge_properties_ = subgraph_edge_properties;
      graph_edge_properties_ = graph_edge_properties;
      initialize(subgraph_vertex, graph_vertex); 
    }
    void initialize(
      subgraph_vertex_descriptor subgraph_vertex
      , graph_vertex_descriptor graph_vertex
    ) {
	  trace_.clear();
      std::deque<search_node> stack;
      subgraph_adjacency_iterator sit, send, tend = adjacent_vertices(graph_vertex, *graph_).second;
      // create trace
      stack.push_back(boost::make_tuple(subgraph_vertex, subgraph_vertex, tend, tend));
      placed_.insert(subgraph_vertex);
      while (stack.size()) {
        trace_.push_back(stack.front());
        stack.pop_front();
        for (tie(sit, send) = adjacent_vertices(boost::get<0>(trace_.back()), *subgraph_); sit != send; ++sit)
          if (placed_.find(*sit) == placed_.end()) {
            stack.push_back(boost::make_tuple(*sit, boost::get<0>(trace_.back()), tend, tend));
            placed_.insert(*sit);
          }
      }
      first(graph_vertex);
    }
    void first(
      graph_vertex_descriptor graph_vertex    
    ) {
      // find embedding for trace
      placed_.clear();
      occupied_.clear();
      occupied_.insert(std::make_pair(graph_vertex, boost::get<0>(trace_.front())));
      placed_.insert(boost::get<0>(trace_.front()));
      boost::put(*property_map_, boost::get<0>(trace_.front()), graph_vertex);
      typename detail::coloring_tag<
          boost::is_same<graph_vertex_coloring_type, boost::no_property>::value
        , boost::is_same<graph_edge_coloring_type, boost::no_property>::value
      >::type tag;
      typename std::vector<search_node>::iterator tit = trace_.begin();
      if (trace_.size() > 1) {
        ++tit;
        boost::tie(boost::get<2>(*tit), boost::get<3>(*tit)) = adjacent_vertices(graph_vertex, *graph_);
      }
      next_leaf(tit);
      if (!closed_ && !has_equal_coloring(tag))
        next();
    }
    void next() {
      typename detail::coloring_tag<
          boost::is_same<graph_vertex_coloring_type, boost::no_property>::value
        , boost::is_same<graph_edge_coloring_type, boost::no_property>::value
      >::type tag;
      do {
        typename std::vector<search_node>::iterator tit = trace_.end() - 1;
        occupied_.erase(*boost::get<2>(*tit));
        placed_.erase(boost::get<0>(*tit));
        ++boost::get<2>(*tit);
        next_leaf(tit);
        if (!fixed_graph_vertex_ && closed_ && ++vertex_it_ != vertex_end_)
          first(*vertex_it_);
      } while(!closed_ && (!has_equal_coloring(tag) || (!fixed_graph_vertex_ && !has_canonical_order())));
    }    
    void next_leaf(
      typename std::vector<search_node>::iterator & tit
    ) {
      subgraph_adjacency_iterator sit, send;
      while (tit != trace_.begin() && tit != trace_.end()) {
        do {
          if (boost::get<2>(*tit) == boost::get<3>(*tit)) {
            occupied_.erase(*boost::get<2>(*(--tit)));
            placed_.erase(boost::get<0>(*tit));
            ++boost::get<2>(*tit);
          }
          while (boost::get<2>(*tit) != boost::get<3>(*tit) && occupied_.find(*boost::get<2>(*tit)) != occupied_.end())
           ++boost::get<2>(*tit);
        } while (tit != trace_.begin() && boost::get<2>(*tit) == boost::get<3>(*tit));
        if (tit != trace_.begin()) {
          occupied_.insert(std::make_pair(*boost::get<2>(*tit), boost::get<0>(*tit)));
          placed_.insert(boost::get<0>(*tit));
          boost::put(*property_map_, boost::get<0>(*tit), *boost::get<2>(*tit));
          // check if all placed neighbors are avalable
          bool valid = true;
          for (tie(sit, send) = adjacent_vertices(boost::get<0>(*tit), *subgraph_); sit != send; ++sit)
            if (placed_.find(*sit) != placed_.end() && !(valid = edge(boost::get(*property_map_, *sit), *boost::get<2>(*tit), *graph_).second))
               break;
          // backtracking
          if (!valid) {
            occupied_.erase(*boost::get<2>(*tit));
            placed_.erase(boost::get<0>(*tit));
            ++boost::get<2>(*tit);
            while (tit != trace_.begin() && boost::get<2>(*tit) == boost::get<3>(*tit)) {
              occupied_.erase(*boost::get<2>(*(--tit)));
              placed_.erase(boost::get<0>(*tit));
              ++boost::get<2>(*tit);
              while (boost::get<2>(*tit) != boost::get<3>(*tit) && occupied_.find(*boost::get<2>(*tit)) != occupied_.end())
               ++boost::get<2>(*tit);
            }
          // next vertex
          } else if (++tit != trace_.end())
            boost::tie(boost::get<2>(*tit), boost::get<3>(*tit)) = adjacent_vertices(boost::get(*property_map_, boost::get<1>(*tit)), *graph_);
        }
      }
      closed_ = (tit == trace_.begin());
    }
    inline bool has_canonical_order() {
      std::vector<graph_edge_descriptor> label;
      subgraph_edge_iterator eit, eend;
      for (boost::tie(eit, eend) = boost::edges(*subgraph_); eit != eend; ++eit)
        label.push_back(boost::edge(
            boost::get(*property_map_, boost::source(*eit, *subgraph_))
          , boost::get(*property_map_, boost::target(*eit, *subgraph_))
          , *graph_
        ).first);
      std::sort(label.begin(), label.end());
      if (visited_.find(label) != visited_.end())
          return false;
      visited_.insert(label);
      return true;
    }    
    inline bool has_equal_vertex_coloring() {
      subgraph_vertex_iterator vit, vend;
      for (boost::tie(vit, vend) = boost::vertices(*subgraph_); vit != vend; ++vit)
        if (
             boost::get(graph_vertex_properties_, *graph_, boost::get(*property_map_, *vit)) 
          == boost::get(subgraph_vertex_properties_, *subgraph_, *vit)
        )
          return false;
      return true;
    }
    inline bool has_equal_edge_coloring() {
      subgraph_edge_iterator eit, eend;
      for (boost::tie(eit, eend) = boost::edges(*subgraph_); eit != eend; ++eit)
        if (
          boost::get(
            graph_edge_properties_
            , *graph_
            , boost::edge(
                boost::get(*property_map_, boost::source(*eit, *subgraph_))
              , boost::get(*property_map_, boost::target(*eit, *subgraph_))
              , *graph_).first
          ) != boost::get(subgraph_edge_properties_, *subgraph_ , *eit)
        )
          return false;
       return true;
    }
    inline bool has_equal_coloring(detail::no_coloring_tag) {
      return true;
    }    
    inline bool has_equal_coloring(detail::vertex_coloring_tag) {
        return has_equal_vertex_coloring();
    }    
    inline bool has_equal_coloring(detail::edge_coloring_tag) {
        return has_equal_edge_coloring();
    }    
    inline bool has_equal_coloring(detail::vertex_edge_coloring_tag) {
        return has_equal_vertex_coloring() && has_equal_edge_coloring();
    }
    bool closed_;
    bool fixed_graph_vertex_;
    std::vector<search_node> trace_;
    std::set<std::vector<graph_edge_descriptor> > visited_;
    std::map<graph_vertex_descriptor, subgraph_vertex_descriptor> occupied_;
    std::set<subgraph_vertex_descriptor> placed_;
    property_map_type * property_map_;
    subgraph_type const * subgraph_;
    graph_type const * graph_;
    subgraph_vertex_coloring_type subgraph_vertex_properties_;
    graph_vertex_coloring_type graph_vertex_properties_;
    subgraph_edge_coloring_type subgraph_edge_properties_;
    graph_edge_coloring_type graph_edge_properties_;
    graph_vertex_iterator vertex_it_, vertex_end_;  
};
template<
    class property_map_type 
  , class subgraph_type
  , class graph_type 
  , class subgraph_prop0_type
  , class graph_prop0_type
  , class subgraph_prop1_type
  , class graph_prop1_type
  , class subgraph_prop2_type
  , class graph_prop2_type
> typename std::pair<
    embedding_iterator<
        property_map_type
      , subgraph_type
      , graph_type
      , subgraph_prop0_type
      , graph_prop0_type
      , subgraph_prop1_type
      , graph_prop1_type
      , subgraph_prop2_type
      , graph_prop2_type
    >
  , embedding_iterator<
        property_map_type
      , subgraph_type
      , graph_type
      , subgraph_prop0_type
      , graph_prop0_type
      , subgraph_prop1_type
      , graph_prop1_type
      , subgraph_prop2_type
      , graph_prop2_type
    >
> embedding (
    property_map_type & property_map
  , subgraph_type const & subgraph
  , graph_type const & graph
  , subgraph_prop0_type const & subgraph_prop0
  , graph_prop0_type const & graph_prop0
  , subgraph_prop1_type const & subgraph_prop1
  , graph_prop1_type const & graph_prop1
  , subgraph_prop2_type const & subgraph_prop2
  , graph_prop2_type const & graph_prop2
) {
  typedef embedding_iterator<
      property_map_type
    , subgraph_type
    , graph_type
    , subgraph_prop0_type
    , graph_prop0_type
    , subgraph_prop1_type
    , graph_prop1_type
    , subgraph_prop2_type
    , graph_prop2_type
  > iterator;
  return std::make_pair(iterator(
      property_map
    , subgraph
    , graph
    , subgraph_prop0
    , graph_prop0
    , subgraph_prop1
    , graph_prop1
    , subgraph_prop2
    , graph_prop2
  ), iterator());
}
template<
    class property_map_type 
  , class subgraph_type
  , class graph_type 
  , class subgraph_prop0_type
  , class graph_prop0_type
  , class subgraph_prop1_type
  , class graph_prop1_type
> std::pair<
    embedding_iterator<
        property_map_type
      , subgraph_type
      , graph_type
      , subgraph_prop0_type
      , graph_prop0_type
      , subgraph_prop1_type
      , graph_prop1_type
    >
  , embedding_iterator<
        property_map_type
      , subgraph_type
      , graph_type
      , subgraph_prop0_type
      , graph_prop0_type
      , subgraph_prop1_type
      , graph_prop1_type
    >
> embedding (
    property_map_type & property_map
  , subgraph_type const & subgraph
  , graph_type const & graph
  , subgraph_prop0_type const & subgraph_prop0
  , graph_prop0_type const & graph_prop0
  , subgraph_prop1_type const & subgraph_prop1
  , graph_prop1_type const & graph_prop1
) {
  return embedding(
      property_map
    , subgraph
    , graph
    , subgraph_prop0
    , graph_prop0
    , subgraph_prop1
    , graph_prop1
    , boost::no_property()
    , boost::no_property()
  );
}
template<
    class property_map_type 
  , class subgraph_type
  , class graph_type 
  , class subgraph_prop0_type
  , class graph_prop0_type
> std::pair<
    embedding_iterator<
        property_map_type
      , subgraph_type
      , graph_type
      , subgraph_prop0_type
      , graph_prop0_type
    >
  , embedding_iterator<
        property_map_type
      , subgraph_type
      , graph_type
      , subgraph_prop0_type
      , graph_prop0_type
    >
> embedding (
    property_map_type & property_map
  , subgraph_type const & subgraph
  , graph_type const & graph
  , subgraph_prop0_type const & subgraph_prop0
  , graph_prop0_type const & graph_prop0
) {
  return embedding(
      property_map
    , subgraph
    , graph
    , subgraph_prop0
    , graph_prop0
    , boost::no_property()
    , boost::no_property()
  );
}
template<
    class property_map_type 
  , class subgraph_type
  , class graph_type 
> std::pair<
    embedding_iterator<
        property_map_type
      , subgraph_type
      , graph_type
    >
  , embedding_iterator<
        property_map_type
      , subgraph_type
      , graph_type
    >
> embedding (
    property_map_type & property_map
  , subgraph_type const & subgraph
  , graph_type const & graph
) {
  return embedding(
      property_map
    , subgraph
    , graph
    , boost::no_property()
    , boost::no_property()
  );
}

#endif // EMBEDDING_HPP
