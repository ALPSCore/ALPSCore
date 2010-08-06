//[ embedding_counter_hpp
// file: embedding.counter.hpp
#ifndef EMBEDDING_COUNTER_HPP
#define EMBEDDING_COUNTER_HPP

#include "../src/embedding.hpp"

#include <utility>
#include <vector>
#include <map>

#include <boost/property_map.hpp>

template <class subgraph_type, class graph_type> class embedding_counter
{
  private:
    typedef std::map<
        typename boost::graph_traits<subgraph_type>::vertex_descriptor
      , typename boost::graph_traits<graph_type>::vertex_descriptor
    > map_type;
    typedef boost::associative_property_map<map_type> property_map_type;
  public:
    embedding_counter(
        subgraph_type const & subgraph
      , graph_type const & graph
    )
      : subgraph_(subgraph) 
      , graph_(graph) 
    {}
    inline std::size_t count() {
      embedding_iterator<
          property_map_type
        , subgraph_type
        , graph_type
      > emb_it, emb_end; 
      map_type map_store;
      property_map_type mapping(map_store);
      std::size_t count_emb = 0;
      boost::tie(emb_it, emb_end) = embedding(
          mapping
        , subgraph_
        , graph_
      );
      for (; emb_it != emb_end; ++emb_it)
        ++count_emb;
      return count_emb;
    }
    inline std::size_t count(
        typename boost::graph_traits<graph_type>::vertex_descriptor graph_vertex
      , typename boost::graph_traits<subgraph_type>::vertex_descriptor subgraph_vertex
    ) {
      embedding_iterator<
          property_map_type
        , subgraph_type
        , graph_type
        , typename boost::graph_traits<subgraph_type>::vertex_descriptor
        , typename boost::graph_traits<graph_type>::vertex_descriptor
      > emb_it, emb_end; 
      map_type map_store;
      property_map_type mapping(map_store);
      std::size_t count_emb = 0;
      boost::tie(emb_it, emb_end) = embedding(
          mapping
        , subgraph_
        , graph_
        , subgraph_vertex
        , graph_vertex
      );
      for (; emb_it != emb_end; ++emb_it)
        ++count_emb;
      return count_emb;      
    }
  private:
    template<typename it_type> inline std::size_t count_impl(
        std::pair<it_type, it_type> it_pair
      , std::size_t count_emb = 0
    ) {
      for (; it_pair.first != it_pair.second; ++it_pair.first)
        ++count_emb;    	
      return count_emb;
    }

    subgraph_type const & subgraph_;
    graph_type const & graph_;
};

#endif // embedding_COUNTER_HPP
//]
