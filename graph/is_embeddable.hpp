#ifndef IS_EMBEDDABLE_HPP
#define IS_EMBEDDABLE_HPP
#include <alps/graph/deprecated/embedding.hpp>

namespace alps
{
namespace graph
{

    
template <typename SubGraph, typename SuperGraph>
struct no_pin
{
};

template <typename SubGraph, typename SuperGraph>
struct vertex_pin : public std::pair<
                      typename boost::graph_traits<SubGraph>::vertex_descriptor
                    , typename boost::graph_traits<SuperGraph>::vertex_descriptor
                    >
{
    vertex_pin(
              typename boost::graph_traits<SubGraph>::vertex_descriptor sub_v
            , typename boost::graph_traits<SuperGraph>::vertex_descriptor super_v
            )
    : std::pair<
              typename boost::graph_traits<SubGraph>::vertex_descriptor
            , typename boost::graph_traits<SuperGraph>::vertex_descriptor
            >(sub_v, super_v)
    {
    }
};


namespace detail{
// TODO: this code should be removed/rewritten
template <class SubGraph, class SuperGraph>
class embedding_counter
{
  public:
    typedef SubGraph subgraph_type;
    typedef SuperGraph supergraph_type;
  private:
    typedef std::map<
        typename boost::graph_traits<subgraph_type>::vertex_descriptor,
        typename boost::graph_traits<supergraph_type>::vertex_descriptor
        > map_type;
    typedef boost::associative_property_map<map_type> property_map_type;
  
  public:
    embedding_counter(subgraph_type const& subgraph, supergraph_type const& supergraph)
      : subgraph_(subgraph), supergraph_(supergraph)
    {}

    bool is_embeddable()
    {
        alps::graph::embedding_iterator<property_map_type, subgraph_type, supergraph_type> emb_it, emb_end;
        map_type map_store;
        property_map_type mapping(map_store);
        boost::tie(emb_it, emb_end) = embedding(mapping, subgraph_, supergraph_);
        return (emb_it != emb_end);
    }

    bool is_embeddable(vertex_pin<subgraph_type, supergraph_type> const& pin)
    {
        alps::graph::embedding_iterator<
          property_map_type,
          subgraph_type,
          supergraph_type,
          typename boost::graph_traits<subgraph_type>::vertex_descriptor,
          typename boost::graph_traits<supergraph_type>::vertex_descriptor
        > emb_it, emb_end;
        map_type map_store;
        property_map_type mapping(map_store);
        boost::tie(emb_it, emb_end) = embedding(mapping, subgraph_, supergraph_, pin.first, pin.second);
        return (emb_it != emb_end);
    }
    
//    inline bool is_embeddable(no_pin<subgraph_type, supergraph_type> const& pin)
//    {
//        return is_embeddable();
//    }

    std::size_t count()
    {
        alps::graph::embedding_iterator<property_map_type, subgraph_type, supergraph_type> emb_it, emb_end;
        map_type map_store;
        property_map_type mapping(map_store);
        std::size_t count_emb = 0;
        boost::tie(emb_it, emb_end) = embedding(mapping, subgraph_, supergraph_);
        for (; emb_it != emb_end; ++emb_it)
            ++count_emb;
        return count_emb;
    }

    std::size_t count(
          typename boost::graph_traits<subgraph_type>::vertex_descriptor subgraph_vertex
        , typename boost::graph_traits<supergraph_type>::vertex_descriptor supergraph_vertex
    )
    {
        // TODO get types straight in assert statements.
        assert( subgraph_vertex < num_vertices(subgraph_) );
        assert( supergraph_vertex < num_vertices(supergraph_) );
        alps::graph::embedding_iterator<
            property_map_type,
            subgraph_type,
            supergraph_type,
            typename boost::graph_traits<subgraph_type>::vertex_descriptor,
            typename boost::graph_traits<supergraph_type>::vertex_descriptor
        > emb_it, emb_end;
        map_type map_store;
        property_map_type mapping(map_store);
        std::size_t count_emb = 0;
        boost::tie(emb_it, emb_end) = embedding(mapping, subgraph_, supergraph_, subgraph_vertex, supergraph_vertex);
        for (; emb_it != emb_end; ++emb_it)
            ++count_emb;
        return count_emb;
    }

  private:
    
    subgraph_type const& subgraph_;
    supergraph_type const& supergraph_;
};
} // end namespace detail

template <class SubGraph, class SuperGraph>
bool is_embeddable(SubGraph const& s, SuperGraph const& g,vertex_pin<SubGraph,SuperGraph> const& pin)
{
    return detail::embedding_counter<SubGraph,SuperGraph>(s,g).is_embeddable(pin);
}

template <class SubGraph, class SuperGraph>
bool is_embeddable(SubGraph const& s, SuperGraph const& g, no_pin<SubGraph,SuperGraph> const& pin = no_pin<SubGraph,SuperGraph>())
{
    return detail::embedding_counter<SubGraph,SuperGraph>(s,g).is_embeddable();
}

} // end namespace graph
} // end namespace alps

#endif //IS_EMBEDDABLE_HPP
