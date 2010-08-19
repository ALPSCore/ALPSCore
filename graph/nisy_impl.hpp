#ifndef NISY_IMPL_HPP
#define NISY_IMPL_HPP

#include "util.hpp"

namespace detail {

  template <
      class graph_type 
    , class vertex_color_type
    , class edge_color_type
  > class nisy_base : public graph_type 
  {
    protected:
      enum { CREATED, CLEARED, CACHING };
      typedef typename boost::graph_traits<graph_type>::vertex_descriptor vertex_descriptor_type;
      typedef typename boost::graph_traits<graph_type>::edge_descriptor edge_descriptor_type;
      typedef typename boost::graph_traits<graph_type>::adjacency_iterator adjacency_iterator_type;
      typedef typename boost::graph_traits<graph_type>::vertex_iterator vertex_iterator_type;
      typedef typename boost::graph_traits<graph_type>::edge_iterator edge_iterator_type;
      typedef typename boost::graph_traits<graph_type>::out_edge_iterator out_edge_iterator_type;
      typedef std::map<vertex_descriptor_type, std::size_t> mapping_type;
    public:
      typedef typename std::list<vertex_descriptor_type> cell_type;
      typedef typename std::list<cell_type> partition_type;
      typedef typename partition_type::iterator partition_iterator_type;
      typedef typename canonical_label_type<vertex_color_type, edge_color_type>::type canonical_label_type;
      typedef canonical_ordering_iterator<partition_type> canonical_ordering_iterator;
      nisy_base(
          graph_type const & graph
      )
        : graph_type(graph)
        , state_(CREATED)
      {}
      virtual ~nisy_base() {}
      inline void invalidate() const {
        state_ = CLEARED;
      }
      inline std::pair<canonical_ordering_iterator, canonical_ordering_iterator> get_canonical_ordering() const {
        if (state_ != CACHING)
          build_cache();
        return std::make_pair(
            canonical_ordering_iterator(canonical_partition_.begin())
          , canonical_ordering_iterator(canonical_partition_.end())
        );
      }
      inline canonical_label_type const & get_canonical_label() const {
        if (state_ != CACHING)
          build_cache();
        return canonical_label_;
      }
      inline partition_type const & get_orbit_partition() const { 
        if (state_ != CACHING)
          build_cache();
        return orbit_;
      }
    protected:
      virtual void build_cache() const = 0;
      mutable partition_type initial_partition_;
      mutable std::size_t state_;
      mutable partition_type canonical_partition_;
      mutable canonical_label_type canonical_label_;
      mutable partition_type orbit_;
      mutable mapping_type orbit_mapping_;
  };
  template<class nisy_type> class serach_tree_node;
  template <
      class graph_type 
    , class vertex_color_type
    , class edge_color_type
    , class vertex_color_property_map_type
    , class edge_color_property_map_type
  > class nisy_derived : public nisy_base<graph_type, vertex_color_type, edge_color_type> 
  {
    private:
      typedef nisy_base<graph_type, vertex_color_type, edge_color_type> base;
      friend class serach_tree_node<nisy_derived<
          graph_type
        , vertex_color_type
        , edge_color_type
        , vertex_color_property_map_type
        , edge_color_property_map_type
      > >;
    public:
      nisy_derived(
          no_coloring_tag
        , graph_type const & graph
      )
        : base(graph) 
      {}
      nisy_derived(
          vertex_coloring_tag
        , graph_type const & graph
        , vertex_color_property_map_type const vertex_property
      )
        : base(graph) 
        , vertex_properties_(vertex_property)
      {
        BOOST_STATIC_ASSERT((boost::is_same<
            vertex_color_type
          , typename boost::property_traits<vertex_color_property_map_type>::value_type
        >::value)); 
      }
      nisy_derived(
          edge_coloring_tag
        , graph_type const & graph
        , edge_color_property_map_type const edge_property
      )
        : base(graph) 
        , edge_properties_(edge_property)
      {
        BOOST_STATIC_ASSERT((boost::is_same<
            vertex_color_type
          , typename boost::property_traits<vertex_color_property_map_type>::value_type
        >::value)); 
        BOOST_STATIC_ASSERT((boost::is_same<
            edge_color_type
          , typename boost::property_traits<edge_color_property_map_type>::value_type
        >::value)); 
      }
      nisy_derived(
          vertex_edge_coloring_tag
        , graph_type const & graph
        , vertex_color_property_map_type const vertex_property
        , edge_color_property_map_type const edge_property
      )
        : base(graph) 
        , vertex_properties_(vertex_property)
        , edge_properties_(edge_property)
      {
        BOOST_STATIC_ASSERT((boost::is_same<
            vertex_color_type
          , typename boost::property_traits<vertex_color_property_map_type>::value_type
        >::value)); 
        BOOST_STATIC_ASSERT((boost::is_same<
            edge_color_type
          , typename boost::property_traits<edge_color_property_map_type>::value_type
        >::value)); 
      }
    private:
      virtual void build_cache() const {
        base::state_ = base::CACHING;
        base::initial_partition_.clear();
        typename coloring_tag<
            boost::is_void<vertex_color_type>::value
          , boost::is_void<edge_color_type>::value
        >::type tag;
        build_inital_partition(tag);
        base::orbit_.clear();
        typename base::vertex_iterator_type oit, oend;
        for (boost::tie(oit, oend) = boost::vertices(*this); oit != oend; ++oit) 
          base::orbit_.push_back(typename base::cell_type(1, *oit));
        build_partition_mapping(base::orbit_, base::orbit_mapping_);
        base::canonical_partition_.clear();
        base::canonical_label_ = typename base::canonical_label_type();
        serach_tree_node<nisy_derived<
            graph_type 
          , vertex_color_type
          , edge_color_type
          , vertex_color_property_map_type
          , edge_color_property_map_type
        > > root(*this, base::initial_partition_, base::orbit_, base::orbit_mapping_);
        typename base::partition_type next(root.next_partition());
        typename base::partition_type first_partition(base::canonical_partition_ = next);
        typename base::canonical_label_type next_label, first_label;
        build_label(next_label, next, tag);
        base::canonical_label_ = first_label = next_label;
        while (root.has_next()) {
          next = root.next_partition();
          build_label(next_label, next, tag);
          if (base::canonical_label_ > next_label) {
            base::canonical_partition_ = next;
            base::canonical_label_ = next_label;
          } else if(first_label == next_label) {
            coarse_orbit(first_partition, next);
          } else if(base::canonical_label_ != first_label && base::canonical_label_ == next_label)
            coarse_orbit(base::canonical_partition_, next);
        }
      }
      bool is_le_edgelist(
            typename base::vertex_descriptor_type T1
          , typename base::vertex_descriptor_type T2
      ) {
        typename base::out_edge_iterator_type it1, it2, end1, end2;
        boost::tie(it1, end1) = boost::out_edges(T1, *this);
        boost::tie(it2, end2) = boost::out_edges(T1, *this);
        for (; it1 != end1 && it2 != end2; ++it1, ++it2)
          if (boost::get(*edge_properties_, *it2) < boost::get(*edge_properties_, *it1))
            return false;
        return it1 == end1;
      }
      void build_inital_partition(no_coloring_tag) const {
        typename base::vertex_iterator_type it, end;
        base::initial_partition_.push_back(typename base::cell_type());
        for (boost::tie(it, end) = boost::vertices(*this); it != end; ++it)
          base::initial_partition_.front().push_back(*it);
      }
      void build_inital_partition(vertex_coloring_tag) const {
        typename base::vertex_iterator_type it, end;
        typename base::partition_iterator_type pit;
        for (boost::tie(it, end) = boost::vertices(*this); it != end; ++it) {
          for (pit = base::initial_partition_.begin(); pit != base::initial_partition_.end(); ++pit)
            if (boost::get(vertex_properties_, *it) <= boost::get(vertex_properties_, pit->front())) {
              if (boost::get(vertex_properties_, *it) == boost::get(vertex_properties_, pit->front()))
                pit->push_back(*it);
              else
                base::initial_partition_.insert(pit, typename base::cell_type(1, *it));
              break;  
            }
          if (pit == base::initial_partition_.end())
            base::initial_partition_.push_back(typename base::cell_type(1, *it));
        }
      }
      void build_inital_partition(edge_coloring_tag) const {
        typename base::vertex_iterator_type it, end;
        typename base::partition_iterator_type pit;
        for (boost::tie(it, end) = boost::vertices(*this); it != end; ++it) {
          for (pit = base::initial_partition_.begin(); pit != base::initial_partition_.end(); ++pit)
            if (is_le_edgelist(*it, pit->front())) {
              if (is_le_edgelist(pit->front(), *it))
                pit->push_back(*it);
              else
                base::initial_partition_.insert(it, cell_type(1, *it));
              break;  
            }
          if (pit == base::initial_partition_.end())
            base::initial_partition_.push_back(cell_type(1, *pit));
        }
      }
      void build_inital_partition(vertex_edge_coloring_tag) const {
        typename base::vertex_iterator_type it, end;
        typename base::partition_iterator_type pit;
        for (boost::tie(it, end) = boost::vertices(*this); it != end; ++it) {
          for (pit = base::initial_partition_.begin(); pit != base::initial_partition_.end(); ++pit)
            if (
              boost::get(vertex_properties_, *it) <= boost::get(vertex_properties_, pit->front())
              && is_le_edgelist(*it, pit->front())
            ) {
              if (
                boost::get(vertex_properties_, *it) == boost::get(vertex_properties_, pit->front())
                && is_le_edgelist(pit->front(), *it)
              )
                pit->push_back(*it);
              else
                base::initial_partition_.insert(it, cell_type(1, *it));
              break;  
            }
          if (pit == base::initial_partition_.end())
            base::initial_partition_.push_back(cell_type(1, *pit));
        }
      }    
      void build_label_nauty(
          boost::dynamic_bitset<> & matrix
        , typename base::partition_type const & partition
        , typename base::mapping_type & mapping
      ) const {
        matrix.resize(boost::num_vertices(*this) * boost::num_vertices(*this));
        std::size_t row_size = boost::num_vertices(*this);
        for (typename base::mapping_type::iterator it1 = mapping.begin(); it1 != mapping.end(); ++it1) {
          std::pair<typename base::adjacency_iterator_type, typename base::adjacency_iterator_type> adjacent_vertices = boost::adjacent_vertices(it1->first, *this);
          for (typename base::adjacency_iterator_type & it2 = adjacent_vertices.first; it2 != adjacent_vertices.second; ++it2)
            matrix[it1->second * row_size + mapping[*it2]] = true;
        }
      }
      template<class vector_type> void build_label_vertices(
          boost::dynamic_bitset<> & matrix
        , vector_type & vector
        , typename base::partition_type const & partition
        , typename base::mapping_type const & mapping
      ) const {
        typename base::vertex_iterator_type vit, vend;
        vector.clear(); 
        for (boost::tie(vit, vend) = boost::vertices(*this); vit != vend; ++vit)
          if (std::find(vector.begin(), vector.end(), boost::get(vertex_properties_, *vit)) == vector.end())
            vector.push_back(boost::get(vertex_properties_, *vit));
        std::sort(vector.begin(), vector.end());
        matrix.resize(boost::num_vertices(*this) * vector.size());
        for (boost::tie(vit, vend) = boost::vertices(*this); vit != vend; ++vit)
          matrix[
            mapping.find(*vit)->second * vector.size() + (std::find(vector.begin(), vector.end(), boost::get(vertex_properties_, *vit)) - vector.begin())
          ] = true;
      }
      template<class vector_type> void build_label_edges(
          boost::dynamic_bitset<> & matrix
        , vector_type & vector
        , typename base::partition_type const & partition
        , typename base::mapping_type const & mapping
      ) const {
        typename base::edge_iterator_type eit, eend;
        vector.clear(); 
        for (boost::tie(eit, eend) = boost::edges(*this); eit != eend; ++eit)
          if (std::find(
            vector.begin(), vector.end(), boost::get(base::edge_properties_, *eit)
          ) == vector.end())
            vector.push_back(boost::get(base::edge_properties_, *eit));
        std::sort(vector.begin(), vector.end());
        matrix.resize(boost::num_edges(*this) * vector.size());
        for (boost::tie(eit, eend) = boost::edges(*this); eit != eend; ++eit)
          matrix[
            *mapping.find(*eit)->second * vector.size() + (std::find(vector.begin(), vector.end(), boost::get(base::edge_properties_, *eit)) - vector.begin())
          ] = true;    
      }
      void build_label(
          typename base::canonical_label_type & label
        , typename base::partition_type const & partition
        , no_coloring_tag
      ) const {
        typename base::mapping_type mapping;
        build_partition_mapping(partition, mapping);
        build_label_nauty(label, partition, mapping);
      }
      void build_label(
          typename base::canonical_label_type & label
        , typename base::partition_type const & partition
        , vertex_coloring_tag
      ) const {
        typename base::mapping_type mapping;
        build_partition_mapping(partition, mapping);
        build_label_nauty(boost::get<0>(label), partition, mapping);
        build_label_vertices(boost::get<1>(label), boost::get<2>(label), partition, mapping);
      }
      void build_label(
          typename base::canonical_label_type & label
        , typename base::partition_type const & partition
        , edge_coloring_tag
      ) const {
        typename base::mapping_type mapping;
        build_partition_mapping(partition, mapping);
        build_label_nauty(boost::get<0>(label), partition, mapping);
        build_label_edges(boost::get<1>(label), boost::get<2>(label), partition, mapping);
      }
      void build_label(
          typename base::canonical_label_type & label
        , typename base::partition_type const & partition
        , vertex_edge_coloring_tag
      ) const {
        typename base::mapping_type mapping;
        build_partition_mapping(partition, mapping);
        build_label_nauty(boost::get<0>(label), partition, mapping);
        build_label_vertices(boost::get<1>(label), boost::get<2>(label), partition, mapping);
        build_label_edges(boost::get<3>(label), boost::get<4>(label), partition, mapping);
      }
      inline void build_partition_mapping(
          typename base::partition_type const & partition
        , typename base::mapping_type & mapping
      ) const {
        mapping.clear();
        typename base::partition_type::size_type index = 0;
        for (typename base::partition_type::const_iterator it1 = partition.begin(); it1 != partition.end(); ++it1, ++index)
          for (typename base::cell_type::const_iterator it2 = it1->begin(); it2 != it1->end(); ++it2)
            mapping[*it2] = index;
      }
      void coarse_orbit(
          typename base::partition_type const & P1
        , typename base::partition_type const & P2
      ) const {
        if (P1 != P2) {
          typename base::partition_type::size_type index = 0;
          typename base::partition_type::iterator it3;
          for (typename base::partition_type::const_iterator it1 = P1.begin(), it2 = P2.begin(); it1 != P1.end(); ++it1, ++it2)
            if (*it1 != *it2 && base::orbit_mapping_[it1->front()] != base::orbit_mapping_[it2->front()]) {
              typename base::cell_type * mergabel = NULL;
              for (it3 = base::orbit_.begin(), index = 0; it3 != base::orbit_.end(); ++it3, ++index)
                if (index == std::min(base::orbit_mapping_[it1->front()], base::orbit_mapping_[it2->front()]))
                  mergabel = &(*it3);
                else if (index == std::max(base::orbit_mapping_[it1->front()], base::orbit_mapping_[it2->front()])) {
                  it3->merge(*mergabel);
                  base::orbit_.remove(typename base::cell_type());
                  build_partition_mapping(base::orbit_, base::orbit_mapping_);
                  break;
                }
            }
        }
      }
      vertex_color_property_map_type vertex_properties_;
      edge_color_property_map_type edge_properties_;
  };
  template<class nisy_type> class serach_tree_node {
    private:
      enum { INITIALIZED, OPEN, CLOSED };
      typedef typename nisy_type::adjacency_iterator_type adjacency_iterator_type;
      typedef typename nisy_type::partition_type partition_type;
      typedef typename nisy_type::cell_type cell_type;
      typedef typename nisy_type::mapping_type mapping_type;
      typedef std::list<std::pair<typename partition_type::size_type[2], typename cell_type::value_type> > children_type;
    public:
      serach_tree_node(
        nisy_type const & G
        , partition_type const & P
        , partition_type & orbit
        , mapping_type & orbit_mapping
      )
        : state_(INITIALIZED)
        , graph_(G)
        , partition_(P)
        , orbit_(orbit)
        , orbit_mapping_(orbit_mapping)
        , active_(NULL) 
      {
        equitable_partition();
      }
      serach_tree_node(
        nisy_type const & G
        , partition_type const & P
        , typename cell_type::value_type v
        , partition_type & orbit
        , mapping_type & orbit_mapping
      )
        : state_(INITIALIZED)
        , graph_(G)
        , partition_(P)
        , orbit_(orbit)
        , orbit_mapping_(orbit_mapping)
        , active_(NULL) 
      {
        for (typename partition_type::iterator it = partition_.begin(); it != partition_.end(); it++)
          if (find(it->begin(), it->end(), v) != it->end()) {
            partition_.insert(it, cell_type(1, v));
            it->remove(v);
            break;
          }
        equitable_partition();
      }
      virtual ~serach_tree_node() {
        if (active_ != NULL)
          delete active_;
      }
      inline bool has_next() {
        return state_ == INITIALIZED 
          || (active_ != NULL && active_->has_next()) 
          || children_.size() > 1;
      }
      partition_type & next_partition() {
        if (state_ == INITIALIZED) {
          for (typename partition_type::iterator it1 = partition_.begin(); it1 != partition_.end(); it1++) {
            std::map<typename partition_type::size_type, typename cell_type::value_type> reduced_cell;
            for (typename cell_type::iterator it2 = it1->begin(); it1->size() > 1 && it2 != it1->end(); it2++)
              if (reduced_cell.find(orbit_mapping_[*it2]) == reduced_cell.end())
                reduced_cell.insert(std::make_pair(orbit_mapping_[*it2], *it2));
              else if (reduced_cell[orbit_mapping_[*it2]] > *it2)
                reduced_cell[orbit_mapping_[*it2]] = *it2;
            for (typename std::map<typename partition_type::size_type, typename cell_type::value_type>::iterator 
              it3 = reduced_cell.begin(); it3 != reduced_cell.end(); it3++
            ) {
              children_.push_back(std::pair<typename partition_type::size_type[2], typename cell_type::value_type>());
              children_.back().first[0] = it3->first;
              children_.back().first[1] = 0;
              children_.back().second = it3->second;
            }
          }
          state_ = children_.size() > 0 ? OPEN : CLOSED;
          if (children_.size() == 0)
            return partition_;
          else
            active_ = new serach_tree_node<nisy_type>(graph_, partition_, children_.front().second, orbit_, orbit_mapping_);
        } else if (!active_->has_next() && children_.size() > 1) {
          delete active_;
          children_.pop_front();
          typename children_type::iterator it1, it2;
          for (it1 = children_.begin(); it1 != children_.end(); it1++)
            it1->first[1] = orbit_mapping_[it1->second];
          children_.sort();
          for (it1 = ++children_.begin(), it2 = children_.begin(); it1 != children_.end(); it1++, it2++)
            if (it1->first[0] == it2->first[0] && it1->first[1] == it2->first[1])
              children_.erase(it1++);
          active_ = new serach_tree_node<nisy_type>(graph_, partition_, children_.front().second, orbit_, orbit_mapping_);
        }
        return active_->next_partition();
      }
    private:
      void equitable_partition() {
        if (partition_.size() < num_vertices(graph_)) {
          mapping_type mapping;
          typename partition_type::size_type partition_size, adjacents_size, index, it4;
          std::pair<adjacency_iterator_type, adjacency_iterator_type> adjacent_vertices;
          typename std::deque<cell_type> refined_cells;
          std::vector<typename partition_type::size_type> current;
          std::pair<std::vector<typename partition_type::size_type>, typename cell_type::value_type> adjacents[num_vertices(graph_)];
          do {
            index = 0;
            mapping.clear();
            for (typename partition_type::iterator it1 = partition_.begin(); it1 != partition_.end(); ++it1, ++index)
              for (typename cell_type::iterator it2 = it1->begin(); it2 != it1->end(); ++it2)
                mapping[*it2] = index;
            partition_size = partition_.size();
            for (typename partition_type::iterator it1 = partition_.begin(); it1 != partition_.end(); ) {
              adjacents_size = 0;
              for (typename cell_type::iterator it2 = it1->begin(); it2 != it1->end(); it2++, adjacents_size++) {
                adjacents[adjacents_size] = std::make_pair(std::vector<typename partition_type::size_type>(partition_.size(), 0), *it2);
                adjacent_vertices = boost::adjacent_vertices(*it2, graph_);
                for (adjacency_iterator_type &it3 = adjacent_vertices.first; it3 != adjacent_vertices.second; ++it3)
                  ++adjacents[adjacents_size].first[mapping[*it3]];
              }
              std::sort(adjacents, adjacents + adjacents_size);
              refined_cells.clear();
              refined_cells.push_back(cell_type(1, adjacents[0].second));
              current = adjacents[0].first;
              for (it4 = 1; it4 < adjacents_size; ++it4)
                if (current == adjacents[it4].first)
                  refined_cells.begin()->push_back(adjacents[it4].second);
                else {
                  current = adjacents[it4].first;
                  refined_cells.push_front(cell_type(1, adjacents[it4].second));
                }
              partition_.insert(it1, refined_cells.rbegin(), refined_cells.rend());
              partition_.erase(it1++);
            }
          } while (partition_size != partition_.size());
        }
      }
      unsigned int state_;
      const nisy_type &graph_;
      partition_type partition_;
      partition_type & orbit_;
      mapping_type & orbit_mapping_;
      children_type children_;
      serach_tree_node<nisy_type> *active_;
  };
} // detail

#endif // NISY_IMPL_HPP
