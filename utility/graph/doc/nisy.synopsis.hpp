//[ nisy_synopsis

template <
    class graph_type 
  , class vertex_color_type = void
  , class edge_color_type = void
> class nisy {
  public:
    typedef typename std::list<std::list<vertex_descriptor_type> > partition_type;
    typedef typename partition_type::iterator partition_iterator_type;
    typedef typename /*implementation-defined*/ canonical_label_type; // LessThanComparable
    typedef typename /*implementation-defined*/ canonical_ordering_iterator; // ForwardIterator
    nisy(
        graph_type const & graph
    ); // only use iif(vertex_color_type == void and edge_color_type == void)
    template <
        class color_property_map_type
    > nisy(
        graph_type const & graph
      , vertex_color_property_map_type vertex_property
    ); // only use iif(vertex_color_type != void and edge_color_type == void)
    template <
        class color_property_map_type
    > nisy(
        graph_type const & graph
      , edge_color_property_map_type edge_property
    ); // only use iif(vertex_color_type == void and edge_color_type != void)
    template <
        class vertex_color_property_map_type
      , class edge_color_property_map_type
    > nisy(
        graph_type const & graph
      , vertex_color_property_map_type vertex_property
      , edge_color_property_map_type edge_property
    ); // only use iif(vertex_color_type != void and edge_color_type != void)
    virtual ~nisy();
    inline void invalidate();
    inline std::pair<canonical_ordering_iterator, canonical_ordering_iterator> get_canonical_ordering() const;
    inline canonical_label_type const & get_canonical_label() const;
    inline partition_type const & get_orbit_partition() const;
    template<class graph_type1> inline bool operator==(nisy<graph_type1, vertex_color_type, edge_color_type> const & T) const;
    template<class graph_type1> inline bool operator!=(nisy<graph_type1, vertex_color_type, edge_color_type> const & T) const;
};

template<
    class graph_type1
  , class graph_type2
  , class vertex_coloring_type1
  , class vertex_coloring_type2
  , class edge_coloring_type1
  , class edge_coloring_type2
> inline std::map<
      typename boost::graph_traits<graph_type1>::vertex_descriptor
    , typename boost::graph_traits<graph_type2>::vertex_descriptor
  > isomorphism(
      nisy<graph_type1, vertex_coloring_type1, edge_coloring_type1> const & T1
    , nisy<graph_type2, vertex_coloring_type2, edge_coloring_type2> const & T2
  );  
//]