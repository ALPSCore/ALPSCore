//[ embedding_synopsis
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
> class embedding_iterator; // ForwardIterator with property_map_type const & operator*() const;

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
    embedding_iterator</*...*/>
  , embedding_iterator</*...*/>
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
);

template<
    class property_map_type 
  , class subgraph_type
  , class graph_type 
  , class subgraph_prop0_type
  , class graph_prop0_type
  , class subgraph_prop1_type
  , class graph_prop1_type
> std::pair<
    embedding_iterator</*...*/>
  , embedding_iterator</*...*/>
> embedding (
    property_map_type & property_map
  , subgraph_type const & subgraph
  , graph_type const & graph
  , subgraph_prop0_type const & subgraph_prop0
  , graph_prop0_type const & graph_prop0
  , subgraph_prop1_type const & subgraph_prop1
  , graph_prop1_type const & graph_prop1
);

template<
    class property_map_type 
  , class subgraph_type
  , class graph_type 
  , class subgraph_prop0_type
  , class graph_prop0_type
> std::pair<
    embedding_iterator</*...*/>
  , embedding_iterator</*...*/>
> embedding (
    property_map_type & property_map
  , subgraph_type const & subgraph
  , graph_type const & graph
  , subgraph_prop0_type const & subgraph_prop0
  , graph_prop0_type const & graph_prop0
);

template<
    class property_map_type 
  , class subgraph_type
  , class graph_type 
> std::pair<
    embedding_iterator</*...*/>
  , embedding_iterator</*...*/>
> embedding (
    property_map_type & property_map
  , subgraph_type const & subgraph
  , graph_type const & graph
);
//]