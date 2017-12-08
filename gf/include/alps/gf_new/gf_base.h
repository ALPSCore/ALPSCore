//
// Created by iskakoff on 08/10/17.
//

#ifndef GREENSFUNCTIONS_GF_H
#define GREENSFUNCTIONS_GF_H


#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include <alps/hdf5/archive.hpp>
#ifdef ALPS_HAVE_MPI
#include <alps/utilities/mpi.hpp>
#endif

#include "tensors/TensorBase.h"
#include "type_traits.h"


namespace alps {
  namespace gf {
    namespace detail {
      /**
       * Definition of base GF
       */
      template<class VTYPE, class Storage, class ...MESHES>
      class gf_base;
    }

    /**
     * Definition of regular GF with dedicated storage
     */
    template<class VTYPE, class ...MESHES>
    using greenf = detail::gf_base<VTYPE, detail::Tensor<VTYPE, sizeof...(MESHES)>, MESHES...>;
    /**
     * Definition of GF as view of existent data array
     */
    template<class VTYPE, class ...MESHES>
    using greenf_view = detail::gf_base<VTYPE, detail::TensorView<VTYPE, sizeof...(MESHES)>, MESHES...>;

    namespace detail {
      /**
       * @brief gf class
       *
       * @author iskakoff
       */
      template<class VTYPE, class Storage, class ...MESHES>
      class gf_base {
      private:
        /// GF version
        static const int minor_version = 1;
        static const int major_version = 0;
        /// Dimension of grid
        static constexpr int _N = sizeof...(MESHES);
        /// data_storage
        Storage data_;
        /// stored meshes
        std::tuple < MESHES... > meshes_;
        /// unitialized state flag
        bool empty_;
        /// mesh types tuple
        using _mesh_types = std::tuple < MESHES... >;
        /// current GF type
        using gf_type   = gf_base < VTYPE, Storage, MESHES... >;
        /// storage types
        using data_storage = Tensor < VTYPE, sizeof...(MESHES) >;
        using data_view    = TensorView < VTYPE, sizeof...(MESHES) >;
        /// Generic GF type
        template<typename St>
        using generic_gf   = gf_base < VTYPE, St, MESHES... >;

        // this is a helper function that should never be called. It used to get compile time type declaration.
        // using the index_sequence we create the proper set of meshes by unrolling tuple into parameter pack with
        // the following template code: "typename std::tuple_element<sizeof...(Ti) - Trim + I, _mesh_types >::type..."
        template <typename S, int Trim, typename... Ti, std::size_t... I>
        gf_base<S, TensorView < S, Trim>,
          typename std::tuple_element<sizeof...(Ti) - Trim + I, _mesh_types >::type...>
        subpack_(S x, const std::tuple<Ti...>& t, index_sequence<I...>) {
          throw std::runtime_error("This function is not intended to be called. The only purpose of this function is to get type declaration.");
        }

        // this is a helper function that should never be called. It used to get compile time type declaration
        // Using the size of current GF mesh and number of indices in new GF we create an index sequence and
        // evaluate the return type of %subpack_% method.
        template <typename S, int Trim, typename... T>
        auto subpack(S x, const std::tuple<T...>& t) -> decltype(subpack_<S, Trim, T...>(x, t, make_index_sequence<Trim>())) {
          throw std::runtime_error("This function is not intended to be called. The only purpose of this function is to get type declaration.");
        }

      public:

        /**
         * Copy-constructor
         * @param g - GF to copy from
         */
        gf_base(const gf_type &g) : data_(g.data_), meshes_(g.meshes_), empty_(g.empty_) {}

        /**
         * Move-constructor
         */
        gf_base(gf_type &&g) : data_(g.data_), meshes_(g.meshes_), empty_(g.empty_) {}

        /**
         * Default constructor. Create uninitilized GF
         */
        gf_base() : data_(std::array < size_t, _N >{{0}}), empty_(true) {}

        /**
         * Create Green's function object with given meshes
         * @param meshes - list of meshes
         */
        gf_base(MESHES...meshes) : gf_base(std::forward_as_tuple(meshes...)) {}
        /// tuple version of the previous function
        gf_base(const _mesh_types &meshes) : data_(get_sizes(meshes)), meshes_(meshes), empty_(false) {}

        /// construct new GF object by copy data from another GF object defined with different storage type
        template<typename St, typename = std::enable_if<!std::is_same<St, Storage>::value && std::is_same<St, data_view>::value > >
        gf_base(const gf_base<VTYPE, data_view, MESHES...> &g) : data_(g.data()), meshes_(g.meshes()), empty_(g.is_empty()) {}
        template<typename St, typename = std::enable_if<!std::is_same<St, Storage>::value && std::is_same<St, data_storage>::value > >
        gf_base(gf_base<VTYPE, St, MESHES...> &g) : data_(g.data()), meshes_(g.meshes()), empty_(g.is_empty()) {}

        /// construct new green's function from index slice of GF with higher dimension
        template<typename St, typename...OLDMESHES, typename ...Indices>
        gf_base(gf_base<VTYPE, TensorBase < VTYPE, sizeof...(OLDMESHES), St >, OLDMESHES...> & g, std::tuple<OLDMESHES...>& oldmesh, const _mesh_types &meshes, const Indices... idx) :
          data_(g.data()(idx()...)), meshes_(meshes), empty_(false) {}

        /// copy assignment
        gf_type& operator=(const gf_type & rhs) {
          data_   = rhs.data_;
          meshes_ = rhs.meshes_;
          empty_  = rhs.empty_;
          return *this;
        }

        /// initialize with zeros
        void initialize() {
          data_ *= VTYPE(0);
        }

        /*
         * Data access operations
         */
        /**
         * @tparam Indices - types of indices
         * @param inds     - indices
         * @return value at the specific position
         */
        template<class...Indices>
        typename std::enable_if < (sizeof...(Indices) == sizeof...(MESHES)), VTYPE >::type
        operator()(Indices...inds) const {
          // check that index types are the same as mesh indices
          check(std::integral_constant < int, 0 >(), std::forward < Indices >(inds)...);
          return value(std::forward < Indices >(inds)...);
        }

        /// @return reference to the specific value in the GF object
        template<class...Indices>
        typename std::enable_if < (sizeof...(Indices) == sizeof...(MESHES)), VTYPE & >::type
        operator()(Indices...inds) {
          // check that index types are the same as mesh indices
          check(std::integral_constant < int, 0 >(), std::forward < Indices >(inds)...);
          return value(std::forward < Indices >(inds)...);
        }

        /**
         * Create GF view for the first indices (ind, inds...), e.g., for the fixed indices (x,y)
         * we get the view object G(:) = G(x, y, :)
         *
         * @param ind  - first index
         * @param inds - other indices
         * @return GF view object
         */
        template<class...Indices>
        auto operator()(typename std::enable_if<(sizeof...(Indices)+1 < _N), typename std::tuple_element<0,_mesh_types>::type::index_type >::type ind,
                    Indices...inds) -> decltype(subpack<VTYPE, _N - sizeof...(Indices) - 1>(VTYPE(0), meshes_)) {
          // get new mesh tuple
          auto t = subtuple<sizeof...(Indices) + 1>(meshes_);
          return decltype(subpack<VTYPE, _N - sizeof...(Indices) - 1>(VTYPE(0), meshes_))(*this, meshes_, t, ind, std::forward<Indices>(inds)...);
        }


        /*
         * Basic GF arithmetics
         */

        /**
         * Compute sum of current GF object and rhs
         *
         * @tparam RHS_GF - type of the right hand side GF
         * @param  rhs    - GF object to sum with the current GF
         * @return new GF object equals to sum of this and rhs
         */
        template<typename RHS_GF>
        typename std::enable_if < std::is_same < RHS_GF, generic_gf<data_storage> >::value || std::is_same < RHS_GF, generic_gf<data_view> >::value, gf_type >::type
        operator+(const RHS_GF &rhs) const {
          throw_if_empty();
          gf_type res(*this);
          return res += rhs;
        }

        /**
         * Update current GF with sum of rhs and current GF.
         *
         * @tparam RHS_GF - type of the right hand side GF
         * @param  rhs    - GF object to sum with the current GF
         * @return updated GF object
         */
        template<typename RHS_GF>
        typename std::enable_if < std::is_same < RHS_GF, generic_gf<data_storage> >::value || std::is_same < RHS_GF, generic_gf<data_view> >::value, gf_type & >::type
        operator+=(const RHS_GF &rhs) {
          throw_if_empty();
          data_ += rhs.data_;
          return *this;
        }

        /**
         * Compute difference of current GF object and rhs
         *
         * @tparam RHS_GF - type of the right hand side GF
         * @param  rhs    - GF object to sum with the current GF
         * @return new GF object equals to sum of this and rhs
         */
        template<typename RHS_GF>
        typename std::enable_if < std::is_same < RHS_GF, generic_gf<data_storage> >::value || std::is_same < RHS_GF, generic_gf<data_view> >::value, gf_type >::type
        operator-(const RHS_GF &rhs) const {
          throw_if_empty();
          gf_type res(*this);
          return res -= rhs;
        }

        /**
         * Inplace subtraction
         */
        template<typename RHS_GF>
        typename std::enable_if < std::is_same < RHS_GF, generic_gf<data_storage> >::value || std::is_same < RHS_GF, generic_gf<data_view> >::value, gf_type & >::type
        operator-=(const RHS_GF &rhs) {
          throw_if_empty();
          data_ -= rhs.data_;
          return *this;
        }

        /**
         * Scaling by scalar inplace
         *
         * @tparam RHS - type of the scalar
         * @param rhs - scaling factor
         * @return scaled Green's function
         */
        template<typename RHS>
        typename std::enable_if < std::is_scalar < RHS >::value || std::is_same < VTYPE, RHS >::value, gf_type & >::type operator*=(RHS rhs) {
          throw_if_empty();
          data_ *= VTYPE(rhs);
          return *this;
        }

        /**
         * Scaling by scalar
         *
         * @tparam RHS - type of the scalar
         * @param rhs - scaling factor
         * @return new scaled Green's function
         */
        template<typename RHS>
        typename std::enable_if < std::is_scalar < RHS >::value || std::is_same < VTYPE, RHS >::value, gf_type >::type operator*(RHS rhs) const {
          throw_if_empty();
          gf_type res(*this);
          return res *= rhs;
        }

        /**
         * Scaling real GF by complex scalar. This method will create new complex-valued GF object defined on the same mesh
         * and fill it with current GF scaled by complex scalar
         *
         * @tparam RHS - type of the scalar
         * @param rhs - scaling factor
         * @return scaled Green's function
         */
        template<typename RHS>
        typename std::enable_if < is_complex < RHS >::value && !std::is_same < VTYPE, RHS >::value, gf_base < RHS, Tensor<RHS, _N>, MESHES... > >::type operator*(RHS rhs) const {
          throw_if_empty();
          gf_base < RHS, Tensor<RHS, _N>, MESHES... > res(meshes_);
          res.data() += this->data();
          return res *= rhs;
        }

        /**
         * Inversed inplace scaling. If the Green's function is real, rhs should be real or integer scalar.
         * If GF is complex, rhs should be castable to complex.
         *
         * @tparam RHS  - type of the scalar
         * @param rhs   - scaling factor
         * @return scaled Green's function
         */
        template<typename RHS>
        typename std::enable_if < std::is_scalar < RHS >::value || std::is_same < VTYPE, RHS >::value, gf_type & >::type operator/=(RHS rhs) {
          throw_if_empty();
          data_ /= VTYPE(rhs);
          return *this;
        }

        /**
        * Inversed scaling
        *
        * @tparam RHS - type of the scalar
        * @param rhs - scaling factor
        * @return scaled Green's function
        */
        template<typename RHS>
        typename std::enable_if < std::is_scalar < RHS >::value || std::is_same < VTYPE, RHS >::value, gf_type >::type operator/(RHS rhs) const {
          throw_if_empty();
          gf_type res(meshes_);
          return res /= rhs;
        }

        /**
        * Inversed scaling of real GF by complex value
        *
        * @tparam RHS - type of the scalar
        * @param rhs - scaling factor
        * @return scaled Green's function
        */
        template<typename RHS>
        typename std::enable_if < is_complex < RHS >::value && !std::is_same < VTYPE, RHS >::value, gf_base < RHS, Tensor<RHS, _N>, MESHES... > >::type operator/(RHS rhs) const {
          throw_if_empty();
          gf_base < RHS, Tensor<RHS, _N>, MESHES... > res(*this);
          res.data_ = this->data_;
          return res /= rhs;
        }

        /**
         * @returns Negated Green's Function (a new copy).
         */
        gf_base < VTYPE, Tensor<VTYPE, _N>, MESHES... > operator-() const {
          throw_if_empty();
          return (*this)*(VTYPE(-1.0));
        }

        /**
         * Comparison. GFs that are defined on the same MESH and have the value return type will be equal if they are:
         *  A. Both empty
         *    or
         *  B.1. Have the same size
         *  B.2. And have the same values
         */
        template<typename RHS_GF>
        typename std::enable_if < std::is_same < RHS_GF, generic_gf<data_storage> >::value || std::is_same < RHS_GF, generic_gf<data_view> >::value, bool >::type
        operator==(const RHS_GF &rhs) const {
          return (empty_ && rhs.is_empty()) || (data_.sizes() == rhs.data().sizes() && data_.data() == rhs.data().data() );
        }

        /**
         * HDF5 storage
         */

        /// Save the GF to HDF5
        void save(alps::hdf5::archive &ar) const {
          save(ar, ar.get_context());
        }

        /// Load the GF from HDF5
        void load(alps::hdf5::archive &ar) {
          load(ar, ar.get_context());
        }

        /// Save the GF to HDF5
        void save(alps::hdf5::archive &ar, const std::string &path) const {
          throw_if_empty();
          save_version(ar, path);
          ar[path + "/data"] << data_.data().data();
          ar[path + "/mesh/N"] << int(_N);
          save_meshes(ar, path, std::integral_constant < int, 0 >());
        }

        /// Load the GF from HDF5
        void load(alps::hdf5::archive &ar, const std::string &path) {
          if (!check_version(ar, path)) throw std::runtime_error("Incompatible archive version");
          empty_ = false;
          int ndim;
          ar[path + "/mesh/N"] >> ndim;
          if (ndim != _N) throw std::runtime_error("Wrong number of dimension reading Matsubara GF, ndim=" + boost::lexical_cast < std::string >(ndim));
          load_meshes(ar, path, std::integral_constant < int, 0 >());
          data_ = Tensor < VTYPE, _N >(get_sizes(meshes_));
          ar[path + "/data"] >> data_.data().data();
        }

        /// Save version of the GF object to maintain compatibility
        void save_version(alps::hdf5::archive &ar, const std::string &path) const {
          std::string vp = path + "/version/";
          ar[vp + "minor"] << int(minor_version);
          ar[vp + "major"] << int(major_version);
          ar[vp + "reference"] << "https://github.com/ALPSCore/H5GF/blob/master/H5GF.rst";
          ar[vp + "originator"] << "ALPSCore GF library, see http://www.alpscore.org";
        }

        /// Check that version of the GF object in the HDF5 archive is the same as current version
        bool check_version(alps::hdf5::archive &ar, const std::string &path) const {
          std::string vp = path + "/version/";
          int ver;
          ar[vp + "major"] >> ver;
          return (major_version == ver);
        }

        /*
         * Access to storage
         */
        /**
         * @return const reference to the storage object
         */
        const Storage &data() const { return data_; };
        /**
         * @return reference to the storage object
         */
        Storage &data() { return data_; };

        /**
         * @return true if GF object was initilized as empty
         */
        bool is_empty() const {
          return empty_;
        }

        /**
         * @return const-reference to the GF meshes tuple
         */
        const _mesh_types& meshes() const {return meshes_;}

        /*
         *  MPI routines
         */
#ifdef ALPS_HAVE_MPI
        /**
         * Broadcast GF object
         *
         * @param comm - MPI communicator
         * @param root - root process number for broadcast
         */
        void broadcast(const alps::mpi::communicator& comm, int root) {
          // check that root GF has been initilized
          if(comm.rank() == root) throw_if_empty();
          // broadcast grid meshes
          broadcast_mesh(comm, root);
          size_t root_sz=data_.size();
          alps::mpi::broadcast(comm, root_sz, root);
          // as long as all grids have been broadcasted we can define tensor object
          if(comm.rank() != root) data_ = Tensor < VTYPE, _N >(get_sizes(meshes_));
          alps::mpi::broadcast(comm, &data_.data().data(0), root_sz, root);
        }
#endif

      private:

        /*
         *  MPI routines
         */
#ifdef ALPS_HAVE_MPI
        /**
         * Perform recursive call for mesh broadcasting
         */
        void broadcast_mesh(const alps::mpi::communicator& comm, int root) {
          // start from the zeroth mesh
          bcast_mesh(comm, root, std::integral_constant < int, 0 >());
        }
        template<int M>
        void bcast_mesh(const alps::mpi::communicator& comm, int root, std::integral_constant < int, M > i) {
          std::get<M>(meshes_).broadcast(comm, root);
          bcast_mesh(comm, root, std::integral_constant < int, M+1 >());
        }
        // Until we reach the last mesh object
        void bcast_mesh(const alps::mpi::communicator& comm, int root, std::integral_constant < int, _N - 1> i) {
          std::get<_N - 1>(meshes_).broadcast(comm, root);
        }
#endif

        /**
         * Throw an exception if the GF object is empty
         */
        inline void throw_if_empty() const {
          if (empty_) {
            throw std::runtime_error("gf is empty");
          }
        }

        /**
         * Get value from underlined data-structure
         *
         * @tparam Indices - types of GF indices
         * @param inds - indices
         * @return element for the provided indices
         */
        template<class ... Indices>
        VTYPE value(Indices...inds) const {
          return data_(inds()...);
        }

        template<class ... Indices>
        VTYPE &value(Indices...inds) {
          return data_(inds()...);
        }

        /**
         * Compute sizes for all grid meshes
         *
         * @param meshes - meshes to get sizes
         * @return sizes for each grid mesh
         */
        std::array < size_t, _N > get_sizes(MESHES...meshes) {
          return get_sizes(std::forward_as_tuple(meshes...));
        };

        /**
         * Tuple version of the previous method
         *
         * @param meshes - tuple of meshes
         * @return array that contains the size of each grid
         */
        std::array < size_t, _N > get_sizes(const _mesh_types &meshes) {
          std::array < size_t, _N > sizes;
          fill_sizes(sizes, std::integral_constant < int, 0 >(), meshes);
          return sizes;
        };

        /**
         * The following two methods fill array with the sizes of each grid
         *
         * @tparam M     - current leading grid number
         * @tparam Grid  - leading grid type
         * @tparam Grids - tail grids types
         * @param sizes  - array to fill sizes of each grid
         * @param index  - current leading grid index
         * @param g1     - leading grid
         * @param grids  - tail grids
         */
        template<int M>
        void fill_sizes(std::array < size_t, _N > &sizes, std::integral_constant < int, M > index, const _mesh_types &grids) {
          sizes[index.value] = std::get < M >(grids).extent();
          fill_sizes(sizes, std::integral_constant < int, M + 1 >(), grids);
        };

        void fill_sizes(std::array < size_t, _N > &sizes, std::integral_constant < int, _N - 1 > index, const _mesh_types &grids) {
          sizes[index.value] = std::get < _N - 1 >(grids).extent();
        };

        /**
         *
         * The following two methods check that indices provided are consistent with the indices of the GF mesh
         *
         * @tparam M - current leading index to check
         * @tparam Index - current leading index type
         * @tparam Indices - tail indices
         * @param i - current leading index number
         * @param id - current leading index value
         * @param inds - tail indices
         */
        template<int M, class Index, class...Indices>
        void check(std::integral_constant < int, M > i, const Index &id, const Indices &...inds) const {
          static_assert(std::is_same < Index, typename std::tuple_element < i.value, _mesh_types >::type::index_type >::value, "Index type is inconsistent with mesh index type.");
          check(std::integral_constant < int, M + 1 >(), std::forward < const Indices & >(inds)...);
        }

        template<class Index>
        void check(std::integral_constant < int, _N - 1 > i, const Index &id) const {
          static_assert(std::is_same < Index, typename std::tuple_element < i.value, _mesh_types >::type::index_type >::value, "Index type is inconsistent with mesh index type.");
        }

        /**
         * Save meshes into hdf5
         *
         * @tparam D   - current mesh number
         * @param ar   - hdf5 archive object
         * @param path - relative context
         * @param i    - current mesh number
         */
        template<int D>
        void save_meshes(alps::hdf5::archive &ar, const std::string &path, std::integral_constant < int, D > i) const {
          ar[path + "/mesh/" + boost::lexical_cast < std::string >(i.value)] << std::get < i.value >(meshes_);
          save_meshes(ar, path, std::integral_constant < int, D + 1 >());
        }

        /**
         * Save the last mesh into hdf5
         */
        void save_meshes(alps::hdf5::archive &ar, const std::string &path, std::integral_constant < int, _N - 1 > i) const {
          ar[path + "/mesh/" + boost::lexical_cast < std::string >(i.value)] << std::get < i.value >(meshes_);
        }

        /**
         * Load meshes from hdf5
         *
         * @tparam D   - current mesh number
         * @param ar   - hdf5 archive object
         * @param path - relative context
         * @param i    - current mesh number
         */
        template<int D>
        void load_meshes(alps::hdf5::archive &ar, const std::string &path, std::integral_constant < int, D > i) {
          ar[path + "/mesh/" + boost::lexical_cast < std::string >(i.value)] >> std::get < i.value >(meshes_);
          load_meshes(ar, path, std::integral_constant < int, D + 1 >());
        }

        /**
         * Load the last mesh from hdf5
         */
        void load_meshes(alps::hdf5::archive &ar, const std::string &path, std::integral_constant < int, _N - 1 > i) {
          ar[path + "/mesh/" + boost::lexical_cast < std::string >(i.value)] >> std::get < i.value >(meshes_);
        }

        /**
         * Extract types from the tuple
         * We need this trick to provide intermediate level for enable_if to avoid type resolving for dimensions
         * larger than we have
         *
         * void_type is special wrapper to make able enable_if to disable MESH<N>()
         * if N is larger than number of meshes
         */
        struct void_type {
          using type = void;
        };
        // Argument types.
        template<size_t i>
        struct args {
          using type
          = typename std::conditional < (i <= _N),
            std::tuple_element < i - 1, _mesh_types >,
            void_type >::type::type;
        };

      public:

        /**
         * Create MESH<N>() functions for compatibility with old interface
         */
        #define MESH_FUNCTION(z, num, c) \
        template<typename = typename std::enable_if< (_N >= num)> >\
        const typename args<num>::type mesh##num() const {\
          return std::get<int(num-1)>(meshes_); \
        }

        /*
         * I guess 10 functions would be enough
         */
        BOOST_PP_REPEAT_FROM_TO (1, 11, MESH_FUNCTION, int)

      };

    }
  }
}

#endif //GREENSFUNCTIONS_GF_H
