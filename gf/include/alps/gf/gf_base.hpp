/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPSCORE_GF_H
#define ALPSCORE_GF_H


#include <tuple>
#include <vector>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/tensor.hpp>
#ifdef ALPS_HAVE_MPI
#include <alps/utilities/mpi.hpp>
#endif

#include <alps/gf/mesh.hpp>
#include <alps/numeric/tensors/tensor_base.hpp>
#include <alps/type_traits/index_sequence.hpp>
#include <alps/type_traits/tuple_traits.hpp>


namespace alps {
  namespace gf {
    namespace detail {
      /**
       * Definition of base GF container
       */
      template<class VTYPE, class Storage, class ...MESHES>
      class gf_base;

      template <typename Tuple, std::size_t...Is>
      std::ostream & output(std::ostream & os, const Tuple& t, index_sequence<Is...>) {
        using swallow = int[];
        (void)swallow{0, (void(os << std::get<Is>(t)), 0)...};
        return os;
      };

      template <typename ...Args>
      auto operator<<(std::ostream & os, const std::tuple<Args...>& t) ->
                                                DECLTYPE(detail::output(os, t, make_index_sequence<sizeof...(Args)>()))

      template<typename T> inline void print_no_complex(std::ostream &os, const T &z){
        os<<z;
      }
      template<> inline void print_no_complex(std::ostream &os, const std::complex<double> &z){
        //specialization for printing complex as ... real imag ...
        os<<z.real()<<" "<<z.imag();
      }
    }

    /**
     * Definition of regular GF with dedicated storage
     */
    template<class VTYPE, class ...MESHES>
    using greenf = detail::gf_base<VTYPE, numerics::tensor<VTYPE, sizeof...(MESHES)>, MESHES...>;
    /**
     * Definition of GF as view of existent data array
     */
    template<class VTYPE, class ...MESHES>
    using greenf_view = detail::gf_base<VTYPE, numerics::tensor_view<VTYPE, sizeof...(MESHES)>, MESHES...>;

    namespace detail {
      /**
       * @brief This class implements general container for Green's functions storage
       */
      template<class VTYPE, class Storage, class ...MESHES>
      class gf_base {
        // types definition
      public:
        // Current GF types
        /// Value type
        using value_type   = VTYPE;
        /// Storage type
        using storage_type = Storage;
        /// mesh types tuple
        using mesh_types = std::tuple < MESHES... >;
        /// index types tuple
        using index_types = std::tuple < typename MESHES::index_type... >;
      private:
        /// current GF type
        using gf_type   = gf_base < VTYPE, Storage, MESHES... >;
        /// storage types
        using data_storage = numerics::tensor < VTYPE, sizeof...(MESHES) >;
        using data_view    = numerics::tensor_view < VTYPE, sizeof...(MESHES) >;
        /// Generic GF type
        template<typename St>
        using generic_gf   = gf_base < VTYPE, St, MESHES... >;
        /// Greens function return type for arithmetic operations with different type
        template<typename RHS_VTYPE>
        using gf_op_type   = gf_base < decltype(RHS_VTYPE{} + VTYPE{}), numerics::tensor<decltype(RHS_VTYPE{} + VTYPE{}),
            sizeof...(MESHES)>, MESHES... >;

        // fields definition
      private:
        /// GF version
        static constexpr int minor_version = 1;
        static constexpr int major_version = 0;
        /// Dimension of grid
        static constexpr int N_ = sizeof...(MESHES);
        /// data_storage
        Storage data_;
        /// stored meshes
        std::tuple < MESHES... > meshes_;
        /// uninitialized state flag
        bool empty_;

        // template hacks
        /**
         * Create Greens function view class based on the mesh tuple types.
         *
         * @tparam S   - type of the stored data
         * @tparam Tup - Mesh tuple
         */
        template <typename S, class Tup> struct subpack_impl;
        template<typename S, template<class...> class Tup, class... T>
        struct subpack_impl<S, Tup<T...> >
        {
          using type = gf_base<S, numerics::tensor_view < S, sizeof...(T)>,  T...>;
        };
        template<typename S, size_t I>
        using subpack = typename subpack_impl<S, decltype(tuple_tail < I, MESHES... >(meshes_) )>::type;

        template<typename RHS_VTYPE, typename LHS_VTYPE>
        struct convert {
          typedef typename std::conditional<std::is_integral<RHS_VTYPE>::value, VTYPE, RHS_VTYPE>::type type;
        };

        /**
         * Check that indices' types are consistent with mesh indices' types
         *
         * @tparam IndicesTypes - types of indices template parameter pack
         */
        template<typename ...IndicesTypes>
        struct check_mesh {
          using type = typename std::conditional<std::is_convertible<std::tuple<IndicesTypes...>, index_types>::value,
              std::true_type, std::false_type>::type;
        };

      public:

        /**
         * Copy-constructor
         * @param g - GF to copy from
         */
        gf_base(const gf_type &g) = default;

        /**
         * Move-constructor
         */
        gf_base(gf_type &&g) = default;

        /**
         * Default constructor. Create uninitilized GF
         */
        gf_base() : data_(std::array < size_t, N_ >{{0}}), empty_(true) {}

        /**
         * Create Green's function object with given meshes
         * @param meshes - list of meshes
         */
        gf_base(MESHES...meshes) : gf_base(std::make_tuple(meshes...)) {}
        /**
         * Create Green's function object with given meshes
         * @param meshes - tuple of meshes
         */
        gf_base(const mesh_types &meshes) : data_(get_sizes(meshes)), meshes_(meshes), empty_(false) {}
        /// Create GF with the provided data
        gf_base(VTYPE* data, const mesh_types &meshes) : data_(data, get_sizes(meshes)), meshes_(meshes), empty_(false) {}
        /// Create GF with the provided data and meshes
        gf_base(data_storage const &data, MESHES...meshes) : data_(data), meshes_(std::make_tuple(meshes...)), empty_(false) {}
        /// Create GF with the provided data and meshes
        gf_base(data_storage && data, MESHES...meshes) : data_(std::move(data)), meshes_(std::make_tuple(meshes...)), empty_(false) {}

        /// construct new GF object by copy data from another GF object defined with different storage type
        template<typename St, typename = std::enable_if<!std::is_same<St, Storage>::value && std::is_same<St, data_view>::value > >
        gf_base(const gf_base<VTYPE, St, MESHES...> &g) : data_(g.data()), meshes_(g.meshes()), empty_(g.is_empty()) {}
        template<typename St, typename = std::enable_if<!std::is_same<St, Storage>::value && std::is_same<St, data_storage>::value > >
        gf_base(gf_base<VTYPE, St, MESHES...> &g) : data_(g.data()), meshes_(g.meshes()), empty_(g.is_empty()) {}

        /// construct new GF by copy/move of GF with a different type
        template<typename RHS_VTYPE, typename St, typename = std::enable_if<std::is_same<data_storage, Storage>::value> >
        gf_base(const gf_base<RHS_VTYPE, St, MESHES...> &g) : data_(g.data()), meshes_(g.meshes()), empty_(g.is_empty()) {
          static_assert(std::is_convertible<RHS_VTYPE, VTYPE>::value, "Right-hand side data type is not convertible into left-hand side.");
        }
        template<typename RHS_VTYPE, typename St, typename = std::enable_if<std::is_same<data_storage, Storage>::value>>
        gf_base(gf_base<RHS_VTYPE, St, MESHES...> &&g) noexcept : data_(std::move(g.data())), meshes_(std::move(g.meshes())), empty_(g.is_empty()) {
          static_assert(std::is_convertible<RHS_VTYPE, VTYPE>::value, "Right-hand side data type is not convertible into left-hand side.");
        }

        /// construct new green's function from index slice of GF with higher dimension
        template<typename St, typename...OLDMESHES, typename Index, typename ...Indices>
        gf_base(gf_base<VTYPE, numerics::detail::tensor_base < VTYPE, sizeof...(OLDMESHES), St >, OLDMESHES...> & g,
                std::tuple<OLDMESHES...>& oldmesh, const mesh_types &meshes, const Index ind, const Indices... idx) :
          data_(g.data()(ind(), idx()...)), meshes_(meshes), empty_(false) {}

        /// construct const view
        template<typename RHSTYPE ,typename St, typename...OLDMESHES, typename ...Indices>
        gf_base(const gf_base<RHSTYPE, numerics::detail::tensor_base < RHSTYPE, sizeof...(OLDMESHES), St >, OLDMESHES...> & g,
                const std::tuple<OLDMESHES...>& oldmesh, mesh_types &meshes, const Indices... idx) :
            data_(g.data()(idx()...)), meshes_(meshes), empty_(false) {}

        /// copy assignment
        gf_type& operator=(const gf_type & rhs) {
          swap_meshes(rhs.meshes_, make_index_sequence<sizeof...(MESHES)>());
          data_ = rhs.data();
          empty_= rhs.empty_;
          return *this;
        }
        /// move assignment
        gf_type& operator=(gf_type && rhs) noexcept {
          swap_meshes(rhs.meshes_, make_index_sequence<sizeof...(MESHES)>());
          using std::swap;
          swap(data_, rhs.data_);
          swap(empty_, rhs.empty_);
          return *this;
        }

        /// initialize with zeros
        void initialize() {
          data_.set_zero();
        }

        /// Check if meshes are compatible, throw if not
        void check_meshes(const gf_type& rhs) const
        {
          if (meshes_ != rhs.meshes_) {
            throw std::invalid_argument("Green Functions have incompatible meshes");
          }
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
        typename std::enable_if < (sizeof...(Indices) == sizeof...(MESHES)), const VTYPE & >::type
        operator()(Indices...inds) const {
          // check that index types are the same as mesh indices
          static_assert(check_mesh<Indices...>::type::value, "Index type is inconsistent with mesh index type.");
          return value(std::forward < Indices >(inds)...);
        }

        /// @return reference to the specific value in the GF object
        template<class...Indices>
        typename std::enable_if < (sizeof...(Indices) == sizeof...(MESHES)), VTYPE & >::type
        operator()(Indices...inds) {
          // check that index types are the same as mesh indices
          static_assert(check_mesh<Indices...>::type::value, "Index type is inconsistent with mesh index type.");
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
        auto operator()(typename std::enable_if<(sizeof...(Indices)+1 < N_),
            typename std::tuple_element<0,mesh_types>::type::index_type >::type ind, Indices...inds) ->
                                                                            subpack<VTYPE, sizeof...(Indices) + 1> {
          return subpack<VTYPE, sizeof...(Indices) + 1 >
                          (*this, meshes_, tuple_tail < sizeof...(Indices) + 1, MESHES...>(meshes_), ind, std::forward<Indices>(inds)...);
        }

        template<class...Indices>
        auto operator()(typename std::enable_if<(sizeof...(Indices)+1 < N_),
            typename std::tuple_element<0,mesh_types>::type::index_type >::type ind, Indices...inds) const ->
                                                                            subpack<const VTYPE, sizeof...(Indices) + 1> {
          auto t = tuple_tail < sizeof...(Indices) + 1, MESHES...>(meshes_);
          return subpack<const VTYPE, sizeof...(Indices) + 1>  (*this, meshes_, t, ind, std::forward<Indices>(inds)...);
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
        template<typename RHS_VTYPE, typename RHS_STORAGE>
        gf_op_type<RHS_VTYPE> operator+(const gf_base<RHS_VTYPE, RHS_STORAGE, MESHES...> &rhs) const {
          throw_if_empty();
          gf_op_type<RHS_VTYPE> res(*this);
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
        typename std::enable_if < std::is_convertible < RHS_GF, generic_gf<data_storage>>::value ||
                                  std::is_convertible < RHS_GF, generic_gf<data_view>>::value, gf_type & >::type
        operator+=(const RHS_GF &rhs) {
          throw_if_empty();
          data_ += rhs.data();
          return *this;
        }

        /**
         * Compute difference of current GF object and rhs
         *
         * @tparam RHS_GF - type of the right hand side GF
         * @param  rhs    - GF object to sum with the current GF
         * @return new GF object equals to sum of this and rhs
         */
        template<typename RHS_VTYPE, typename RHS_STORAGE>
        gf_op_type<RHS_VTYPE> operator-(const gf_base<RHS_VTYPE, RHS_STORAGE, MESHES...> &rhs) const {
          throw_if_empty();
          gf_op_type<RHS_VTYPE> res(*this);
          return res -= rhs;
        }

        /**
         * Inplace subtraction
         */
        template<typename RHS_GF>
        typename std::enable_if < std::is_convertible < RHS_GF, generic_gf<data_storage>>::value ||
                                  std::is_convertible < RHS_GF, generic_gf<data_view>>::value, gf_type & >::type
        operator-=(const RHS_GF &rhs) {
          throw_if_empty();
          data_ -= rhs.data();
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
        gf_op_type<typename convert<RHS,VTYPE>::type> operator*(RHS rhs) const {
          throw_if_empty();
          gf_op_type<typename convert<RHS,VTYPE>::type> res(*this);
          return res *= rhs;
        }

        template<typename LHS>
        friend gf_op_type<typename convert<LHS,VTYPE>::type> operator*(LHS lhs, const gf_type & g) {
          return g * lhs;
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
        gf_op_type<typename convert<RHS,VTYPE>::type> operator/(RHS rhs) const {
          throw_if_empty();
          gf_op_type<typename convert<RHS,VTYPE>::type> res(*this);
          return res /= rhs;
        }

        template<typename LHS>
        friend gf_op_type<typename convert<LHS,VTYPE>::type> operator/(LHS lhs, const gf_type & g) {
          return g / lhs;
        }

        /**
         * @returns Negated Green's Function (a new copy).
         */
        gf_base < VTYPE, numerics::tensor<VTYPE, N_>, MESHES... > operator-() const {
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
        typename std::enable_if < std::is_convertible < RHS_GF, generic_gf<data_storage> >::value
                               || std::is_convertible < RHS_GF, generic_gf<data_view> >::value, bool >::type
        operator==(const RHS_GF &rhs) const {
          return (empty_ && rhs.is_empty()) || (data_.shape() == rhs.data().shape() && data_.storage() == rhs.data().storage() );
        }

        template<typename RHS_GF>
        bool operator!=(const RHS_GF &rhs) const {
          return !(*this==rhs);
        }

        /**
         * @return norm of the Green's function
         */
        double norm() const {
          throw_if_empty();
          return std::abs(*std::max_element(data_.data(), data_.num_elements() + data_.data(),
                                            [](VTYPE a, VTYPE b) {return std::abs(a) < std::abs(b);} ) );
        }

        // reshape green's function
        void reshape(MESHES...meshes) {
          std::array<size_t, N_> new_shape = get_sizes(std::make_tuple(meshes...));
          size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), size_t(1), std::multiplies<size_t>());
          if(std::is_same<data_view, Storage>::value) {
            if (data_.size() != new_size) {
              throw std::invalid_argument("The total size of resulting Green's function view should be the same as before.");
            }
          }
          data_.reshape(new_shape);
          meshes_ = std::make_tuple(meshes...);
          empty_ = false;
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
          ar[path + "/data"] << data_;
          ar[path + "/mesh/N"] << int(N_);
          save_meshes(ar, path, make_index_sequence<sizeof...(MESHES)>());
        }

        /// Load the GF from HDF5
        void load(alps::hdf5::archive &ar, const std::string &path) {
          if (!check_version(ar, path)) throw std::runtime_error("Incompatible archive version");
          int ndim;
          ar[path + "/mesh/N"] >> ndim;
          if (ndim != N_) throw std::runtime_error("Wrong number of dimension reading GF, ndim=" + std::to_string(ndim)
                                                   + ", should be N=" + std::to_string(N_));
          load_meshes(ar, path, make_index_sequence<sizeof...(MESHES)>());
          data_ = numerics::tensor < VTYPE, N_ >(get_sizes(meshes_));
          ar[path + "/data"] >> data_;
          empty_ = false;
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
        const mesh_types& meshes() const {return meshes_;}

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
          if(comm.rank() != root) data_ = numerics::tensor < VTYPE, N_ >(get_sizes(meshes_));
          alps::mpi::broadcast(comm, &data_.storage().data(0), root_sz, root);
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
        void bcast_mesh(const alps::mpi::communicator& comm, int root, std::integral_constant < int, N_ - 1> i) {
          std::get<N_ - 1>(meshes_).broadcast(comm, root);
        }
#endif

        /**
         * Throw an exception if the GF object is empty
         */
        inline void throw_if_empty() const {
#ifndef NDEBUG
          if (empty_) {
            throw std::runtime_error("gf is empty");
          }
#endif
        }

        /**
         * Get value from underlined data-structure
         *
         * @tparam Indices - types of GF indices
         * @param inds - indices
         * @return element for the provided indices
         */
        template<class ... Indices>
        const VTYPE & value(Indices...inds) const {
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
        std::array < size_t, N_ > get_sizes(MESHES...meshes) {
          return get_sizes(std::forward_as_tuple(meshes...));
        };

        /**
         * Tuple version of the previous method
         *
         * @param meshes - tuple of meshes
         * @return array that contains the size of each grid
         */
        std::array < size_t, N_ > get_sizes(const mesh_types &meshes) {
          std::array < size_t, N_ > sizes;
          sizes = fill_sizes(meshes, make_index_sequence<sizeof...(MESHES)>());
          return sizes;
        };

        /**
         * The following method fill array with the sizes of each grid
         *
         * @tparam Is    - indices to be filled
         * @param sizes  - array to fill sizes of each grid
         * @param index  - current leading grid index
         * @param g1     - leading grid
         * @param grids  - tail grids
         * @return new sizes
         */
        template<size_t...Is>
        inline std::array < size_t, N_ > fill_sizes(const mesh_types &grids, index_sequence<Is...>) {
          return {{size_t(std::get<Is>(grids).extent())...}};
        };

        /**
         * Swap meshes between
         * @tparam Is
         * @param old_meshes
         */
        template<size_t...Is>
        void swap_meshes(const mesh_types & old_meshes, index_sequence<Is...>) {
          std::tie(std::get < Is >(meshes_) = std::get < Is >(old_meshes)...);
        }

        /**
         * Save meshes into hdf5
         *
         * @tparam Is  - mesh index list
         * @param ar   - hdf5 archive object
         * @param path - relative context
         */
        template<size_t...Is>
        void save_meshes(alps::hdf5::archive &ar, const std::string &path, index_sequence<Is...>) const {
          std::tie(ar[path + "/mesh/" + std::to_string(Is+1)] << std::get < Is >(meshes_)...);
        }


        /**
         * Load meshes from hdf5
         *
         * @tparam Is  - mesh index list
         * @param ar   - hdf5 archive object
         * @param path - relative context
         */
        template<size_t...Is>
        void load_meshes(alps::hdf5::archive &ar, const std::string &path, index_sequence<Is...>) {
          std::tie(ar[path + "/mesh/" + std::to_string(Is+1)] >> std::get < Is >(meshes_)...);
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
          = typename std::conditional < (i <= N_),
            std::tuple_element < i - 1, mesh_types >,
            void_type>::type::type;
        };

      public:

        /**
         * Create MESH<N>() functions for compatibility with old interface
         */
        #define MESH_FUNCTION(z, num, c) \
        template<typename = typename std::enable_if< (N_ >= num)> >\
        typename std::add_lvalue_reference<const typename args<num>::type>::type mesh##num() const {\
          return std::get<int(num-1)>(meshes_); \
        }

        #define MESH_TYPES(z, num, c) \
        /*template<typename = typename std::enable_if< (N_ >= num)> >*/\
        typedef typename args<num>::type mesh##num##_type;
        // = typename args<num>::type;

        /*
         * I guess 10 functions would be enough
         */
        BOOST_PP_REPEAT_FROM_TO (1, 11, MESH_FUNCTION, int)
        BOOST_PP_REPEAT_FROM_TO (1, 11, MESH_TYPES, int)

        /**
         * Print Green's function object into stream
         */
        friend std::ostream &operator<<(std::ostream &os, const gf_type & G ){
          using detail::operator<<;
          os<<G.meshes();
          for(int i=0;i<G.mesh1().extent();++i){
            os<<(G.mesh1().points()[i])<<" ";
            size_t points = G.data().size()/G.data().shape()[0];
            for(size_t j = 0; j< points; ++j) {
              detail::print_no_complex<value_type>(os, G.data().data()[j + G.data().index(i)]);
              os<<" ";
            }
            os<<std::endl;
          }
          return os;
        }
      };
    }
  }
}

#endif //ALPSCORE_GF_H
