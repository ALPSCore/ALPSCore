/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPSCORE_GF_TAIL_BASE_H
#define ALPSCORE_GF_TAIL_BASE_H

#include <alps/gf/gf_base.hpp>

namespace alps {
  namespace gf {
    static const int TAIL_NOT_SET=-1; ///< Special tail order meaning the tail is not set

    // forward declarations
    namespace detail {
      template<typename HEADGF, typename TAILGF>
      class gf_tail_base;

#ifdef ALPS_HAVE_MPI
      template <typename TAILT>
      void broadcast_tail(const alps::mpi::communicator& comm,
                          int& min_order, int& max_order,
                          std::vector<TAILT>& tails,
                          const TAILT& tail_init,
                          int root)
      {
        using alps::mpi::broadcast;
        broadcast(comm, min_order, root);
        broadcast(comm, max_order, root);

        if (min_order==TAIL_NOT_SET) return;
        if (comm.rank()!=root) {
          tails.resize(max_order+1, tail_init);
        }
        for (int i=min_order; i<=max_order; ++i) {
          tails[i].broadcast(comm,root);
        }
      }
#endif
    }
    template<typename HEADGF, typename TAILGF>
    using gf_tail      = detail::gf_tail_base<HEADGF, TAILGF>;
    // TODO: think how to implement tailed GF view
//    template<typename VTYPE, typename ...Meshes>
//    using gf_tail_view = detail::gf_tail_base<VTYPE, detail::tensor_view<VTYPE, sizeof...(Meshes)>, Meshes...>;

    namespace detail {
      /**
       * @brief Implementation of the Green's function with high-frequency tails.
       * We assume that Green's function has first leading index in frequency or
       * imaginary time. And for the rest indices we should have high-frequency
       * information.
       *
       * @author iskakoff
       */
      template<typename HEADGF, typename TAILGF>
      class gf_tail_base : public HEADGF {
        // parent class method using
        using HEADGF::meshes;
      public:
        // types definition
        /// GF storage type
        using Storage = typename HEADGF::storage_type;
        /// GF value type
        using VTYPE = typename HEADGF::value_type;
        using mesh_tuple = typename HEADGF::mesh_types;
        /// green's function type
        typedef HEADGF gf_type;
        /// tail type
        typedef TAILGF tail_type;
        /// type of the current GF object
        typedef gf_tail_base< HEADGF, TAILGF > gf_type_with_tail;
      private:
        // fields definition
        std::vector<tail_type> tails_;
        int min_tail_order_;
        int max_tail_order_;

      public:

        /// default constructor
        gf_tail_base() : gf_type(), min_tail_order_(TAIL_NOT_SET), max_tail_order_(TAIL_NOT_SET) {}
        /// construct gf with tail from simple gf object
        gf_tail_base(const gf_type& gf): gf_type(gf), min_tail_order_(TAIL_NOT_SET), max_tail_order_(TAIL_NOT_SET)
        { }
        /// copy-constructor
        gf_tail_base(const gf_type_with_tail& gft): gf_type(gft), tails_(gft.tails_),
                                                           min_tail_order_(gft.min_tail_order_),
                                                           max_tail_order_(gft.max_tail_order_)
        { }

        /// minimum non-empty tail order
        int min_tail_order() const { return min_tail_order_; }
        /// maximum non-empty tail order
        int max_tail_order() const { return max_tail_order_; }

        /// Returns tail component of the given order
        const tail_type& tail(int order) const{
          if (order<min_tail_order_ || order > max_tail_order_)
            throw std::runtime_error("tails are known between min and max order, your order is outside.");
          return tails_[order];
        }

        /// Returns tail as a vector
        const std::vector<tail_type>& tail() const{
          return tails_;
        }

        /**
         * Check for equality. Two GF with tails defined on the same meshes are the same if:
         * 1. They both have the same tail or they both do not have tail.
         * 2. The 'head' Green's functions are equal to each other.
         *
         *
         * @param rhs - GF to compare with
         * @return
         */
        bool operator==(const gf_type_with_tail &rhs) const {
          bool tail_eq = (min_tail_order_ == rhs.min_tail_order_) && (max_tail_order_ == rhs.max_tail_order_);
          if(tail_eq && min_tail_order_!=TAIL_NOT_SET) {
            for (int i = min_tail_order_; i <= max_tail_order_; ++i) {
              tail_eq &= (tails_[i] == rhs.tails_[i]);
            }
          }
          return tail_eq && HEADGF::operator==(rhs);
        }

        template<typename RHS_GF>
        bool operator!=(const RHS_GF &rhs) const {
          return !(this->operator==(rhs));
        }

        /**
         * Set new tail of high-frequency order = %order%
         * @param order - tail order
         * @param tail  - tail object
         * @return      - return current GF with new tail of order %order% set
         */
        gf_type_with_tail& set_tail(int order, const tail_type &tail){
          static_assert(std::is_same<typename tail_type::mesh_types,
              decltype(tuple_tail < 1, std::tuple_size<typename HEADGF::mesh_types>::value >(HEADGF::meshes())) >::value, "Incorrect tail mesh types" );

          int tail_size=tails_.size();
          if(order>=tail_size){
            tails_.resize(order+1, tail_type (tuple_tail < 1, std::tuple_size<typename HEADGF::mesh_types>::value >(meshes())));
            for(int i=tail_size;i<=order;++i) tails_[i].initialize();
          }
          tails_[order]=tail;

          //set minimum and maximum known coefficients if needed
          if(min_tail_order_==TAIL_NOT_SET || min_tail_order_>order) min_tail_order_=order;
          if(max_tail_order_==TAIL_NOT_SET || max_tail_order_<=order) max_tail_order_=order;
          return *this;
        }

        /// Save the GF with tail to HDF5
        void save(alps::hdf5::archive& ar, const std::string& path) const
        {
          gf_type::save(ar,path);
          ar[path+"/tail/descriptor"]="INFINITY_TAIL";
          ar[path+"/tail/min_tail_order"]=min_tail_order_;
          ar[path+"/tail/max_tail_order"]=max_tail_order_;
          if(min_tail_order_==TAIL_NOT_SET) return;
          for (int i=min_tail_order_; i<=max_tail_order_; ++i) {
            ar[path+"/tail/"+std::to_string(i)] << tails_[i].data();
          }
        }

        /// Load the GF with tail from HDF5
        void load(alps::hdf5::archive& ar, const std::string& path)
        {
          gf_type::load(ar,path);
          std::string descr; ar[path+"/tail/descriptor"] >> descr;
          if (descr!="INFINITY_TAIL") throw std::runtime_error("Wrong tail format '"+descr+"', expected INFINITY_TAIL");

          // FIXME!FIXME! Rewrite using clone-swap for exception safety.
          ar[path+"/tail/min_tail_order"] >> min_tail_order_;
          ar[path+"/tail/max_tail_order"] >> max_tail_order_;

          tails_.clear();
          if(min_tail_order_==TAIL_NOT_SET) return;

          if(min_tail_order_>0) tails_.resize(min_tail_order_, tail_type (tuple_tail < 1, std::tuple_size<typename HEADGF::mesh_types>::value >(meshes())));

          for (int i=min_tail_order_; i<=max_tail_order_; ++i) {
            typename tail_type::storage_type buffer;
            ar[path+"/tail/"+std::to_string(i)] >> buffer;
            tails_.push_back(tail_type(buffer, tuple_tail < 1, std::tuple_size<typename HEADGF::mesh_types>::value >(meshes())));
          }
        }

        /// Save the GF to HDF5
        void save(alps::hdf5::archive& ar) const
        {
          save(ar, ar.get_context());
        }

        /// Load the GF from HDF5
        void load(alps::hdf5::archive& ar)
        {
          load(ar, ar.get_context());
        }

#ifdef ALPS_HAVE_MPI
        /// Broadcast the tail and the GF
        void broadcast(const alps::mpi::communicator& comm, int root)
        {
          // FIXME: use clone-swap?
          gf_type::broadcast(comm,root);
          detail::broadcast_tail(comm,
                                 min_tail_order_, max_tail_order_,
                                 tails_, tail_type(tuple_tail < 1, std::tuple_size<typename HEADGF::mesh_types>::value >(meshes())),
                                 root);
        }
#endif
      };
    }
  }
}

#endif //ALPSCORE_GF_TAIL_BASE_H
