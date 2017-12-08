//
// Created by iskakoff on 08/12/17.
//

#ifndef ALPSCORE_GF_TAIL_H
#define ALPSCORE_GF_TAIL_H

#include "gf_base.h"

namespace alps {
  namespace gf {
    static const int TAIL_NOT_SET=-1; ///< Special tail order meaning the tail is not set

    // forward declarations
    namespace detail {
      template<typename VTYPE, typename Storage, typename Mesh1, typename ...Meshes>
      class gf_tail_base;
    }
    template<typename VTYPE, typename ...Meshes>
    using gf_tail = detail::gf_tail_base<VTYPE, detail::Tensor<VTYPE, sizeof...(Meshes)>, Meshes...>;

    namespace detail {
      /**
       * @brief Implementation of the Green's function with high-frequency tails.
       * We assume that Green's function has first leading index in frequency or
       * imaginary time. And for the rest indices we should have high-frequency
       * information.
       *
       * @author iskakoff
       */
      template<typename VTYPE, typename Storage, typename Mesh1, typename ...Meshes>
      class gf_tail_base : public gf_base < VTYPE, Storage, Mesh1, Meshes... > {
      public:
        /// green's function type
        typedef gf_base < VTYPE, Storage, Mesh1, Meshes... > gf_type;
        /// tail type
        typedef gf_base < VTYPE, Storage, Meshes... > tail_type;
        /// type of the current GF object
        typedef gf_tail_base< VTYPE, Storage, Mesh1, Meshes... > gf_type_with_tail;
      private:
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
                                                           min_tail_order_(gft.min_tail_order_), max_tail_order_(gft.max_tail_order_)
        { }

        /// minimum non-empty tail order
        int min_tail_order() const { return min_tail_order_; }
        /// maximum non-empty tail order
        int max_tail_order() const { return max_tail_order_; }

        /// Returns tail component of the given order
        const tail_type& tail(int order) const{
          if (order<min_tail_order_ || order > max_tail_order_) throw std::runtime_error("tails are known between min and max order, your order is outside.");
          return tails_[order];
        }

        /// Returns tail as a vector
        const std::vector<tail_type>& tail() const{
          return tails_;
        }

        /**
         * Set new tail of high-frequency order = %order%
         * @param order - tail order
         * @param tail  - tail object
         * @return      - return current GF with new tail of order %order% set
         */
        gf_type_with_tail& set_tail(int order, const tail_type &tail){
          if(this->mesh2()!=tail.mesh1())
            throw std::runtime_error("invalid mesh type in tail assignment");

          int tail_size=tails_.size();
          if(order>=tail_size){
            tails_.resize(order+1, tail_type(this->mesh2()));
            for(int i=tail_size;i<=order;++i) tails_[i].initialize();
          }
          tails_[order]=tail;

          //set minimum and maximum known coefficients if needed
          if(min_tail_order_==TAIL_NOT_SET || min_tail_order_>order) min_tail_order_=order;
          if(max_tail_order_==TAIL_NOT_SET || max_tail_order_<=order) max_tail_order_=order;
          return *this;
        }
      };
    }
  }
}

#endif //ALPSCORE_GF_TAIL_H
