/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <vector>

namespace alps {namespace gf {
  namespace statistics {
    enum statistics_type {
      BOSONIC=0,
      FERMIONIC=1
    };
  }
  namespace detail {
    /* The following is an in-house implementation of a static_assert
       with the intent to generate a compile-time error message
       that has some relevance to the asserted condition
       (unlike BOOST_STATIC_ASSERT).
    */
    /// A helper class: indicator that a mesh can have a tail
    struct can_have_tail_yes {typedef bool mesh_can_have_tail;};
    /// A helper class: indicator that a mesh can NOT have a tail
    struct can_have_tail_no {typedef bool mesh_cannot_have_tail;};
    /// Trait: whether a mesh can have a tail (general meshes cannot have tails)
    template<typename>
    struct can_have_tail : public can_have_tail_no {};
    /* ^^^^ End of static_assert code */
  }
/// Common part of interface and implementation for GF meshes
  class base_mesh {
  public:
    base_mesh() {}
    base_mesh(const base_mesh& rhs) : points_(rhs.points_) {}
    /// Const access to mesh points
    const std::vector<double> &points() const{return points_;}
  protected:
    // we do not want external functions be able to change a grid.
    std::vector<double> &_points() {return points_;}
    void swap(base_mesh &other) {
      using std::swap;
      swap(this->points_, other.points_);
    }
  private:
    std::vector<double> points_;
  };
}}

