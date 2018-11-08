/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#ifdef ALPS_HAVE_MPI
#include <alps/gf/mpi_bcast.hpp>
#endif
#include <alps/gf/mesh/index.hpp>
#include <alps/gf/mesh/mesh_base.hpp>

namespace alps {namespace gf {
  class chebyshev_mesh {
    ///inverse temperature
    double beta_;
    ///chebyshev polynomial order
    int k_;
    ///mesh statistics: bosons or fermions
    alps::gf::statistics::statistics_type statistics_;

    ///zeros of the chebyshev polynomial of order k
    std::vector<double> zeros_;

    inline void throw_if_empty() const {
      if (extent() == 0) {
        throw std::runtime_error("chebyshev mesh is empty");
      }
    }

  public:
    typedef alps::gf::generic_index<chebyshev_mesh> index_type;

    chebyshev_mesh() : beta_(0.0), k_(0), statistics_(alps::gf::statistics::FERMIONIC) {
    }

    chebyshev_mesh(double beta, int k) : beta_(beta), k_(k), statistics_(alps::gf::statistics::FERMIONIC) {
      compute_zeros();
    }

    int operator()(index_type idx) const {
#ifndef NDEBUG
      throw_if_empty();
#endif
      return idx();
    }

    int extent() const { return k_; }

    /// Comparison operators
    bool operator==(const chebyshev_mesh &mesh) const {
      throw_if_empty();
      return beta_ == mesh.beta_ && k_ == mesh.k_ && statistics_ == mesh.statistics_;
    }

    ///Getter variables for members
    double beta() const { return beta_; }

    alps::gf::statistics::statistics_type statistics() const { return statistics_; }

    const std::vector<double> &points() const { return zeros_; }

    /// Comparison operators
    bool operator!=(const chebyshev_mesh &mesh) const {
      throw_if_empty();
      return !(*this == mesh);
    }

    void save(alps::hdf5::archive &ar, const std::string &path) const {
      throw_if_empty();
      ar[path + "/kind"] << "IMAGINARY_TIME_CHEBYSHEV";
      ar[path + "/k"] << k_;
      ar[path + "/statistics"] << int(statistics_); //
      ar[path + "/beta"] << beta_;
      ar[path + "/points"] << zeros_;
    }

    void load(alps::hdf5::archive &ar, const std::string &path) {
      std::string kind;
      int stat;
      ar[path + "/kind"] >> kind;
      if (kind != "IMAGINARY_TIME_CHEBYSHEV")
        throw std::runtime_error("Attempt to read Chebyshev mesh from non-Chebyshev data, kind=" + kind); // FIXME: specific exception
      double beta;
      int k;

      ar[path + "/k"] >> k;
      ar[path + "/statistics"] >> stat;
      ar[path + "/beta"] >> beta;

      statistics_ = alps::gf::statistics::statistics_type(stat); // FIXME: check range
      beta_ = beta;
      k_ = k;
      compute_zeros();
    }

    /// Save to HDF5
    void save(alps::hdf5::archive &ar) const {
      save(ar, ar.get_context());
    }

    /// Load from HDF5
    void load(alps::hdf5::archive &ar) {
      load(ar, ar.get_context());
    }

#ifdef ALPS_HAVE_MPI
    void broadcast(const alps::mpi::communicator& comm, int root)
    {
      using alps::mpi::broadcast;
      throw_if_empty();
      // FIXME: introduce (debug-only?) consistency check, like type checking? akin to load()?
      broadcast(comm, beta_, root);
      broadcast(comm, k_, root);
      int stat=statistics_;
      broadcast(comm, stat, root);
      statistics_=static_cast<alps::gf::statistics::statistics_type>(stat);
      compute_zeros();
    }
#endif

    void compute_zeros() {
      zeros_.resize(k_);
      for (int i = 0; i < k_; ++i) {
        double z = cos((2 * i + 1) * M_PI / (2 * k_));
        zeros_[i] = z * beta_ / 2. + beta_ / 2.;
      }

    }
  };

  ///Stream output operator, e.g. for printing to file
  std::ostream &operator<<(std::ostream &os, const chebyshev_mesh &M);
}}