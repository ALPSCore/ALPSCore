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


  class legendre_mesh : public base_mesh {
    double beta_;
    int n_max_;//number of legendre polynomials

    statistics::statistics_type statistics_;

    inline void throw_if_empty() const {
      if (extent() == 0) {
        throw std::runtime_error("legendre_mesh is empty");
      }
    }

  public:
    typedef generic_index<legendre_mesh> index_type;
    legendre_mesh(const legendre_mesh& rhs) : beta_(rhs.beta_), n_max_(rhs.n_max_), statistics_(rhs.statistics_) {compute_points();}
    legendre_mesh(gf::statistics::statistics_type statistics=statistics::FERMIONIC):
        beta_(0.0), n_max_(0), statistics_(statistics) {}

    legendre_mesh(double b, int n_max, gf::statistics::statistics_type statistics=statistics::FERMIONIC):
        beta_(b), n_max_(n_max), statistics_(statistics) {
      check_range();
      compute_points();
    }
    int extent() const{return n_max_;}

    int operator()(index_type idx) const {
#ifndef NDEBUG
      throw_if_empty();
#endif
      return idx();
    }

    /// Comparison operators
    bool operator==(const legendre_mesh &mesh) const {
      throw_if_empty();
      return beta_==mesh.beta_ && n_max_==mesh.n_max_ && statistics_==mesh.statistics_;
    }

    /// Comparison operators
    bool operator!=(const legendre_mesh &mesh) const {
      throw_if_empty();
      return !(*this==mesh);
    }

    ///getter functions for member variables
    double beta() const{ return beta_;}
    statistics::statistics_type statistics() const{ return statistics_;}

    /// Swaps this and another mesh
    // It's a member function to avoid dealing with templated friend decalration.
    void swap(legendre_mesh& other) {
      using std::swap;
      throw_if_empty();
      if (this->statistics_ != other.statistics_) {
        throw std::runtime_error("Attempt to swap LEGENDRE meshes with different statistics");
      }
      swap(this->beta_, other.beta_);
      swap(this->n_max_, other.n_max_);
      base_mesh::swap(other);
    }

    void save(alps::hdf5::archive& ar, const std::string& path) const
    {
      throw_if_empty();
      ar[path+"/kind"] << "LEGENDRE";
      ar[path+"/N"] << n_max_;
      ar[path+"/statistics"] << int(statistics_);
      ar[path+"/beta"] << beta_;
    }

    void load(alps::hdf5::archive& ar, const std::string& path)
    {
      std::string kind;
      ar[path+"/kind"] >> kind;
      if (kind!="LEGENDRE") throw std::runtime_error("Attempt to read LEGENDRE mesh from non-LEGENDRE data, kind="+kind); // FIXME: specific exception
      double n_max, beta;
      int stat;

      ar[path+"/N"] >> n_max;
      ar[path+"/statistics"] >> stat;
      ar[path+"/beta"] >> beta;

      statistics_=statistics::statistics_type(stat);
      beta_=beta;
      n_max_=n_max;
      check_range();
      compute_points();
    }

    /// Save to HDF5
    void save(alps::hdf5::archive& ar) const
    {
      save(ar, ar.get_context());
    }

    /// Load from HDF5
    void load(alps::hdf5::archive& ar)
    {
      load(ar, ar.get_context());
    }

#ifdef ALPS_HAVE_MPI
    void broadcast(const alps::mpi::communicator& comm, int root)
    {
      using alps::mpi::broadcast;
      if(comm.rank() == root) throw_if_empty();
      broadcast(comm, beta_, root);
      broadcast(comm, n_max_, root);
      {
        int stat = static_cast<int>(statistics_);
        broadcast(comm, stat, root);
        statistics_=statistics::statistics_type(stat);
      }

      try {
        check_range();
      } catch (const std::exception& exc) {
        // FIXME? Try to communiucate the error with all ranks, at least in debug mode?
        int wrank=alps::mpi::communicator().rank();
        std::cerr << "legendre_mesh<>::broadcast() exception at WORLD rank=" << wrank << std::endl
                  << exc.what()
                  << "\nAborting." << std::endl;
        MPI_Abort(MPI_COMM_WORLD,1);
      }
      compute_points(); // recompute points rather than sending them over MPI
    }
#endif

    void check_range(){
      if(statistics_!=statistics::FERMIONIC && statistics_!=statistics::BOSONIC) throw std::invalid_argument("statistics should be bosonic or fermionic");
    }
    void compute_points(){
      //This is sort of trivial in the current implementation.
      //We use P_0, P_1, P_2, ..., P_{n_max_-1}
      _points().resize(extent());
      for(int i=0;i<n_max_;++i){
        _points()[i]=i;
      }
    }
  };
  ///Stream output operator, e.g. for printing to file
  std::ostream &operator<<(std::ostream &os, const legendre_mesh &M);


  typedef legendre_mesh::index_type legendre_index;

}}

