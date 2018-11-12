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
  namespace mesh {
    enum frequency_positivity_type {
      POSITIVE_NEGATIVE=0,
      POSITIVE_ONLY=1
    };
  }

/// Mesh of real frequencies
  class real_frequency_mesh: public base_mesh{
    inline void throw_if_empty() const {
      if (extent() == 0) {
        throw std::runtime_error("real_frequency_mesh is empty");
      }
    }
  public:
    typedef generic_index<real_frequency_mesh> index_type;
    real_frequency_mesh() {};
    real_frequency_mesh(const real_frequency_mesh& rhs) : base_mesh(rhs) {}

    template<typename GRID>
    explicit real_frequency_mesh(const GRID & grid)  {
      grid.compute_points(_points());
    }
    int extent() const {return points().size();}

    int operator()(index_type idx) const {
#ifndef NDEBUG
      throw_if_empty();
#endif
      return idx();
    }
    void save(alps::hdf5::archive& ar, const std::string& path) const {
      throw_if_empty();
      ar[path+"/kind"] << "REAL_FREQUENCY";
      ar[path+"/points"] << points();
    }

    void load(alps::hdf5::archive& ar, const std::string& path) {
      std::string kind;
      ar[path+"/kind"]   >> kind;
      if (kind!="REAL_FREQUENCY") throw std::runtime_error("Attempt to read real frequency mesh from non-real frequency data, kind="+kind);
      ar[path+"/points"] >> _points();
    }

    /// Save to HDF5
    void save(alps::hdf5::archive& ar) const {
      save(ar, ar.get_context());
    }

    /// Load from HDF5
    void load(alps::hdf5::archive& ar) {
      load(ar, ar.get_context());
    }

    /// Comparison operators
    bool operator==(const real_frequency_mesh &mesh) const {
      throw_if_empty();
      return extent()==mesh.extent() && std::equal ( mesh.points().begin(), mesh.points().end(), points().begin() );;
    }

    /// Comparison operators
    bool operator!=(const real_frequency_mesh &mesh) const {
      throw_if_empty();
      return !(*this==mesh);
    }
#ifdef ALPS_HAVE_MPI
    void broadcast(const alps::mpi::communicator& comm, int root) {
      using alps::mpi::broadcast;
      if(comm.rank() == root) throw_if_empty();
      int size = extent();
      broadcast(comm, size, root);
      /// since real frequency mesh can be generated differently we should broadcast points
      if(root!=comm.rank()) {
        /// adjust target array size
        _points().resize(size);
      }
      broadcast(comm, _points().data(), extent(), root);
    }
#endif
    };
    std::ostream &operator<<(std::ostream &os, const real_frequency_mesh &M);

  template <mesh::frequency_positivity_type PTYPE>
  class matsubara_mesh : public base_mesh {
    double beta_;
    int nfreq_;

    statistics::statistics_type statistics_;
    static const mesh::frequency_positivity_type positivity_=PTYPE;

    //FIXME: we had this const, but that prevents copying.
    int offset_;

    inline void throw_if_empty() const {
      if (extent() == 0) {
        throw std::runtime_error("matsubara_mesh is empty");
      }
    }

  public:
    typedef generic_index<matsubara_mesh> index_type;
    /// copy constructor
    matsubara_mesh(const matsubara_mesh& rhs) : beta_(rhs.beta_), nfreq_(rhs.nfreq_), statistics_(rhs.statistics_), offset_(rhs.offset_) {check_range();compute_points();}
    matsubara_mesh():
        beta_(0.0), nfreq_(0), statistics_(statistics::FERMIONIC), offset_(-1)
    {
    }

    matsubara_mesh(double b, int nfr, gf::statistics::statistics_type statistics=statistics::FERMIONIC):
        beta_(b), nfreq_(nfr), statistics_(statistics), offset_((PTYPE==mesh::POSITIVE_ONLY)?0:nfr/2) {
      check_range();
      compute_points();
    }
    int extent() const{return nfreq_;}


    int operator()(index_type idx) const {
#ifndef NDEBUG
      throw_if_empty();
#endif
      return offset_+idx(); // FIXME: can be improved by specialization?
    }

    /// Comparison operators
    bool operator==(const matsubara_mesh &mesh) const {
      throw_if_empty();
      return beta_==mesh.beta_ && nfreq_==mesh.nfreq_ && statistics_==mesh.statistics_ && offset_ == mesh.offset_;
    }

    /// Comparison operators
    bool operator!=(const matsubara_mesh &mesh) const {
      throw_if_empty();
      return !(*this==mesh);
    }

    ///getter functions for member variables
    double beta() const{ return beta_;}
    statistics::statistics_type statistics() const{ return statistics_;}
    mesh::frequency_positivity_type positivity() const{ return positivity_;}

    /// Swaps this and another mesh
    // It's a member function to avoid dealing with templated friend decalration.
    void swap(matsubara_mesh& other) {
      throw_if_empty();
      if(statistics_!=other.statistics_)
        throw std::runtime_error("Attempt to swap two meshes with different statistics.");// FIXME: specific exception
      using std::swap;
      swap(this->beta_, other.beta_);
      swap(this->nfreq_, other.nfreq_);
      base_mesh::swap(other);
    }

    void save(alps::hdf5::archive& ar, const std::string& path) const
    {
      throw_if_empty();
      ar[path+"/kind"] << "MATSUBARA";
      ar[path+"/N"] << nfreq_;
      ar[path+"/statistics"] << int(statistics_); //
      ar[path+"/beta"] << beta_;
      ar[path+"/positive_only"] << int(positivity_);
      ar[path+"/points"] << points();
    }

    void load(alps::hdf5::archive& ar, const std::string& path)
    {
      std::string kind;
      ar[path+"/kind"] >> kind;
      if (kind!="MATSUBARA") throw std::runtime_error("Attempt to read Matsubara mesh from non-Matsubara data, kind="+kind); // FIXME: specific exception
      double nfr, beta;
      int stat, posonly;

      ar[path+"/N"] >> nfr;
      ar[path+"/statistics"] >> stat;
      ar[path+"/beta"] >> beta;
      ar[path+"/positive_only"] >> posonly;

      statistics_=statistics::statistics_type(stat);
      if (mesh::frequency_positivity_type(posonly)!=positivity_) {
        throw std::invalid_argument("Attempt to read Matsubara mesh with the wrong positivity type "+std::to_string(posonly) ); // FIXME: specific exception? Verbose positivity?
      };
      beta_=beta;
      nfreq_=nfr;
      offset_ = ((PTYPE==mesh::POSITIVE_ONLY)?0:nfr/2);
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
      // FIXME: introduce (debug-only?) consistency check, like type checking? akin to load()?
      broadcast(comm, beta_, root);
      broadcast(comm, nfreq_, root);
      int stat = int(statistics_);
      broadcast(comm, stat, root);
      statistics_ = statistics::statistics_type(stat);
      int pos = int(positivity_);
      broadcast(comm, pos, root);
      if (mesh::frequency_positivity_type(pos)!=positivity_) {
        throw std::invalid_argument("Attempt to broadcast Matsubara mesh with the wrong positivity type "+std::to_string(pos) ); // FIXME: specific exception? Verbose positivity?
      };
      offset_ = ((PTYPE==mesh::POSITIVE_ONLY)?0:nfreq_/2);

      try {
        check_range();
      } catch (const std::exception& exc) {
        int wrank=alps::mpi::communicator().rank();
        // FIXME? Try to communicate the error with all ranks, at least in debug mode?
        std::cerr << "matsubara_mesh<>::broadcast() exception at WORLD rank=" << wrank << std::endl
                  << exc.what()
                  << "\nAborting." << std::endl;
        MPI_Abort(MPI_COMM_WORLD,1);
      }
      compute_points(); // recompute points rather than sending them over MPI
    }
#endif

    void check_range(){
      if(statistics_!=statistics::FERMIONIC && statistics_!=statistics::BOSONIC) throw std::invalid_argument("statistics should be bosonic or fermionic");
      if(positivity_!=mesh::POSITIVE_ONLY &&
         positivity_!=mesh::POSITIVE_NEGATIVE) {
        throw std::invalid_argument("positivity should be POSITIVE_ONLY or POSITIVE_NEGATIVE");
      }
      throw_if_empty();
    }
    void compute_points(){
      throw_if_empty();
      _points().resize(extent());
      for(int i=0;i<nfreq_;++i){
        _points()[i]=(2*(i-offset_)+statistics_)*M_PI/beta_;
      }
    }
  };
  ///Stream output operator, e.g. for printing to file
  template<mesh::frequency_positivity_type PTYPE> std::ostream &operator<<(std::ostream &os, const matsubara_mesh<PTYPE> &M){
    os<<"# "<<"MATSUBARA"<<" mesh: N: "<<M.extent()<<" beta: "<<M.beta()<<" statistics: ";
    os<<(M.statistics()==statistics::FERMIONIC?"FERMIONIC":"BOSONIC")<<" ";
    os<<(M.positivity()==mesh::POSITIVE_ONLY?"POSITIVE_ONLY":"POSITIVE_NEGATIVE");
    os<<std::endl;
    return os;
  }

  /// Swaps two Matsubara meshes
  template <mesh::frequency_positivity_type PTYPE>
  void swap(matsubara_mesh<PTYPE>& a, matsubara_mesh<PTYPE>& b) {
    a.swap(b);
  }
  typedef matsubara_mesh<mesh::POSITIVE_ONLY> matsubara_positive_mesh;
  typedef matsubara_mesh<mesh::POSITIVE_NEGATIVE> matsubara_pn_mesh;
  typedef matsubara_mesh<mesh::POSITIVE_ONLY>::index_type matsubara_index;
  typedef matsubara_mesh<mesh::POSITIVE_NEGATIVE>::index_type matsubara_pn_index;
  typedef real_frequency_mesh::index_type real_freq_index;

  namespace detail{
    /// Trait: Matsubara meshes can have tails
    template <> struct can_have_tail<matsubara_positive_mesh>: public can_have_tail_yes {};
    /// Trait: Matsubara meshes can have tails
    template <> struct can_have_tail<matsubara_pn_mesh>: public can_have_tail_yes {};
  }

}}
