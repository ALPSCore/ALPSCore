/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#ifdef ALPS_HAVE_MPI
#include <alps/gf/mpi_bcast.hpp>
#endif
#include <alps/gf/piecewise_polynomial.hpp>
#include <alps/gf/mesh/index.hpp>
#include <alps/gf/mesh/mesh_base.hpp>

namespace alps {namespace gf {

  /**
   * Numerical mesh (T = double or std::complex<double>)
   */
  template<class T>
  class numerical_mesh : public base_mesh {
    double beta_;
    int dim_;//dimension of the basis

    statistics::statistics_type statistics_;

    std::vector<piecewise_polynomial<T> > basis_functions_;

    bool valid_;

    void set_validity() {
      valid_ = true;
      valid_ = valid_ && dim_ > 0;
      valid_ = valid_ && std::size_t(dim_) == basis_functions_.size();
      valid_ = valid_ && beta_ >= 0.0;
      valid_ = valid_ && (statistics_==statistics::FERMIONIC || statistics_==statistics::BOSONIC);

      if (basis_functions_.size() > 1u) {
        for (std::size_t l=0; l < basis_functions_.size()-1; ++l) {
          valid_ = valid_ && (basis_functions_[l].section_edges() == basis_functions_[l+1].section_edges());
        }
      }
    }

    void check_validity() const {
      if (!valid_) {
        throw std::runtime_error("numerical mesh has not been properly constructed!");
      }
    }

  public:
    typedef generic_index<numerical_mesh> index_type;
    numerical_mesh(const numerical_mesh<T>& rhs) : beta_(rhs.beta_), dim_(rhs.dim_), basis_functions_(rhs.basis_functions_), statistics_(rhs.statistics_), valid_(rhs.valid_) {
      set_validity();
      compute_points();
    }
    numerical_mesh(gf::statistics::statistics_type statistics=statistics::FERMIONIC):
        beta_(0.0), dim_(0), statistics_(statistics), basis_functions_(), valid_(false) {}

    numerical_mesh(double b,  const std::vector<piecewise_polynomial<T> >&basis_functions,
                   gf::statistics::statistics_type statistics=statistics::FERMIONIC):
        beta_(b), dim_(basis_functions.size()), statistics_(statistics), basis_functions_(basis_functions), valid_(false) {
      set_validity();
      //check_range();
      compute_points();
    }

    int extent() const{
      check_validity();
      return dim_;
    }

    int operator()(index_type idx) const {
      check_validity();
      return idx();
    }

    /// Comparison operators
    bool operator==(const numerical_mesh &mesh) const {
      check_validity();
      return beta_==mesh.beta_ && dim_==mesh.dim_ && statistics_==mesh.statistics_ && basis_functions_==mesh.basis_functions_;
    }

    /// Comparison operators
    bool operator!=(const numerical_mesh &mesh) const {
      check_validity();
      return !(*this==mesh);
    }

    ///getter functions for member variables
    double beta() const{ return beta_;}
    statistics::statistics_type statistics() const{ return statistics_;}
    const piecewise_polynomial<T>& basis_function(int l) const {
      assert(l>=0 && l < dim_);
      check_validity();
      return basis_functions_[l];
    }


    /// Swaps this and another mesh
    // It's a member function to avoid dealing with templated friend decalration.
    void swap(numerical_mesh& other) {
      using std::swap;
      check_validity();
      swap(this->beta_, other.beta_);
      swap(this->dim_, other.dim_);
      if (this->statistics_ != other.statistics_) {
        throw std::runtime_error("Do not swap numerical meshes with different statistics!");
      }
      swap(this->basis_functions_, other.basis_functions_);
      base_mesh::swap(other);
    }

    numerical_mesh& operator=(const numerical_mesh& other) {
      this->beta_ = other.beta_;
      this->dim_ = other.dim_;
      this->statistics_ = other.statistics_;
      this->basis_functions_ = other.basis_functions_;
      base_mesh::operator=(other);
    }

    void save(alps::hdf5::archive& ar, const std::string& path) const
    {
      check_validity();
      ar[path+"/kind"] << "NUMERICAL_MESH";
      ar[path+"/N"] << dim_;
      ar[path+"/statistics"] << int(statistics_);
      ar[path+"/beta"] << beta_;
      for (int l=0; l < dim_; ++l) {
        basis_functions_[l].save(ar, path+"/basis_functions"+std::to_string(l));
      }
    }

    void load(alps::hdf5::archive& ar, const std::string& path)
    {
      std::string kind;
      ar[path+"/kind"] >> kind;
      if (kind!="NUMERICAL_MESH") throw std::runtime_error("Attempt to read NUMERICAL_MESH mesh from non-numerical-mesh data, kind="+kind); // FIXME: specific exception
      double dim, beta;
      int stat;

      ar[path+"/N"] >> dim;
      ar[path+"/statistics"] >> stat;
      if (valid_ && stat != statistics_) {
        throw std::runtime_error("Attemp to load data with different statistics!");
      }

      ar[path+"/beta"] >> beta;
      basis_functions_.resize(dim);
      for (int l=0; l < dim; ++l) {
        basis_functions_[l].load(ar, path+"/basis_functions"+std::to_string(l));
      }

      statistics_ = static_cast<statistics::statistics_type>(stat);
      beta_=beta;
      dim_=dim;
      set_validity();
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
      if(comm.rank() == root) check_validity();

      broadcast(comm, beta_, root);
      broadcast(comm, dim_, root);
      {
        int stat = static_cast<int>(statistics_);
        broadcast(comm, stat, root);
        statistics_=statistics::statistics_type(stat);
      }

      basis_functions_.resize(dim_);
      for (int l=0; l < dim_; ++l) {
        basis_functions_[l].broadcast(comm, root);
      }

      set_validity();
      try {
        check_validity();
      } catch (const std::exception& exc) {
        // FIXME? Try to communiucate the error with all ranks, at least in debug mode?
        int wrank=alps::mpi::communicator().rank();
        std::cerr << "numerical_mesh<>::broadcast() exception at WORLD rank=" << wrank << std::endl
                  << exc.what()
                  << "\nAborting." << std::endl;
        MPI_Abort(MPI_COMM_WORLD,1);
      }
      compute_points(); // recompute points rather than sending them over MPI
    }
#endif

    void compute_points(){
      _points().resize(extent());
      for(int i=0;i<dim_;++i){
        _points()[i]=i;
      }
    }
  };

  typedef numerical_mesh<double>::index_type numerical_mesh_index;

}}