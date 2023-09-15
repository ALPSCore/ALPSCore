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

  class itime_mesh {
    double beta_;
    int ntau_;
    bool last_point_included_;
    bool half_point_mesh_;
    statistics::statistics_type statistics_;
    std::vector<double> points_;

    inline void throw_if_empty() const {
      if (extent() == 0) {
        throw std::runtime_error("itime_mesh is empty");
      }
    }

  public:
    typedef generic_index<itime_mesh> index_type;

    itime_mesh(): beta_(0.0), ntau_(0), last_point_included_(true), half_point_mesh_(false), statistics_(statistics::FERMIONIC)
    {
    }

    itime_mesh(const itime_mesh&rhs): beta_(rhs.beta_), ntau_(rhs.ntau_),
                                      last_point_included_(rhs.last_point_included_), half_point_mesh_(rhs.half_point_mesh_), statistics_(rhs.statistics_){
      compute_points();

    }

    itime_mesh(double beta, int ntau): beta_(beta), ntau_(ntau), last_point_included_(true), half_point_mesh_(false), statistics_(statistics::FERMIONIC){
      compute_points();

    }
    int operator()(index_type idx) const {
#ifndef NDEBUG
      throw_if_empty();
#endif
      return idx();
    }
    int extent() const{return ntau_;}

    /// Comparison operators
    bool operator==(const itime_mesh &mesh) const {
      throw_if_empty();
      return beta_==mesh.beta_ && ntau_==mesh.ntau_ && last_point_included_==mesh.last_point_included_ &&
             half_point_mesh_ == mesh.half_point_mesh_ && statistics_==mesh.statistics_;
    }

    ///Getter variables for members
    double beta() const{ return beta_;}
    statistics::statistics_type statistics() const{ return statistics_;}
    const std::vector<double> &points() const{return points_;}

    /// Comparison operators
    bool operator!=(const itime_mesh &mesh) const {
      throw_if_empty();
      return !(*this==mesh);
    }

    void save(alps::hdf5::archive& ar, const std::string& path) const
    {
      throw_if_empty();
      ar[path+"/kind"] << "IMAGINARY_TIME";
      ar[path+"/N"] << ntau_;
      ar[path+"/statistics"] << int(statistics_); //
      ar[path+"/beta"] << beta_;
      ar[path+"/half_point_mesh"] << int(half_point_mesh_);
      ar[path+"/last_point_included"] << int(last_point_included_);
      ar[path+"/points"] << points_;
    }

    void load(alps::hdf5::archive& ar, const std::string& path)
    {
      std::string kind;
      ar[path+"/kind"] >> kind;
      if (kind!="IMAGINARY_TIME") throw std::runtime_error("Attempt to read Imaginary time mesh from non-itime data, kind="+kind); // FIXME: specific exception
      double ntau, beta;
      int stat, half_point_mesh, last_point_included;

      ar[path+"/N"] >> ntau;
      ar[path+"/statistics"] >> stat;
      ar[path+"/beta"] >> beta;
      ar[path+"/half_point_mesh"] >> half_point_mesh;
      ar[path+"/last_point_included"] >> last_point_included;

      statistics_=statistics::statistics_type(stat); // FIXME: check range
      half_point_mesh_=half_point_mesh;
      last_point_included_=last_point_included;
      beta_=beta;
      ntau_=ntau;
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
      broadcast(comm, ntau_, root);
      broadcast(comm, last_point_included_, root);
      broadcast(comm, half_point_mesh_, root);
      int stat=statistics_;
      broadcast(comm, stat, root);
      statistics_=static_cast<statistics::statistics_type>(stat);
      compute_points(); // recompute points rather than sending them over MPI
    }
#endif

    void compute_points(){
      points_.resize(extent());
      if(half_point_mesh_){
        double dtau=beta_/ntau_;
        for(int i=0;i<ntau_;++i){
          points_[i]=(i+0.5)*dtau;
        }
      }
      for(int i=0;i<ntau_;++i){
        double dtau=last_point_included_?beta_/(ntau_-1):beta_/ntau_;
        for(int i=0;i<ntau_;++i){
          points_[i]=i*dtau;
        }
      }
    }
  };
  ///Stream output operator, e.g. for printing to file
  std::ostream &operator<<(std::ostream &os, const itime_mesh &M);

  class power_mesh {
    double beta_;
    int ntau_;
    int power_;
    int uniform_;

    statistics::statistics_type statistics_;
    std::vector<double> points_;
    std::vector<double> weights_;

    inline void throw_if_empty() const {
      if (extent() == 0) {
        throw std::runtime_error("power_mesh is empty");
      }
    }

  public:
    typedef generic_index<power_mesh> index_type;

    power_mesh(const power_mesh& rhs): beta_(rhs.beta_), power_(rhs.power_), uniform_(rhs.uniform_), statistics_(rhs.statistics_){
      compute_points();
      compute_weights();
    }

    power_mesh(): beta_(0.0), ntau_(0), power_(0), uniform_(0), statistics_(statistics::FERMIONIC){
    }

    power_mesh(double beta, int power, int uniform): beta_(beta), power_(power), uniform_(uniform), statistics_(statistics::FERMIONIC){
      compute_points();
      compute_weights();
    }
    int operator()(index_type idx) const {
#ifndef NDEBUG
      throw_if_empty();
#endif
      return idx();
    }
    int extent() const{return ntau_;}
    int power() const{return power_;}
    int uniform() const{return uniform_;}

    /// Comparison operators
    bool operator==(const power_mesh &mesh) const {
      throw_if_empty();
      return beta_==mesh.beta_ && ntau_==mesh.ntau_ && power_==mesh.power_ &&
             uniform_ == mesh.uniform_ && statistics_==mesh.statistics_;
    }

    ///Getter variables for members
    double beta() const{ return beta_;}
    statistics::statistics_type statistics() const{ return statistics_;}
    const std::vector<double> &points() const{return points_;}
    const std::vector<double> &weights() const{return weights_;}

    /// Comparison operators
    bool operator!=(const power_mesh &mesh) const {
      throw_if_empty();
      return !(*this==mesh);
    }

    void save(alps::hdf5::archive& ar, const std::string& path) const
    {
      throw_if_empty();
      ar[path+"/kind"] << "IMAGINARY_TIME_POWER";
      ar[path+"/N"] << ntau_;
      ar[path+"/statistics"] << int(statistics_); //
      ar[path+"/beta"] << beta_;
      ar[path+"/power"] << power_;
      ar[path+"/uniform"] << uniform_;
      ar[path+"/points"] << points_;
    }

    void load(alps::hdf5::archive& ar, const std::string& path)
    {
      std::string kind;
      ar[path+"/kind"] >> kind;
      if (kind!="IMAGINARY_TIME_POWER") throw std::runtime_error("Attempt to read Imaginary time power mesh from non-itime power data, kind="+kind); // FIXME: specific exception
      double ntau, beta;
      int stat, power, uniform;

      ar[path+"/N"] >> ntau;
      ar[path+"/statistics"] >> stat;
      ar[path+"/beta"] >> beta;
      ar[path+"/power"] >> power;
      ar[path+"/uniform"] >> uniform;

      statistics_=statistics::statistics_type(stat); // FIXME: check range
      power_=power;
      uniform_=uniform;
      beta_=beta;
      ntau_=ntau;
      compute_points();
      compute_weights();
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
      broadcast(comm, ntau_, root);
      broadcast(comm, power_, root);
      broadcast(comm, uniform_, root);
      int stat=statistics_;
      broadcast(comm, stat, root);
      statistics_=static_cast<statistics::statistics_type>(stat);
      compute_points(); // recompute points rather than sending them over MPI
      compute_weights();
    }
#endif

    void compute_points(){
      //create a power grid spacing
      if(uniform_%2 !=0) throw std::invalid_argument("Simpson weights in power grid only work for even uniform spacing.");
      std::vector<double> power_points;
      power_points.push_back(0);
      for(int i=power_;i>=0;--i){
        power_points.push_back(beta_*0.5*std::pow(2.,-i));
      }
      for(int i=power_;i>0;--i){
        power_points.push_back(beta_*(1.-0.5*std::pow(2.,-i)));
      }
      power_points.push_back(beta_);
      std::sort(power_points.begin(),power_points.end());

      //create the uniform grid within each power grid
      points_.resize(0);
      for(std::size_t i=0;i<power_points.size()-1;++i){
        for(int j=0;j<uniform_;++j){
          double dtau=(power_points[i+1]-power_points[i])/(double)(uniform_);
          points_.push_back(power_points[i]+dtau*j);
        }
      }
      points_.push_back(power_points.back());
      ntau_=points_.size();
    }
    void compute_weights(){
      weights_.resize(extent());
      weights_[0        ]=(points_[1]    -points_[0        ])/(2.*beta_);
      weights_[extent()-1]=(points_.back()-points_[extent()-2])/(2.*beta_);

      for(int i=1;i<extent()-1;++i){
        weights_[i]=(points_[i+1]-points_[i-1])/(2.*beta_);
      }
    }
  };
  ///Stream output operator, e.g. for printing to file
  std::ostream &operator<<(std::ostream &os, const power_mesh &M);

  namespace detail{
    /// Trait: Imaginary time meshes can have tails
    template <> struct can_have_tail<itime_mesh>: public can_have_tail_yes {};
  }

  typedef itime_mesh::index_type itime_index;

}}

