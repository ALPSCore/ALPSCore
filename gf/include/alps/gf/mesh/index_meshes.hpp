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
  class momentum_realspace_index_mesh {
  public:
    typedef alps::numerics::tensor<double,2> container_type;
    momentum_realspace_index_mesh(const momentum_realspace_index_mesh& rhs) : points_(rhs.points_.shape()[0], rhs.points_.shape()[1]),
                                                                              kind_(rhs.kind_){
      points_ = rhs.points_;
    }
    momentum_realspace_index_mesh& operator=(const momentum_realspace_index_mesh& rhs) {
      points_.reshape(rhs.points_.shape());
      points_ = rhs.points_;
      kind_   = rhs.kind_;
      return *this;
    }
  protected:
    container_type points_;
  private:
    std::string kind_;

    inline void throw_if_empty() const {
      if (extent() == 0) {
        throw std::runtime_error("momentum_realspace_index_mesh is empty");
      }
    }

  protected:
    momentum_realspace_index_mesh(): points_(0, 0), kind_("")
    {
    }

    momentum_realspace_index_mesh(const std::string& kind, int ns,int ndim): points_(ns, ndim), kind_(kind)
    {
    }

    momentum_realspace_index_mesh(const std::string& kind, const container_type& mesh_points): points_(mesh_points), kind_(kind)
    {
    }

  public:
    /// Returns the number of points
    int extent() const { return points_.shape()[0];}
    ///returns the spatial dimension
    int dimension() const { return points_.shape()[1];}
    ///returns the mesh kind
    const std::string &kind() const{return kind_;}

    /// Comparison operators
    bool operator==(const momentum_realspace_index_mesh &mesh) const {
      throw_if_empty();
      return kind_ == mesh.kind_ &&
             points_ == mesh.points_;
    }

    /// Comparison operators
    bool operator!=(const momentum_realspace_index_mesh &mesh) const {
      throw_if_empty();
      return !(*this==mesh);
    }

    const container_type &points() const{return points_;}
    container_type &points() {return points_;}

    void save(alps::hdf5::archive& ar, const std::string& path) const
    {
      throw_if_empty();
      ar[path+"/kind"] << kind_;
      ar[path+"/points"] << points_;
    }

    void load(alps::hdf5::archive& ar, const std::string& path)
    {
      std::string kind;
      ar[path+"/kind"] >> kind;
      if (kind!=kind_) throw std::runtime_error("Attempt to load momentum/realspace index mesh from incorrect mesh kind="+kind+ " (expected: "+kind_+")");
      ar[path+"/points"] >> points_;
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
      if (comm.rank()==root) {
        throw_if_empty();
      }
      // FIXME: introduce (debug-only?) consistency check, like type checking? akin to load()?
      std::array<size_t, 2> sizes{{points_.shape()[0], points_.shape()[1]}};
      alps::mpi::broadcast(comm, &sizes[0], 2, root);
      if (comm.rank()!=root) points_.reshape(sizes);
      detail::broadcast(comm, points_, root);
      broadcast(comm, kind_, root);
    }
#endif

  };
  ///Stream output operator, e.g. for printing to file
  std::ostream &operator<<(std::ostream &os, const momentum_realspace_index_mesh &M);


  class momentum_index_mesh: public momentum_realspace_index_mesh {
    typedef momentum_realspace_index_mesh base_type;

  public:

    typedef generic_index<momentum_index_mesh> index_type;
    momentum_index_mesh(const momentum_index_mesh& rhs) : base_type(rhs) {}

    momentum_index_mesh& operator=(const momentum_index_mesh& rhs) {
      base_type::operator=(rhs);
      return *this;
    }

    momentum_index_mesh(): base_type("MOMENTUM_INDEX",0,0)
    {
    }

    momentum_index_mesh(int ns,int ndim): base_type("MOMENTUM_INDEX",ns,ndim)
    {
    }

    momentum_index_mesh(const container_type& mesh_points): base_type("MOMENTUM_INDEX",mesh_points)
    {
    }

    /// Returns the index of the mesh point in the data array
    int operator()(index_type idx) const { return idx(); }
  };

  class real_space_index_mesh: public momentum_realspace_index_mesh {
    typedef momentum_realspace_index_mesh base_type;

  public:

    typedef generic_index<momentum_index_mesh> index_type;
    real_space_index_mesh(const real_space_index_mesh& rhs) : base_type(rhs) {}

    real_space_index_mesh(): base_type("REAL_SPACE_INDEX",0,0)
    {
    }

    real_space_index_mesh(int ns,int ndim): base_type("REAL_SPACE_INDEX",ns,ndim)
    {
    }

    real_space_index_mesh(const container_type& mesh_points): base_type("REAL_SPACE_INDEX",mesh_points)
    {
    }

    /// Returns the index of the mesh point in the data array
    int operator()(index_type idx) const { return idx(); }
  };

  class index_mesh {
    int npoints_;
    std::vector<int> points_;
  public:
    typedef generic_index<index_mesh> index_type;

    index_mesh(const index_mesh& rhs) : npoints_(rhs.npoints_) {compute_points();}
    index_mesh& operator=(const index_mesh& rhs) {npoints_ = rhs.npoints_; compute_points();return * this;}
    index_mesh(): npoints_(0) { compute_points();}
    index_mesh(int np): npoints_(np) { compute_points();}
    int extent() const{return npoints_;}
    int operator()(index_type idx) const { return idx(); }
    const std::vector<int> &points() const{return points_;}

    inline void throw_if_empty() const {
      if (extent() == 0) {
        throw std::runtime_error("index_mesh is empty");
      }
    }

    /// Comparison operators
    bool operator==(const index_mesh &mesh) const {
      throw_if_empty();
      return npoints_==mesh.npoints_;
    }

    /// Comparison operators
    bool operator!=(const index_mesh &mesh) const {
      throw_if_empty();
      return !(*this==mesh);
    }

    void save(alps::hdf5::archive& ar, const std::string& path) const
    {
      throw_if_empty();
      ar[path+"/kind"] << "INDEX";
      ar[path+"/N"] << npoints_;
    }

    void load(alps::hdf5::archive& ar, const std::string& path)
    {
      std::string kind;
      ar[path+"/kind"] >> kind;
      if (kind!="INDEX") throw std::runtime_error("Attempt to read Index mesh from non-Index data, kind="+kind); // FIXME: specific exception

      int np;
      ar[path+"/N"] >> np;
      npoints_=np;
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

    void compute_points(){ points_.resize(npoints_); for(int i=0;i<npoints_;++i){points_[i]=i;} }

#ifdef ALPS_HAVE_MPI
    void broadcast(const alps::mpi::communicator& comm, int root)
    {
      using alps::mpi::broadcast;
      if(comm.rank() == root) throw_if_empty();
      broadcast(comm, npoints_, root);
      compute_points();
    }
#endif
  };
  ///Stream output operator, e.g. for printing to file
  std::ostream &operator<<(std::ostream &os, const index_mesh &M);

  typedef momentum_index_mesh::index_type momentum_index;
  typedef real_space_index_mesh::index_type real_space_index;
  typedef index_mesh::index_type index;
}}

