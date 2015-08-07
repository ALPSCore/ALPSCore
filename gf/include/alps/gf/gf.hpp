#pragma once
#include <complex>
#include <boost/multi_array.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/hdf5/multi_array.hpp>

#include "mesh.hpp"


namespace alps {
    namespace gf {
        
        /// Matsubara GF(omega, k1_2d, k2_2d, spin)
        class matsubara_gf {
            static const int minor_version_=1;
            static const int major_version_=0;
            typedef std::complex<double> value_type;
            typedef boost::multi_array<value_type,4> container_type;

            matsubara_mesh mesh1_;
            momentum_index_mesh mesh2_;
            momentum_index_mesh mesh3_;
            index_mesh mesh4_;

            container_type data_;
          public:
            matsubara_gf(const matsubara_mesh& mesh1,
                         const momentum_index_mesh& mesh2,
                         const momentum_index_mesh& mesh3,
                         const index_mesh& mesh4)
                : mesh1_(mesh1), mesh2_(mesh2), mesh3_(mesh3), mesh4_(mesh4),
                  data_(boost::extents[mesh1_.extent()][mesh2_.extent()][mesh3_.extent()][mesh4_.extent()])
            {
            }
            
            const value_type& operator()(matsubara_index omega, momentum_index i, momentum_index j, index sigma) const
            {
                return data_[omega()][i()][j()][sigma()];
            }

            value_type& operator()(matsubara_index omega, momentum_index i, momentum_index j, index sigma)
            {
                return data_[omega()][i()][j()][sigma()];
            }

            /// Initialize the GF data to value_type(0.)
            void initialize()
            {
                for (int i=0; i<mesh1_.extent(); ++i) {
                    for (int j=0; j<mesh2_.extent(); ++j) {
                        for (int k=0; k<mesh3_.extent(); ++k) {
                            for (int l=0; l<mesh4_.extent(); ++l) {
                                data_[i][j][k][l]=value_type(0.0);
                            }
                        }
                    }
                }
            }

            /// Save the GF to HDF5
            void save(alps::hdf5::archive& ar, const std::string& path) const
            {
                save_version(ar,path);
                ar[path+"/data"] << data_;
                ar[path+"/mesh/N"] << int(container_type::dimensionality);
                mesh1_.save(ar,path+"/mesh/1");
                mesh2_.save(ar,path+"/mesh/2");
                mesh3_.save(ar,path+"/mesh/3");
                mesh4_.save(ar,path+"/mesh/4");
            }

            /// Load the GF from HDF5
            void load(alps::hdf5::archive& ar, const std::string& path)
            {
                if (!check_version(ar,path)) throw std::runtime_error("Incompatible archive version");

                int ndim;
                ar[path+"/mesh/N"] >> ndim;
                if (ndim != container_type::dimensionality) throw std::runtime_error("Wrong number of dimension reading Matsubara GF, ndim="+ndim);

                mesh1_.load(ar,path+"/mesh/1");
                mesh2_.load(ar,path+"/mesh/2");
                mesh3_.load(ar,path+"/mesh/3");
                mesh4_.load(ar,path+"/mesh/4");

                data_.resize(boost::extents[mesh1_.extent()][mesh2_.extent()][mesh3_.extent()][mesh4_.extent()]);
                
                ar[path+"/data"] >> data_;
            }

            static void save_version(alps::hdf5::archive& ar, const std::string& path)
            {
                std::string vp=path+"/version/";
                ar[vp+"minor"]<< int(minor_version_);
                ar[vp+"major"]<< int(major_version_);
                ar[vp+"reference"]<<"https://github.com/ALPSCore/H5GF/blob/master/H5GF.rst";
                ar[vp+"originator"]<<"ALPSCore GF library, see http://www.alpscore.org";
            }

            static bool check_version(alps::hdf5::archive& ar, const std::string& path)
            {
                std::string vp=path+"/version/";
                int ver;
                ar[vp+"major"]>>ver;
                return (major_version_==ver);
            }
                
        };
        class itime_gf {
            static const int minor_version_=1;
            static const int major_version_=0;
            typedef double value_type;
            typedef boost::multi_array<value_type,4> container_type;

            itime_mesh mesh1_;
            momentum_index_mesh mesh2_;
            momentum_index_mesh mesh3_;
            index_mesh mesh4_;

            container_type data_;
          public:
            itime_gf(const itime_mesh& mesh1,
                         const momentum_index_mesh& mesh2,
                         const momentum_index_mesh& mesh3,
                         const index_mesh& mesh4)
                : mesh1_(mesh1), mesh2_(mesh2), mesh3_(mesh3), mesh4_(mesh4),
                  data_(boost::extents[mesh1_.extent()][mesh2_.extent()][mesh3_.extent()][mesh4_.extent()])
            {
            }
            
            const value_type & operator()(itime_index tau, momentum_index i, momentum_index j, index sigma) const
            {
                return data_[tau()][i()][j()][sigma()];
            }

            value_type & operator()(itime_index tau, momentum_index i, momentum_index j, index sigma)
            {
                return data_[tau()][i()][j()][sigma()];
            }

            /// Initialize the GF data to value_type(0.)
            void initialize()
            {
                for (int i=0; i<mesh1_.extent(); ++i) {
                    for (int j=0; j<mesh2_.extent(); ++j) {
                        for (int k=0; k<mesh3_.extent(); ++k) {
                            for (int l=0; l<mesh4_.extent(); ++l) {
                                data_[i][j][k][l]=value_type(0.0);
                            }
                        }
                    }
                }
            }

            /// Save the GF to HDF5
            void save(alps::hdf5::archive& ar, const std::string& path) const
            {
                save_version(ar,path);
                ar[path+"/data"] << data_;
                ar[path+"/mesh/N"] << int(container_type::dimensionality);
                mesh1_.save(ar,path+"/mesh/1");
                mesh2_.save(ar,path+"/mesh/2");
                mesh3_.save(ar,path+"/mesh/3");
                mesh4_.save(ar,path+"/mesh/4");
            }

            /// Load the GF from HDF5
            void load(alps::hdf5::archive& ar, const std::string& path)
            {
                if (!check_version(ar,path)) throw std::runtime_error("Incompatible archive version");

                int ndim;
                ar[path+"/mesh/N"] >> ndim;
                if (ndim != container_type::dimensionality) throw std::runtime_error("Wrong number of dimension reading Matsubara GF, ndim="+ndim);

                mesh1_.load(ar,path+"/mesh/1");
                mesh2_.load(ar,path+"/mesh/2");
                mesh3_.load(ar,path+"/mesh/3");
                mesh4_.load(ar,path+"/mesh/4");

                data_.resize(boost::extents[mesh1_.extent()][mesh2_.extent()][mesh3_.extent()][mesh4_.extent()]);
                
                ar[path+"/data"] >> data_;
            }

            static void save_version(alps::hdf5::archive& ar, const std::string& path)
            {
                std::string vp=path+"/version/";
                ar[vp+"minor"]<< int(minor_version_);
                ar[vp+"major"]<< int(major_version_);
                ar[vp+"reference"]<<"https://github.com/ALPSCore/H5GF/blob/master/H5GF.rst";
                ar[vp+"originator"]<<"ALPSCore GF library, see http://www.alpscore.org";
            }

            static bool check_version(alps::hdf5::archive& ar, const std::string& path)
            {
                std::string vp=path+"/version/";
                int ver;
                ar[vp+"major"]>>ver;
                return (major_version_==ver);
            }
        };
    }
}
