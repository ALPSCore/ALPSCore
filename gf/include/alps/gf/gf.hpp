#pragma once
#include <complex>
#include <boost/multi_array.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/hdf5/multi_array.hpp>

#include "mesh.hpp"


namespace alps {
    namespace gf {

        const int minor_version=1;
        const int major_version=0;
       
        //FIXME: problem here when we have multiple includes 
        inline void save_version(alps::hdf5::archive& ar, const std::string& path)
        {
            std::string vp=path+"/version/";
            ar[vp+"minor"]<< int(minor_version);
            ar[vp+"major"]<< int(major_version);
            ar[vp+"reference"]<<"https://github.com/ALPSCore/H5GF/blob/master/H5GF.rst";
            ar[vp+"originator"]<<"ALPSCore GF library, see http://www.alpscore.org";
        }
        
        inline bool check_version(alps::hdf5::archive& ar, const std::string& path)
        {
            std::string vp=path+"/version/";
            int ver;
            ar[vp+"major"]>>ver;
            return (major_version==ver);
        }
        
        template<class value_type, class MESH1, class MESH2, class MESH3> class two_index_gf {
            typedef boost::multi_array<value_type,2> container_type;
        
            MESH1 mesh1_;
            MESH2 mesh2_;
        
            container_type data_;
            public:
            two_index_gf(const MESH1& mesh1,
                           const MESH2& mesh2)
                         : mesh1_(mesh1), mesh2_(mesh2),
                           data_(boost::extents[mesh1_.extent()][mesh2_.extent()])
            {
            }
        
            const value_type& operator()(typename MESH1::index_type i1, typename MESH2::index_type i2) const
            {
                return data_[i1()][i2()];
            }
        
            value_type& operator()(typename MESH1::index_type i1, typename MESH2::index_type i2)
            {
                return data_[i1()][i2()];
            }
        
            /// Initialize the GF data to value_type(0.)
            void initialize()
            {
                for (int i=0; i<mesh1_.extent(); ++i) {
                    for (int j=0; j<mesh2_.extent(); ++j) {
                        data_[i][j]=value_type(0.0);
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
          
                data_.resize(boost::extents[mesh1_.extent()][mesh2_.extent()]);
          
                ar[path+"/data"] >> data_;
            }
        
        };

        template<class value_type, class MESH1, class MESH2, class MESH3> class three_index_gf {
            typedef boost::multi_array<value_type,3> container_type;
        
            MESH1 mesh1_;
            MESH2 mesh2_;
            MESH3 mesh3_;
        
            container_type data_;
            public:
            three_index_gf(const MESH1& mesh1,
                           const MESH2& mesh2,
                           const MESH3& mesh3)
                : mesh1_(mesh1), mesh2_(mesh2), mesh3_(mesh3),
                  data_(boost::extents[mesh1_.extent()][mesh2_.extent()][mesh3_.extent()])
            {
            }
        
            const value_type& operator()(typename MESH1::index_type i1, typename MESH2::index_type i2, typename MESH3::index_type i3) const
            {
                return data_[i1()][i2()][i3()];
            }
        
            value_type& operator()(typename MESH1::index_type i1, typename MESH2::index_type i2, typename MESH3::index_type i3)
            {
                return data_[i1()][i2()][i3()];
            }
        
            /// Initialize the GF data to value_type(0.)
            void initialize()
            {
                for (int i=0; i<mesh1_.extent(); ++i) {
                    for (int j=0; j<mesh2_.extent(); ++j) {
                        for (int k=0; k<mesh3_.extent(); ++k) {
                            data_[i][j][k]=value_type(0.0);
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
          
                data_.resize(boost::extents[mesh1_.extent()][mesh2_.extent()][mesh3_.extent()]);
          
                ar[path+"/data"] >> data_;
            }
        
        };

        template<class value_type, class MESH1, class MESH2, class MESH3, class MESH4> class four_index_gf {
            typedef boost::multi_array<value_type,4> container_type;

            MESH1 mesh1_;
            MESH2 mesh2_;
            MESH3 mesh3_;
            MESH4 mesh4_;

            container_type data_;
            public:
            four_index_gf(const MESH1& mesh1,
                          const MESH2& mesh2,
                          const MESH3& mesh3,
                          const MESH4& mesh4)
                : mesh1_(mesh1), mesh2_(mesh2), mesh3_(mesh3), mesh4_(mesh4),
                  data_(boost::extents[mesh1_.extent()][mesh2_.extent()][mesh3_.extent()][mesh4_.extent()])
            {
            }

            const MESH1& mesh1() const { return mesh1_; } 
            
            const value_type& operator()(typename MESH1::index_type i1, typename MESH2::index_type i2, typename MESH3::index_type i3, typename MESH4::index_type i4) const
            {
                return data_[mesh1_(i1)][mesh2_(i2)][mesh3_(i3)][mesh4_(i4)];
            }

            value_type& operator()(typename MESH1::index_type i1, typename MESH2::index_type i2, typename MESH3::index_type i3, typename MESH4::index_type i4)
            {
                return data_[mesh1_(i1)][mesh2_(i2)][mesh3_(i3)][mesh4_(i4)];
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
        };

        typedef four_index_gf<std::complex<double>, matsubara_mesh<mesh::POSITIVE_ONLY>, momentum_index_mesh, momentum_index_mesh, index_mesh> omega_k1_k2_sigma_gf;
        typedef four_index_gf<             double , itime_mesh    , momentum_index_mesh, momentum_index_mesh, index_mesh> itime_k1_k2_sigma_gf;
        typedef four_index_gf<std::complex<double>, matsubara_mesh<mesh::POSITIVE_ONLY>, real_space_index_mesh, real_space_index_mesh, index_mesh> omega_r1_r2_sigma_gf;
        typedef four_index_gf<             double , itime_mesh    , real_space_index_mesh, real_space_index_mesh, index_mesh> itime_r1_r2_sigma_gf;

        typedef three_index_gf<std::complex<double>, matsubara_mesh<mesh::POSITIVE_ONLY>, momentum_index_mesh, index_mesh> omega_k_sigma_gf;
        typedef three_index_gf<             double , itime_mesh    , momentum_index_mesh, index_mesh> itime_k_sigma_gf;
        
        typedef omega_k1_k2_sigma_gf matsubara_gf;
        typedef itime_k1_k2_sigma_gf itime_gf;
    }
}
