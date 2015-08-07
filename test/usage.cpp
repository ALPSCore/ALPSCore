#include "gtest/gtest.h"

#include <complex>
#include <boost/multi_array.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/hdf5/multi_array.hpp>


namespace alps {
    namespace gf {

        namespace statistics {
            enum statistics_type {
                BOSONIC=0,
                FERMIONIC=1
            };
        }
        
        /// A generic index
        template <typename>
        class generic_index {
            private:
            int index_;
            public:
            explicit generic_index(int i): index_(i) {}
            generic_index() : index_(0) {}
            void operator=(int i) { index_=i; }
            int operator()(){return get();}
            int get() { return index_; }
        };

        class matsubara_mesh {
            double beta_;
            int nfreq_;
          public:

            enum frequency_positivity_type {
                POSITIVE_NEGATIVE=0,
                POSITIVE_ONLY=1
            };

           private:
            statistics::statistics_type statistics_;
            frequency_positivity_type positivity_;
            
           public:
            matsubara_mesh(double b, int nfr): beta_(b), nfreq_(nfr), statistics_(statistics::FERMIONIC), positivity_(POSITIVE_ONLY) {}
            int extent() const{return nfreq_;}
            typedef generic_index<matsubara_mesh> index_type;

            void save(alps::hdf5::archive& ar, const std::string& path) const
            {
                ar[path+"/kind"] << "MATSUBARA";
                ar[path+"/N"] << nfreq_;
                ar[path+"/statistics"] << int(statistics_); // 
                ar[path+"/beta"] << beta_;
                ar[path+"/positive_only"] << int(positivity_);
                // ...and optional ["points"]
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

                statistics_=statistics::statistics_type(stat); // FIXME: check range
                positivity_=frequency_positivity_type(posonly); // FIXME: check range
                beta_=beta;
                nfreq_=nfr;
            }
        };

        class momentum_index_mesh {
          public:
            typedef boost::multi_array<double,2> container_type;
          private:
            container_type points_;
          public:
            momentum_index_mesh(int ns,int ndim): points_(boost::extents[ns][ndim])
            {
            }

            momentum_index_mesh(const container_type& mesh_points): points_(mesh_points)
            {
            }

            typedef generic_index<momentum_index_mesh> index_type;

            // Returns the number of points
            int extent() const { return points_.shape()[0];}

            void save(alps::hdf5::archive& ar, const std::string& path) const
            {
                ar[path+"/kind"] << "MOMENTUM_INDEX";
                ar[path+"/points"] << points_;
            }

            void load(alps::hdf5::archive& ar, const std::string& path)
            {
                std::string kind;
                ar[path+"/kind"] >> kind;
                if (kind!="MOMENTUM_INDEX") throw std::runtime_error("Attempt to load momentum index mesh from incorrect mesh kind="+kind);
                ar[path+"/points"] >> points_;
            }
        };
        
        class index_mesh {
            int npoints_;
          public:
            index_mesh(int np): npoints_(np) {}
            typedef generic_index<index_mesh> index_type;
            int extent() const{return npoints_;}

            void save(alps::hdf5::archive& ar, const std::string& path) const
            {
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
            }
        };
        
        typedef matsubara_mesh::index_type matsubara_index;
        typedef momentum_index_mesh::index_type momentum_index;
        typedef index_mesh::index_type index;


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
            
            const std::complex<double>& operator()(matsubara_index omega, momentum_index i, momentum_index j, index sigma) const
            {
                return data_[omega()][i()][j()][sigma()];
            }

            std::complex<double>& operator()(matsubara_index omega, momentum_index i, momentum_index j, index sigma)
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
    }
}
//int alps::gf::matsubara_gf::minor_version_;
//int alps::gf::matsubara_gf::major_version_;


/// This generates some "outside" data to fill the mesh: 4 2-d points
alps::gf::momentum_index_mesh::container_type get_data_for_mesh()
{
    alps::gf::momentum_index_mesh::container_type points(boost::extents[4][2]);
    points[0][0]=0; points[0][1]=0; 
    points[1][0]=M_PI; points[1][1]=M_PI;
    points[2][0]=M_PI; points[2][1]=0; 
    points[3][0]=0; points[3][1]=M_PI;

    return points;
}


class TestGF : public ::testing::Test
{
  public:
    const double beta;
    const int nsites;
    const int nfreq ;
    const int nspins;
    alps::gf::matsubara_gf gf;
    alps::gf::matsubara_gf gf2;

    TestGF():beta(10), nsites(4), nfreq(10), nspins(2),
             gf(alps::gf::matsubara_mesh(beta,nfreq),
                alps::gf::momentum_index_mesh(get_data_for_mesh()),
                alps::gf::momentum_index_mesh(get_data_for_mesh()),
                alps::gf::index_mesh(nspins)),
             gf2(gf) {}
};
    

TEST_F(TestGF,access)
{
    alps::gf::matsubara_index omega; omega=4;
    alps::gf::momentum_index i; i=2;
    alps::gf::momentum_index j=alps::gf::momentum_index(3);
    alps::gf::index sigma(1);

    gf(omega, i,j,sigma)=std::complex<double>(3,4);
    std::complex<double> x=gf(omega,i,j,sigma);
    EXPECT_EQ(3, x.real());
    EXPECT_EQ(4, x.imag());
}

TEST_F(TestGF,init)
{
    alps::gf::matsubara_index omega; omega=4;
    alps::gf::momentum_index i; i=2;
    alps::gf::momentum_index j=alps::gf::momentum_index(3);
    alps::gf::index sigma(1);

    gf.initialize();
    std::complex<double> x=gf(omega,i,j,sigma);
    EXPECT_EQ(0, x.real());
    EXPECT_EQ(0, x.imag());
}

TEST_F(TestGF,saveload)
{
    namespace g=alps::gf;
    {
        alps::hdf5::archive oar("gf.h5","w");
        gf(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1))=std::complex<double>(7., 3.);
        gf.save(oar,"/gf");
    }
    {
        alps::hdf5::archive iar("gf.h5");
        gf2.load(iar,"/gf");
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).imag());
    {
        alps::hdf5::archive oar("gf.h5","rw");
        oar["/gf/version/major"]<<7;
        EXPECT_THROW(gf2.load(oar,"/gf"),std::runtime_error);
    }
    EXPECT_EQ(7, gf2(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).real());
    EXPECT_EQ(3, gf2(g::matsubara_index(4),g::momentum_index(3), g::momentum_index(2), g::index(1)).imag());
    
    
    //boost::filesystem::remove("g5.h5");
}
