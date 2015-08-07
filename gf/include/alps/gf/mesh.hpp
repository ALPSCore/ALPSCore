#pragma once
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

        class itime_mesh {
            double beta_;
            int ntau_;
            bool last_point_included_;
            bool half_point_mesh_;
            statistics::statistics_type statistics_;
            
           public:
            itime_mesh(double beta, int ntau): beta_(beta), ntau_(ntau), statistics_(statistics::FERMIONIC), last_point_included_(true), half_point_mesh_(false){}
            int extent() const{return ntau_;}
            typedef generic_index<itime_mesh> index_type;

            void save(alps::hdf5::archive& ar, const std::string& path) const
            {
                ar[path+"/kind"] << "IMAGINARY_TIME";
                ar[path+"/N"] << ntau_;
                ar[path+"/statistics"] << int(statistics_); // 
                ar[path+"/beta"] << beta_;
                ar[path+"/half_point_mesh"] << int(half_point_mesh_);
                ar[path+"/last_point_included"] << int(last_point_included_);
                // ...and optional ["points"]
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
        typedef itime_mesh::index_type itime_index;
        typedef momentum_index_mesh::index_type momentum_index;
        typedef index_mesh::index_type index;


    }
}
