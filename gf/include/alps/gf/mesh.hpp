/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once
#include <complex>
#include <boost/multi_array.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/operators.hpp>
#include <boost/type_traits/integral_constant.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/multi_array.hpp>

#ifdef ALPS_HAVE_MPI
#include "mpi_bcast.hpp"
#endif

#include"flagcheck.hpp"

namespace alps {
    namespace gf {
    
        namespace statistics {
            enum statistics_type {
                BOSONIC=0,
                FERMIONIC=1
            };
        }

        /// A generic index
        template <typename X>
        class generic_index :
            boost::additive2<generic_index<X>, int,
            boost::unit_steppable< generic_index<X>,
            boost::totally_ordered2< generic_index<X>, int> > >
        {
            private:
            int index_;
            public:
            explicit generic_index(int i): index_(i) {}
            generic_index() : index_(0) {}
      
            void operator=(int i) { index_=i; }
      
            generic_index& operator++() { index_++; return *this; }
            generic_index& operator--() { index_--; return *this; }
      
            generic_index& operator+=(int i) { index_+=i; return *this; }
            generic_index& operator-=(int i) { index_-=i; return *this; }
      
            bool operator<(int x) const { return index_ <x; }
            bool operator>(int x) const { return index_ >x; }
            bool operator==(int x) const { return index_==x; }
      
            int operator()() const { return index_; }

#ifdef ALPS_HAVE_MPI
          void broadcast(const alps::mpi::communicator& comm, int root) {
                alps::mpi::broadcast(comm, index_, root);
            }
#endif
        };
    
        //    template <typename T> bool operator==(int q, const generic_index<T> &p){ return p.operator==(q);}
        namespace mesh {
            enum frequency_positivity_type {
                POSITIVE_NEGATIVE=0,
                POSITIVE_ONLY=1
            };
        }
        class base_mesh {
        public:
            const std::vector<double> &points() const{return points_;}
        protected:
            // we do not want external functions be able to change a grid.
            std::vector<double> &_points() {return points_;}
            void swap(base_mesh &other) {
                using std::swap;
                swap(this->points_, other.points_);
            }
        private:
            std::vector<double> points_;
        };
        class real_frequency_mesh: public base_mesh{
        public:
            typedef generic_index<real_frequency_mesh> index_type;
            real_frequency_mesh() {};

            template<typename GRID>
            real_frequency_mesh(GRID grid)  {
                grid.compute_points(_points());
            }
            int extent() const {return points().size();}

            int operator()(index_type idx) const {
              return idx();
            }
            void save(alps::hdf5::archive& ar, const std::string& path) const
            {
                ar[path+"/kind"] << "REAL_FREQUENCY";
                ar[path+"/points"] << points();
            }

            void load(alps::hdf5::archive& ar, const std::string& path)
            {
                std::string kind;
                ar[path+"/kind"]   >> kind;
                if (kind!="REAL_FREQUENCY") throw std::runtime_error("Attempt to read real frequency mesh from non-real frequency data, kind="+kind);
                ar[path+"/points"] >> _points();
            }

            /// Comparison operators
            bool operator==(const real_frequency_mesh &mesh) const {
                return extent()==mesh.extent() && std::equal ( mesh.points().begin(), mesh.points().end(), points().begin() );;
            }

            /// Comparison operators
            bool operator!=(const real_frequency_mesh &mesh) const {
                return !(*this==mesh);
            }
#ifdef ALPS_HAVE_MPI
            void broadcast(const alps::mpi::communicator& comm, int root)
            {
                using alps::mpi::broadcast;
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

        template <mesh::frequency_positivity_type PTYPE>
        class matsubara_mesh : public base_mesh {
            double beta_;
            int nfreq_;
      
            statistics::statistics_type statistics_;
            static const mesh::frequency_positivity_type positivity_=PTYPE;
      
            //FIXME: we had this const, but that prevents copying.
            int offset_;

            public:
            typedef generic_index<matsubara_mesh> index_type;
            matsubara_mesh(double b, int nfr, gf::statistics::statistics_type statistics=statistics::FERMIONIC):
                beta_(b), nfreq_(nfr), statistics_(statistics), offset_((PTYPE==mesh::POSITIVE_ONLY)?0:nfr) {
                check_range();
                compute_points();
            }
            int extent() const{return nfreq_;}


            int operator()(index_type idx) const {
                return offset_+idx(); // FIXME: can be improved by specialization?
            }

            /// Comparison operators
            bool operator==(const matsubara_mesh &mesh) const {
                return beta_==mesh.beta_ && nfreq_==mesh.nfreq_ && statistics_==mesh.statistics_;
            }

            /// Comparison operators
            bool operator!=(const matsubara_mesh &mesh) const {
                return !(*this==mesh);
            }

            ///getter functions for member variables
            double beta() const{ return beta_;}
            statistics::statistics_type statistics() const{ return statistics_;}
            mesh::frequency_positivity_type positivity() const{ return positivity_;}

            /// Swaps this and another mesh
            // It's a member function to avoid dealing with templated friend decalration.
            void swap(matsubara_mesh& other) {
                using std::swap;
                swap(this->beta_, other.beta_);
                swap(this->nfreq_, other.nfreq_);
                base_mesh::swap(other);
            }
          
            void save(alps::hdf5::archive& ar, const std::string& path) const
            {
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
                    throw std::invalid_argument("Attempt to read Matsubara mesh with the wrong positivity type "+boost::lexical_cast<std::string>(posonly) ); // FIXME: specific exception? Verbose positivity?
                };
                beta_=beta;
                nfreq_=nfr;
                check_range();
                compute_points();
            }

#ifdef ALPS_HAVE_MPI
          void broadcast(const alps::mpi::communicator& comm, int root)
            {
                using alps::mpi::broadcast;
                // FIXME: introduce (debug-only?) consistency check, like type checking? akin to load()?
                broadcast(comm, beta_, root);
                broadcast(comm, nfreq_, root);
                try {
                    check_range();
                } catch (const std::exception& exc) {
                    // FIXME? Try to communiucate the error with all ranks, at least in debug mode?
                    int wrank=alps::mpi::communicator().rank();
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
            }
            void compute_points(){
                _points().resize(extent());
                for(int i=0;i<nfreq_;++i){
                    _points()[i]=(2*i+statistics_)*M_PI/beta_;
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

        class itime_mesh {
            double beta_;
            int ntau_;
            bool last_point_included_;
            bool half_point_mesh_;
            statistics::statistics_type statistics_;
            std::vector<double> points_;
      
            public:
            typedef generic_index<itime_mesh> index_type;

            itime_mesh(double beta, int ntau): beta_(beta), ntau_(ntau), last_point_included_(true), half_point_mesh_(false), statistics_(statistics::FERMIONIC){
              compute_points();

            }
                int operator()(index_type idx) const { return idx(); }
            int extent() const{return ntau_;}
      
            /// Comparison operators
            bool operator==(const itime_mesh &mesh) const {
                return beta_==mesh.beta_ && ntau_==mesh.ntau_ && last_point_included_==mesh.last_point_included_ &&
                    half_point_mesh_ == mesh.half_point_mesh_ && statistics_==mesh.statistics_;
            }
          
            ///Getter variables for members
            double beta() const{ return beta_;}
            statistics::statistics_type statistics() const{ return statistics_;}
            const std::vector<double> &points() const{return points_;}

            /// Comparison operators
            bool operator!=(const itime_mesh &mesh) const {
                return !(*this==mesh);
            }

            void save(alps::hdf5::archive& ar, const std::string& path) const
            {
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

#ifdef ALPS_HAVE_MPI
          void broadcast(const alps::mpi::communicator& comm, int root)
            {
                using alps::mpi::broadcast;
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

            public:
            typedef generic_index<power_mesh> index_type;

            power_mesh(double beta, int power, int uniform): beta_(beta), power_(power), uniform_(uniform), statistics_(statistics::FERMIONIC){
              compute_points();
              compute_weights();
            }
                int operator()(index_type idx) const { return idx(); }
                int extent() const{return ntau_;}
                int power() const{return power_;}
                int uniform() const{return uniform_;}

            /// Comparison operators
            bool operator==(const power_mesh &mesh) const {
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
                return !(*this==mesh);
            }

            void save(alps::hdf5::archive& ar, const std::string& path) const
            {
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

#ifdef ALPS_HAVE_MPI
          void broadcast(const alps::mpi::communicator& comm, int root)
            {
                using alps::mpi::broadcast;
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


        class momentum_realspace_index_mesh {
            public:
            typedef boost::multi_array<double,2> container_type;
            protected:
            container_type points_;
            private:
            std::string kind_;

            protected:
            momentum_realspace_index_mesh(const std::string& kind, int ns,int ndim): points_(boost::extents[ns][ndim]), kind_(kind)
            {
            }
      
            momentum_realspace_index_mesh(const std::string& kind, const container_type& mesh_points): points_(mesh_points), kind_(kind)
            {
            }
      
            public:
            // Returns the number of points
            int extent() const { return points_.shape()[0];}
            ///returns the spatial dimension
            int dimension() const { return points_.shape()[1];}
            ///returns the mesh kind
            const std::string &kind() const{return kind_;}

            /// Comparison operators
            bool operator==(const momentum_realspace_index_mesh &mesh) const {
                return kind_ == mesh.kind_ &&
                    points_ == mesh.points_;
            }
          
            /// Comparison operators
            bool operator!=(const momentum_realspace_index_mesh &mesh) const {
                return !(*this==mesh);
            }
            
            const container_type &points() const{return points_;}

            void save(alps::hdf5::archive& ar, const std::string& path) const
            {
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

#ifdef ALPS_HAVE_MPI
          void broadcast(const alps::mpi::communicator& comm, int root)
            {
                using alps::mpi::broadcast;
                // FIXME: introduce (debug-only?) consistency check, like type checking? akin to load()?
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

            index_mesh(int np): npoints_(np) { compute_points();}
            int extent() const{return npoints_;}
            int operator()(index_type idx) const { return idx(); }
            const std::vector<int> &points() const{return points_;}
      
            /// Comparison operators
            bool operator==(const index_mesh &mesh) const {
                return npoints_==mesh.npoints_;
            }
          
            /// Comparison operators
            bool operator!=(const index_mesh &mesh) const {
                return !(*this==mesh);
            }

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
                compute_points();
            }
            void compute_points(){ points_.resize(npoints_); for(int i=0;i<npoints_;++i){points_[i]=i;} }

#ifdef ALPS_HAVE_MPI
          void broadcast(const alps::mpi::communicator& comm, int root)
            {
                using alps::mpi::broadcast;
                broadcast(comm, npoints_, root);
            }
#endif
        };
        ///Stream output operator, e.g. for printing to file
        std::ostream &operator<<(std::ostream &os, const index_mesh &M);
    
        typedef matsubara_mesh<mesh::POSITIVE_ONLY> matsubara_positive_mesh;
        typedef matsubara_mesh<mesh::POSITIVE_NEGATIVE> matsubara_pn_mesh;
        typedef matsubara_mesh<mesh::POSITIVE_ONLY>::index_type matsubara_index;
        typedef matsubara_mesh<mesh::POSITIVE_NEGATIVE>::index_type matsubara_pn_index;
        typedef itime_mesh::index_type itime_index;
        typedef momentum_index_mesh::index_type momentum_index;
        typedef real_space_index_mesh::index_type real_space_index;
        typedef index_mesh::index_type index;
        typedef real_frequency_mesh::index_type real_freq_index;

        namespace detail {
            /* The following is an in-house implementation of a static_assert
               with the intent to generate a compile-time error message
               that has some relevance to the asserted condition
               (unlike BOOST_STATIC_ASSERT).
            */
            
            /// A helper class: indicator that a mesh can have a tail
            struct can_have_tail_yes { typedef bool mesh_can_have_tail; };
            /// A helper class: indicator that a mesh can NOT have a tail
            struct can_have_tail_no  { typedef bool mesh_cannot_have_tail; };
                
            /// Trait: whether a mesh can have a tail (general meshes cannot have tails)
            template <typename> struct can_have_tail: public can_have_tail_no {};

            /// Trait: Matsubara meshes can have tails
            template <> struct can_have_tail<matsubara_positive_mesh>: public can_have_tail_yes {};
            /// Trait: Matsubara meshes can have tails
            template <> struct can_have_tail<matsubara_pn_mesh>: public can_have_tail_yes {};
            /// Trait: Imaginary time meshes can have tails
            template <> struct can_have_tail<itime_mesh>: public can_have_tail_yes {};

            /* ^^^^ End of static_assert code */
            
        } // ::detail
    }
}
