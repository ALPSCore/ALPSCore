#pragma once
#include <complex>
#include <algorithm>
#include <functional>
#include <boost/multi_array.hpp>
#include <boost/lexical_cast.hpp>

// #include <alps/type_traits/is_complex.hpp>
#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/hdf5/multi_array.hpp>

// FIXME: make conditional
#include <mpi.h>

#include "mesh.hpp"

namespace alps {
    namespace gf {

        namespace detail {
            // FIXME: make condidional
            /// Broadcast a multi-array
            template <typename T, size_t N>
            void broadcast_multiarray(boost::multi_array<T,N>& data, int root, MPI_Comm comm)
            {
                size_t nbytes=data.num_elements()*sizeof(T);
                // { // DEBUG:
                //     int rank;
                //     MPI_Comm_rank(comm,&rank);
                //     std::cout << "DEBUG: root=" << root << " rank=" << rank << " broadcast of nbytes=" << nbytes << std::endl;
                // }
                // This is an additional broadcast, but we need to ensure MPI broadcast correctness
                unsigned long nbytes_root=nbytes;
                MPI_Bcast(&nbytes_root, 1, MPI_UNSIGNED_LONG, root, comm);
                if (nbytes_root!=nbytes) {
                    int rank;
                    MPI_Comm_rank(comm,&rank);
                    // FIXME!!
                    // Here we have a mismatched broadcast, and the following options:
                    // 1) Call MPI_Abort() here as we cannot recover from a mismatched broadcast.
                    // 2) Communicate with root rank to NOT to attempt broadcast (e.g., use MPI_Alltoall?)
                    // 3) Temporary establish MPI error handler, do broadcast, get an error from MPI, continue.
                    throw std::runtime_error("Broadcast of incompatible GF data detected on rank "+boost::lexical_cast<std::string>(rank)+
                                             ".\nRoot sends "+boost::lexical_cast<std::string>(nbytes_root)+" bytes,"+
                                             " this process expects "+boost::lexical_cast<std::string>(nbytes)+" bytes.");
                }
                MPI_Bcast(data.origin(), nbytes, MPI_BYTE, root, comm);
            }
        }

        const int minor_version=1;
        const int major_version=0;
       
        void save_version(alps::hdf5::archive& ar, const std::string& path);
        
        bool check_version(alps::hdf5::archive& ar, const std::string& path);
        
        namespace detail{
            template<typename T> void print_no_complex(std::ostream &os, const T &z){
                os<<z;
            }

        }

        template<class VTYPE, class MESH1> class one_index_gf
        :boost::additive<one_index_gf<VTYPE,MESH1>,
         boost::multiplicative2<one_index_gf<VTYPE,MESH1>,VTYPE> >
        {
            public:
            typedef boost::multi_array<VTYPE,1> container_type;
            typedef MESH1 mesh1_type;
            typedef VTYPE value_type;

            private:
            MESH1 mesh1_;

            container_type data_;
            public:
            one_index_gf(const MESH1& mesh1)
                : mesh1_(mesh1),
                  data_(boost::extents[mesh1_.extent()])
            {
            }

            one_index_gf(const MESH1& mesh1,
                         const container_type& data)
                : mesh1_(mesh1),
                  data_(data)
            {
                if (mesh1_.extent()!=data_.shape()[0])
                    throw std::invalid_argument("Initialization of GF with the data of incorrect size");
            }

            const container_type& data() const { return data_; }

            const MESH1& mesh1() const { return mesh1_; }

            const value_type& operator()(typename MESH1::index_type i1) const
            {
                return data_[i1()];
            }

            value_type& operator()(typename MESH1::index_type i1)
            {
                return data_[i1()];
            }

            /// Initialize the GF data to value_type(0.)
            void initialize()
            {
                for (int i=0; i<mesh1_.extent(); ++i) {
                        data_[i]=value_type(0.0);
                }
            }
            /// Norm operation (FIXME: is it always double??)
            double norm() const
            {
                using std::abs;
                double v=0;
                for (const value_type* ptr=data_.origin(); ptr!=data_.origin()+data_.num_elements(); ++ptr) {
                    v=std::max(abs(*ptr), v);
                }
                return v;
            }

            /// Assignment-op with scalar
            template <typename op>
            one_index_gf& do_op(const value_type& scalar)
            {

                std::transform(data_.origin(), data_.origin()+data_.num_elements(), // inputs
                               data_.origin(), // output
                               std::bind2nd(op(), scalar)); // bound binary(?,scalar)

                return *this;
            }

            /// Assignment-op with another GF
            template <typename op>
            one_index_gf& do_op(const one_index_gf& rhs)
            {
                if (mesh1_!=rhs.mesh1_) {

                    throw std::runtime_error("Incompatible meshes in one_index_gf::do_op");
                }

                std::transform(data_.origin(), data_.origin()+data_.num_elements(), rhs.data_.origin(), // inputs
                               data_.origin(), // output
                               op());

                return *this;
            }

            /// Element-wise addition
            one_index_gf& operator+=(const one_index_gf& rhs)
            {
                return do_op< std::plus<value_type> >(rhs);
            }

            /// Element-wise subtraction
            one_index_gf& operator-=(const one_index_gf& rhs)
            {
                return do_op< std::minus<value_type> >(rhs);
            }

            /// Element-wise scaling 
            one_index_gf& operator*=(const value_type& scalar)
            {
                return do_op< std::multiplies<value_type> >(scalar);
            }

            /// Element-wise scaling 
            one_index_gf& operator/=(const value_type& scalar)
            {
                return do_op< std::divides<value_type> >(scalar);
            }

            /// Save the GF to HDF5
            void save(alps::hdf5::archive& ar, const std::string& path) const
            {
                save_version(ar,path);
                ar[path+"/data"] << data_;
                ar[path+"/mesh/N"] << int(container_type::dimensionality);
                mesh1_.save(ar,path+"/mesh/1");
            }

            /// Load the GF from HDF5
            void load(alps::hdf5::archive& ar, const std::string& path)
            {
                if (!check_version(ar,path)) throw std::runtime_error("Incompatible archive version");

                int ndim;
                ar[path+"/mesh/N"] >> ndim;
                if (ndim != container_type::dimensionality) throw std::runtime_error("Wrong number of dimension reading Matsubara GF, ndim="+boost::lexical_cast<std::string>(ndim) );

                mesh1_.load(ar,path+"/mesh/1");

                data_.resize(boost::extents[mesh1_.extent()]);

                ar[path+"/data"] >> data_;
            }

            /// Broadcast the data portion of GF (assuming identical meshes)
            void broadcast_data(int root, MPI_Comm comm)
            {
                detail::broadcast_multiarray(data_, root, comm);
            }

        };
        template<class value_type, class MESH1> std::ostream &operator<<(std::ostream &os, one_index_gf<value_type,MESH1> G){
          os<<G.mesh1();
          for(int i=0;i<G.mesh1().extent();++i){
            os<<G.mesh1().points()[i]<<" ";
            detail::print_no_complex<value_type>(os, G(typename MESH1::index_type(i)));
            os<<std::endl;
          }
          return os;
        }

        template<class VTYPE, class MESH1, class MESH2> class two_index_gf
        :boost::additive<two_index_gf<VTYPE,MESH1,MESH2>,
         boost::multiplicative2<two_index_gf<VTYPE,MESH1,MESH2>,VTYPE> >
        {
            public:
            typedef boost::multi_array<VTYPE,2> container_type;
            typedef MESH1 mesh1_type;
            typedef MESH2 mesh2_type;
            typedef VTYPE value_type;

            private: 
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

            two_index_gf(const MESH1& mesh1,
                         const MESH2& mesh2,
                         const container_type& data)
                : mesh1_(mesh1), mesh2_(mesh2),
                  data_(data)
            {
                if (mesh1_.extent()!=data_.shape()[0] || mesh2_.extent()!=data_.shape()[1])
                    throw std::invalid_argument("Initialization of GF with the data of incorrect size");
            }

            const container_type& data() const { return data_; }
            
            const MESH1& mesh1() const { return mesh1_; } 
            const MESH2& mesh2() const { return mesh2_; } 

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
            /// Norm operation (FIXME: is it always double??)
            double norm() const
            {
                using std::abs;
                double v=0;
                for (const value_type* ptr=data_.origin(); ptr!=data_.origin()+data_.num_elements(); ++ptr) {
                    v=std::max(abs(*ptr), v);
                }
                return v;
            }

            /// Assignment-op with another GF
            template <typename op>
            two_index_gf& do_op(const two_index_gf& rhs)
            {
                if (mesh1_!=rhs.mesh1_ ||
                    mesh2_!=rhs.mesh2_ ) {
                    
                    throw std::runtime_error("Incompatible meshes in two_index_gf::operator+=");
                }

                std::transform(data_.origin(), data_.origin()+data_.num_elements(), rhs.data_.origin(), // inputs
                               data_.origin(), // output
                               op());

                return *this;
            }

            /// Element-wise addition
            two_index_gf& operator+=(const two_index_gf& rhs)
            {
                return do_op< std::plus<value_type> >(rhs);
            }

            /// Element-wise subtraction
            two_index_gf& operator-=(const two_index_gf& rhs)
            {
                return do_op< std::minus<value_type> >(rhs);
            }
        
            /// Assignment-op with scalar
            template <typename op>
            two_index_gf& do_op(const value_type& scalar)
            {

                std::transform(data_.origin(), data_.origin()+data_.num_elements(), // inputs
                               data_.origin(), // output
                               std::bind2nd(op(), scalar)); // bound binary(?,scalar)

                return *this;
            }

            /// Element-wise scaling 
            two_index_gf& operator*=(const value_type& scalar)
            {
                return do_op< std::multiplies<value_type> >(scalar);
            }

            /// Element-wise scaling 
            two_index_gf& operator/=(const value_type& scalar)
            {
                return do_op< std::divides<value_type> >(scalar);
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
                if (ndim != container_type::dimensionality) throw std::runtime_error("Wrong number of dimension reading Matsubara GF, ndim="+boost::lexical_cast<std::string>(ndim) );
          
                mesh1_.load(ar,path+"/mesh/1");
                mesh2_.load(ar,path+"/mesh/2");
          
                data_.resize(boost::extents[mesh1_.extent()][mesh2_.extent()]);
          
                ar[path+"/data"] >> data_;
            }
        
            /// Broadcast the data portion of GF (assuming identical meshes)
            void broadcast_data(int root, MPI_Comm comm)
            {
                detail::broadcast_multiarray(data_, root, comm);
            }

        };

        template<class value_type, class MESH1, class MESH2> std::ostream &operator<<(std::ostream &os, two_index_gf<value_type,MESH1,MESH2> G){
          os<<G.mesh1()<<G.mesh2();
          for(int i=0;i<G.mesh1().extent();++i){
            os<<G.mesh1().points()[i]<<" ";
            for(int k=0;k<G.mesh2().extent();++k){
              detail::print_no_complex<value_type>(os, G(typename MESH1::index_type(i),typename MESH2::index_type(k))); os<<" ";
            }
            os<<std::endl;
          }
          return os;
        }


        template<class VTYPE, class MESH1, class MESH2, class MESH3> class three_index_gf
        :boost::additive<three_index_gf<VTYPE,MESH1,MESH2,MESH3>,
         boost::multiplicative2<three_index_gf<VTYPE,MESH1,MESH2,MESH3>,VTYPE> >
        {
            public:
            typedef VTYPE value_type;
            typedef boost::multi_array<value_type,3> container_type;
            typedef MESH1 mesh1_type;
            typedef MESH2 mesh2_type;
            typedef MESH3 mesh3_type;

            private:
            mesh1_type mesh1_;
            mesh2_type mesh2_;
            mesh3_type mesh3_;
        
            container_type data_;
            
            public:
            three_index_gf(const MESH1& mesh1,
                           const MESH2& mesh2,
                           const MESH3& mesh3)
                : mesh1_(mesh1), mesh2_(mesh2), mesh3_(mesh3),
                  data_(boost::extents[mesh1_.extent()][mesh2_.extent()][mesh3_.extent()])
            {
            }

            three_index_gf(const MESH1& mesh1,
                           const MESH2& mesh2,
                           const MESH3& mesh3,
                           const container_type& data)
                : mesh1_(mesh1), mesh2_(mesh2), mesh3_(mesh3),
                  data_(data)
            {
                if (mesh1_.extent()!=data_.shape()[0] || mesh2_.extent()!=data_.shape()[1] || mesh3_.extent()!=data_.shape()[2])
                    throw std::invalid_argument("Initialization of GF with the data of incorrect size");
            }


            const MESH1& mesh1() const { return mesh1_; } 
            const MESH2& mesh2() const { return mesh2_; } 
            const MESH3& mesh3() const { return mesh3_; } 
            const container_type& data() const { return data_; }
            
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
        
            /// Norm operation (FIXME: is it always double??)
            double norm() const
            {
                using std::abs;
                double v=0;
                for (const value_type* ptr=data_.origin(); ptr!=data_.origin()+data_.num_elements(); ++ptr) {
                    v=std::max(abs(*ptr), v);
                }
                return v;
            }

            /// Assignment-op with another GF
            template <typename op>
            three_index_gf& do_op(const three_index_gf& rhs)
            {
                if (mesh1_!=rhs.mesh1_ ||
                    mesh2_!=rhs.mesh2_ ||
                    mesh3_!=rhs.mesh3_ ) {
                    
                    throw std::runtime_error("Incompatible meshes in three_index_gf::operator+=");
                }

                std::transform(data_.origin(), data_.origin()+data_.num_elements(), rhs.data_.origin(), // inputs
                               data_.origin(), // output
                               op());

                return *this;
            }

            /// Element-wise addition
            three_index_gf& operator+=(const three_index_gf& rhs)
            {
                return do_op< std::plus<value_type> >(rhs);
            }

            /// Element-wise subtraction
            three_index_gf& operator-=(const three_index_gf& rhs)
            {
                return do_op< std::minus<value_type> >(rhs);
            }

            /// Assignment-op with scalar
            template <typename op>
            three_index_gf& do_op(const value_type& scalar)
            {

                std::transform(data_.origin(), data_.origin()+data_.num_elements(), // inputs
                               data_.origin(), // output
                               std::bind2nd(op(), scalar)); // bound binary(?,scalar)

                return *this;
            }

            /// Element-wise scaling 
            three_index_gf& operator*=(const value_type& scalar)
            {
                return do_op< std::multiplies<value_type> >(scalar);
            }

            /// Element-wise scaling 
            three_index_gf& operator/=(const value_type& scalar)
            {
                return do_op< std::divides<value_type> >(scalar);
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
                if (ndim != container_type::dimensionality) throw std::runtime_error("Wrong number of dimension reading Matsubara GF, ndim="+boost::lexical_cast<std::string>(ndim));
          
                mesh1_.load(ar,path+"/mesh/1");
                mesh2_.load(ar,path+"/mesh/2");
                mesh3_.load(ar,path+"/mesh/3");
          
                data_.resize(boost::extents[mesh1_.extent()][mesh2_.extent()][mesh3_.extent()]);
          
                ar[path+"/data"] >> data_;
            }
        
            /// Broadcast the data portion of GF (assuming identical meshes)
            void broadcast_data(int root, MPI_Comm comm)
            {
                detail::broadcast_multiarray(data_, root, comm);
            }
        };

        template<class value_type, class MESH1, class MESH2, class MESH3> std::ostream &operator<<(std::ostream &os, three_index_gf<value_type,MESH1,MESH2,MESH3> G){
          os<<G.mesh1()<<G.mesh2()<<G.mesh3();
          for(int i=0;i<G.mesh1().extent();++i){
            os<<G.mesh1().points()[i]<<" ";
            for(int j=0;j<G.mesh2().extent();++j){
              for(int k=0;k<G.mesh3().extent();++k){
                detail::print_no_complex<value_type>(os, G(typename MESH1::index_type(i),typename MESH2::index_type(j),typename MESH3::index_type(k))); os<<" ";
              }
            }
            os<<std::endl;
          }
          return os;
        }

        template<class VTYPE, class MESH1, class MESH2, class MESH3, class MESH4> class four_index_gf 
        :boost::additive<four_index_gf<VTYPE,MESH1,MESH2,MESH3,MESH4>,
         boost::multiplicative2<four_index_gf<VTYPE,MESH1,MESH2,MESH3,MESH4>,VTYPE> > {
            public:
            typedef VTYPE value_type;
            typedef boost::multi_array<value_type,4> container_type;
            typedef MESH1 mesh1_type;
            typedef MESH2 mesh2_type;
            typedef MESH3 mesh3_type;
            typedef MESH4 mesh4_type;


            private:

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

            four_index_gf(const MESH1& mesh1,
                          const MESH2& mesh2,
                          const MESH3& mesh3,
                          const MESH3& mesh4,
                          const container_type& data)
                : mesh1_(mesh1), mesh2_(mesh2), mesh3_(mesh3), mesh4_(mesh4),
                  data_(data)
            {
                if (mesh1_.extent()!=data_.shape()[0] || mesh2_.extent()!=data_.shape()[1] ||
                    mesh3_.extent()!=data_.shape()[2] || mesh4_.extent()!=data_.shape()[3])
                    throw std::invalid_argument("Initialization of GF with the data of incorrect size");
            }

            const MESH1& mesh1() const { return mesh1_; } 
            const MESH2& mesh2() const { return mesh2_; } 
            const MESH3& mesh3() const { return mesh3_; } 
            const MESH4& mesh4() const { return mesh4_; } 
            const container_type& data() const { return data_; }
            
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

            /// Norm operation (FIXME: is it always double??)
            double norm() const
            {
                using std::abs;
                double v=0;
                for (const value_type* ptr=data_.origin(); ptr!=data_.origin()+data_.num_elements(); ++ptr) {
                    v=std::max(abs(*ptr), v);
                }
                return v;
            }

            /// Assignment-op with another GF
            template <typename op>
            four_index_gf& do_op(const four_index_gf& rhs)
            {
                if (mesh1_!=rhs.mesh1_ ||
                    mesh2_!=rhs.mesh2_ ||
                    mesh3_!=rhs.mesh3_ ||
                    mesh4_!=rhs.mesh4_ ) {
                    
                    throw std::runtime_error("Incompatible meshes in three_index_gf::operator+=");
                }

                std::transform(data_.origin(), data_.origin()+data_.num_elements(), rhs.data_.origin(), // inputs
                               data_.origin(), // output
                               op());

                return *this;
            }

            /// Element-wise addition
            four_index_gf& operator+=(const four_index_gf& rhs)
            {
                return do_op< std::plus<value_type> >(rhs);
            }

            /// Element-wise subtraction
            four_index_gf& operator-=(const four_index_gf& rhs)
            {
                return do_op< std::minus<value_type> >(rhs);
            }

            /// Assignment-op with scalar
            template <typename op>
            four_index_gf& do_op(const value_type& scalar)
            {

                std::transform(data_.origin(), data_.origin()+data_.num_elements(), // inputs
                               data_.origin(), // output
                               std::bind2nd(op(), scalar)); // bound binary(?,scalar)

                return *this;
            }

            /// Element-wise scaling 
            four_index_gf& operator*=(const value_type& scalar)
            {
                return do_op< std::multiplies<value_type> >(scalar);
            }

            /// Element-wise scaling 
            four_index_gf& operator/=(const value_type& scalar)
            {
                return do_op< std::divides<value_type> >(scalar);
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
                if (ndim != container_type::dimensionality) throw std::runtime_error("Wrong number of dimension reading Matsubara GF, ndim="+boost::lexical_cast<std::string>(ndim));

                mesh1_.load(ar,path+"/mesh/1");
                mesh2_.load(ar,path+"/mesh/2");
                mesh3_.load(ar,path+"/mesh/3");
                mesh4_.load(ar,path+"/mesh/4");

                data_.resize(boost::extents[mesh1_.extent()][mesh2_.extent()][mesh3_.extent()][mesh4_.extent()]);
                
                ar[path+"/data"] >> data_;
            }

            /// Broadcast the data portion of GF (assuming identical meshes)
            void broadcast_data(int root, MPI_Comm comm)
            {
                detail::broadcast_multiarray(data_, root, comm);
            }
        };


        template<class value_type, class MESH1, class MESH2, class MESH3, class MESH4> std::ostream &operator<<(std::ostream &os, four_index_gf<value_type,MESH1,MESH2,MESH3,MESH4> G){
          os<<G.mesh1()<<G.mesh2()<<G.mesh3()<<G.mesh4();
          for(int i=0;i<G.mesh1().extent();++i){
            os<<G.mesh1().points()[i]<<" ";
            for(int j=0;j<G.mesh2().extent();++j){
              for(int k=0;k<G.mesh3().extent();++k){
                for(int l=0;l<G.mesh4().extent();++l){
                  detail::print_no_complex<value_type>(os, G(typename MESH1::index_type(i),typename MESH2::index_type(j),typename MESH3::index_type(k),typename MESH4::index_type(l))); os<<" ";
                }
              }
            }
            os<<std::endl;
          }
          return os;
        }

        typedef four_index_gf<std::complex<double>, matsubara_mesh<mesh::POSITIVE_ONLY>, momentum_index_mesh, momentum_index_mesh, index_mesh> omega_k1_k2_sigma_gf;
        typedef four_index_gf<             double , itime_mesh    , momentum_index_mesh, momentum_index_mesh, index_mesh> itime_k1_k2_sigma_gf;
        typedef four_index_gf<std::complex<double>, matsubara_mesh<mesh::POSITIVE_ONLY>, real_space_index_mesh, real_space_index_mesh, index_mesh> omega_r1_r2_sigma_gf;
        typedef four_index_gf<             double , itime_mesh    , real_space_index_mesh, real_space_index_mesh, index_mesh> itime_r1_r2_sigma_gf;
        typedef four_index_gf<std::complex<double>, matsubara_mesh<mesh::POSITIVE_ONLY>, momentum_index_mesh, index_mesh, index_mesh> omega_k_sigma1_sigma2_gf;
        typedef four_index_gf<             double , itime_mesh    , momentum_index_mesh, index_mesh, index_mesh> itime_k_sigma1_sigma2_gf;

        typedef three_index_gf<std::complex<double>, matsubara_mesh<mesh::POSITIVE_ONLY>, momentum_index_mesh, index_mesh> omega_k_sigma_gf;
        typedef three_index_gf<             double , itime_mesh    , momentum_index_mesh, index_mesh> itime_k_sigma_gf;
        
        typedef two_index_gf<std::complex<double>, matsubara_mesh<mesh::POSITIVE_ONLY>, index_mesh> omega_sigma_gf;
        typedef two_index_gf<             double , itime_mesh, index_mesh> itime_sigma_gf;

        typedef one_index_gf<std::complex<double>, matsubara_mesh<mesh::POSITIVE_ONLY> >omega_gf;
        typedef one_index_gf<             double , itime_mesh> itime_gf;
        typedef one_index_gf<std::complex<double>, index_mesh> sigma_gf;

        typedef omega_k1_k2_sigma_gf matsubara_gf;

    }
}
