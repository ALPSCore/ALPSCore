/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once
#include "gf.hpp"

namespace alps {
    namespace gf {

        static const int TAIL_NOT_SET=-1; ///< Special tail order meaning the tail is not set
        
        namespace detail {
#ifdef ALPS_HAVE_MPI
            template <typename TAILT>
            void broadcast_tail(const alps::mpi::communicator& comm,
                                int& min_order, int& max_order,
                                std::vector<TAILT>& tails,
                                const TAILT& tail_init,
                                int root)
            {
                using alps::mpi::broadcast;
                broadcast(comm, min_order, root);
                broadcast(comm, max_order, root);
                
                if (min_order==TAIL_NOT_SET) return;
                if (comm.rank()!=root) {
                    tails.resize(max_order+1, tail_init);
                }
                for (int i=min_order; i<=max_order; ++i) {
                    tails[i].broadcast(comm,root);
                }
            }
#endif
        
        } // detail::

        /// 2-index Green's function (of type GFT) with a tail (which is a Green's function of type TAILT)
        /** The *first* mesh of GFT is assumed to be a Matsubara frequency or imaginary time mesh */
        template <typename GFT, typename TAILT>
        class two_index_gf_with_tail : public GFT {
          public:
            typedef TAILT tail_type;
            typedef GFT gf_type;

          private:
            std::vector<tail_type> tails_;
            int min_tail_order_;
            int max_tail_order_;

            // If you see an error here, the first mesh of your GFT cannot have a tail!
            static const typename detail::can_have_tail<typename gf_type::mesh1_type>::mesh_can_have_tail mesh1_can_have_tail=true;
            // If you see an error here, the second mesh of your GFT should not have a tail!
            static const typename detail::can_have_tail<typename gf_type::mesh2_type>::mesh_cannot_have_tail mesh2_cannot_have_tail=true;

          public:

            two_index_gf_with_tail(const gf_type& gf): gf_type(gf), min_tail_order_(TAIL_NOT_SET), max_tail_order_(TAIL_NOT_SET)
            { }

            two_index_gf_with_tail(const two_index_gf_with_tail& gft): gf_type(gft), tails_(gft.tails_), min_tail_order_(gft.min_tail_order_), max_tail_order_(gft.max_tail_order_)
            { }

            int min_tail_order() const { return min_tail_order_; }
            int max_tail_order() const { return max_tail_order_; }

            /// Returns tail component of the given order
            const tail_type& tail(int order) const{
                if (order<min_tail_order_ || order > max_tail_order_) throw std::runtime_error("tails are known between min and max order, your order is outside.");
                return tails_[order];
            }

            /// Returns tail as a vector
            const std::vector<tail_type>& tail() const{
                return tails_;
            }

            two_index_gf_with_tail& set_tail(int order, const tail_type &tail){
                if(this->mesh2()!=tail.mesh1())
                    throw std::runtime_error("invalid mesh type in tail assignment");

                int tail_size=tails_.size();
                if(order>=tail_size){
                    tails_.resize(order+1, tail_type(this->mesh2()));
                    for(int i=tail_size;i<=order;++i) tails_[i].initialize();
                }
                tails_[order]=tail;

                //set minimum and maximum known coefficients if needed
                if(min_tail_order_==TAIL_NOT_SET || min_tail_order_>order) min_tail_order_=order;
                if(max_tail_order_==TAIL_NOT_SET || max_tail_order_<=order) max_tail_order_=order;
                return *this;
            }

            /// Save the GF with tail to HDF5
            void save(alps::hdf5::archive& ar, const std::string& path) const
            {
                gf_type::save(ar,path);
                ar[path+"/tail/descriptor"]="INFINITY_TAIL";
                ar[path+"/tail/min_tail_order"]=min_tail_order_;
                ar[path+"/tail/max_tail_order"]=max_tail_order_;
                if(min_tail_order_==TAIL_NOT_SET) return;
                for (int i=min_tail_order_; i<=max_tail_order_; ++i) {
                    ar[path+"/tail/"+boost::lexical_cast<std::string>(i)] << tails_[i].data();
                }
            }

            /// Load the GF with tail from HDF5
            void load(alps::hdf5::archive& ar, const std::string& path)
            {
                gf_type::load(ar,path);
                std::string descr; ar[path+"/tail/descriptor"] >> descr;
                if (descr!="INFINITY_TAIL") throw std::runtime_error("Wrong tail format '"+descr+"', expected INFINITY_TAIL");

                // FIXME!FIXME! Rewrite using clone-swap for exception safety.
                ar[path+"/tail/min_tail_order"] >> min_tail_order_;
                ar[path+"/tail/max_tail_order"] >> max_tail_order_;

                tails_.clear();
                if (min_tail_order_==TAIL_NOT_SET) return;
                if (min_tail_order_>0) tails_.resize(min_tail_order_,tail_type(this->mesh2()));

                typename tail_type::container_type buffer;
                for (int i=min_tail_order_; i<=max_tail_order_; ++i) {
                    ar[path+"/tail/"+boost::lexical_cast<std::string>(i)] >> buffer;
                    tails_.push_back(tail_type(this->mesh2(), buffer));
                }
            }

            /// Save the GF to HDF5
            void save(alps::hdf5::archive& ar) const
            {
                save(ar, ar.get_context());
            }
            
            /// Load the GF from HDF5
            void load(alps::hdf5::archive& ar)
            {
                load(ar, ar.get_context());
            }
            
#ifdef ALPS_HAVE_MPI
            /// Broadcast the tail and the GF
          void broadcast(const alps::mpi::communicator& comm, int root)
            {
                // FIXME: use clone-swap?
                gf_type::broadcast(comm,root);
                detail::broadcast_tail(comm,
                                       min_tail_order_, max_tail_order_,
                                       tails_, tail_type(this->mesh2()),
                                       root);
            }
#endif
        };


        /// 3-index Green's function (of type GFT) with a tail (which is a Green's function of type TAILT)
        /** The *first* mesh of GFT is assumed to be a Matsubara frequency or imaginary time mesh */
        template <typename GFT, typename TAILT>
        class three_index_gf_with_tail : public GFT {
          public:
            typedef TAILT tail_type;
            typedef GFT gf_type;

          private:
            std::vector<tail_type> tails_;
            int min_tail_order_;
            int max_tail_order_;

            // If you see an error here, the first mesh of your GFT cannot have a tail!
            static const typename detail::can_have_tail<typename gf_type::mesh1_type>::mesh_can_have_tail mesh1_can_have_tail=true;
            // If you see an error here, the second mesh of your GFT should not have a tail!
            static const typename detail::can_have_tail<typename gf_type::mesh2_type>::mesh_cannot_have_tail mesh2_cannot_have_tail=true;
            // If you see an error here, the third mesh of your GFT should not have a tail!
            static const typename detail::can_have_tail<typename gf_type::mesh3_type>::mesh_cannot_have_tail mesh3_cannot_have_tail=true;
            
          public:

            three_index_gf_with_tail(const gf_type& gf): gf_type(gf), min_tail_order_(TAIL_NOT_SET), max_tail_order_(TAIL_NOT_SET)
            { }
            
            three_index_gf_with_tail(const three_index_gf_with_tail& gft): gf_type(gft), tails_(gft.tails_), min_tail_order_(gft.min_tail_order_), max_tail_order_(gft.max_tail_order_)
            { }

            int min_tail_order() const { return min_tail_order_; }
            int max_tail_order() const { return max_tail_order_; }
            
            /// Returns tail component of the given order
            const tail_type& tail(int order) const{
                if (order<min_tail_order_ || order > max_tail_order_) throw std::runtime_error("tails are known between min and max order, your order is outside.");
                return tails_[order];
            }
    
            /// Returns tail as a vector
            const std::vector<tail_type>& tail() const{
                return tails_;
            }
    
            three_index_gf_with_tail& set_tail(int order, const tail_type &tail){
                if(this->mesh2()!=tail.mesh1() || this->mesh3()!=tail.mesh2())
                    throw std::runtime_error("invalid mesh type in tail assignment");
                
                int tail_size=tails_.size();
                if(order>=tail_size){
                    tails_.resize(order+1, tail_type(this->mesh2(), this->mesh3()));
                    for(int i=tail_size;i<=order;++i) tails_[i].initialize();
                }
                tails_[order]=tail;

                //set minimum and maximum known coefficients if needed
                if(min_tail_order_==TAIL_NOT_SET || min_tail_order_>order) min_tail_order_=order;
                if(max_tail_order_==TAIL_NOT_SET || max_tail_order_<=order) max_tail_order_=order;
                return *this;
            }

            /// Save the GF with tail to HDF5
            void save(alps::hdf5::archive& ar, const std::string& path) const
            {
                gf_type::save(ar,path);
                ar[path+"/tail/descriptor"]="INFINITY_TAIL";
                ar[path+"/tail/min_tail_order"]=min_tail_order_;
                ar[path+"/tail/max_tail_order"]=max_tail_order_;
                if(min_tail_order_==TAIL_NOT_SET) return;
                for (int i=min_tail_order_; i<=max_tail_order_; ++i) {
                    ar[path+"/tail/"+boost::lexical_cast<std::string>(i)] << tails_[i].data();
                }
            }
   
            /// Load the GF with tail from HDF5
            void load(alps::hdf5::archive& ar, const std::string& path)
            {
                gf_type::load(ar,path);
                std::string descr; ar[path+"/tail/descriptor"] >> descr;
                if (descr!="INFINITY_TAIL") throw std::runtime_error("Wrong tail format '"+descr+"', expected INFINITY_TAIL");

                // FIXME!FIXME! Rewrite using clone-swap for exception safety.
                ar[path+"/tail/min_tail_order"] >> min_tail_order_;
                ar[path+"/tail/max_tail_order"] >> max_tail_order_;

                tails_.clear();
                if(min_tail_order_==TAIL_NOT_SET) return;

                if(min_tail_order_>0) tails_.resize(min_tail_order_,tail_type(this->mesh2(), this->mesh3()));

                typename tail_type::container_type buffer;
                for (int i=min_tail_order_; i<=max_tail_order_; ++i) {
                    ar[path+"/tail/"+boost::lexical_cast<std::string>(i)] >> buffer;
                    tails_.push_back(tail_type(this->mesh2(), this->mesh3(), buffer));
                }
            }

            /// Save the GF to HDF5
            void save(alps::hdf5::archive& ar) const
            {
                save(ar, ar.get_context());
            }
            
            /// Load the GF from HDF5
            void load(alps::hdf5::archive& ar)
            {
                load(ar, ar.get_context());
            }
            
#ifdef ALPS_HAVE_MPI
            /// Broadcast the tail and the GF
          void broadcast(const alps::mpi::communicator& comm, int root)
            {
                // FIXME: use clone-swap?
                gf_type::broadcast(comm,root);
                detail::broadcast_tail(comm,
                                       min_tail_order_, max_tail_order_,
                                       tails_, tail_type(this->mesh2(), this->mesh3()),
                                       root);
            }
#endif

        };

        /// 4-index Green's function (ot type GFT) with a tail (which is a Green's function of type TAILT)
        /** The *first* mesh of GFT is assumed to be a Matsubara frequency or imaginary time mesh */
        template <typename GFT, typename TAILT>
        class four_index_gf_with_tail : public GFT {
          public:
            typedef TAILT tail_type;
            typedef GFT gf_type;

          private:
            std::vector<tail_type> tails_;
            int min_tail_order_;
            int max_tail_order_;

            // If you see an error here, the first mesh of your GFT cannot have a tail!
            static const typename detail::can_have_tail<typename gf_type::mesh1_type>::mesh_can_have_tail mesh1_can_have_tail=true;
            // If you see an error here, the second mesh of your GFT should not have a tail!
            static const typename detail::can_have_tail<typename gf_type::mesh2_type>::mesh_cannot_have_tail mesh2_cannot_have_tail=true;
            // If you see an error here, the third mesh of your GFT should not have a tail!
            static const typename detail::can_have_tail<typename gf_type::mesh3_type>::mesh_cannot_have_tail mesh3_cannot_have_tail=true;
            // If you see an error here, the fourth mesh of your GFT should not have a tail!
            static const typename detail::can_have_tail<typename gf_type::mesh4_type>::mesh_cannot_have_tail mesh4_cannot_have_tail=true;
            
          public:

            four_index_gf_with_tail(const gf_type& gf): gf_type(gf), min_tail_order_(TAIL_NOT_SET), max_tail_order_(TAIL_NOT_SET)
            { }
            
            four_index_gf_with_tail(const four_index_gf_with_tail& gft): gf_type(gft), tails_(gft.tails_), min_tail_order_(gft.min_tail_order_), max_tail_order_(gft.max_tail_order_)
            { }

            int min_tail_order() const { return min_tail_order_; }
            int max_tail_order() const { return max_tail_order_; }
            
            /// Returns tail component of the given order
            const tail_type& tail(int order) const{
                if (order<min_tail_order_ || order > max_tail_order_) throw std::runtime_error("tails are known between min and max order, your order is outside.");
                return tails_[order];
            }
    
            /// Returns tail as a vector
            const std::vector<tail_type>& tail() const{
                return tails_;
            }
    
            four_index_gf_with_tail& set_tail(int order, const tail_type &tail){
                if(this->mesh2()!=tail.mesh1() || this->mesh3()!=tail.mesh2() || this->mesh4()!=tail.mesh3())
                    throw std::runtime_error("invalid mesh type in tail assignment");
                
                int tail_size=tails_.size();
                if(order>=tail_size){
                    tails_.resize(order+1, tail_type(this->mesh2(), this->mesh3(), this->mesh4()));
                    for(int i=tail_size;i<=order;++i) tails_[i].initialize();
                }
                tails_[order]=tail;

                //set minimum and maximum known coefficients if needed
                if(min_tail_order_==TAIL_NOT_SET || min_tail_order_>order) min_tail_order_=order;
                if(max_tail_order_==TAIL_NOT_SET || max_tail_order_<=order) max_tail_order_=order;
                return *this;
            }

            /// Save the GF with tail to HDF5
            void save(alps::hdf5::archive& ar, const std::string& path) const
            {
                gf_type::save(ar,path);
                ar[path+"/tail/descriptor"]="INFINITY_TAIL";
                ar[path+"/tail/min_tail_order"]=min_tail_order_;
                ar[path+"/tail/max_tail_order"]=max_tail_order_;
                if(min_tail_order_==TAIL_NOT_SET) return;
                for (int i=min_tail_order_; i<=max_tail_order_; ++i) {
                    ar[path+"/tail/"+boost::lexical_cast<std::string>(i)] << tails_[i].data();
                }
            }
   
            /// Load the GF with tail from HDF5
            void load(alps::hdf5::archive& ar, const std::string& path)
            {
                gf_type::load(ar,path);
                std::string descr; ar[path+"/tail/descriptor"] >> descr;
                if (descr!="INFINITY_TAIL") throw std::runtime_error("Wrong tail format '"+descr+"', expected INFINITY_TAIL");

                // FIXME!FIXME! Rewrite using clone-swap for exception safety.
                ar[path+"/tail/min_tail_order"] >> min_tail_order_;
                ar[path+"/tail/max_tail_order"] >> max_tail_order_;

                tails_.clear();
                if(min_tail_order_==TAIL_NOT_SET) return;
                if(min_tail_order_>0) tails_.resize(min_tail_order_,tail_type(this->mesh2(), this->mesh3(), this->mesh4()));

                typename tail_type::container_type buffer;
                for (int i=min_tail_order_; i<=max_tail_order_; ++i) {
                    ar[path+"/tail/"+boost::lexical_cast<std::string>(i)] >> buffer;
                    tails_.push_back(tail_type(this->mesh2(), this->mesh3(), this->mesh4(), buffer));
                }
            }

            /// Save the GF to HDF5
            void save(alps::hdf5::archive& ar) const
            {
                save(ar, ar.get_context());
            }
            
            /// Load the GF from HDF5
            void load(alps::hdf5::archive& ar)
            {
                load(ar, ar.get_context());
            }
            
#ifdef ALPS_HAVE_MPI
            /// Broadcast the tail and the GF
          void broadcast(const alps::mpi::communicator& comm, int root)
            {
                // FIXME: use clone-swap?
                gf_type::broadcast(comm,root);
                detail::broadcast_tail(comm,
                                       min_tail_order_, max_tail_order_,
                                       tails_, tail_type(this->mesh2(),this->mesh3(),this->mesh4()),
                                       root);
            }
#endif

        };

        /// 5-index Green's function (of type GFT) with a tail (which is a Green's function of type TAILT)
        /** The *first* mesh of GFT is assumed to be a Matsubara frequency or imaginary time mesh */
        template <typename GFT, typename TAILT>
        class five_index_gf_with_tail : public GFT {
          public:
            typedef TAILT tail_type;
            typedef GFT gf_type;

          private:
            std::vector<tail_type> tails_;
            int min_tail_order_;
            int max_tail_order_;

            // If you see an error here, the first mesh of your GFT cannot have a tail!
            static const typename detail::can_have_tail<typename gf_type::mesh1_type>::mesh_can_have_tail mesh1_can_have_tail=true;
            // If you see an error here, the second mesh of your GFT should not have a tail!
            static const typename detail::can_have_tail<typename gf_type::mesh2_type>::mesh_cannot_have_tail mesh2_cannot_have_tail=true;
            // If you see an error here, the third mesh of your GFT should not have a tail!
            static const typename detail::can_have_tail<typename gf_type::mesh3_type>::mesh_cannot_have_tail mesh3_cannot_have_tail=true;
            // If you see an error here, the fourth mesh of your GFT should not have a tail!
            static const typename detail::can_have_tail<typename gf_type::mesh4_type>::mesh_cannot_have_tail mesh4_cannot_have_tail=true;
            // If you see an error here, the fourth mesh of your GFT should not have a tail!
            static const typename detail::can_have_tail<typename gf_type::mesh5_type>::mesh_cannot_have_tail mesh5_cannot_have_tail=true;

          public:

            five_index_gf_with_tail(const gf_type& gf): gf_type(gf), min_tail_order_(TAIL_NOT_SET), max_tail_order_(TAIL_NOT_SET)
            { }

            five_index_gf_with_tail(const five_index_gf_with_tail& gft): gf_type(gft), tails_(gft.tails_), min_tail_order_(gft.min_tail_order_), max_tail_order_(gft.max_tail_order_)
            { }

            int min_tail_order() const { return min_tail_order_; }
            int max_tail_order() const { return max_tail_order_; }

            /// Returns tail component of the given order
            const tail_type& tail(int order) const{
                if (order<min_tail_order_ || order > max_tail_order_) throw std::runtime_error("tails are known between min and max order, your order is outside.");
                return tails_[order];
            }

            /// Returns tail as a vector
            const std::vector<tail_type>& tail() const{
                return tails_;
            }

            five_index_gf_with_tail& set_tail(int order, const tail_type &tail){
                if(this->mesh2()!=tail.mesh1() || this->mesh3()!=tail.mesh2() || this->mesh4()!=tail.mesh3() || this->mesh5()!=tail.mesh4())
                    throw std::runtime_error("invalid mesh type in tail assignment");

                int tail_size=tails_.size();
                if(order>=tail_size){
                    tails_.resize(order+1, tail_type(this->mesh2(), this->mesh3(), this->mesh4(), this->mesh5()));
                    for(int i=tail_size;i<=order;++i) tails_[i].initialize();
                }
                tails_[order]=tail;

                //set minimum and maximum known coefficients if needed
                if(min_tail_order_==TAIL_NOT_SET || min_tail_order_>order) min_tail_order_=order;
                if(max_tail_order_==TAIL_NOT_SET || max_tail_order_<=order) max_tail_order_=order;
                return *this;
            }

            /// Save the GF with tail to HDF5
            void save(alps::hdf5::archive& ar, const std::string& path) const
            {
                gf_type::save(ar,path);
                ar[path+"/tail/descriptor"]="INFINITY_TAIL";
                ar[path+"/tail/min_tail_order"]=min_tail_order_;
                ar[path+"/tail/max_tail_order"]=max_tail_order_;
                if(min_tail_order_==TAIL_NOT_SET) return;
                for (int i=min_tail_order_; i<=max_tail_order_; ++i) {
                    ar[path+"/tail/"+boost::lexical_cast<std::string>(i)] << tails_[i].data();
                }
            }

            /// Load the GF with tail from HDF5
            void load(alps::hdf5::archive& ar, const std::string& path)
            {
                gf_type::load(ar,path);
                std::string descr; ar[path+"/tail/descriptor"] >> descr;
                if (descr!="INFINITY_TAIL") throw std::runtime_error("Wrong tail format '"+descr+"', expected INFINITY_TAIL");

                // FIXME!FIXME! Rewrite using clone-swap for exception safety.
                ar[path+"/tail/min_tail_order"] >> min_tail_order_;
                ar[path+"/tail/max_tail_order"] >> max_tail_order_;

                tails_.clear();
                if(min_tail_order_==TAIL_NOT_SET) return;
                if(min_tail_order_>0) tails_.resize(min_tail_order_,tail_type(this->mesh2(), this->mesh3(), this->mesh4()));

                typename tail_type::container_type buffer;
                for (int i=min_tail_order_; i<=max_tail_order_; ++i) {
                    ar[path+"/tail/"+boost::lexical_cast<std::string>(i)] >> buffer;
                    tails_.push_back(tail_type(this->mesh2(), this->mesh3(), this->mesh4(), this->mesh5(), buffer));
                }
            }

            /// Save the GF to HDF5
            void save(alps::hdf5::archive& ar) const
            {
                save(ar, ar.get_context());
            }
            
            /// Load the GF from HDF5
            void load(alps::hdf5::archive& ar)
            {
                load(ar, ar.get_context());
            }
            
        };

        typedef two_index_gf_with_tail<omega_sigma_gf, one_index_gf<double, index_mesh> > omega_sigma_gf_with_tail;
        typedef two_index_gf_with_tail<itime_sigma_gf, one_index_gf<double, index_mesh> > itime_sigma_gf_with_tail;

        typedef three_index_gf_with_tail<omega_k_sigma_gf, two_index_gf<double, momentum_index_mesh, index_mesh> > omega_k_sigma_gf_with_tail;
        typedef three_index_gf_with_tail<itime_k_sigma_gf, two_index_gf<double, momentum_index_mesh, index_mesh> > itime_k_sigma_gf_with_tail;

        typedef four_index_gf_with_tail<omega_k1_k2_sigma_gf, three_index_gf<double, momentum_index_mesh, momentum_index_mesh, index_mesh> > omega_k1_k2_sigma_gf_with_tail;
        typedef four_index_gf_with_tail<itime_k1_k2_sigma_gf, three_index_gf<double, momentum_index_mesh, momentum_index_mesh, index_mesh> > itime_k1_k2_sigma_gf_with_tail;

        typedef four_index_gf_with_tail<omega_k_sigma1_sigma2_gf, three_index_gf<double, momentum_index_mesh, index_mesh, index_mesh> > omega_k_sigma1_sigma2_gf_with_tail;
        typedef four_index_gf_with_tail<itime_k_sigma1_sigma2_gf, three_index_gf<double, momentum_index_mesh, index_mesh, index_mesh> > itime_k_sigma1_sigma2_gf_with_tail;

        typedef four_index_gf_with_tail<omega_r1_r2_sigma_gf, three_index_gf<double, real_space_index_mesh, real_space_index_mesh, index_mesh> > omega_r1_r2_sigma_gf_with_tail;
        typedef four_index_gf_with_tail<itime_r1_r2_sigma_gf, three_index_gf<double, real_space_index_mesh, real_space_index_mesh, index_mesh> > itime_r1_r2_sigma_gf_with_tail;
        typedef four_index_gf_with_tail<itime_r1_r2_sigma_complex_gf, three_index_gf<double, real_space_index_mesh, real_space_index_mesh, index_mesh> > itime_r1_r2_sigma_complex_gf_with_tail;

        typedef five_index_gf_with_tail<omega_r1_r2_sigma1_sigma2_gf, four_index_gf<double, real_space_index_mesh, real_space_index_mesh, index_mesh,index_mesh> > omega_r1_r2_sigma1_sigma2_gf_with_tail;
        typedef five_index_gf_with_tail<itime_r1_r2_sigma1_sigma2_gf, four_index_gf<double, real_space_index_mesh, real_space_index_mesh, index_mesh,index_mesh> > itime_r1_r2_sigma1_sigma2_gf_with_tail;
    } // end namespace gf
} // end alps::
