#pragma once
#include "gf.hpp"

namespace alps {
    namespace gf {

        /// Green's function (ot type GFT) with a tail (which is a Green's function of type TAILT)
        /** The *first* mesh of GFT is assumed to be a Matsubara frquency or imaginary time mesh */
        template <typename GFT, typename TAILT>
        class three_index_gf_with_tail : public GFT {
            public:
            typedef TAILT tail_type;
            typedef GFT gf_type;

            private:
            std::vector<tail_type> tails_;

            // If you see an error here, the first mesh of your GFT cannot have a tail!
            static const typename detail::can_have_tail<typename gf_type::mesh1_type>::mesh_can_have_tail mesh1_can_have_tail=true;
            // If you see an error here, the second mesh of your GFT should not have a tail!
            static const typename detail::can_have_tail<typename gf_type::mesh2_type>::mesh_cannot_have_tail mesh2_cannot_have_tail=true;
            // If you see an error here, the third mesh of your GFT should not have a tail!
            static const typename detail::can_have_tail<typename gf_type::mesh3_type>::mesh_cannot_have_tail mesh3_cannot_have_tail=true;
            
            public:

            three_index_gf_with_tail(const gf_type& gf): gf_type(gf)
            { }
            
            const tail_type& tail(int order) const{
                return tails_[order];
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
                return *this;
            }
    
        };

        typedef three_index_gf_with_tail<omega_k_sigma_gf, two_index_gf<double, momentum_index_mesh, index_mesh> > omega_k_sigma_gf_with_tail;
        typedef three_index_gf_with_tail<itime_k_sigma_gf, two_index_gf<double, momentum_index_mesh, index_mesh> > itime_k_sigma_gf_with_tail;
    } // end namespace gf
} // end alps::
