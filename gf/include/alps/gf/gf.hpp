/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPSCORE_NEW_GF_H
#define ALPSCORE_NEW_GF_H

#include <alps/gf/gf_base.hpp>
#include <alps/gf/tail_base.hpp>
#include <alps/gf/mesh.hpp>

namespace alps {namespace gf {

    template<class VTYPE, class MESH1>
    using one_index_gf = greenf<VTYPE, MESH1>;
    template<class VTYPE, class MESH1, class MESH2>
    using two_index_gf = greenf<VTYPE, MESH1, MESH2>;
    template<class VTYPE, class MESH1, class MESH2, class MESH3>
    using three_index_gf = greenf<VTYPE, MESH1, MESH2, MESH3>;
    template<class VTYPE, class MESH1, class MESH2, class MESH3, class MESH4>
    using four_index_gf = greenf<VTYPE, MESH1, MESH2, MESH3, MESH4>;
    template<class VTYPE, class MESH1, class MESH2, class MESH3, class MESH4, class MESH5>
    using five_index_gf = greenf<VTYPE, MESH1, MESH2, MESH3, MESH4, MESH5>;
    template<class VTYPE, class MESH1, class MESH2, class MESH3, class MESH4, class MESH5, class MESH6>
    using six_index_gf = greenf<VTYPE, MESH1, MESH2, MESH3, MESH4, MESH5, MESH6>;
    template<class VTYPE, class MESH1, class MESH2, class MESH3, class MESH4, class MESH5, class MESH6, class MESH7>
    using seven_index_gf = greenf<VTYPE, MESH1, MESH2, MESH3, MESH4, MESH5, MESH6, MESH7>;

    typedef greenf<std::complex<double>, matsubara_mesh<mesh::POSITIVE_ONLY>, real_space_index_mesh,real_space_index_mesh, index_mesh, index_mesh> omega_r1_r2_sigma1_sigma2_gf;
    typedef greenf<             double , itime_mesh    , real_space_index_mesh,real_space_index_mesh, index_mesh, index_mesh> itime_r1_r2_sigma1_sigma2_gf;

    typedef greenf<std::complex<double>, matsubara_mesh<mesh::POSITIVE_ONLY>, momentum_index_mesh, momentum_index_mesh, index_mesh> omega_k1_k2_sigma_gf;
    typedef greenf<             double , itime_mesh    , momentum_index_mesh, momentum_index_mesh, index_mesh> itime_k1_k2_sigma_gf;
    typedef greenf<std::complex<double>, matsubara_mesh<mesh::POSITIVE_ONLY>, real_space_index_mesh, real_space_index_mesh, index_mesh> omega_r1_r2_sigma_gf;
    typedef greenf<             double , itime_mesh    , real_space_index_mesh, real_space_index_mesh, index_mesh> itime_r1_r2_sigma_gf;
    typedef greenf<std::complex<double>, itime_mesh    , real_space_index_mesh, real_space_index_mesh, index_mesh> itime_r1_r2_sigma_complex_gf;
    typedef greenf<std::complex<double>, matsubara_mesh<mesh::POSITIVE_ONLY>, momentum_index_mesh, index_mesh, index_mesh> omega_k_sigma1_sigma2_gf;
    typedef greenf<             double , itime_mesh    , momentum_index_mesh, index_mesh, index_mesh> itime_k_sigma1_sigma2_gf;

    typedef greenf<std::complex<double>, matsubara_mesh<mesh::POSITIVE_ONLY>, momentum_index_mesh, index_mesh> omega_k_sigma_gf;
    typedef greenf<             double , itime_mesh    , momentum_index_mesh, index_mesh> itime_k_sigma_gf;
    typedef greenf<             double , momentum_index_mesh, index_mesh, index_mesh> k_sigma1_sigma2_gf;

    typedef greenf<std::complex<double>, matsubara_mesh<mesh::POSITIVE_ONLY>, index_mesh> omega_sigma_gf;
    typedef greenf<             double , itime_mesh, index_mesh> itime_sigma_gf;
    typedef greenf<double, momentum_index_mesh, index_mesh> k_sigma_gf;

    typedef greenf<std::complex<double>, matsubara_mesh<mesh::POSITIVE_ONLY> >omega_gf;
    typedef greenf<             double , itime_mesh> itime_gf;
    typedef greenf<             double , index_mesh> sigma_gf;
    typedef greenf<             double , legendre_mesh> legendre_gf;
    typedef greenf<std::complex<double>, legendre_mesh> complex_legendre_gf;

    typedef omega_k1_k2_sigma_gf matsubara_gf;

    template <typename GFT, typename TAILT>
    using two_index_gf_with_tail = gf_tail<GFT, TAILT>;
    template <typename GFT, typename TAILT>
    using three_index_gf_with_tail = gf_tail<GFT, TAILT>;
    template <typename GFT, typename TAILT>
    using four_index_gf_with_tail = gf_tail<GFT, TAILT>;
    template <typename GFT, typename TAILT>
    using five_index_gf_with_tail = gf_tail<GFT, TAILT>;

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

  }
}


#endif //ALPSCORE_NEW_GF_H