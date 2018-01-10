/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#include <alps/gf/gf.hpp>
#include <alps/gf/tail.hpp>

#include <boost/array.hpp>

namespace g=alps::gf;

// Generates points for momentum mesh
g::momentum_index_mesh::container_type generate_momentum_mesh();

void Demo() {
    const int nspins=2;
    const int nfreq=10;
    const double beta=5;

    // Construct the meshes:
    g::matsubara_positive_mesh m_mesh(beta, nfreq);
    g::momentum_index_mesh k_mesh(generate_momentum_mesh());
    g::index_mesh s_mesh(nspins);
    
    // construct a GF using a pre-defined convenience type
    g::omega_k_sigma_gf gf(m_mesh, k_mesh, s_mesh);
    // initialize a GF to all-zeros
    gf.initialize();

    // Make indices:
    g::matsubara_index omega;
    omega=4;
    g::momentum_index ii(2);
    g::index sigma(0);

    // Assign a GF element:
    gf(omega,ii,sigma)=std::complex<double>(3,4);

    // Density matrix as a double-valued GF
    // on momentum and integer-index space:
    typedef
        g::two_index_gf<double, g::momentum_index_mesh, g::index_mesh>
        density_matrix_type;
    
    // Construct the object:
    density_matrix_type denmat=density_matrix_type(k_mesh,s_mesh);

    // prepare diagonal matrix
    const double U=3.0;
    denmat.initialize();
    // loop over first mesh index:
    for (g::momentum_index i=g::momentum_index(0);
         i<denmat.mesh1().extent(); ++i) {
        denmat(i,g::index(0))=0.5*U;
        denmat(i,g::index(1))=0.5*U;
    }

    // construct a tailed GF using predefined convenience type:
    g::omega_k_sigma_gf_with_tail gft(gf); 
    gft.set_tail(0, denmat);              // set the tail

    density_matrix_type gftail=gft.tail(0); // retrieve the tail

    // access the tailed GF element
    std::complex<double> x=gft(omega,ii,sigma);
}

/// Generates 4 2-D points k_1,...,k_4 where each point k=(k_x,k_y)
g::momentum_index_mesh::container_type generate_momentum_mesh() {
    const boost::array<int,2> dimensions={{4,2}};
    g::momentum_index_mesh::container_type mesh_points(dimensions);
    mesh_points[0][0]=0;    mesh_points[0][1]=0;    // (0,0)
    mesh_points[1][0]=M_PI; mesh_points[1][1]=M_PI; // (pi,pi)
    mesh_points[2][0]=M_PI; mesh_points[2][1]=0;    // (pi,0)
    mesh_points[3][0]=0;    mesh_points[3][1]=M_PI; // (0,pi)
    return mesh_points;
}
