/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "gtest/gtest.h"
#include <alps/gf/gf.hpp>
#include "alps/gf/fourier.hpp"

class AtomicFourierTestGF : public ::testing::Test
{
public:
  const double beta;
  double U;
  double mu;
  const int nfreq;
  const int ntau;
  typedef alps::gf::omega_sigma_gf_with_tail matsubara_gf_type;
  typedef alps::gf::itime_sigma_gf_with_tail itime_gf_type;
  typedef alps::gf::one_index_gf<double, alps::gf::index_mesh> density_matrix_type;
  matsubara_gf_type g_omega;
  itime_gf_type g_tau;
  matsubara_gf_type gf2;
  itime_gf_type g_tau_2;

  AtomicFourierTestGF():beta(10), U(5), mu(2.5), nfreq(1000),ntau(1001),
      g_omega(alps::gf::omega_sigma_gf(alps::gf::matsubara_positive_mesh(beta,nfreq),
          alps::gf::index_mesh(2))),
          g_tau(alps::gf::itime_sigma_gf(alps::gf::itime_mesh(beta,ntau),alps::gf::index_mesh(2))),
          gf2(g_omega),
          g_tau_2(g_tau){}
  double Z(){
    return 1. + 2*std::exp(beta*mu) + std::exp(beta*(2*mu - U));
  }
  double density(){
    return (std::exp(beta*mu)+std::exp(beta*(2*mu-U)))/Z();
  }
  double wn(int n){
    return (2.*n+1)*M_PI/beta;
  }
  std::complex<double> iwn(int n){
    return std::complex<double>(0., wn(n));
  }
  std::complex<double> atomic_matsubara(int n){
    return (1-density())/(iwn(n)+mu) +density()/(iwn(n)+mu-U);
  }
  double atomic_itime(double tau){
    return -1/Z()*(std::exp(tau*mu)+std::exp(-(beta-tau)*(-mu))*std::exp(-tau*(U-2*mu)));
  }
  double tau(int n){
    return beta/(ntau-1)*n;
  }

  void initialize_as_atomic_matsubara(matsubara_gf_type &g){
    for(alps::gf::matsubara_positive_mesh::index_type n(0);n<nfreq;++n){
      g(n,alps::gf::index(0))=atomic_matsubara(n());
      g(n,alps::gf::index(1))=atomic_matsubara(n());
    }
  }
  void initialize_as_atomic_itime(itime_gf_type &g){
    for(alps::gf::itime_mesh::index_type n(0);n<ntau;++n){
      g(n,alps::gf::index(0))=atomic_itime(tau(n()));
      g(n,alps::gf::index(1))=atomic_itime(tau(n()));
    }
  }
};

TEST_F(AtomicFourierTestGF,ZeroIsDensity){
  initialize_as_atomic_itime(g_tau);
  EXPECT_NEAR(g_tau(alps::gf::itime_mesh::index_type(0     ),alps::gf::index(0)), -(1-density()), 1.e-8);
  EXPECT_NEAR(g_tau(alps::gf::itime_mesh::index_type(ntau-1),alps::gf::index(0)),    -density() , 1.e-8);
}
TEST_F(AtomicFourierTestGF,MatsubaraToTimeFourierHalfFilling){
  initialize_as_atomic_matsubara(g_omega);
  density_matrix_type unity=density_matrix_type(alps::gf::index_mesh(2));
  unity.initialize();
  unity(alps::gf::index(0))=1;
  unity(alps::gf::index(1))=1;
  g_omega.set_tail(1,unity);

  fourier_frequency_to_time(g_omega, g_tau);

  initialize_as_atomic_itime(g_tau_2);

  EXPECT_NEAR((g_tau-g_tau_2).norm(), 0, 1.e-5);
  //that's the missing c2 term
  EXPECT_NEAR(U*density()-mu, 0, 1.e-5);
}
TEST_F(AtomicFourierTestGF,MatsubaraToTimeFourierAwayHalfFilling){
  mu=0; //this makes it away from half filling
  U=0.2;
  initialize_as_atomic_matsubara(g_omega);
  density_matrix_type unity=density_matrix_type(alps::gf::index_mesh(2));
  unity.initialize();
  unity(alps::gf::index(0))=1;
  unity(alps::gf::index(1))=1;
  g_omega.set_tail(1,unity);

  density_matrix_type c2=density_matrix_type(alps::gf::index_mesh(2));
  c2.initialize();
  c2(alps::gf::index(0))=U*density()-mu;
  c2(alps::gf::index(1))=U*density()-mu;
  g_omega.set_tail(2,c2);


  fourier_frequency_to_time(g_omega, g_tau);

  initialize_as_atomic_itime(g_tau_2);

  EXPECT_NEAR((g_tau-g_tau_2).norm(), 0, 1.e-7);
}
