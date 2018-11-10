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

class NoninteractingFourierTestGF : public ::testing::Test
{
public:
  const double beta;
  double mu;
  const int nfreq;
  const int ntau;
  const int nk;
  typedef alps::gf::omega_k_sigma_gf_with_tail matsubara_gf_type;
  typedef alps::gf::itime_k_sigma_gf_with_tail itime_gf_type;
  typedef alps::gf::two_index_gf<double, alps::gf::momentum_index_mesh, alps::gf::index_mesh> tail_type;
  matsubara_gf_type g_omega;
  itime_gf_type g_tau;
  itime_gf_type g_tau_2;

  NoninteractingFourierTestGF():beta(10), nk(200), mu(1.0), nfreq(2000),ntau(2001),
                        g_omega(alps::gf::omega_k_sigma_gf(alps::gf::matsubara_positive_mesh(beta,nfreq),
                                                           alps::gf::momentum_index_mesh(nk, 1),
                                                           alps::gf::index_mesh(2))),
                        g_tau(alps::gf::itime_k_sigma_gf(alps::gf::itime_mesh(beta,ntau),
                                                         alps::gf::momentum_index_mesh(nk, 1),
                                                         alps::gf::index_mesh(2))), g_tau_2(g_tau){}
  double wn(int n){
    return (2.*n+1)*M_PI/beta;
  }
  std::complex<double> iwn(int n){
    return std::complex<double>(0., wn(n));
  }
  double epsilon(int k) {
    return 2.0*cos(2.0*k*M_PI/nk);
  }
  std::complex<double> matsubara(int n, int k){
    return 1.0/(iwn(n) + mu - epsilon(k));
  }
  double itime(double tau, int k){
    double epsk_m_mu = epsilon(k) - mu;
    double fermi_1 = -1.0/(1.0 + std::exp(-beta * epsk_m_mu) );
    double fermi_2 = -1.0/(1.0 + std::exp( beta * epsk_m_mu) );
    return (epsk_m_mu<0 ? std::exp((beta-tau) * epsk_m_mu) * fermi_2:
                          std::exp(-tau       * epsk_m_mu) * fermi_1);
  }
  double tau(int n){
    return beta/(ntau-1)*n;
  }

  void initialize_matsubara(matsubara_gf_type &g){
    for(alps::gf::matsubara_positive_mesh::index_type n(0);n<nfreq;++n){
      for (alps::gf::momentum_index ik(0); ik < nk; ++ik) {
        g(n, ik, alps::gf::index(0))=matsubara(n(), ik());
        g(n, ik, alps::gf::index(1))=matsubara(n(), ik());
      }
    }
  }
  void initialize_itime(itime_gf_type &g){
    for(alps::gf::itime_mesh::index_type n(0);n<ntau;++n){
      for (alps::gf::momentum_index ik(0); ik < nk; ++ik) {
        g(n, ik, alps::gf::index(0)) = itime(tau(n()), ik());
        g(n, ik, alps::gf::index(1)) = itime(tau(n()), ik());
      }
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

TEST_F(NoninteractingFourierTestGF,MatsubaraToTimeFourier){
  initialize_matsubara(g_omega);
  tail_type unity=tail_type(alps::gf::momentum_index_mesh(nk, 1), alps::gf::index_mesh(2));
  tail_type second=tail_type(alps::gf::momentum_index_mesh(nk, 1), alps::gf::index_mesh(2));
  tail_type third=tail_type(alps::gf::momentum_index_mesh(nk, 1), alps::gf::index_mesh(2));
  unity.initialize();
  second.initialize();
  unity.data().set_number(1.0);
  for (int ik = 0; ik < nk; ++ik) {
    second.data()(ik, 0) = epsilon(ik) - mu;
    second.data()(ik, 1) = epsilon(ik) - mu;
    third.data()(ik, 0)  = (epsilon(ik) - mu) * (epsilon(ik) - mu);
    third.data()(ik, 1)  = (epsilon(ik) - mu) * (epsilon(ik) - mu);
  }
  g_omega.set_tail(1,unity);
  g_omega.set_tail(2,second);
  g_omega.set_tail(3,third);
  fourier_frequency_to_time(g_omega, g_tau);

  initialize_itime(g_tau_2);

  EXPECT_NEAR((g_tau-g_tau_2).norm(), 0, 1.e-7);
}

TEST(FourierTestGF, FourierStrategy) {
  double beta = 100;
  int nts = 1001;
  int iwmax = 1000;
  int nk = 10;
  double mu = 0.1;
  alps::gf::omega_k_sigma_gf matsubara_gf(alps::gf::matsubara_positive_mesh(beta,iwmax),
                                          alps::gf::momentum_index_mesh(nk, 1),
                                          alps::gf::index_mesh(2));
  alps::gf::itime_k_sigma_gf itime_gf_1(alps::gf::itime_mesh(beta,nts),
                                           alps::gf::momentum_index_mesh(nk, 1),
                                           alps::gf::index_mesh(2));
  alps::gf::itime_k_sigma_gf itime_gf_2(alps::gf::itime_mesh(beta,nts),
                                           alps::gf::momentum_index_mesh(nk, 1),
                                           alps::gf::index_mesh(2));
  for (alps::gf::matsubara_index iw(0); iw < matsubara_gf.mesh1().extent(); ++iw) {
    for (alps::gf::momentum_index ik(0); ik < matsubara_gf.mesh2().extent(); ++ik) {
      matsubara_gf(iw, ik, alps::gf::index(0)) = 1.0 / (std::complex<double>(0., matsubara_gf.mesh1().points()[iw()]) + mu - cos(2.0 * ik() * M_PI / nk) );
      matsubara_gf(iw, ik, alps::gf::index(1)) = 1.0/  (std::complex<double>(0., matsubara_gf.mesh1().points()[iw()]) - mu - cos(2.0 * ik() * M_PI / nk) );
    }
  }
  alps::gf::transform_vector_no_tail_loop(matsubara_gf.data(),matsubara_gf.mesh1().points(), itime_gf_1.data(), itime_gf_1.mesh1().points(), itime_gf_1.mesh1().beta());
  alps::gf::transform_vector_no_tail_matrix(matsubara_gf.data(),matsubara_gf.mesh1().points(), itime_gf_2.data(), itime_gf_2.mesh1().points(), itime_gf_2.mesh1().beta());
  EXPECT_NEAR((itime_gf_1-itime_gf_2).norm(), 0, 1.e-11);
}
