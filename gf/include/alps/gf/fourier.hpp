/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once
#include <alps/gf/gf.hpp>

namespace alps {
namespace gf {

  inline std::complex<double> f_omega(double wn, double c1, double c2, double c3) {
    std::complex<double> iwn(0., wn);
    std::complex<double> iwnsq=iwn*iwn;
    return c1/iwn + c2/(iwnsq) + c3/(iwn*iwnsq);
  }

  inline double f_tau(double tau, double beta, double c1, double c2, double c3) {
    return -0.5*c1 + (c2*0.25)*(-beta+2.*tau) + (c3*0.25)*(beta*tau-tau*tau);
  }


  template<size_t D>
  inline alps::numerics::tensor<std::complex<double>, D> f_omega(double wn, const alps::numerics::tensor<double, D>& c1, const alps::numerics::tensor<double, D>& c2, const alps::numerics::tensor<double, D>&c3) {
    std::complex<double> iwn(0., wn);
    std::complex<double> iwnsq=iwn*iwn;
    return c1/iwn + c2/(iwnsq) + c3/(iwn*iwnsq);
  }

  template<size_t D>
  inline alps::numerics::tensor<double, D> f_tau(double tau, double beta,
                                                 const alps::numerics::tensor<double, D>& c1,
                                                 const alps::numerics::tensor<double, D>& c2,
                                                 const alps::numerics::tensor<double, D>& c3) {
    return c1 *(-0.5) + (c2*0.25)*(-beta+2.*tau) + (c3*0.25)*(beta*tau-tau*tau);
  }

  ///Fourier transform kernel of the omega -> tau transform
  template<size_t D>
  inline void transform_vector_no_tail_matrix(const alps::numerics::tensor<std::complex<double>, D>&input_data, const std::vector<double> &omega,
                                              alps::numerics::tensor<double, D> &output_data, const std::vector<double> &tau, double beta){
    using Matrix  = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixX = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Matrix Cos(tau.size(), omega.size());
    Matrix Sin(tau.size(), omega.size());
    for (size_t i=0; i<tau.size(); ++i) {
      for (size_t k=0; k<omega.size(); ++k) {
        double wt=omega[k]*tau[i];
        Cos(i, k) = cos(wt);
        Sin(i, k) = sin(wt);
      }
    }
    size_t ld_in = omega.size();
    size_t rest_in = input_data.size() / ld_in;
    size_t ld_out = tau.size();
    size_t rest_out = output_data.size() / ld_out;

    assert(rest_in == rest_out);

    Eigen::Map<const MatrixX> In(input_data.data(), ld_in, rest_in);
    Eigen::Map<Matrix>        Out(output_data.data(), ld_out, rest_out);

    // + sign comes from the -i in the phase
    Out = 2.0*(Cos * In.real() + Sin * In.imag())/beta;
  }

  ///Fourier transform kernel of the omega -> tau transform
  template<size_t D>
  inline void transform_vector_no_tail_loop(const alps::numerics::tensor<std::complex<double>, D> &input_data,
                                            const std::vector<double> &omega,
                                            alps::numerics::tensor<double, D> &output_data, const std::vector<double> &tau, double beta){
    size_t ld_in = omega.size();
    size_t rest_in = input_data.size() / ld_in;
#ifndef NDEBUG
    size_t ld_out = tau.size();
    size_t rest_out = output_data.size() / ld_out;
#endif
    assert(rest_in == rest_out);

    output_data.set_zero();
    for (size_t r = 0; r < rest_in; ++r) {
      for (size_t i=0; i<tau.size(); ++i) {
        for (size_t k = 0; k < omega.size(); ++k) {
          double wt=omega[k]*tau[i];
          output_data.data()[i * rest_in + r] += 2.0 * (cos(wt) * input_data.data()[k * rest_in + r].real() + sin(wt) * input_data.data()[k * rest_in + r].imag()) / beta;
        }
      }
    }
  }

  ///Fourier transform a matsubara gf to an imag time gf
  template<class...MESHES> void fourier_frequency_to_time(
      const gf_tail<
      detail::gf_base<std::complex<double>, numerics::tensor<std::complex<double>, sizeof...(MESHES) + 1>, matsubara_positive_mesh, MESHES...>,
      greenf<double,MESHES...> > &g_omega,
      gf_tail<
      detail::gf_base<double, numerics::tensor<double, sizeof...(MESHES) + 1>, itime_mesh, MESHES...>,
      greenf<double, MESHES...> > &g_tau){
    alps::numerics::tensor<std::complex<double>, (sizeof...(MESHES)) + 1> in_data(g_omega.data().shape());

    using tail_data = alps::numerics::tensor<double, sizeof...(MESHES)>;

    std::array<size_t, sizeof...(MESHES)> tail_shape;
    for (size_t i = 0; i < sizeof...(MESHES); ++i) {
      tail_shape[i] = g_omega.data().shape()[i+1];
    }

    tail_data zero_tail(tail_shape);

    const alps::numerics::tensor<double, sizeof...(MESHES)>& c0=(g_omega.min_tail_order()==0 && g_omega.max_tail_order()>=0 )? g_omega.tail(0).data():zero_tail;
    const alps::numerics::tensor<double, sizeof...(MESHES)>& c1=(g_omega.min_tail_order()<=1 && g_omega.max_tail_order()>=1 )? g_omega.tail(1).data():zero_tail;
    const alps::numerics::tensor<double, sizeof...(MESHES)>& c2=(g_omega.min_tail_order()<=2 && g_omega.max_tail_order()>=2 )? g_omega.tail(2).data():zero_tail;
    const alps::numerics::tensor<double, sizeof...(MESHES)>& c3=(g_omega.min_tail_order()<=3 && g_omega.max_tail_order()>=3 )? g_omega.tail(3).data():zero_tail;
    for (size_t i = 0; i < c0.size(); ++i) {
      if(c0.data()[i] != 0) throw std::runtime_error("attempt to Fourier transform an object which goes to a constant. FT is ill defined");
    }
    for(int n=0;n<g_omega.mesh1().extent();++n) {
      in_data(size_t(n)) = g_omega(matsubara_index(size_t(n))).data() - f_omega(g_omega.mesh1().points()[size_t(n)],c1,c2,c3);
    }

    if(g_omega.mesh1().extent() * g_tau.mesh1().extent() > 5000000 || g_omega.data().size()/g_omega.mesh1().extent() < 100)
      transform_vector_no_tail_loop(in_data,g_omega.mesh1().points(), g_tau.data(), g_tau.mesh1().points(), g_tau.mesh1().beta());
    else
      transform_vector_no_tail_matrix(in_data,g_omega.mesh1().points(), g_tau.data(), g_tau.mesh1().points(), g_tau.mesh1().beta());

    for(int t=0;t<g_tau.mesh1().extent();++t){
      g_tau(itime_index(t)).data() += f_tau(g_tau.mesh1().points()[t],g_tau.mesh1().beta(),c1,c2,c3);
    }
  }
}
} // end alps::
