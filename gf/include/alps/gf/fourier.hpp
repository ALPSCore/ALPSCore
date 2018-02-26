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

///Fourier transform kernel of the omega -> tau transform
inline void transform_vector_no_tail(const std::vector<std::complex<double> >&input_data, const std::vector<double> &omega, std::vector<double> &output_data, const std::vector<double> &tau, double beta){
  for (unsigned int i=0; i<output_data.size(); ++i) {
    output_data[i] = 0.;
    for (unsigned int k=0; k<input_data.size(); ++k) {
      double wt=omega[k]*tau[i];
      output_data[i]+= 2/beta*(cos(wt)*input_data[k].real()+sin(wt)*input_data[k].imag());
    }
  }
}
///Fourier transform a two-index matsubara gf to an imag time gf
template<class MESH1> void fourier_frequency_to_time(const two_index_gf_with_tail<
    two_index_gf<std::complex<double>, matsubara_positive_mesh, MESH1>, one_index_gf<double, MESH1> > &g_omega,
    two_index_gf_with_tail<two_index_gf<double, itime_mesh, MESH1>, one_index_gf<double, MESH1> > &g_tau){

  std::vector<std::complex<double> > input_data(g_omega.mesh1().extent(), 0.);
  std::vector<double >               output_data(g_tau.mesh1().extent(), 0.);

  for(int i=0;i<g_omega.mesh2().extent();++i){
    double c0=(g_omega.min_tail_order()==0 && g_omega.max_tail_order()>=0 )? g_omega.tail(0)(typename MESH1::index_type(i)):0;
    double c1=(g_omega.min_tail_order()<=1 && g_omega.max_tail_order()>=1 )? g_omega.tail(1)(typename MESH1::index_type(i)):0;
    double c2=(g_omega.min_tail_order()<=2 && g_omega.max_tail_order()>=2 )? g_omega.tail(2)(typename MESH1::index_type(i)):0;
    double c3=(g_omega.min_tail_order()<=3 && g_omega.max_tail_order()>=3 )? g_omega.tail(3)(typename MESH1::index_type(i)):0;
    if(c0 != 0) throw std::runtime_error("attempt to Fourier transform an object which goes to a constant. FT is ill defined");
    for(int n=0;n<g_omega.mesh1().extent();++n){
      input_data[n]=g_omega(matsubara_index(n),typename MESH1::index_type(i))-f_omega(g_omega.mesh1().points()[n],c1,c2,c3);
    }
    transform_vector_no_tail(input_data,g_omega.mesh1().points(), output_data, g_tau.mesh1().points(), g_tau.mesh1().beta());
    for(int t=0;t<g_tau.mesh1().extent();++t){
      g_tau(itime_index(t),typename MESH1::index_type(i))=output_data[t]+f_tau(g_tau.mesh1().points()[t],g_tau.mesh1().beta(),c1,c2,c3);
    }
  }
}
///Fourier transform a three-index matsubara gf to an imag time gf
template<class MESH1, class MESH2> void fourier_frequency_to_time(const three_index_gf_with_tail<
    three_index_gf<std::complex<double>, matsubara_positive_mesh, MESH1,MESH2>, two_index_gf<double, MESH1,MESH2> > &g_omega,
    three_index_gf_with_tail<three_index_gf<double, itime_mesh, MESH1,MESH2>, two_index_gf<double, MESH1,MESH2> > &g_tau){

  std::vector<std::complex<double> > input_data(g_omega.mesh1().extent(), 0.);
  std::vector<double >               output_data(g_tau.mesh1().extent(), 0.);

  for(int i=0;i<g_omega.mesh2().extent();++i){
    for(int j=0;j<g_omega.mesh3().extent();++j){
      double c0=(g_omega.min_tail_order()==0 && g_omega.max_tail_order()>=0 )? g_omega.tail(0)(typename MESH1::index_type(i),typename MESH2::index_type(j)):0;
      double c1=(g_omega.min_tail_order()<=1 && g_omega.max_tail_order()>=1 )? g_omega.tail(1)(typename MESH1::index_type(i),typename MESH2::index_type(j)):0;
      double c2=(g_omega.min_tail_order()<=2 && g_omega.max_tail_order()>=2 )? g_omega.tail(2)(typename MESH1::index_type(i),typename MESH2::index_type(j)):0;
      double c3=(g_omega.min_tail_order()<=3 && g_omega.max_tail_order()>=3 )? g_omega.tail(3)(typename MESH1::index_type(i),typename MESH2::index_type(j)):0;
      if(c0 != 0) throw std::runtime_error("attempt to Fourier transform an object which goes to a constant. FT is ill defined");
      for(int n=0;n<g_omega.mesh1().extent();++n){
        input_data[n]=g_omega(matsubara_index(n),typename MESH1::index_type(i),typename MESH2::index_type(j))-f_omega(g_omega.mesh1().points()[n],c1,c2,c3);
      }
      transform_vector_no_tail(input_data,g_omega.mesh1().points(), output_data, g_tau.mesh1().points(), g_tau.mesh1().beta());
      for(int t=0;t<g_tau.mesh1().extent();++t){
        g_tau(itime_index(t),typename MESH1::index_type(i),typename MESH2::index_type(j))=output_data[t]+f_tau(g_tau.mesh1().points()[t],g_tau.mesh1().beta(),c1,c2,c3);
      }
    }
  }
}
///Fourier transform a four-index matsubara gf to an imag time gf
template<class MESH1, class MESH2, class MESH3> void fourier_frequency_to_time(const four_index_gf_with_tail<
    four_index_gf<std::complex<double>, matsubara_positive_mesh, MESH1,MESH2,MESH3>, three_index_gf<double, MESH1,MESH2,MESH3> > &g_omega,
    four_index_gf_with_tail<four_index_gf<double, itime_mesh, MESH1,MESH2,MESH3>, three_index_gf<double, MESH1,MESH2,MESH3> > &g_tau){

  std::vector<std::complex<double> > input_data(g_omega.mesh1().extent(), 0.);
  std::vector<double >               output_data(g_tau.mesh1().extent(), 0.);

  for(int i=0;i<g_omega.mesh2().extent();++i){
    for(int j=0;j<g_omega.mesh3().extent();++j){
      for(int k=0;k<g_omega.mesh4().extent();++k){
        double c0=(g_omega.min_tail_order()==0 && g_omega.max_tail_order()>=0 )? g_omega.tail(0)(typename MESH1::index_type(i),typename MESH2::index_type(j),typename MESH3::index_type(k)):0;
        double c1=(g_omega.min_tail_order()<=1 && g_omega.max_tail_order()>=1 )? g_omega.tail(1)(typename MESH1::index_type(i),typename MESH2::index_type(j),typename MESH3::index_type(k)):0;
        double c2=(g_omega.min_tail_order()<=2 && g_omega.max_tail_order()>=2 )? g_omega.tail(2)(typename MESH1::index_type(i),typename MESH2::index_type(j),typename MESH3::index_type(k)):0;
        double c3=(g_omega.min_tail_order()<=3 && g_omega.max_tail_order()>=3 )? g_omega.tail(3)(typename MESH1::index_type(i),typename MESH2::index_type(j),typename MESH3::index_type(k)):0;
        if(c0 != 0) throw std::runtime_error("attempt to Fourier transform an object which goes to a constant. FT is ill defined");
        for(int n=0;n<g_omega.mesh1().extent();++n){
          input_data[n]=g_omega(matsubara_index(n),typename MESH1::index_type(i),typename MESH2::index_type(j),typename MESH3::index_type(k))-f_omega(g_omega.mesh1().points()[n],c1,c2,c3);
        }
        transform_vector_no_tail(input_data,g_omega.mesh1().points(), output_data, g_tau.mesh1().points(), g_tau.mesh1().beta());
        for(int t=0;t<g_tau.mesh1().extent();++t){
          g_tau(itime_index(t),typename MESH1::index_type(i),typename MESH2::index_type(j),typename MESH3::index_type(k))=output_data[t]+f_tau(g_tau.mesh1().points()[t],g_tau.mesh1().beta(),c1,c2,c3);
        }
      }
    }
  }
}
///Fourier transform a five-index matsubara gf to an imag time gf
template<class MESH1, class MESH2, class MESH3,class MESH4> void fourier_frequency_to_time(const five_index_gf_with_tail<
    five_index_gf<std::complex<double>, matsubara_positive_mesh, MESH1,MESH2,MESH3,MESH4>, four_index_gf<double, MESH1,MESH2,MESH3,MESH4> > &g_omega,
    five_index_gf_with_tail<five_index_gf<double, itime_mesh, MESH1,MESH2,MESH3,MESH4>, four_index_gf<double, MESH1,MESH2,MESH3,MESH4> > &g_tau){

  std::vector<std::complex<double> > input_data(g_omega.mesh1().extent(), 0.);
  std::vector<double >               output_data(g_tau.mesh1().extent(), 0.);

  for(int i=0;i<g_omega.mesh2().extent();++i){
    for(int j=0;j<g_omega.mesh3().extent();++j){
      for(int k=0;k<g_omega.mesh4().extent();++k){
        for(int l=0;l<g_omega.mesh5().extent();++l){
        double c0=(g_omega.min_tail_order()==0 && g_omega.max_tail_order()>=0 )? g_omega.tail(0)(typename MESH1::index_type(i),typename MESH2::index_type(j),typename MESH3::index_type(k),typename MESH4::index_type(l)):0;
        double c1=(g_omega.min_tail_order()<=1 && g_omega.max_tail_order()>=1 )? g_omega.tail(1)(typename MESH1::index_type(i),typename MESH2::index_type(j),typename MESH3::index_type(k),typename MESH4::index_type(l)):0;
        double c2=(g_omega.min_tail_order()<=2 && g_omega.max_tail_order()>=2 )? g_omega.tail(2)(typename MESH1::index_type(i),typename MESH2::index_type(j),typename MESH3::index_type(k),typename MESH4::index_type(l)):0;
        double c3=(g_omega.min_tail_order()<=3 && g_omega.max_tail_order()>=3 )? g_omega.tail(3)(typename MESH1::index_type(i),typename MESH2::index_type(j),typename MESH3::index_type(k),typename MESH4::index_type(l)):0;
        if(c0 != 0) throw std::runtime_error("attempt to Fourier transform an object which goes to a constant. FT is ill defined");
        for(int n=0;n<g_omega.mesh1().extent();++n){
          input_data[n]=g_omega(matsubara_index(n),typename MESH1::index_type(i),typename MESH2::index_type(j),typename MESH3::index_type(k),typename MESH4::index_type(l))-f_omega(g_omega.mesh1().points()[n],c1,c2,c3);
        }
        transform_vector_no_tail(input_data,g_omega.mesh1().points(), output_data, g_tau.mesh1().points(), g_tau.mesh1().beta());
        for(int t=0;t<g_tau.mesh1().extent();++t){
          g_tau(itime_index(t),typename MESH1::index_type(i),typename MESH2::index_type(j),typename MESH3::index_type(k),typename MESH4::index_type(l))=output_data[t]+f_tau(g_tau.mesh1().points()[t],g_tau.mesh1().beta(),c1,c2,c3);
        }
        }
      }
    }
  }
}

}
} // end alps::
