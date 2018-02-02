/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#include"alps/gf/mesh.hpp"

namespace alps{
namespace gf{

std::ostream &operator<<(std::ostream &os, const itime_mesh &M){
  os<<"# "<<"IMAGINARY_TIME"<<" mesh: N: "<<M.extent()<<" beta: "<<M.beta()<<" statistics: ";
  os<<(M.statistics()==statistics::FERMIONIC?"FERMIONIC":"BOSONIC");
  os<<std::endl;
  return os;
}

std::ostream &operator<<(std::ostream &os, const power_mesh &M){
  os<<"# "<<"POWER"<<" mesh: power: "<<M.power()<<" uniform: "<<M.uniform()<<" N: "<<M.extent()<<" beta: "<<M.beta()<<" statistics: ";
  os<<(M.statistics()==statistics::FERMIONIC?"FERMIONIC":"BOSONIC");
  os<<std::endl;
  return os;
}

std::ostream &operator<<(std::ostream &os, const momentum_realspace_index_mesh &M){
  os << "# "<<M.kind()<<" mesh: N: "<<M.extent()<<" dimension: "<<M.dimension()<<" points: ";
  for(int i=0;i<M.extent();++i){
    os<<"(";
    for(int d=0;d<M.dimension()-1;++d){ os<<M.points()[i][d]<<","; } os<<M.points()[i][M.dimension()-1]<<") ";
  }
  os<<std::endl;
  return os;
}

std::ostream &operator<<(std::ostream &os, const index_mesh &M){
  os << "# "<<"INDEX"<<" mesh: N: "<<M.extent()<<std::endl;
  return os;
}

std::ostream &operator<<(std::ostream &os, const real_frequency_mesh &M){
  os<<"# "<<"REAL_FREQUENCY"<<" mesh: N: "<<M.extent();
  os<<std::endl;
  return os;
}

std::ostream &operator<<(std::ostream &os, const legendre_mesh &M){
  os<<"# "<<"LEGENDRE"<<" mesh: N: "<<M.extent()<<" beta: "<<M.beta()<<" statistics: ";
  os<<(M.statistics()==statistics::FERMIONIC?"FERMIONIC":"BOSONIC");
  os<<std::endl;
  return os;
}

  namespace detail {
  // print 1D boost multiarray --- a 2D-point of a mesh
  std::ostream& operator<<(std::ostream& s, const boost::multi_array<double, 1>& data)
  {
    typedef boost::multi_array<double, 1> data_type;
    typedef data_type::const_iterator iterator_type;
    s << "";
    iterator_type it=data.begin();
    if (data.end()!=it) s << *(it++);
    for (; it!=data.end(); ++it) {
      s << " " << *it;
    }
    s << " "; // << std::endl;
    return s;
  }
  } // detail::

} // gf::
} // alps::
