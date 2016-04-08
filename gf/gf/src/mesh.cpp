#include"alps/gf/mesh.hpp"

///Stream output operator, e.g. for printing to file
namespace alps{
namespace gf{

std::ostream &operator<<(std::ostream &os, const itime_mesh &M){
  os<<"# "<<"IMAGINARY_TIME"<<" mesh: N: "<<M.extent()<<" beta: "<<M.beta()<<" statistics: ";
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


}
}
