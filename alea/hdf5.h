/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2004 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Beat Ammon <ammon@ginnan.issp.u-tokyo.ac.jp>,
*                            Andreas Laeuchli <laeuchli@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

/* $Id$ */
#ifndef ALPS_ALEA_HDF5_H
#define ALPS_ALEA_HDF5_H

#ifdef ALPS_HAVE_MOCASITO

///New ALPS hdf5 interface
#ifdef ALPS_HAVE_HDF5
#include<hdf5.h>
#include<alps/alea/observableset.h>
#include<alps/alea/simpleobservable.h>
#include<alps/alea/abstractsimpleobservable.h>
#include<mocasito/src/io/container.hpp>
#include<mocasito/src/io/hdf5.hpp>
#include<mocasito/src/io/util.hpp>
namespace mocasito {
  namespace io {
    template<typename E> context<E> & assign(context<E> & c, const alps::ObservableSet &measurements){
      //For now: no need to write observable set specific information.
      return c;
    }
    template<typename E> context<E> & assign(context<E> & c, const alps::Observable &obs){
      //For now: no need to write observable specific information.
      return c;
    }
    template<typename E, typename T> context<E> & assign(context<E> & c, const std::valarray<T> &val){
      c.set(&(val[0]), val.size());
      return c;
    }
    template<typename E, typename T> context<E> & assign(context<E> & c, alps::AbstractSimpleObservable<T> const &obs){
      assign(c,*((alps::Observable*)&obs)); //dump observable first.
      (c+"count")=obs.count();
      (c+"mean/value")=obs.mean();
      (c+"mean/error")=obs.error();
      if(obs.has_variance())
        (c+"mean/variance")=obs.variance();
      if(obs.has_tau())
        (c+"mean/tau_int")=obs.tau();
      (c+"/mean/bins/bin_size")=obs.bin_size();
      (c+"/mean/bins/number")=obs.bin_number();
      (c+"/mean/bins2/number")=obs.bin_number2();
      for(int k=0;k<obs.bin_number();++k){
        std::stringstream path; path<<"mean/bins/"<<k<<"/value";
        std::stringstream path2; path2<<"mean/bins2/"<<k<<"/value";
        (c+path.str()) =obs.bin_value(k);
        (c+path2.str())=obs.bin_value2(k);
      }
      return c;
    }
    template<typename E, typename T, typename B> context<E> & assign(context<E> & c, const alps::SimpleObservable<T,B> &obs){
      assign(c,*((alps::AbstractSimpleObservable<T>*)&obs)); //dump observable first.
      //For now: no need to write observable specific information.
      return c;
    }
  }
}
namespace alps{
void ObservableSet::write_hdf5(boost::filesystem::path const & path, std::size_t realization, std::size_t clone) const {
  mocasito::io::container<mocasito::io::hdf5> container(path.string());
  std::stringstream set_path;
  set_path<<"/simulation/realizations/"<<realization<<"/clones/"<<clone<<"/results";
  assign(container[set_path.str()],*this);
  for(ObservableSet::const_iterator it=begin();it!=end();++it){
    it->second->write_hdf5(path, realization, clone);
  }
}
template <typename T> void AbstractSimpleObservable<T>::write_hdf5(boost::filesystem::path const & path, std::size_t realization, std::size_t clone) const{
  mocasito::io::container<mocasito::io::hdf5> container(path.string());
  std::stringstream obs_path;
  obs_path<<"/simulation/realizations/"<<realization<<"/clones/"<<clone<<"/results/"<<name();
  std::cerr<<"path is: "<<obs_path.str()<<std::endl;
  assign(container[obs_path.str()],*this);
}
template <typename T, typename B> void SimpleObservable<T,B>::write_hdf5(boost::filesystem::path const & path, std::size_t realization, std::size_t clone) const{
  mocasito::io::container<mocasito::io::hdf5> container(path.string());
  std::stringstream obs_path;
  obs_path<<"/simulation/realizations/"<<realization<<"/clones/"<<clone<<"/results/"<<this->name();
  std::cerr<<"path is: "<<obs_path.str()<<std::endl;
  assign(container[obs_path.str()],*this);
}
void ObservableSet::read_hdf5(boost::filesystem::path const &, std::size_t realization, std::size_t clone){
  
}
} //namespace
#endif //ALPS_HAVE_MOCASITO
#endif //ALPS_HAVE_HDF5

#endif // ALPS_ALEA_HDF5_H
