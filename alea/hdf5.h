/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C)      2008 by Emanuel Gull < gull@itp.phys.ethz.ch>
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


///New ALPS ALEA hdf5 interface
#ifdef ALPS_HAVE_HDF5
#include<hdf5.h>
#include<alps/alea/observableset.h>
#include<alps/alea/simpleobservable.h>
#include<alps/alea/simplebinning.h>
#include<alps/alea/detailedbinning.h>
#include<alps/alea/nobinning.h>
#include<alps/alea/abstractsimpleobservable.h>
#include<mocasito/io/container.hpp>
#include<mocasito/io/hdf5.hpp>
#include<mocasito/io/util.hpp>
#include<mocasito/io/context.hpp>
namespace mocasito {
  namespace io {
    using namespace detail;
    template<typename E> context<E>& assign(context<E>& c, const alps::ObservableSet &measurements){
      //For now: no need to write observable set specific information.
      return c;
    }
    template<typename E> const context<E>& assign(alps::ObservableSet &measurements, const context<E> & c){
      return c;
    }
    template<typename E> context<E>& assign(context<E> & c, const alps::Observable &obs){
      //For now: no need to write observable specific information.
      return c;
    }
    template<typename E, typename T> context<E>& assign(context<E>& c, const std::valarray<T> &val){
      c.set(&(val[0]), val.size());
      return c;
    }
    template<typename E, typename T> std::valarray<T> & assign(std::valarray<T> &val, const context<E>& c){
      if (c.dimensions() != 1) throw(std::invalid_argument("this variable has dimension != 1"));
      val.resize(c.extends()[0]);
      c.get(&(val[0]));
      return val;
    }
    template<typename E, typename T> context<E> & assign(context<E> & c, alps::AbstractSimpleObservable<T> const &obs){
      assign(c,*((alps::Observable*)&obs)); //dump observable first.
      (c+std::string("count"))=obs.count();
      if(obs.count()==0){ 
        std::cerr<<"warning: observable "<<obs.name()<<" has a count of zero!"<<std::endl;
        return c;
        //throw(std::invalid_argument("trying to save observable "+obs.name()+" that has a count of zero")); 
      }
      (c+std::string("mean/value"))=obs.mean();
      (c+std::string("mean/error"))=obs.error();
      if(obs.has_variance())
        (c+std::string("mean/variance"))=obs.variance();
      if(obs.has_tau())
        (c+std::string("mean/tau_int"))=obs.tau();
      (c+std::string("mean/bins/bin_size"))=obs.bin_size();
      (c+std::string("mean/bins/number"))=obs.bin_number();
      (c+std::string("mean/bins2/number"))=obs.bin_number2();
      for(std::size_t k=0;k<obs.bin_number();++k){
        std::stringstream path; path<<"mean/bins/"<<k<<"/value";
        std::stringstream path2; path2<<"mean/bins2/"<<k<<"/value";
        (c+path.str()) =obs.bin_value(k);
        (c+path2.str())=obs.bin_value2(k);
      }
      return c;
    }
    template<typename E, typename T, typename B> context<E> & assign(context<E>& c, const alps::SimpleObservable<T,B> &obs){
      assign(c,*((alps::Observable*)&obs)); //dump observable first.
      //if we have 'binning', even if it is 'nobinning', then all the
      //information is hidden within that binning class. let's get it from
      //there!
      write_hdf5(c);
    }
  }
}
namespace alps{
template <typename T> void AbstractSimpleObservable<T>::write_hdf5(boost::filesystem::path const & path, std::size_t realization, std::size_t clone) const{
  mocasito::io::container<mocasito::io::hdf5> container(path.string());
  std::stringstream obs_path;
  obs_path<<"/simulation/realizations/"<<realization<<"/clones/"<<clone<<"/results/"<<name();
  assign(container[obs_path.str()],*this);
}
template <typename T, typename B> void SimpleObservable<T,B>::write_hdf5(boost::filesystem::path const & path, std::size_t realization, std::size_t clone) const{
  mocasito::io::container<mocasito::io::hdf5> container(path.string());
  std::stringstream obs_path;
  obs_path<<"/simulation/realizations/"<<realization<<"/clones/"<<clone<<"/results/"<<this->name();
  b_.write_hdf5(container[obs_path.str()]);
}
template<typename T, typename B> void SimpleObservable<T,B>::read_hdf5(boost::filesystem::path const & path, std::size_t realization, std::size_t clone){
  mocasito::io::container<mocasito::io::hdf5> container(path.string());
  std::stringstream obs_path;
  obs_path<<"/simulation/realizations/"<<realization<<"/clones/"<<clone<<"/results/"<<this->name();
  b_.read_hdf5(container[obs_path.str()]);
}
template<typename T> template<typename E> void SimpleBinning<T>::read_hdf5(const E &c){
  AbstractBinning<T>::read_hdf5(c); //call base class function
  count_=(c+std::string("count"));
  std::cout<<"simplebinning read: "<<count_<<std::endl;
  thermal_count_=(c+std::string("thermal_count"));
  bin_entries_=(c+std::string("mean/bins/bin_entries"));
  last_bin_.resize((c+std::string("mean/bins/size")));
  sum_.resize((c+std::string("mean/bins/size")));
  sum2_.resize((c+std::string("mean/bins/size")));
  for(std::size_t i=0;i<last_bin_.size();++i){
    std::stringstream path ; path <<"mean/bins/last_bin/"<<i;
    std::stringstream path1; path1<<"mean/bins/sum/"<<i;
    std::stringstream path2; path2<<"mean/bins/sum2/"<<i;
    assign(last_bin_[i],(c+path .str()));
    assign(sum_[i],(c+path1.str()));
    assign(sum2_[i],(c+path2.str()));
  }
}
template<typename T> template<typename E> void BasicDetailedBinning<T>::write_hdf5(E &c) const {
  SimpleBinning<T>::write_hdf5(c); //call base class function
  (c+std::string("mean/bins/binsize"))=binsize_;
  (c+std::string("mean/bins/minbinsize"))=minbinsize_;
  (c+std::string("mean/bins/maxbinnum"))=maxbinnum_;
  (c+std::string("mean/bins/binentries"))=binentries_;
  (c+std::string("mean/bins/values/size"))=values_.size();
  for(std::size_t i=0;i<values_.size();++i){
    std::stringstream path ; path <<"mean/bins/values/"<<i;
    std::stringstream path1; path1<<"mean/bins/values2/"<<i;
    (c+path .str())=values_[i];
    (c+path1.str())=values2_[i];
  }
}
template<typename T> template<typename E> void SimpleBinning<T>::write_hdf5(E &c) const {
  AbstractBinning<T>::write_hdf5(c); //call base class function
  (c+std::string("count"))=count_;
  (c+std::string("thermal_count"))=thermal_count_;
  (c+std::string("mean/bins/bin_entries"))=bin_entries_;
  (c+std::string("mean/bins/size"))=last_bin_.size();
  for(std::size_t i=0;i<last_bin_.size();++i){
    std::stringstream path ; path <<"mean/bins/last_bin/"<<i;
    std::stringstream path1; path1<<"mean/bins/sum/"<<i;
    std::stringstream path2; path2<<"mean/bins/sum2/"<<i;
    (c+path .str())=last_bin_[i];
    (c+path1.str())=sum_[i];
    (c+path2.str())=sum2_[i];
  }
}
template<typename T> template<typename E> void NoBinning<T>::write_hdf5(const E &c) const{
  AbstractBinning<T>::write_hdf5(c); //call base class function
  (c+std::string("count"))=count_;
  (c+std::string("thermal_count"))=thermal_count_;
  (c+std::string("mean/sum"))=sum_;
  (c+std::string("mean/sum2"))=sum2_;
}
template<typename T> template<typename E> void NoBinning<T>::read_hdf5(const E &c){
  AbstractBinning<T>::read_hdf5(c); //call base class function
  count_=(c+std::string("count"));
  thermal_count_=(c+std::string("thermal_count"));
  assign(sum_,(c+std::string("mean/sum")));
  assign(sum2_,(c+std::string("mean/sum2")));
}
template<typename T> template<typename E> void BasicDetailedBinning<T>::read_hdf5(const E &c){
  SimpleBinning<T>::read_hdf5(c); //call base class function
  binsize_=(c+"mean/bins/binsize");
  minbinsize_=(c+"mean/bins/minbinsize");
  maxbinnum_=(c+"mean/bins/maxbinnum");
  binentries_=(c+"mean/bins/binentries");
  values_.resize(c+"mean/bins/values/size");
  values2_.resize(c+"mean/bins/values/size");
  for(std::size_t i=0;i<values_.size();++i){
    std::stringstream path ; path <<"mean/bins/values/"<<i;
    std::stringstream path1; path1<<"mean/bins/values2/"<<i;
    assign(values_[i],(c+path .str()));
    assign(values2_[i],(c+path1.str()));
  }
}
template<typename T> template<typename E> void AbstractBinning<T>::write_hdf5(const E &c) const{
  (c+std::string("thermalized"))=thermalized_;
}
template<typename T> template<typename E> void AbstractBinning<T>::read_hdf5(const E &c){
  thermalized_=(c+std::string("thermalized"));
}

} //namespace
#endif //ALPS_HAVE_HDF5

#endif // ALPS_ALEA_HDF5_H
