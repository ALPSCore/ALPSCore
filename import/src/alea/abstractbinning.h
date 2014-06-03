/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_ALEA_ABSTRACTBINNING_H
#define ALPS_ALEA_ABSTRACTBINNING_H

#include <alps/osiris/dump.h>
#include <alps/type_traits/change_value_type.hpp>

namespace alps {

//=======================================================================
// AbstractBinning
//-----------------------------------------------------------------------

template <class T>
class AbstractBinning {
 public: 
  typedef T value_type;
  typedef typename change_value_type<T,double>::type time_type;
  typedef typename change_value_type<T,int>::type convergence_type;
  AbstractBinning(std::size_t=0) {}

  time_type tau()                  const { boost::throw_exception(std::logic_error("Called non-implemented function of AbstractBinning")); return time_type(); }
  uint32_t max_bin_number()        const { return 0; }
  uint32_t bin_number()            const { return 0; }
  uint32_t filled_bin_number()     const { return 0; }
  uint32_t filled_bin_number2()     const { return 0; }
  uint32_t bin_size()              const { return 0; }
  const value_type& bin_value(uint32_t  ) const {
    boost::throw_exception(std::logic_error("Binning is not supported for this observable"));
    return *(new value_type); // dummy return
  } 
  const value_type& bin_value2(uint32_t  ) const {
    boost::throw_exception(std::logic_error("Binning is not supported for this observable"));
    return *(new value_type); // dummy return
  } 
  const std::vector<value_type>& bins() const {
    boost::throw_exception(std::logic_error("Binning is not supported for this observable"));
    return *(new std::vector<value_type>); // dummy return
  }
  void extract_timeseries(ODump& dump) const { dump << 0 << 0 << 0;}


  bool has_variance() const { return true;} // for now
  void write_scalar_xml(oxstream&) const {}
  template <class IT> void write_vector_xml(oxstream&, IT) const {}

  void save(ODump& /* dump */) const {}
  void load(IDump& dump) 
  { 
    bool thermalized_; 
    if (dump.version() < 306 && dump.version() != 0) 
      dump >> thermalized_;
  }


  void save(hdf5::archive &) const {};
  void load(hdf5::archive &) {};

  std::string evaluation_method() const { return "simple";}
};

} // end namespace alps

#endif // ALPS_ALEA_ABSTRACTBINNING_H
