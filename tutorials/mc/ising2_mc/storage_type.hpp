/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_TUTORIALS_MC_ISING2_STORAGE_TYPE_HPP_228662e62e9247d5903f52f9644b6455
#define ALPS_TUTORIALS_MC_ISING2_STORAGE_TYPE_HPP_228662e62e9247d5903f52f9644b6455

#include <vector>
#include <alps/hdf5.hpp>

// Storage class for 2D spin array.
// Implemented as vector of vectors for simplicity.
class storage_type {
  private:
    std::vector< std::vector<int> > data_;
  public:
    // Constructor
    storage_type(int nrows, int ncols):
      data_(nrows, std::vector<int>(ncols))
    {}

    // Read access
    int operator()(int i, int j) const {
        return data_[i][j];
    }
    // Read/Write access
    int& operator()(int i, int j) {
        return data_[i][j];
    }

    // Custom save
    void save(alps::hdf5::archive& ar) const {
        ar["2Darray"] << data_;
    }
    // Custom load
    void load(alps::hdf5::archive& ar) {
        ar["2Darray"] >> data_;
    }
};


#endif /* ALPS_TUTORIALS_MC_ISING2_STORAGE_TYPE_HPP_228662e62e9247d5903f52f9644b6455 */
