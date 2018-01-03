/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <algorithm>
#include <alps/hdf5.hpp>
#include <alps/utilities/short_print.hpp>

#include <vector>

/// User-defined class: a naive matrix.
/**
   For the sake of simplicity and clarity of the example:
   1. The matrix is represented as a vector of columns.
   2. The matrix objects stores its dimensions.
*/
template <typename T>
class NaiveMatrix {
  public:
    typedef std::vector<T> column_type;
  private:
    std::size_t nrows_;
    std::size_t ncols_;
    std::vector<column_type> columns_;
  public:
    /// Zero-initilizing constructor.
    NaiveMatrix(std::size_t nr, std::size_t nc):
        nrows_(nr),
        ncols_(nc),
        columns_(ncols_,column_type(nrows_))
    {}

    /// Number of rows
    std::size_t nrows() const { return nrows_; }
    
    /// Number of columns
    std::size_t ncols() const { return ncols_; }

    /// Const accessor.
    /** @param i row index
        @param j column index
        @return Value of matrix element at (i,j)
        @throws std::out_of_range An index is out of range
    */
    T operator()(std::size_t i, std::size_t j) const {
        return columns_.at(j).at(i);
    }

    /// Non-const accessor
    /** @param i row index
        @param j column index
        @return Reference to matrix element at (i,j)
        @throws std::out_of_range An index is out of range
    */
    T& operator()(std::size_t i, std::size_t j) {
        return columns_.at(j).at(i);
    }

    /// Swaps this matrix with another one.
    /** @param other Another matrix of the same class
        @note Not required, but makes it easy to implement load() in a safe manner. */
    void swap(NaiveMatrix& other) {
        using std::swap;
        swap(columns_, other.columns_);
        swap(ncols_, other.ncols_);
        swap(nrows_, other.nrows_);
    }

    /// Requred to interface with HDF5: Saves the object to an archive
    void save(alps::hdf5::archive& ar) const {
        ar["nrows"] << nrows_;
        ar["ncols"] << ncols_;
        // Note that alps::hdf5 knows how to handle vector of vectors
        ar["data"] << columns_; 
    }

    /// Requred to interface with HDF5: Loads the object from an archive
    void load(alps::hdf5::archive& ar) {
        std::size_t nr, nc;
        ar["nrows"] >> nr;
        ar["ncols"] >> nc;
        NaiveMatrix loaded(nr,nc);
        ar["data"] >> loaded.columns_;
        swap(loaded);
    }
};

/// Convenience function to print the matrix to a stream
/** @param ostr Stream to print to
    @param mtx The matrix to print
*/
template <typename T>
std::ostream& operator<<(std::ostream& ostr, const NaiveMatrix<T>& mtx) {
    for (std::size_t i=0; i<mtx.nrows(); ++i) {
        for (std::size_t j=0; j<mtx.ncols(); ++j) {
            ostr << mtx(i,j) << " ";
        }
        endl(ostr);
    }
    return ostr;
}


/**
 * This example shows how to save and restore instances of user-defined classes using hdf5.
 * 
 * @param argc the number of arguments
 * @param argv the argument array
 * @return the exit code
 */
int main(int argc, char** argv)
{
    // The filename for the hdf5 file
    std::string filename("measurements.h5");
    
    // Sample data to write
    std::cout << "Creating a user-defined object..." << std::endl;
    NaiveMatrix<double> mtx(3,2);
    for (std::size_t j=0; j<mtx.ncols(); ++j) {
        for (std::size_t i=0; i<mtx.nrows(); ++i) {
            mtx(i,j)=i*10+j;
        }
    }
    std::cout << "The original matrix:\n" << mtx << std::endl;
    
    // Open the hdf5 file with write permission.
    std::cout << "Opening parameters.h5..." << std::endl;
    alps::hdf5::archive oar(filename, "w");
    
    // We write some data 
    // Each value is written a path, which corresponds to the location where
    // the value is stored in the hdf5 hierarchy within the file.
    // The object-specific names are under that path.

    std::cout << "Writing data..." << std::endl;
    oar["/user/mymatrix"] << mtx;
    
    // Close the file
    std::cout << "Closing parameters.h5..." << std::endl;
    oar.close();

    // Declare variables for reading the data
    // We need a placeholder to load the data. The types must match.
    NaiveMatrix<double> mtx2(0,0);
    
    // Open the hdf5 file with read permission
    std::cout << "Opening parameters.h5..." << std::endl;
    alps::hdf5::archive iar(filename, "r");
    
    // Read the data back
    std::cout << "Reading the data:" << std::endl;
    iar["/user/mymatrix"] >> mtx2;

    // Close the file
    std::cout << "Closing parameters.h5..." << std::endl;
    iar.close();

    std::cout << "The loaded matrix:\n" << mtx2 << std::endl;


    
    // Check that content matches the original object.
    if (mtx.nrows()!=mtx2.nrows()) {
        std::cerr << "Mismatched number of rows.\n";
        return 1;
    }
    if (mtx.ncols()!=mtx2.ncols()) {
        std::cerr << "Mismatched number of columns.\n";
        return 1;
    }
    for (std::size_t j=0; j<mtx2.ncols(); ++j) {
        for (std::size_t i=0; i<mtx2.nrows(); ++i) {
            if (i*10+j != mtx2(i,j)) {
                std::cerr << "Mismatch at (i="<<i<<", j="<<j<<")\n";
            }
        }
    }

    return 0;
}
