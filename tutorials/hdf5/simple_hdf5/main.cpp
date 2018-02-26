/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <alps/hdf5.hpp>
#include <alps/utilities/short_print.hpp>

/**
 * This example shows how to save and restore scalars and vectors using hdf5.
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
    // We are going to use different data types to show that the library can
    // handle them. For a full list of type supported consult the reference
    // documentation (Doxygen).
    double sDoubleSample = 6.28;
    int sIntSample = 500;
    std::complex<double> sComplexSample(1.0, -1.0);
    std::string sStringSample("ALPSCore");
    std::vector<double> vDoubleSample(10, 0);
    for (int i = 0; i < vDoubleSample.size(); i++) {
        vDoubleSample[i] = i / 2.0;
    }
    std::vector<int> vIntSample(10, 0);
    for (int i = 0; i < vIntSample.size(); i++) {
        vIntSample[i] = 2 * i;
    }
    
    // Open the hdf5 file with write permission.
    std::cout << "Opening parameters.h5..." << std::endl;
    alps::hdf5::archive oar(filename, "w");
    
    // We write some data 
    // Each value is written a path, which corresponds to the location where
    // the value is stored in the hdf5 hierarchy within the file. Note how
    // the syntax is the same regardless of the data being saved.
    std::cout << "Writing data..." << std::endl;
    oar["/scalars/double"] << sDoubleSample;
    oar["/scalars/int"] << sIntSample;
    oar["/scalars/complex"] << sComplexSample;
    oar["/scalars/string"] << sStringSample;
    oar["/vectors/double"] << vDoubleSample;
    oar["/vectors/int"] << vIntSample;
    
    // Close the file
    std::cout << "Closing parameters.h5..." << std::endl;
    oar.close();

    // Declare variables for reading the data
    // We need a placeholder to load the data. The type does not need to match
    // the original as long as it can be converted. E.g. int -> double,
    // real -> complex. Consult the reference documentation (Doxygen) for
    // a full list of supported conversions.
    double sDoubleValue;
    int sIntValue;
    double sIntConvertedValue;
    std::complex<double> sComplexValue;
    std::string sStringValue;
    std::vector<double> vDoubleValue;
    std::vector<int> vIntValue;

    // Open the hdf5 file with read permission
    std::cout << "Opening parameters.h5..." << std::endl;
    alps::hdf5::archive iar(filename, "r");
    
    // Read the data back
    // Note that the data does not have to be read in the same order it was.
    // written. We do it here as it is easier to maintain.
    std::cout << "Reading the data:" << std::endl;
    iar["/scalars/double"] >> sDoubleValue;
    std::cout << "/scalars/double: " << sDoubleValue << std::endl;
    iar["/scalars/int"] >> sIntValue;
    std::cout << "/scalars/int: " << sIntValue << std::endl;
    iar["/scalars/int"] >> sIntConvertedValue;
    std::cout << "/scalars/int (as double): " << sIntConvertedValue << std::endl;
    iar["/scalars/complex"] >> sComplexValue;
    std::cout << "/scalars/complex: " << sComplexValue << std::endl;
    iar["/scalars/string"] >> sStringValue;
    std::cout << "/scalars/string: " << sStringValue << std::endl;
    iar["/vectors/double"] >> vDoubleValue;
    std::cout << "/vectors/double: " << alps::short_print(vDoubleValue) << std::endl;
    iar["/vectors/int"] >> vIntValue;
    std::cout << "/vectors/int: " << alps::short_print(vIntValue) << std::endl;
    
    // Close the file
    std::cout << "Closing parameters.h5..." << std::endl;
    iar.close();

    return 0;
}
