/** @file main.cpp
    @brief alps::params cmd_params
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
int main(int argc, const char* argv[])
{
    // The filename for the hdf5 file
    std::string filename("measurements.h5");
    
    // Sample data to write
    std::complex<double> sComplexSample(1.0, -1.0);
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
    
    // Write some data
    std::cout << "Write some data..." << std::endl;
    oar["/scalars/double"] << 6.28;
    oar["/scalars/int"] << 500;
    oar["/scalars/complex"] << sComplexSample;
    oar["/scalars/string"] << filename;
    oar["/vectors/double"] << vDoubleSample;
    oar["/vectors/int"] << vIntSample;
    
    // Close the file
    std::cout << "Closing parameters.h5..." << std::endl;
    oar.close();

    // Open the hdf5 file with read permission
    std::cout << "Opening parameters.h5..." << std::endl;
    alps::hdf5::archive iar(filename, "r");
    
    // Read the data back
    std::cout << "Reading the data:" << std::endl;
    double dValue;
    int iValue;
    std::complex<double> sComplexValue;
    std::string sStringValue;
    std::vector<double> vDoubleValue;
    std::vector<double> vIntValue;
    iar["/scalars/double"] >> dValue;
    std::cout << "/scalars/double: " << dValue << std::endl;
    iar["/scalars/int"] >> iValue;
    std::cout << "/scalars/int: " << iValue << std::endl;
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
