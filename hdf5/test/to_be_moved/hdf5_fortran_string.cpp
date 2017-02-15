/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/hdf5.hpp>
#include <alps/version.h>

#include <boost/filesystem.hpp>

#include <iostream>
#include <string>

int main() {
    boost::filesystem::path infile(ALPS_SRCDIR);
    infile = infile / "test" / "hdf5" / "hdf5_fortran_string.h5";
    if (!boost::filesystem::exists(infile)) {
        std::cout << "Reference file " << infile << " not found." << std::endl;
        return -1;
    }
	alps::hdf5::archive ia(infile);
	std::string path("/fortran_string");
	std::string test;
	std::cout<<"is datatype<string>: " << (ia.is_datatype<std::string>(path) ? "True" : "False")<<std::endl;
	ia[path] >> test;
	std::cout << test << std::endl;
	return ia.is_datatype<std::string>(path) ? 0 : -1;
}