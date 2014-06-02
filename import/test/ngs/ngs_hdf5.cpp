/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/complex.hpp>

#include <boost/filesystem.hpp>

#include <iostream>

using namespace alps;

int main() {

    std::string const filename = "ngs.h5";
    if (boost::filesystem::exists(boost::filesystem::path(filename)))
        boost::filesystem::remove(boost::filesystem::path(filename));
    {
        hdf5::archive ar(filename, "a");
        ar << make_pvp("/to/to", 3.14159);
    }
    {
        hdf5::archive ar(filename, "r");
        double value;
        ar >> make_pvp("/to/to", value);
        std::cout << value << std::endl;
    }

    {
        hdf5::archive ar(filename, "a");
        ar << make_pvp("/to/my/vec/in/a/very/deep/path", std::vector<double>(17, 15.141));
    }
    {
        hdf5::archive ar(filename, "r");
        std::vector<unsigned> value;
        ar >> make_pvp("/to/my/vec/in/a/very/deep/path", value);
        std::cout << value[0] << std::endl;
    }

    {
        hdf5::archive ar(filename, "a");
        ar << make_pvp("/to/to", std::complex<double>(3.14159, 12.34));
    }
    {
        hdf5::archive ar(filename, "r");
        std::complex<double> value;
        ar >> make_pvp("/to/to", value);
        std::cout << value.real() << " " << value.imag() << std::endl;
    }

    {
        hdf5::archive ar(filename, "a");
        ar << make_pvp("/to/str", std::string("asdf"));
    }
    {
        hdf5::archive ar(filename, "r");
        std::string value;
        ar >> make_pvp("/to/str", value);
        std::cout << value << std::endl;
    }

    {
        hdf5::archive ar(filename, "a");
        ar << make_pvp("/to/char", "asdf");
    }
    {
        hdf5::archive ar(filename, "r");
        std::string value;
        ar >> make_pvp("/to/char", value);
        std::cout << value << std::endl;
    }
    {
        hdf5::archive ar(filename, "r");
        std::cout << (ar.is_datatype<double>("/to/to") ? "true" : "false") << std::endl;
    }
    boost::filesystem::remove(boost::filesystem::path(filename));
    return 0;
}
