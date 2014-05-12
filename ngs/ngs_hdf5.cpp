/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
 *                              Matthias Troyer <troyer@comp-phys.org>             *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

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
