/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2012 by Lukas Gamper <gamperl@gmail.com>                   *
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
#include <alps/hdf5/pair.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/shared_array.hpp>

#include <iostream>

namespace detail {
    void copy_data(alps::hdf5::archive & tar, alps::hdf5::archive & sar, std::string const & segment) {
        if (false);
        #define CHECK_TYPE(T)                                                                                                                   \
            else if (sar.is_datatype<T>(segment) && sar.is_null(segment))                                                                       \
                tar[segment] = std::vector<T>();                                                                                                \
            else if (sar.is_datatype<T>(segment) && sar.is_scalar(segment)) {                                                                   \
                T value;                                                                                                                        \
                sar[segment] >> value;                                                                                                          \
                tar[segment] = value;                                                                                                           \
            } else if (sar.is_datatype<T>(segment)) {                                                                                           \
                std::vector<std::size_t> extent = sar.extent(segment);                                                                          \
                std::size_t size = std::accumulate(extent.begin(), extent.end(), std::size_t(1), std::multiplies<std::size_t>());               \
                boost::shared_array<T> array(new T[size]);                                                                                      \
                std::pair<T *, std::vector<std::size_t> > value(array.get(), extent);                                                           \
                sar[segment] >> value;                                                                                                          \
                tar[segment] = value;                                                                                                           \
            }
        ALPS_NGS_FOREACH_NATIVE_HDF5_TYPE(CHECK_TYPE)
        #undef CHECK_TYPE
        else
            throw std::runtime_error("Unknown type in path: " + sar.complete_path(segment) + ALPS_STACKTRACE);
    }
}

void copy(alps::hdf5::archive & tar, std::string const & tpath, alps::hdf5::archive & sar, std::string const & spath) {
    std::string tcontext = tar.get_context();
    std::string scontext = sar.get_context();
    tar.set_context(tar.complete_path(tpath));
    sar.set_context(sar.complete_path(spath));

    std::vector<std::string> children = sar.list_children("");
    for (std::vector<std::string>::const_iterator it = children.begin(); it != children.end(); ++it)
        if (sar.is_group(*it))
            copy(tar, *it, sar, *it);
        else {
            detail::copy_data(tar, sar, *it);
            std::vector<std::string> attributes = sar.list_attributes(*it);
            for (std::vector<std::string>::const_iterator jt = attributes.begin(); jt != attributes.end(); ++jt)
                detail::copy_data(tar, sar, *it + "/@" + *jt);
        }
    std::vector<std::string> attributes = sar.list_attributes("");
    for (std::vector<std::string>::const_iterator it = attributes.begin(); it != attributes.end(); ++it)
        detail::copy_data(tar, sar, "@" + *it);

    tar.set_context(tcontext);
    sar.set_context(scontext);
}

int main() {
    try {
        std::vector<std::vector<int> > a(4);
        a[0] = std::vector<int>(1, 2);
        a[1] = std::vector<int>(3, 4);
        a[2] = std::vector<int>(5, 6);

        {
            alps::hdf5::archive ar("test_hdf5_copy.h5", "w");
            ar["/dat/vec"] = a;
            ar["/dat/vec/@foo"] = 10;
            ar["/dat/cpx"] = std::complex<double>(1., 1.);
            ar["/int"] = 2;
        }
        {
            alps::hdf5::archive tar("test_hdf5_copy2.h5", "w");
            alps::hdf5::archive sar("test_hdf5_copy.h5", "r");
            copy(tar, "/cpy", sar, "/dat");
        }
    } catch (std::runtime_error e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        std::abort();
    }
    return 0;
 }
