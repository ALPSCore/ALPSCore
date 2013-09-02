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


#include <alps/hdf5.hpp>
#include <alps/hdf5/pair.hpp>
#include <alps/hdf5/vector.hpp>

#include <iostream>

void copy(alps::hdf5::archive & tar, std::string const & tpath, alps::hdf5::archive & sar, std::string const & spath) {
    std::string tcontext = tar.get_context();
    std::string scontext = sar.get_context();
    tar.set_context(tar.complete_path(tpath));
    sar.set_context(sar.complete_path(spath));

    std::vector<std::string> children = sar.list_children("");
    for (std::vector<std::string>::const_iterator it = children.begin(); it != children.end(); ++it) {
        if (sar.is_group(*it))
            copy(tar, *it, sar, *it);
        #define CHECK_TYPE(T)                                                                                                                   \
            else if (sar.is_datatype<int>(*it) && sar.is_null(*it))                                                                             \
                tar[*it] = std::vector<int>();                                                                                                  \
            else if (sar.is_datatype<int>(*it) && sar.is_scalar(*it)) {                                                                         \
                int value;                                                                                                                      \
                sar[*it] >> value;                                                                                                              \
                tar[*it] = value;                                                                                                               \
            } else if (sar.is_datatype<int>(*it)) {                                                                                             \
                std::vector<std::size_t> extent = sar.extent(*it);                                                                              \
                std::size_t size = std::accumulate(extent.begin(), extent.end(), std::size_t(1), std::multiplies<std::size_t>());               \
                boost::shared_array<int> array(new int[size]);                                                                                  \
                std::pair<int *, std::vector<std::size_t> > value(array.get(), extent);                                                         \
                sar[*it] >> value;                                                                                                              \
                tar[*it] = value;                                                                                                               \
            }
        ALPS_NGS_FOREACH_NATIVE_HDF5_TYPE(CHECK_TYPE)
        #undef CHECK_TYPE
    }

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
            ar["/vec"] << a;
            ar["/int"] << 2;
        }

        {
            alps::hdf5::archive tar("test_hdf5_copy2.h5", "w");
            alps::hdf5::archive sar("test_hdf5_copy.h5", "r");
            copy(tar, "/cpy", sar, "/vec");
        }
    } catch (std::runtime_error e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        std::abort();
    }
    return 0;
 }
