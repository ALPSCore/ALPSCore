/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */


#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/pair.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/shared_array.hpp>

#include <iostream>
#include "gtest/gtest.h"

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
        ALPS_FOREACH_NATIVE_HDF5_TYPE(CHECK_TYPE)
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

TEST(hdf5_complex, TestingOfHDF5Copy){
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
 }
