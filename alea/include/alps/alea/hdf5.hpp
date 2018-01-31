/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <sstream>
#include <numeric>

#include <alps/alea/core.hpp>

#include <alps/hdf5/archive.hpp>

namespace alps { namespace alea {

inline std::string join_paths(const std::string &base, const std::string &rel)
{
    std::ostringstream maker(base);
    if (*(base.rbegin()) != '/')
        maker << '/';
    if (*(rel.begin()) == '/')
        throw std::runtime_error("Relative path must not begin with '/'");
    maker << rel;
    return maker.str();
}

class hdf5_serializer
    : public serializer
{
public:
    hdf5_serializer(hdf5::archive &ar, const std::string &path)
        : archive_(&ar)
        , path_(path)
    { }

    void write(const std::string &key, const computed<double> &value) {
        do_write(key, value);
    }

    void write(const std::string &key, const computed<std::complex<double> > &value) {
        do_write(key, value);
    }

    void write(const std::string &key, const computed<complex_op<double> > &value) {
        do_write(key, value);
    }

    void write(const std::string &key, const computed<long> &value) {
        do_write(key, value);
    }

    void write(const std::string &key, const computed<unsigned long> &value) {
        do_write(key, value);
    }

protected:
    // Look at:
    // void archive::write(std::string path, T const * value,
    //     , std::vector<std::size_t> size
    //     , std::vector<std::size_t> chunk = std::vector<std::size_t>()
    //     , std::vector<std::size_t> offset = std::vector<std::size_t>()
    //     ) const;

    template <typename T>
    void do_write(const std::string &relpath, const computed<T> &data)
    {
        std::string path = join_paths(path_, relpath);

        std::vector<size_t> shape = data.shape();
        std::vector<size_t> offset(shape.size(), 0);
        std::vector<size_t> chunk = shape;
        size_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                      std::multiplies<size_t>());

        // TODO: use HDF5 dataspaces to avoid copy
        std::vector<T> buffer(size, 0);
        data.add_to(sink<T>(buffer.data(), size));
        (*archive_).write(path, buffer.data(), shape, chunk, offset);
    }

    template <typename T>
    void do_write(const std::string &relpath, const computed<std::complex<T>> &data)
    {
        std::string path = join_paths(path_, relpath);

        std::vector<size_t> shape = data.shape();
        shape.push_back(2);         // last dimension

        std::vector<size_t> offset(shape.size(), 0);
        std::vector<size_t> chunk = shape;
        size_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                      std::multiplies<size_t>());

        // TODO: use HDF5 dataspaces to avoid copy
        std::vector<T> buffer(size, 0);
        data.add_to(sink<std::complex<T>>(
                            reinterpret_cast<std::complex<T>*>(buffer.data()),
                            size/2
                            ));
        (*archive_).write(path, &buffer[0], shape, chunk, offset);
        (*archive_).write(path + "/@__complex__", true);
    }

private:
    hdf5::archive *archive_;
    std::string path_;
};

}}
