/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <sstream>
#include <numeric>
#include <iostream>

#include <alps/alea/core.hpp>

#include <alps/hdf5/archive.hpp>

namespace alps { namespace alea {

class hdf5_serializer
    : public serializer
    , public deserializer
{
public:
    hdf5_serializer(hdf5::archive &ar, const std::string &path)
        : archive_(&ar)
        , path_(path)
        , group_()
    { }

    // Common methods

    void enter(const std::string &group) override
    {
        archive_->create_group(get_path(group));  // TODO: what if exists?
        group_.push_back(group);
    }

    void exit() override
    {
        if (group_.empty())
            throw std::runtime_error("exit without enter");
        group_.pop_back();
    }

    void write(const std::string &key, ndview<const double> value) override {
        do_write(key, value);
    }

    void write(const std::string &key, ndview<const std::complex<double>> value) override {
        do_write(key, value);
    }

    void write(const std::string &key, ndview<const complex_op<double>> value) override {
        throw unsupported_operation();  // FIXME
    }

    void write(const std::string &key, ndview<const long> value) override {
        do_write(key, value);
    }

    void write(const std::string &key, ndview<const unsigned long> value) override {
        do_write(key, value);
    }

    // Deserialization methods

    std::vector<size_t> get_shape(const std::string &key) override {
        return get_extent(get_path(key));
    }

    void read(const std::string &key, ndview<double> value) override {
        do_read(key, value);
    }

    void read(const std::string &key, ndview<std::complex<double>> value) override {
        do_read(key, value);
    }

    void read(const std::string &key, ndview<complex_op<double>> value) override {
        throw unsupported_operation();  // FIXME
    }

    void read(const std::string &key, ndview<long> value) override {
        do_read(key, value);
    }

    void read(const std::string &key, ndview<unsigned long> value) override {
        do_read(key, value);
    }

    ~hdf5_serializer()
    {
        // Cannot do exception because we are in destructor
        if (!group_.empty()) {
            std::cerr << "alps::alea::hdf5_serializer: warning: "
                      << "enter without exit\n Lingering groups:"
                      << get_path("") << "\n\n";
        }
    }

protected:
    template <typename T>
    void do_write(const std::string &relpath, ndview<const T> data)
    {
        std::string path = get_path(relpath);

        std::vector<size_t> shape(data.shape(), data.shape() + data.ndim());
        std::vector<size_t> offset(shape.size(), 0);
        std::vector<size_t> chunk = shape;

        if (data.ndim() == 0)
            archive_->write(path, *data.data());
        else
            archive_->write(path, data.data(), shape, chunk, offset);
    }

    template <typename T>
    void do_write(const std::string &relpath, ndview<const std::complex<T>> data)
    {
        std::string path = get_path(relpath);

        if (data.ndim() == 0)
            throw unsupported_operation();

        std::vector<size_t> shape(data.shape(), data.shape() + data.ndim());
        shape.push_back(2);  // for complex
        std::vector<size_t> offset(shape.size(), 0);
        std::vector<size_t> chunk = shape;

        // hdf5::archive does not support complex
        archive_->write(path, reinterpret_cast<const T*>(data.data()), shape,
                          chunk, offset);
        archive_->write(path + "/@__complex__", true);
    }

    template <typename T>
    void do_read(const std::string &relpath, ndview<T> data)
    {
        std::string path = get_path(relpath);

        // check shape (this is cheap compared to reading)
        std::vector<size_t> shape = get_extent(path);
        if (data.ndim() != shape.size())
            throw size_mismatch();
        for (size_t i = 0; i != shape.size(); ++i)
            if (shape[i] != data.shape()[i])
                throw size_mismatch();

        // discard the data
        if (data.data() == nullptr)
            return;

        if (shape.empty()) {
            archive_->read(path, *data.data());
        } else {
            // vector read
            std::vector<size_t> offset(shape.size(), 0);
            std::vector<size_t> chunk = shape;
            archive_->read(path, data.data(), chunk, offset);
        }
    }

    template <typename T>
    void do_read(const std::string &relpath, ndview<std::complex<T>> data)
    {
        std::string path = get_path(relpath);

        // check shape (this is cheap compared to reading)
        std::vector<size_t> shape = get_extent(path);
        if (data.ndim() != shape.size() - 1)
            throw size_mismatch();
        for (size_t i = 0; i != data.ndim(); ++i)
            if (shape[i] != data.shape()[i])
                throw size_mismatch();
        if (shape[data.ndim()] != 2)
            throw size_mismatch();

        // discard the data
        if (data.data() == nullptr)
            return;

        // vector read
        std::vector<size_t> offset(shape.size(), 0);
        std::vector<size_t> chunk = shape;
        archive_->read(path, reinterpret_cast<T*>(data.data()), chunk, offset);

        bool tag;
        archive_->read(path + "/@__complex__", tag);
    }

    std::string get_path(const std::string &key)
    {
        std::ostringstream maker(path_, std::ios_base::app);
        maker << '/';
        for (const std::string &group : group_)
            maker << group << '/';
        if (key.find('/') != std::string::npos)
            throw std::runtime_error("Key must not contain '/'");
        maker << key;
        return maker.str();
    }

    std::vector<size_t> get_extent(const std::string &path)
    {
        if (archive_->is_scalar(path))
            return std::vector<size_t>();
        else
            return archive_->extent(path);
    }

private:
    hdf5::archive *archive_;
    std::string path_;
    std::vector<std::string> group_;
};

}}
