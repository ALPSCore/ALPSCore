/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <numeric>
#include <type_traits>

#include <alps/alea/core.hpp>
#include <alps/alea/complex_op.hpp>
#include <alps/alea/mean.hpp>
#include <alps/alea/variance.hpp>
#include <alps/alea/covariance.hpp>
#include <alps/alea/autocorr.hpp>
#include <alps/alea/batch.hpp>

namespace alps { namespace alea {

/**
 * stream_serializer
 *
 * This class establishes connection between the ALEA serialization
 * interface and Boost/HPX serialization frameworks. Its instance
 * is used in free function save() (see below) that is called upon
 * Boost/HPX serialization of ALEA *_result<T> types.
 */
template <typename Archive> class stream_serializer : public serializer
{

public:

    stream_serializer(Archive &ar) : ar_(ar) {}

    // Common methods

    // Nothing to be done here: streams have no notion of groups
    void enter(const std::string &) override {}
    void exit() override {}

    // Key names are irrelevant

    void write(const std::string &, ndview<const double> data_view) override
    {
        do_write(data_view);
    }
    void write(const std::string &, ndview<const std::complex<double>> data_view) override
    {
        do_write(data_view);
    }
    void write(const std::string &, ndview<const complex_op<double>> data_view) override
    {
        do_write(data_view);
    }
    void write(const std::string &, ndview<const long> data_view) override
    {
        do_write(data_view);
    }
    void write(const std::string &, ndview<const unsigned long> data_view) override
    {
        do_write(data_view);
    }

protected:

    template <typename T> void do_write(const ndview<const T> &data_view)
    {
        size_t size = data_view.size();
        const T * data = data_view.data();
        for(size_t n = 0; n != size; ++n)
            ar_ << *(data + n);
    }

private:

    Archive &ar_;
};

/**
 * stream_deserializer
 *
 * This class establishes connection between the ALEA serialization
 * interface and Boost/HPX serialization frameworks. Its instance
 * is used in free function load() (see below) that is called upon
 * Boost/HPX deserialization of ALEA *_result<T> types.
 */
template <typename Archive> class stream_deserializer : public deserializer
{

public:

    stream_deserializer(Archive &ar) : ar_(ar) {}

    // Common methods

    // Nothing to be done here: streams have no notion of groups
    void enter(const std::string &) override {}
    void exit() override {}

    // Deserialization methods

    // Key names are irrelevant

    std::vector<size_t> get_shape(const std::string &key) override
    {
        return {{}}; // There is no way to know the shape beforehand
    }

    void read(const std::string &, ndview<double> value) override
    {
        do_read(value);
    }
    void read(const std::string &, ndview<std::complex<double>> value) override
    {
        do_read(value);
    }
    void read(const std::string &, ndview<complex_op<double>> value) override
    {
        do_read(value);
    }
    void read(const std::string &, ndview<long> value) override
    {
        do_read(value);
    }
    void read(const std::string &, ndview<unsigned long> value) override
    {
        do_read(value);
    }

protected:

    template <typename T> void do_read(ndview<T> &data_view)
    {
        size_t size = data_view.size();
        T * data = data_view.data();
        if(data) {
            for(size_t n = 0; n != size; ++n)
                ar_ >> *(data + n);
        } else {
            T tmp;
            for(size_t n = 0; n != size; ++n)
                ar_ >> tmp;
        }
    }

private:

    Archive &ar_;
};

/**
 * This method will be called by Boost/HPX Serialization libraries
 * when serialization of an ALEA result type is requested.
 */
template <typename Archive, typename T>
typename std::enable_if<is_alea_result<T>::value, void>::type
save(Archive &ar, const T &r, const unsigned int) {
    stream_serializer<Archive> ser(ar);
    serialize(ser, "", r);
}

/**
 * This method will be called by Boost/HPX Serialization libraries
 * when deserialization of an ALEA result type is requested.
 */
template <typename Archive, typename T>
typename std::enable_if<is_alea_result<T>::value, void>::type
load(Archive &ar, T &r, const unsigned int) {
    stream_deserializer<Archive> ser(ar);
    deserialize(ser, "", r);
}

/**
 * Serialize/deserialize alps::alea::complex_op<T>.
 * Called by Boost/HPX Serialization libraries.
 */
template <typename Archive, typename T>
void serialize(Archive &ar, complex_op<T> &co, const unsigned int) {
    ar & co.rere() & co.reim() & co.imre() & co.imim();
}

}}
