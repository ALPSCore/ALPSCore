/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>

#include <array>

namespace alps { namespace alea { namespace internal {

/**
 * Allows RAII-type use of groups.
 */
struct serializer_sentry
{
    serializer_sentry(serializer &ser, const std::string &group)
        : ser_(ser)
        , group_(group)
    {
        if (group != "")
            ser_.enter(group);
    }

    ~serializer_sentry()
    {
        if (group_ != "")
            ser_.exit();
    }

private:
    serializer &ser_;
    std::string group_;
};

/**
 * Allows RAII-type use of groups.
 */
struct deserializer_sentry
{
    deserializer_sentry(deserializer &ser, const std::string &group)
        : ser_(ser)
        , group_(group)
    {
        if (group != "")
            ser_.enter(group);
    }

    ~deserializer_sentry()
    {
        if (group_ != "")
            ser_.exit();
    }

private:
    deserializer &ser_;
    std::string group_;
};

/** Helper function for serialization of scalars */
template <typename T>
void scalar_serialize(serializer &ser, const std::string &key, T value)
{
    ser.write(key, ndview<const T>(&value, nullptr, 0));
}

/** Helper function for deserialization of scalars */
template <typename T>
T scalar_deserialize(deserializer &ser, const std::string &key)
{
    T value;
    ser.read(key, ndview<T>(&value, nullptr, 0));
    return value;
}

/** Helper function for deserialization of scalars */
template <typename T>
void scalar_deserialize(deserializer &ser, const std::string &key, T &value)
{
    ser.read(key, ndview<T>(&value, nullptr, 0));
}

}}}

namespace alps { namespace alea {

// Serialization methods

inline void serialize(serializer &ser, const std::string &key, unsigned long value) {
    internal::scalar_serialize(ser, key, value);
}
inline void serialize(serializer &ser, const std::string &key, long value) {
    internal::scalar_serialize(ser, key, value);
}
inline void serialize(serializer &ser, const std::string &key, double value) {
    internal::scalar_serialize(ser, key, value);
}
inline void serialize(serializer &ser, const std::string &key,
                      std::complex<double> value) {
    internal::scalar_serialize(ser, key, value);
}

template <typename Derived>
void serialize(serializer &ser, const std::string &key,
               const Eigen::MatrixBase<Derived> &value)
{
    typedef Eigen::internal::traits<Derived> traits;
    typedef typename traits::Scalar scalar_type;
    typedef Eigen::Matrix<scalar_type, Derived::RowsAtCompileTime,
                          Derived::ColsAtCompileTime> plain_matrix_type;

    // Ensure that evaluated expression will be continuous
    if ((Derived::MaxRowsAtCompileTime != Eigen::Dynamic
                    && Derived::MaxRowsAtCompileTime != value.rows())
            || (Derived::MaxColsAtCompileTime != Eigen::Dynamic
                    && Derived::MaxColsAtCompileTime != value.cols())
            || ((Derived::Options & Eigen::RowMajor)
                    && value.rows() != 1 && value.cols() != 1))
        serialize(ser, key, plain_matrix_type(value));

    // Evaluate to matrix or proxy object if already matrix
    auto temp = value.eval();

    if (Derived::ColsAtCompileTime == 1 || Derived::RowsAtCompileTime == 1) {
        // Omit second dimension for simple vectors
        std::array<size_t, 1> dims = {{(size_t)temp.size()}};
        ser.write(key, ndview<const scalar_type>(temp.data(), dims.data(), 1));
    } else {
        // Eigen arrays are column-major
        std::array<size_t, 2> dims = {{(size_t)temp.cols(), (size_t)temp.rows()}};
        ser.write(key, ndview<const scalar_type>(temp.data(), dims.data(), 2));
    }
}


// Argument-oriented deserialization

inline void deserialize(deserializer &ser, const std::string &key, unsigned long &value) {
    internal::scalar_deserialize(ser, key, value);
}
inline void deserialize(deserializer &ser, const std::string &key, long &value) {
    internal::scalar_deserialize(ser, key, value);
}
inline void deserialize(deserializer &ser, const std::string &key, double &value) {
    internal::scalar_deserialize(ser, key, value);
}
inline void deserialize(deserializer &ser, const std::string &key,
                        std::complex<double> &value) {
    internal::scalar_deserialize(ser, key, value);
}

template <typename T>
void deserialize(deserializer &ser, const std::string &key,
                 Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &value)
{
    std::array<size_t, 2> shape = {{(size_t)value.cols(), (size_t)value.rows()}};
    ser.read(key, ndview<T>(value.data(), shape.data(), shape.size()));
}

template <typename T>
void deserialize(deserializer &ser, const std::string &key,
                 Eigen::Matrix<T, Eigen::Dynamic, 1> &value)
{
    std::array<size_t, 1> shape = {{(size_t)value.rows()}};
    ser.read(key, ndview<T>(value.data(), shape.data(), shape.size()));
}

template <typename T>
void deserialize(deserializer &ser, const std::string &key,
                 Eigen::Matrix<T, 1, Eigen::Dynamic> &value)
{
    std::array<size_t, 1> shape = {{(size_t)value.cols()}};
    ser.read(key, ndview<T>(value.data(), shape.data(), shape.size()));
}


}}
