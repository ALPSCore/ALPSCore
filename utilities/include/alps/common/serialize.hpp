/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <cstddef>
#include <cassert>
#include <string>
#include <complex>
#include <vector>

#include <Eigen/Dense>

#include <alps/common/view.hpp>

namespace alps { namespace common {

/** Class does not support this operation */
struct unsupported_serializer_clone : public std::exception { };

/**
 * Foster the serialization of data to disk.
 *
 * The serialization interface writes a hierarchy of named groups, traversed by
 * `enter()` and `exit()`, each containing a set of primitives or key-value
 * pairs, written by the `write()` family of methods.
 *
 * @see alps::alea::serialize(), alps::alea::deserializer
 */
struct serializer
{
    /** Creates and descends into a group with name `group` */
    virtual void enter(const std::string &group) = 0;

    /** Ascends from the lowermost group */
    virtual void exit() = 0;

    /** Writes a named multi-dimensional array of doubles */
    virtual void write(const std::string &key, ndview<const double>) = 0;

    /** Writes a named multi-dimensional array of complex doubles */
    virtual void write(const std::string &key, ndview<const std::complex<double>>) = 0;

    /** Writes a named multi-dimensional array of longs */
    virtual void write(const std::string &key, ndview<const int64_t>) = 0;

    /** Writes a named multi-dimensional array of unsigned longs */
    virtual void write(const std::string &key, ndview<const uint64_t>) = 0;

    /** Writes a named multi-dimensional array of int */
    virtual void write(const std::string &key, ndview<const int32_t>) = 0;

    /** Writes a named multi-dimensional array of unsigned int */
    virtual void write(const std::string &key, ndview<const uint32_t>) = 0;

    /** Returns a copy of `*this` created using `new` */
    virtual serializer *clone() { throw unsupported_serializer_clone(); }

    /** Destructor */
    virtual ~serializer() { }
};

/**
 * Foster the deserialization of data from disk.
 *
 * The serialization interface writes a hierarchy of named groups, traversed by
 * `enter()` and `exit()`, each containing a set of primitives or key-value
 * pairs, read out by the `read()` family of methods.
 *
 * Each `read()` method read to the `ndview::data()` buffer, if given.  If
 * that field is `nullptr`, it shall instead read but discard the data.
 *
 * @see alps::alea::deserialize(), alps::alea::serializer
 */
struct deserializer
{
    /** Descends into a group with name `group` */
    virtual void enter(const std::string &group) = 0;

    /** Ascends from the lowermost group */
    virtual void exit() = 0;

    /** Retrieves metadata for a primitive */
    virtual std::vector<size_t> get_shape(const std::string &key) = 0;

    /** Reads a named multi-dimensional array of double */
    virtual void read(const std::string &key, ndview<double>) = 0;

    /** Reads a named multi-dimensional array of double complex */
    virtual void read(const std::string &key, ndview<std::complex<double>>) = 0;

    /** Reads a named multi-dimensional array of longs */
    virtual void read(const std::string &key, ndview<int64_t>) = 0;

    /** Reads a named multi-dimensional array of unsigned longs */
    virtual void read(const std::string &key, ndview<uint64_t>) = 0;

    /** Reads a named multi-dimensional array of int */
    virtual void read(const std::string &key, ndview<int32_t>) = 0;

    /** Reads a named multi-dimensional array of unsigned int */
    virtual void read(const std::string &key, ndview<uint32_t>) = 0;

    /** Returns a copy of `*this` created using `new` */
    virtual deserializer *clone() { throw unsupported_serializer_clone(); }

    /** Destructor */
    virtual ~deserializer() { }
};

/**
 * Allows RAII-type use of groups in serializer.
 *
 * Enters the specified group on construction of the sentry, and automatically
 * leaves the group when the object is destroyed.  This also allows recovery in
 * the case of soft exceptions.
 *
 *     void write_to_group(serializer &s, std::string name) {
 *         internal::serializer_sentry group(s, name);
 *         serialize(s, "first_item", 42);
 *         serialize(s, "second_item", 4711);
 *     } // exits group here
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
 * Allows RAII-type use of groups in deserializer.
 *
 * Enters the specified group on construction of the sentry, and automatically
 * leaves the group when the object is destroyed.  This also allows recovery in
 * the case of soft exceptions.
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

}}

namespace alps { namespace common { namespace internal {

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

namespace alps { namespace common {

// Serialization methods

inline void serialize(serializer &ser, const std::string &key, uint32_t value) {
    internal::scalar_serialize(ser, key, value);
}
inline void serialize(serializer &ser, const std::string &key, int32_t value) {
    internal::scalar_serialize(ser, key, value);
}
inline void serialize(serializer &ser, const std::string &key, uint64_t value) {
    internal::scalar_serialize(ser, key, value);
}
inline void serialize(serializer &ser, const std::string &key, int64_t value) {
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

inline void deserialize(deserializer &ser, const std::string &key, uint64_t &value) {
    internal::scalar_deserialize(ser, key, value);
}
inline void deserialize(deserializer &ser, const std::string &key, int64_t &value) {
    internal::scalar_deserialize(ser, key, value);
}
inline void deserialize(deserializer &ser, const std::string &key, uint32_t &value) {
    internal::scalar_deserialize(ser, key, value);
}
inline void deserialize(deserializer &ser, const std::string &key, int32_t &value) {
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
