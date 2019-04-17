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

#include <alps/common/view.hpp>

// Forward declarations

namespace alps { namespace serialization {
    struct unsupported_serializer_clone;
    struct serializer;
    struct deserializer;

    namespace internal {
        template <typename T>
        void scalar_serialize(serializer &, const std::string &, T);

        template <typename T>
        void scalar_deserialize(deserializer &, const std::string &, T &);
    }
}}

// Actual declarations

namespace alps { namespace serialization {

using alps::common::ndview;

/** Class does not support this operation */
struct unsupported_operation : public std::exception { };

/**
 * Foster the serialization of data to disk.
 *
 * The serialization interface writes a hierarchy of named groups, traversed by
 * `enter()` and `exit()`, each containing a set of primitives or key-value
 * pairs, written by the `write()` family of methods.
 *
 * \note You will usually only use the `write()` methods directly when writing
 *       your own serialzation format or when extending serialization for new
 *       matrix containers.  In other cases, refer to the `serialize()`
 *       function.
 *
 * @see serialize(), deserializer
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
    virtual serializer *clone() { throw unsupported_operation(); }

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
 * \note You will usually only use the `read()` methods directly when writing
 *       your own serialzation format or when extending serialization for new
 *       matrix containers.  In other cases, refer to the `deserialize()`
 *       function.
 *
 * @see deserialize(), serializer
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
    virtual deserializer *clone() { throw unsupported_operation(); }

    /** Destructor */
    virtual ~deserializer() { }
};

/**
 * Trait checking whether `T` is a valid serialization primitive.
 *
 * `is_primitive<T>::value` evaluates to `true` if and only if `T` is a
 * serialization primitive, i.e., if `serializer::write` accepts `ndview<T>`
 * and `deserializer::read` accepts `ndview<const T>`.
 */
template <typename T>
struct is_primitive : std::false_type { };

template <> struct is_primitive<double> : std::true_type { };
template <> struct is_primitive<std::complex<double>> : std::true_type { };
template <> struct is_primitive<int64_t> : std::true_type { };
template <> struct is_primitive<uint64_t> : std::true_type { };
template <> struct is_primitive<int32_t> : std::true_type { };
template <> struct is_primitive<uint32_t> : std::true_type { };

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

}}

namespace alps { namespace serialization { namespace internal {

/** Helper function for serialization of scalars */
template <typename T>
void scalar_serialize(serializer &ser, const std::string &key, T value)
{
    ser.write(key, ndview<const T>(&value, nullptr, 0));
}

/** Helper function for deserialization of scalars */
template <typename T>
void scalar_deserialize(deserializer &ser, const std::string &key, T &value)
{
    ser.read(key, ndview<T>(&value, nullptr, 0));
}

}}}
