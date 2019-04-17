/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <Eigen/Dense>
#include <array>

#include <alps/serialization/core.hpp>

namespace alps { namespace serialization {

template <typename Derived>
struct eigen_scalar {
    using type = typename Eigen::internal::traits<Derived>::Scalar;
};

template <typename Derived>
using eigen_scalar_t = typename eigen_scalar<Derived>::type;

template <typename Derived>
struct has_primitive_scalar {
    static const bool value = is_primitive<eigen_scalar_t<Derived>>::value;
};

template <typename Derived, typename T>
struct eigen_scalar_is {
    static const bool value = std::is_same<eigen_scalar_t<Derived>, T>::value;
};

template <typename Derived>
typename std::enable_if<has_primitive_scalar<Derived>::value, void>::type
serialize(serializer &ser, const std::string &key,
          const Eigen::MatrixBase<Derived> &value)
{
    typedef Eigen::internal::traits<Derived> traits;
    typedef typename traits::Scalar scalar_type;
    typedef Eigen::Matrix<scalar_type, Derived::RowsAtCompileTime,
                          Derived::ColsAtCompileTime> plain_matrix_type;

    // Ensure that evaluated expression will be C-contiguous
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

template <typename T>
typename std::enable_if<is_primitive<T>::value, void>::type
deserialize(deserializer &ser, const std::string &key,
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &value)
{
    std::array<size_t, 2> shape = {{(size_t)value.cols(), (size_t)value.rows()}};
    ser.read(key, ndview<T>(value.data(), shape.data(), shape.size()));
}

template <typename T>
typename std::enable_if<is_primitive<T>::value, void>::type
deserialize(deserializer &ser, const std::string &key,
            Eigen::Matrix<T, Eigen::Dynamic, 1> &value)
{
    std::array<size_t, 1> shape = {{(size_t)value.rows()}};
    ser.read(key, ndview<T>(value.data(), shape.data(), shape.size()));
}

template <typename T>
typename std::enable_if<is_primitive<T>::value, void>::type
deserialize(deserializer &ser, const std::string &key,
            Eigen::Matrix<T, 1, Eigen::Dynamic> &value)
{
    std::array<size_t, 1> shape = {{(size_t)value.cols()}};
    ser.read(key, ndview<T>(value.data(), shape.data(), shape.size()));
}

}}
