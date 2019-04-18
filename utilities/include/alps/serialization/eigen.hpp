/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <Eigen/Dense>
#include <array>

#include <iostream> //FIXME

#include <alps/serialization/core.hpp>

namespace alps { namespace serialization {

/** Type trait that extracts the underlying scalar type of array */
template <typename Derived>
struct eigen_scalar {
    using type = typename Eigen::internal::traits<Derived>::Scalar;
};

/** Type trait that extracts the underlying scalar type of array */
template <typename Derived>
using eigen_scalar_t = typename eigen_scalar<Derived>::type;

/** Underlying Eigen scalar type is a serialization primitive */
template <typename Derived>
struct has_primitive_scalar {
    static const bool value = is_primitive<eigen_scalar_t<Derived>>::value;
};

/** Underlying Eigen scalar type is `T` */
template <typename Derived, typename T>
struct eigen_scalar_is {
    static const bool value = std::is_same<eigen_scalar_t<Derived>, T>::value;
};

/** Returns true if and only if Eigen array is Fortran-contiguous */
template <typename Derived>
constexpr bool eigen_is_contiguous(const Eigen::PlainObjectBase<Derived> &matrix)
{
    // check that unless the dimension is trivial (of size 1), the strides
    // must be that of a Fortran-contiguous (column-major) array.
    return (matrix.rows() == 1 || matrix.rowStride() == 1)
        && (matrix.cols() == 1 || matrix.colStride() == matrix.rows());
}

/** Serializes Eigen array or matrix */
template <typename Derived>
typename std::enable_if<has_primitive_scalar<Derived>::value, void>::type
serialize(serializer &ser, const std::string &key,
          const Eigen::PlainObjectBase<Derived> &value)
{
    using scalar_type = eigen_scalar_t<Derived>;
    using matrix_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>;

    // Ensure that evaluated expression will be Fortran-contiguous
    if (!eigen_is_contiguous(value))
        serialize(ser, key, matrix_type(value));

    // Evaluate to matrix or proxy object if already matrix
    if (Derived::ColsAtCompileTime == 1 || Derived::RowsAtCompileTime == 1) {
        // Omit second dimension for simple vectors
        std::array<size_t, 1> dims = {{(size_t)value.size()}};
        ser.write(key, ndview<const scalar_type>(value.data(), dims.data(), 1));
    } else {
        // Eigen arrays are column-major
        std::array<size_t, 2> dims = {{(size_t)value.cols(), (size_t)value.rows()}};
        ser.write(key, ndview<const scalar_type>(value.data(), dims.data(), 2));
    }
}

/** Deserializes Eigen array or matrix */
template <typename Derived>
typename std::enable_if<has_primitive_scalar<Derived>::value, void>::type
deserialize(deserializer &ser, const std::string &key,
            Eigen::PlainObjectBase<Derived> &value)
{
    using scalar_type = eigen_scalar_t<Derived>;

    // Ensure that evaluated expression will be Fortran-contiguous
    if (!eigen_is_contiguous(value)) {
        throw std::runtime_error("Unable to read to array: "
                                 "it must be Fortran contiguous in memory");
    }

    if (Derived::ColsAtCompileTime == 1 || Derived::RowsAtCompileTime == 1) {
        // Omit second dimension for simple vectors
        std::array<size_t, 1> shape = {{(size_t)value.size()}};
        ser.read(key, ndview<scalar_type>(value.data(), shape.data(), shape.size()));
    } else {
        // Extract underlying buffer and read
        std::array<size_t, 2> shape = {{(size_t)value.cols(), (size_t)value.rows()}};
        ser.read(key, ndview<scalar_type>(value.data(), shape.data(), shape.size()));
    }
}

}}
