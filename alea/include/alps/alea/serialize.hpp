/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>
#include <alps/serialization/core.hpp>
#include <alps/serialization/util.hpp>
#include <alps/serialization/eigen.hpp>

#include <array>

namespace alps { namespace serialization {

/** Serializes Eigen array of complex_op<double> */
template <typename Derived>
typename std::enable_if<
        eigen_scalar_is<Derived, alps::alea::complex_op<double>>::value>::type
serialize(serializer &ser, const std::string &key,
          const Eigen::PlainObjectBase<Derived> &value)
{
    using scalar_type = eigen_scalar_t<Derived>;
    using matrix_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>;

    // Ensure that evaluated expression will be Fortran-contiguous
    if (!eigen_is_contiguous(value))
        serialize(ser, key, matrix_type(value));

    // Evaluate to matrix or proxy object if already matrix
    const double *dbl_data = reinterpret_cast<const double *>(value.data());
    if (Derived::ColsAtCompileTime == 1 || Derived::RowsAtCompileTime == 1) {
        // Omit second dimension for simple vectors
        std::array<size_t, 3> dims = {{(size_t)value.size(), 2, 2}};
        ser.write(key, ndview<const double>(dbl_data, dims.data(), dims.size()));
    } else {
        // Eigen arrays are column-major
        std::array<size_t, 4> dims = {{(size_t)value.cols(), value.rows(), 2, 2}};
        ser.write(key, ndview<const double>(dbl_data, dims.data(), dims.size()));
    }
}

/** Deserializes Eigen array or matrix of complex_op<double> */
template <typename Derived>
typename std::enable_if<
        eigen_scalar_is<Derived, alps::alea::complex_op<double>>::value>::type
deserialize(deserializer &ser, const std::string &key,
            Eigen::PlainObjectBase<Derived> &value)
{
    // Ensure that evaluated expression will be Fortran-contiguous
    if (!eigen_is_contiguous(value)) {
        throw std::runtime_error("Unable to read to array: "
                                 "it must be Fortran contiguous in memory");
    }

    // Evaluate to matrix or proxy object if already matrix
    double *dbl_data = reinterpret_cast<double *>(value.data());
    if (Derived::ColsAtCompileTime == 1 || Derived::RowsAtCompileTime == 1) {
        // Omit second dimension for simple vectors
        std::array<size_t, 3> shape = {{(size_t)value.size(), 2, 2}};
        ser.read(key, ndview<double>(dbl_data, shape.data(), shape.size()));
    } else {
        // Extract underlying buffer and read
        std::array<size_t, 4> shape = {{(size_t)value.cols(), (size_t)value.rows(), 2, 2}};
        ser.read(key, ndview<double>(dbl_data, shape.data(), shape.size()));
    }
}

}}
