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

template <typename Derived>
typename std::enable_if<
        eigen_scalar_is<Derived, alps::alea::complex_op<double>>::value>::type
serialize(serializer &ser, const std::string &key,
          const Eigen::MatrixBase<Derived> &value)
{
    throw std::runtime_error("serializing complex_op matrices not implemented.");
}

template <typename T>
typename std::enable_if<
            std::is_same<T, alps::alea::complex_op<double>>::value>::type
deserialize(deserializer &ser, const std::string &key,
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &value)
{
    throw std::runtime_error("serializing complex_op matrices not implemented.");
}

template <typename T>
typename std::enable_if<
            std::is_same<T, alps::alea::complex_op<double>>::value>::type
deserialize(deserializer &ser, const std::string &key,
            Eigen::Matrix<T, Eigen::Dynamic, 1> &value)
{
    throw std::runtime_error("serializing complex_op matrices not implemented.");
}

template <typename T>
typename std::enable_if<
            std::is_same<T, alps::alea::complex_op<double>>::value>::type
deserialize(deserializer &ser, const std::string &key,
            Eigen::Matrix<T, 1, Eigen::Dynamic> &value)
{
    throw std::runtime_error("serializing complex_op matrices not implemented.");
}

}}
