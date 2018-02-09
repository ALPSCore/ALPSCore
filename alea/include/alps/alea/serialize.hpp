/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>

namespace alps { namespace alea { namespace internal {

/**
 * Allows RAII-type use of groups.
 */
struct group_sentry
{
    group_sentry(serializer &ser, const std::string &group)
        : ser_(ser)
        , group_(group)
    {
        if (group != "")
            ser_.enter(group);
    }

    ~group_sentry()
    {
        if (group_ != "")
            ser_.exit();
    }

private:
    serializer &ser_;
    std::string group_;
};

/**
 * Helper function
 */
template <typename T>
void scalar_serialize(serializer &ser, const std::string &key, T value)
{
    ser.write("key", ndview<const T>(&value, nullptr, 0));
}

}}}

namespace alps { namespace alea {

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
               const Eigen::DenseBase<Derived> &value)
{
    typedef Eigen::internal::traits<Derived> traits;
    typedef const typename traits::Scalar scalar_type;

    // TODO figure out whether strided arrays are evaluated
    auto temp = value.eval();

    if (Derived::ColsAtCompileTime == 1 || Derived::RowsAtCompileTime == 1) {
        // Omit second dimension for simple vectors
        std::array<size_t, 1> dims = {(size_t)temp.size()};
        ser.write(key, ndview<scalar_type>(temp.data(), dims.data(), 1));
    } else {
        // Eigen arrays are column-major
        std::array<size_t, 2> dims = {(size_t)temp.cols(), (size_t)temp.rows()};
        ser.write(key, ndview<scalar_type>(temp.data(), dims.data(), 2));
    }
}

}}
