/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file archive_traits.hpp defines traits for the data types loadability */

#pragma once

#include <alps/hdf5/archive.hpp>

namespace alps {
    namespace accumulators {
        namespace detail {

            /// Trait to determine loadability of type T
            template <typename T>
            struct archive_trait {
                typedef typename alps::hdf5::scalar_type<T>::type scalar_type;
                /// Determine if the type can be loaded from @param name
                /** Parameters:
                    @param ar : Archive object
                    @param name : HDF5 path
                    @param dim : expected dimension of the stored array if dim>0;
                                 otherwise a scalar value expected.
                */
                static bool can_load(hdf5::archive& ar,
                                     const std::string& name,
                                     std::size_t dim) {
                    bool ok=ar.is_data(name) &&
                        !ar.is_attribute(name+"/@c++_type") && // plain types should not have the attribute
                        ar.is_datatype<scalar_type>(name) &&
                        ((dim==0 && ar.is_scalar(name)) ||
                         (dim>0  && ar.dimensions(name)==dim));
                    return ok;
                }
            };

        } // detail::
    } // accumulators::
} // alps::
