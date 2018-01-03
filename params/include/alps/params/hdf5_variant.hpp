/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file mpi_variant.hpp
    
    @brief Header for archiving interface for boost::variant

    @todo FIXME: Move to archive/hdf5
*/

#ifndef ALPS_PARAMS_HDF5_VARIANT_HPP_e36c01f03a8b4756a0a0a098877618dd
#define ALPS_PARAMS_HDF5_VARIANT_HPP_e36c01f03a8b4756a0a0a098877618dd

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/pair.hpp>

#include <alps/params/serialize_variant.hpp>

#include <cassert>

namespace alps {
    namespace hdf5 {

        namespace detail {
            /// Consumer class to save an object to archive
            struct to_archive {
                alps::hdf5::archive& ar_;
                std::string context_;

                to_archive(alps::hdf5::archive& ar) : ar_(ar), context_(ar.get_context()) {}

                template <typename T>
                void operator()(const T& val) {
                    ar_[context_] << val;
                }
            };
            
            /// Producer class to load an object from an archive
            struct from_archive {
                alps::hdf5::archive& ar_;
                std::string context_;

                /// Convenience predicate: can we read it as type T?
                template <typename T>
                inline bool can_read(const T*)
                {
                    return is_native_type<T>::value
                        && ar_.is_datatype<T>(context_)
                        && ar_.is_scalar(context_);
                }

                /// Convenience predicate: can we read it as a vector of T?
                template <typename T>
                bool can_read(const std::vector<T>*)
                {
                    return is_native_type<T>::value
                        && ar_.is_datatype<T>(context_)
                        && !ar_.is_scalar(context_);
                }
                
                from_archive(alps::hdf5::archive& ar)
                    : ar_(ar), context_(ar.get_context())
                {}

                template <typename T>
                boost::optional<T> operator()(const T*)
                {
                    boost::optional<T> maybe_val;
                    if (can_read((T*)0)) {
                        T val;
                        ar_[context_] >> val;
                        maybe_val=val;
                    }
                    return maybe_val;
                }
            };

            typedef alps::detail::variant_serializer<alps::params_ns::detail::dict_all_types,
                                                    to_archive, from_archive> var_serializer;
            typedef var_serializer::variant_type variant_type;
            
        } // detail::
        
        /// saving of a boost::variant over MPL type sequence MPLSEQ
        template <typename MPLSEQ>
        inline void write_variant(alps::hdf5::archive& ar, const typename boost::make_variant_over<MPLSEQ>::type& var)
        {
            detail::to_archive consumer(ar);
            detail::var_serializer::consume(consumer, var);
        }

        /// loading of a boost::variant over MPL type sequence MPLSEQ
        template <typename MPLSEQ>
        inline
        typename boost::make_variant_over<MPLSEQ>::type read_variant(alps::hdf5::archive& ar)
        {
            detail::from_archive producer(ar);
            return detail::var_serializer::produce(producer);
        }

    } // mpi::
} // alps::

#endif /* ALPS_PARAMS_HDF5_VARIANT_HPP_e36c01f03a8b4756a0a0a098877618dd */
