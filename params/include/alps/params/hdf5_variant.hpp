/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
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
                int which_;

                to_archive(alps::hdf5::archive& ar, int which) : ar_(ar), which_(which) {}

                template <typename T>
                void operator()(const T& val) {
                    ar_[""] << val;
                    ar_["@which"] << which_; // FIXME!!!! This is proof-of-concept! This is a bad format for long-term archiving!
                }
            };

            /// Producer class to load an object from an archive
            struct from_archive {
                int which_count_;
                alps::hdf5::archive& ar_;

                from_archive(alps::hdf5::archive& ar)
                    : which_count_(0), ar_(ar)
                {}

                template <typename T>
                boost::optional<T> operator()(const T*)
                {
                    boost::optional<T> ret;
                    int target_which;
                    ar_["@which"] >> target_which; // FIXME!!!! This is proof-of-concept! This is a bad format for long-term archiving!
                    if (target_which==which_count_) {
                        T val;
                        ar_[""] >> val;
                        ret=val;
                    }
                    ++which_count_;
                    return ret;
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
            int which=var.which();
            detail::to_archive consumer(ar, which);
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
