/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_HDF5_STD_COMPLEX
#define ALPS_HDF5_STD_COMPLEX

#include <alps/hdf5/archive.hpp>

#include <complex>

namespace alps {
    namespace hdf5 {

        template<typename T> struct scalar_type<std::complex<T> > {
            typedef typename scalar_type<typename std::complex<T>::value_type>::type type;
        };

        template<typename T> struct is_continuous<std::complex<T> >
            : public is_continuous<T>
        {};
        template<typename T> struct is_continuous<std::complex<T> const >
            : public is_continuous<T>
        {};

        template<typename T> struct has_complex_elements<std::complex<T> >
            : public std::true_type
        {};

        namespace detail {

            template<typename T> struct get_extent<std::complex<T> > {
                static std::vector<std::size_t> apply(std::complex<T> const & /*value*/) {
                    return std::vector<std::size_t>(1, 2);
                }
            };

            template<typename T> struct set_extent<std::complex<T> > {
                static void apply(std::complex<T> & /*value*/, std::vector<std::size_t> const & /*extent*/) {}
            };

            template<typename T> struct is_vectorizable<std::complex<T> > {
                static bool apply(std::complex<T> const & /*value*/) {
                    return true;
                }
            };
            template<typename T> struct is_vectorizable<std::complex<T> const> {
                static bool apply(std::complex<T> const & /*value*/) {
                    return true;
                }
            };

            template<typename T> struct get_pointer<std::complex<T> > {
                static typename scalar_type<std::complex<T> >::type * apply(std::complex<T> & value) {
                    using alps::hdf5::get_pointer;
                    return get_pointer(*reinterpret_cast<typename scalar_type<std::complex<T> >::type *> (&value));
                }
            };

            template<typename T> struct get_pointer<std::complex<T> const> {
                static typename scalar_type<std::complex<T> >::type const * apply(std::complex<T> const & value) {
                    using alps::hdf5::get_pointer;
                    return get_pointer(*reinterpret_cast<typename scalar_type<std::complex<T> >::type const *> (&value));
                }
            };
        }

        template<typename T> void save(
              archive & ar
            , std::string const & path
            , std::complex<T> const & value
            , std::vector<std::size_t> size = std::vector<std::size_t>()
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            if (is_continuous<T>::value) {
                size.push_back(2);
                chunk.push_back(2);
                offset.push_back(0);
                ar.write(path, get_pointer(value), size, chunk, offset);
            } else
                throw wrong_type("invalid type" + ALPS_STACKTRACE);
        }

        template<typename T> void load(
              archive & ar
            , std::string const & path
            , std::complex<T> & value
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            if (ar.is_group(path) || !is_continuous<T>::value)
                throw wrong_type("invalid path" + ALPS_STACKTRACE);
            else if (!ar.is_complex(path))
                throw archive_error("no complex value in archive" + ALPS_STACKTRACE);
            else {
                chunk.push_back(2);
                offset.push_back(0);
                ar.read(path, get_pointer(value), chunk, offset);
            }
        }
    }
}

#endif
