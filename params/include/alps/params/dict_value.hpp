/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file dict_value.hpp Defines type(s) used to populate `alps::params_ns::dictionary` container. */

/*  Requirements for the `dict_value_type`:

    1. It can hold a value from a predefined set of scalar types, of
       corresponding vector types, or can be in the "undefined" ("empty") state.

    2. Any value can be assigned to it; the object acquires both the
       type and the value, if it is convertible to one of the
       supported types. The value is converted to a "larger" supported
       type.
       Special case 1: conversion from char is unspecified.
       Special case 2: conversion from char* to string is supported.

    3. If "undefined", it cannot be assigned to anything.

    4. If holds a value of some type, it can be assigned to the same or a "larger" type.
       Special case 1: conversion to char is unspecified (may throw).
       Special case 2: conversion to char* is explicitly unsupported, even for strings
       (the user can use `const char* p=val.as<string>().c_str()` and face the consequences).

    5. It holds its name for error reporting purposes.

    6. It can be streamed to an HDF5 archive member.

    7. It can be broadcast over MPI.

    8. It can be streamed to an `ostream`.
*/

#ifndef ALPS_PARAMS_DICT_VALUE_HPP_a8ecbead92aa4a1995f43adfc6d0aae0
#define ALPS_PARAMS_DICT_VALUE_HPP_a8ecbead92aa4a1995f43adfc6d0aae0

#include <iosfwd>
#include <type_traits>
#include <stdexcept>

#include <boost/variant/variant.hpp>

#include "./dict_exceptions.hpp"
#include "./dict_types.hpp" // Sequences of supported types

#include <alps/hdf5/archive.hpp>

#ifdef ALPS_HAVE_MPI
#include <alps/utilities/mpi.hpp>
#endif

namespace alps {
    namespace params_ns {

        namespace detail {
            template <typename> struct is_allowed;
        }

        class dict_value {
          public:

            typedef boost::make_variant_over<detail::dict_all_types>::type value_type;
            typedef detail::None None; ///< "Empty value" type

          private:
            std::string name_; ///< The option name (for proper error messages)
            value_type val_; ///< Value of the option

          public:

            /// Constructs the empty nameless value
            // FIXME: This is used only for MPI and must be changed
            dict_value(): name_("NO_NAME"), val_() {}

            /// Constructs the empty value
            explicit dict_value(const std::string& name): name_(name), val_() {}

            /// whether the value contains None
            bool empty() const;

            /// check the type of the containing value
            template <typename X>
            bool isType() const;

            /// Assignment operator (with conversion)
            template <typename T>
            const T& operator=(const T& rhs);

            /// Assignment operator (with conversion from `const char*`)
            const char* operator=(const char* rhs);

            /// Shortcut for explicit conversion to a target type
            template <typename T>
            T as() const;

            /// Conversion to a target type, explicit or implicit
            template <typename T
#ifndef BOOST_NO_CXX11_FUNCTION_TEMPLATE_DEFAULT_ARGS
                      ,typename std::enable_if<detail::is_allowed<T>::value, int>::type =0
#endif
                      >
            inline operator T() const {
                // BOOST_STATIC_ASSERT_MSG(detail::is_allowed<T>::value, "The type is not supported by the dictionary");
                return as<T>();
            }

            // template <typename T, int>
            // operator T() const;

            /// Reset to an empty value
            void clear();

            /// Comparison
            /** a.compare(b) returns 0 if a==b, !=0 if a!=b.
                if well-ordered, returns -1 if a<b, +1 if a>b.
            */
            template <typename T>
            int compare(const T& rhs) const;

            int compare(const dict_value& rhs) const;

            /// Returns true if the objects hold the same type and value, false otherwise
            bool equals(const dict_value& rhs) const;

            /// Saves the value to an archive
            void save(alps::hdf5::archive& ar) const;

            /// Loads the value from an archive
            void load(alps::hdf5::archive& ar);

            /// Const-access visitor to the bound value
            /** @param visitor functor should be callable as `R result=visitor(bound_value_const_ref)`

                The functor type `F` must define typename `F::result_type`.
             */
            template <typename F>
            typename F::result_type apply_visitor(F& visitor) const;

            /// Const-access visitor to the bound value
            /** @param visitor functor should be callable as `R result=visitor(bound_value_const_ref)`

                The functor type `F` must define typename `F::result_type`.
             */
            template <typename F>
            typename F::result_type apply_visitor(const F& visitor) const;

            /// Print the value, possibly together with type, in a human-readable format
            friend
            std::ostream& print(std::ostream&, const dict_value&, bool terse);

#ifdef ALPS_HAVE_MPI
            void broadcast(const alps::mpi::communicator& comm, int root);
#endif
        };

        /// Print the value together with type in some human-readable format
        inline std::ostream& operator<<(std::ostream& os, const dict_value& dv) {
            return print(os, dv, false);
        }

    } // params_ns::
} // alps::

#include "./dict_value_impl.hpp"

#endif /* ALPS_PARAMS_DICT_VALUE_HPP_a8ecbead92aa4a1995f43adfc6d0aae0 */
