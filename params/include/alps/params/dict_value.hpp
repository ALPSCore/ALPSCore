/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
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

#include <iostream>
#include <stdexcept>

#include <boost/variant/variant.hpp>
// #include <boost/utility.hpp> /* for enable_if */

#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_signed.hpp>
#include <boost/type_traits/is_unsigned.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/type_traits/make_unsigned.hpp>
#include <boost/type_traits/make_signed.hpp>
#include <boost/integer_traits.hpp>
// using boost::enable_if;
// using boost::enable_if_c;
// using boost::disable_if;
// using boost::disable_if_c;

#include <boost/type_index.hpp> // for pretty-printing exceptions

// #include "alps/utilities/short_print.hpp" // for streaming
// #include "alps/hdf5/archive.hpp"          // archive support
// #include "alps/hdf5/vector.hpp"           // vector archiving support
// #include "alps/hdf5/map.hpp"              // map archiving support

#include "./dict_exceptions.hpp"
#include "./dict_types.hpp" // Sequences of supported types
// #include "alps/params/param_types_ranking.hpp" // for detail::is_convertible<F,T>

// #ifdef ALPS_HAVE_MPI
// #include <alps/utilities/mpi_map.hpp>
// #include <alps/utilities/mpi_optional.hpp>
// #include <alps/utilities/mpi_vector.hpp>
// #endif

namespace alps {
    namespace params_ns {
        namespace detail {
            template <bool V> struct yes_no {};
            template <> struct yes_no<true> : public boost::true_type { typedef bool yes; };
            template <> struct yes_no<false> : public boost::false_type { typedef bool no; };

            template <typename T>
            struct is_bool : public yes_no<boost::is_same<T,bool>::value> {};

            // bool type is NOT integral for the purposes of this code
            template <typename T>
            struct is_integral : public yes_no<boost::is_integral<T>::value && !is_bool<T>::value> {};

            // signed value: integral and signed
            template <typename T>
            struct is_signed : public yes_no<boost::is_signed<T>::value && is_integral<T>::value> {};

            // unsigned value: integral and unsigned
            template <typename T>
            struct is_unsigned : public yes_no<boost::is_unsigned<T>::value && is_integral<T>::value> {};

            // meta-predicate: conversion bool->integral
            template <typename FROM, typename TO>
            struct is_bool_to_integral
                : public yes_no<is_bool<FROM>::value && is_integral<TO>::value>
            {};

            // meta-predicate: conversion signed->integral
            template <typename FROM, typename TO>
            struct is_sig_to_intgl
                : public yes_no<is_signed<FROM>::value && is_integral<TO>::value>
            {};

            // meta-predicate: conversion unsigned->integral
            template <typename FROM, typename TO>
            struct is_unsig_to_intgl
                : public yes_no<is_unsigned<FROM>::value && is_integral<TO>::value>
            {};

            // meta-predicate: conversion integral->floating_point
            template <typename FROM, typename TO>
            struct is_intgl_to_fp
                : public yes_no<is_integral<FROM>::value && boost::is_floating_point<TO>::value>
            {};

            // meta-predicate: general conversion, not caught by other ones
            template <typename FROM, typename TO>
            struct is_other_conversion
                : public yes_no<!is_bool_to_integral<FROM,TO>::value &&
                                !is_sig_to_intgl<FROM,TO>::value &&
                                !is_unsig_to_intgl<FROM,TO>::value &&
                                !is_intgl_to_fp<FROM,TO>::value
                               >
            {};
            
            namespace visitor {
                /// Visitor to get a value (with conversion): returns type LHS_T, converts from the bound type RHS_T
                template <typename LHS_T>
                struct getter: public boost::static_visitor<LHS_T> {

                    /// Simplest case: the values are of the same type
                    LHS_T apply(const LHS_T& val) const {
                        return val; // no conversion 
                    }

                    /// Extracting bool type to an integral type
                    template <typename RHS_T>
                    LHS_T apply(const RHS_T& val, typename is_bool_to_integral<RHS_T,LHS_T>::yes =true) const {
                        return val;
                    }

                    /// Extracting integral type to a floating point type
                    template <typename RHS_T>
                    LHS_T apply(const RHS_T& val, typename is_intgl_to_fp<RHS_T,LHS_T>::yes =true) const {
                        return val;
                    }

                    /// Extracting unsigned integral type to an integral type
                    template <typename RHS_T>
                    LHS_T apply(const RHS_T& val, typename is_unsig_to_intgl<RHS_T,LHS_T>::yes =true) const {
                        typedef typename boost::make_unsigned<LHS_T>::type U_LHS_T;
                        const U_LHS_T max_num=boost::integer_traits<LHS_T>::const_max; // always possible
                        // compare 2 unsigned
                        if (val>max_num)
                            throw exception::value_mismatch("", "Integer overflow detected: unsigned integer too large");
                        return val;
                    }
                
                    /// Extracting signed integral type to an integral type
                    template <typename RHS_T>
                    LHS_T apply(const RHS_T& val, typename is_sig_to_intgl<RHS_T,LHS_T>::yes =true) const {
                        typedef typename boost::make_signed<LHS_T>::type S_LHS_T;
                        typedef typename boost::make_unsigned<LHS_T>::type U_LHS_T;
                        typedef typename boost::make_unsigned<RHS_T>::type U_RHS_T;

                        const S_LHS_T min_num=boost::integer_traits<LHS_T>::const_min; // always possible
                        
                        if (val<min_num)
                            throw exception::value_mismatch("", "Integer underflow detected: signed integer too small");

                        if (val<0) return val; // always within range
                        
                        const U_LHS_T max_num=boost::integer_traits<LHS_T>::const_max; // always possible
                        const U_RHS_T uval=val; // as val>=0, it's always correct
                        // compare 2 unsigned
                        if (uval>max_num)
                            throw exception::value_mismatch("", "Integer overflow detected: signed integer too large");
                        return val;
                    }
                
                    /// Placeholder: extracting any other type
                    template <typename RHS_T>
                    LHS_T apply(const RHS_T& val, typename is_other_conversion<RHS_T,LHS_T>::yes =true) const {
                        std::string rhs_name=boost::typeindex::type_id<RHS_T>().pretty_name();
                        std::string lhs_name=boost::typeindex::type_id<LHS_T>().pretty_name();
                        throw exception::type_mismatch("","Types do not match;"
                                                       " conversion " + rhs_name + " --> " + lhs_name);
                    }
                

                    /// Called by apply_visitor()
                    template <typename RHS_T>
                    LHS_T operator()(const RHS_T& val) const {
                        return apply(val);
                    }
                };

                /// Visitor to get a value (with conversion to bool): returns type bool, converts from the bound type RHS_T
                template <>
                struct getter<bool>: public boost::static_visitor<bool> {

                    /// Simplest case: the value is of type bool
                    bool apply(const bool& val) const {
                        return val; // no conversion 
                    }

                    /// Extracting any other type
                    template <typename RHS_T>
                    bool apply(const RHS_T& val) const {
                        std::string rhs_name=boost::typeindex::type_id<RHS_T>().pretty_name();
                        throw exception::type_mismatch("","Cannot convert non-bool type "+rhs_name+" to bool");
                    }

                    /// Called by apply_visitor()
                    template <typename RHS_T>
                    bool operator()(const RHS_T& val) const {
                        return apply(val);
                    }
                };

                /// Visitor to check if the type is X
                template <typename X>
                class check_type : public boost::static_visitor<bool> {
                  public:
                    /// Called by apply_visitor() if the bound type is X
                    bool operator()(const X& val) const { return true; }
                    
                    /// Called by apply_visitor() for bound type T
                    template <typename T>
                    bool operator()(const T& val) const { return false; }
                };
                    
            }
            

        } //::detail
        
        class dict_value {
          public:

            typedef boost::make_variant_over<detail::dict_all_types>::type value_type;
            typedef detail::None None; ///< "Empty value" type

          private:
            std::string name_; ///< The option name (FIXME: make it "functionally const")
            value_type val_; ///< Value of the option

            
          public:

            /// Constructs the empty value
            explicit dict_value(const std::string& name): name_(name), val_() {}
            
            /// whether the value contains None
            bool empty() const { return val_.which()==0; } // NOTE: relies on `None` being the first type

            /// check the type of the containing value
            template <typename X>
            bool isType() const {
                return boost::apply_visitor(detail::visitor::check_type<X>(), val_);
            }

            /// Assignment operator (with conversion)
            template <typename T>
            const T& operator=(const T& rhs) {
                val_=rhs;
                return rhs;
            }
            
            /// Assignment operator (with conversion from `const char*`)
            const char* operator=(const char* rhs) {
                val_=std::string(rhs);
                return rhs;
            }
            
            /// Shortcut for explicit conversion to a target type
            template <typename T>
            T as() const {
                if (this->empty()) throw exception::uninitialized_value(name_,"Attempt to read uninitialized value");
                try {
                    return boost::apply_visitor(detail::visitor::getter<T>(), val_);
                } catch (exception::exception_base& exc) {
                    exc.set_name(name_);
                    throw;
                }
            }
            
            /// Conversion to a target type, explicit or implicit
            template <typename T>
            operator T() const {
                return as<T>();
            }

            /// Reset to an empty value
            void clear() { val_=None(); }
        };

        template <typename T>
        bool operator==(const T& lhs, const dict_value& rhs) {return false; }
        
        // template <typename T>
        // bool operator<(const T& lhs, const dict_value& rhs) {return false; }
        
        // template <typename T>
        // bool operator>(const T& lhs, const dict_value& rhs) {return false; }
        
        template <typename T>
        bool operator!=(const T& lhs, const dict_value& rhs) {return false; }
        
        // template <typename T>
        // bool operator>=(const T& lhs, const dict_value& rhs) {return false; }
        
        // template <typename T>
        // bool operator<=(const T& lhs, const dict_value& rhs) {return false; }
        
        template <typename T>
        bool operator==(const dict_value& lhs, const T& rhs) {return false; }
        
        // template <typename T>
        // bool operator<(const dict_value& lhs, const T& rhs) {return false; }
        
        // template <typename T>
        // bool operator>(const dict_value& lhs, const T& rhs) {return false; }
        
        template <typename T>
        bool operator!=(const dict_value& lhs, const T& rhs) {return false; }
        
        // template <typename T>
        // bool operator>=(const dict_value& lhs, const T& rhs) {return false; }
        
        // template <typename T>
        // bool operator<=(const dict_value& lhs, const T& rhs) {return false; }
        
        inline bool operator==(const dict_value& lhs, const dict_value& rhs) {return false; }
        
        // inline bool operator<(const dict_value& lhs, const dict_value& rhs) {return false; }
        
        // inline bool operator>(const dict_value& lhs, const dict_value& rhs) {return false; }
        
        inline bool operator!=(const dict_value& lhs, const dict_value& rhs) {return false; }
        
        // inline bool operator>=(const dict_value& lhs, const dict_value& rhs) {return false; }
        
        // inline bool operator<=(const dict_value& lhs, const dict_value& rhs) {return false; }
        
    } // ::params_ns

#if 0
#ifdef ALPS_HAVE_MPI
    namespace mpi {
        inline
        void broadcast(const alps::mpi::communicator &comm, alps::params_ns::option_type& val, int root)
        {
            val.broadcast(comm, root);
        }
    }
#endif
#endif /* 0 */
    
} // ::alps

#endif /* ALPS_PARAMS_DICT_VALUE_HPP_a8ecbead92aa4a1995f43adfc6d0aae0 */