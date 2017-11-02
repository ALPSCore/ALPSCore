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

#ifdef ALPS_HAVE_MPI
#include <alps/utilities/mpi.hpp>
// #include <alps/utilities/mpi_map.hpp>
// #include <alps/utilities/mpi_optional.hpp>
// #include <alps/utilities/mpi_vector.hpp>
#endif



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

            // meta-predicate: both types are unsigned
            template <typename A, typename B>
            struct is_both_unsigned
                : public yes_no<is_unsigned<A>::value && is_unsigned<B>::value>
            {};

            // meta-predicate: both types are signed
            template <typename A, typename B>
            struct is_both_signed
                : public yes_no<is_signed<A>::value && is_signed<B>::value>
            {};

            // meta-predicate: first type is signed, the other is unsigned
            template <typename A, typename B>
            struct is_signed_unsigned
                : public yes_no<is_signed<A>::value && is_unsigned<B>::value>
            {};

            // meta-predicate: one of the types, but not the other, is bool
            template <typename A, typename B>
            struct is_either_bool
                : public yes_no< (is_bool<A>::value && !is_bool<B>::value) ||
                                 (!is_bool<A>::value && is_bool<B>::value) >
            {};

            // meta-predicate: floating-point conversion between different types
            template <typename A, typename B>
            struct is_fp_conv
                : public yes_no< boost::is_floating_point<A>::value &&
                                 boost::is_floating_point<B>::value &&
                                 !boost::is_same<A,B>::value>
            {};
            
            // meta-predicate: other comparison
            template <typename A, typename B>
            struct is_other_cmp
                : public yes_no<!is_either_bool<A,B>::value &&
                                !is_both_signed<A,B>::value && !is_both_unsigned<A,B>::value &&
                                !is_signed_unsigned<A,B>::value && !is_signed_unsigned<B,A>::value &&
                                !is_intgl_to_fp<A,B>::value && !is_intgl_to_fp<B,A>::value &&
                                !is_fp_conv<A,B>::value>
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


                /// Visitor to compare a value of type RHS_T with the value of the bound type LHS_T
                template <typename RHS_T>
                class comparator : public boost::static_visitor<int> {
                    const RHS_T& rhs_;

                    template <typename A, typename B>
                    static bool cmp_(const A& a, const B& b) { return (a==b)? 0 : (a<b)? -1:1; }
                    
                  public:
                    comparator(const RHS_T& rhs): rhs_(rhs) {}

                    /// Called by apply_visitor when the bound type us the same as RHS_T
                    int operator()(const RHS_T& lhs) const {
                        return cmp_(lhs,rhs_);
                    }
                    
                    /// Called by apply_visitor for the bound value of type LHS_T
                    template <typename LHS_T>
                    int operator()(const LHS_T& lhs) const {
                        return apply(lhs,rhs_);
                    }

                    /// Invoked when the bound type is `bool` and is compared with another type
                    template <typename LHS_T>
                    int apply(const LHS_T& lhs, const RHS_T& rhs,
                              typename is_either_bool<LHS_T,RHS_T>::yes =true) const {
                        std::string lhs_name=boost::typeindex::type_id<LHS_T>().pretty_name();
                        std::string rhs_name=boost::typeindex::type_id<RHS_T>().pretty_name();
                        throw exception::type_mismatch("","Attempt to compare a boolean value with an incompatible type "+
                                                       lhs_name + "<=>" + rhs_name);
                    }

                    /// Invoked when both types are signed
                    template <typename LHS_T>
                    int apply(const LHS_T& lhs, const RHS_T& rhs, typename is_both_signed<LHS_T,RHS_T>::yes =true) const {
                        return cmp_(lhs,rhs);
                    }

                    /// Invoked when both types are unsigned
                    template <typename LHS_T>
                    int apply(const LHS_T& lhs, const RHS_T& rhs, typename is_both_unsigned<LHS_T,RHS_T>::yes =true) const {
                        return cmp_(lhs,rhs);
                    }

                    /// Invoked when a signed bound type is compared with an unsigned type
                    template <typename LHS_T>
                    int apply(const LHS_T& lhs, const RHS_T& rhs, typename is_signed_unsigned<LHS_T,RHS_T>::yes =true) const {
                        typedef typename boost::make_unsigned<LHS_T>::type U_LHS_T;
                        if (lhs<0) return -1;
                        // lhs is non-negative..
                        U_LHS_T u_lhs=static_cast<U_LHS_T>(lhs); // always valid for lhs>=0
                        return cmp_(u_lhs, rhs);
                    }

                    /// Invoked when an usigned bound type is compared with a signed type
                    template <typename LHS_T>
                    int apply(const LHS_T& lhs, const RHS_T& rhs, typename is_signed_unsigned<RHS_T,LHS_T>::yes =true) const {
                        return -apply(rhs,lhs);
                    }
                    
                    /// Invoked when an integral bound type is compared with a floating-point type
                    template <typename LHS_T>
                    int apply(const LHS_T& lhs, const RHS_T& rhs, typename is_intgl_to_fp<LHS_T,RHS_T>::yes =true) const {
                        return cmp_(lhs, rhs);
                    }

                    /// Invoked when a floating-point bound type is compared with an integral type
                    template <typename LHS_T>
                    int apply(const LHS_T& lhs, const RHS_T& rhs, typename is_intgl_to_fp<RHS_T,LHS_T>::yes =true) const {
                        return cmp_(lhs, rhs);
                    }

                    /// Invoked when a floating-point bound type is compared with another floating-point type
                    template <typename LHS_T>
                    int apply(const LHS_T& lhs, const RHS_T& rhs, typename is_fp_conv<RHS_T,LHS_T>::yes =true) const {
                        return cmp_(lhs, rhs);
                    }

                    /// Catch-all for all other conversions
                    template <typename LHS_T>
                    int apply(const LHS_T& lhs, const RHS_T& rhs, typename is_other_cmp<LHS_T,RHS_T>::yes =true) const {
                        std::string lhs_name=boost::typeindex::type_id<LHS_T>().pretty_name();
                        std::string rhs_name=boost::typeindex::type_id<RHS_T>().pretty_name();
                        throw exception::type_mismatch("","Attempt to compare incompatible types "+
                                                       lhs_name + "<=>" + rhs_name);
                    }
                };

                /// Visitor to compare 2 value of dict_value type
                class comparator2 : public boost::static_visitor<int> {
                    template <typename A, typename B>
                    static bool cmp_(const A& a, const B& b) { return (a==b)? 0 : (a<b)? -1:1; }
                    
                  public:
                    /// Called by apply_visitor for bound values of different types
                    template <typename LHS_T, typename RHS_T>
                    int operator()(const LHS_T& lhs, const RHS_T& rhs) const {
                        std::string lhs_name=boost::typeindex::type_id<LHS_T>().pretty_name();
                        std::string rhs_name=boost::typeindex::type_id<RHS_T>().pretty_name();
                        throw exception::type_mismatch("","Attempt to compare dictionary values containing "
                                                       "incompatible types "+
                                                       lhs_name + "<=>" + rhs_name);
                    }

                    /// Called by apply_visitor for bound values of the same type
                    template <typename LHS_RHS_T>
                    int operator()(const LHS_RHS_T& lhs, const LHS_RHS_T& rhs) const {
                        return cmp_(lhs,rhs);
                    }

                    /// Called by apply_visitor for bound values both having None type
                    int operator()(const None& lhs, const None& rhs) const {
                        return 1;
                    }
                        

                    // FIXME:TODO:
                    // Same types: compare directly
                    // Integral types: compare using signs (extract it to a separate namespace/class)
                    // FP types: compare directly
                    // Everything else: throw
                };

                /// Visitor to test for exact equality (name and value)
                class equals2 : public boost::static_visitor<bool> {
                  public:
                    /// Called when bound values have the same type
                    template <typename LHS_RHS_T>
                    bool operator()(const LHS_RHS_T& lhs, const LHS_RHS_T& rhs) const {
                        return lhs==rhs;
                    }

                    /// Called when bound types are different
                    template <typename LHS_T, typename RHS_T>
                    bool operator()(const LHS_T& lhs, const RHS_T& rhs) const{
                        return false;
                    }

                    /// Called when LHS is None
                    template <typename RHS_T>
                    bool operator()(const None&, const RHS_T&) const {
                        return false;
                    }
                    
                    /// Called when RHS is None
                    template <typename LHS_T>
                    bool operator()(const LHS_T&, const None&) const {
                        return false;
                    }
                    
                    /// Called when both are None
                    bool operator()(const None&, const None&) const {
                        return true;
                    }
                };

            } // ::visitor
            

        } //::detail
        
        class dict_value {
          public:

            typedef boost::make_variant_over<detail::dict_all_types>::type value_type;
            typedef detail::None None; ///< "Empty value" type

          private:
            std::string name_; ///< The option name (FIXME: make it "functionally const")
            value_type val_; ///< Value of the option

          public:

            /// Constructs the empty nameless value
            // FIXME: This is used only for MPI and must be changed
            dict_value(): name_("NO_NAME"), val_() {}
            
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

            /// Comparison
            /** a.compare(b) returns 0 if a==b, !=0 if a!=b.
                if well-ordered, returns -1 if a<b, +1 if a>b.
            */
            template <typename T>
            int compare(const T& rhs) const
            {
                if (this->empty()) throw exception::uninitialized_value(name_,"Attempt to compare uninitialized value");
                try {
                    return boost::apply_visitor(detail::visitor::comparator<T>(rhs), val_);
                } catch (exception::exception_base& exc) {
                    exc.set_name(name_);
                    throw;
                }
            }

            int compare(const dict_value& rhs) const
            {
                if (this->empty() || rhs.empty()) throw exception::uninitialized_value(name_+"<=>"+rhs.name_,"Attempt to compare uninitialized value");
                
                try {
                    return boost::apply_visitor(detail::visitor::comparator2(), val_, rhs.val_);
                } catch (exception::exception_base& exc) {
                    exc.set_name(name_+"<=>"+rhs.name_);
                    throw;
                } 
            }

            /// Returns true if the objects hold the same type and value, false otherwise
            bool equals(const dict_value& rhs) const
            {
                return boost::apply_visitor(detail::visitor::equals2(), val_, rhs.val_);
            }

#ifdef ALPS_HAVE_MPI
            void broadcast(const alps::mpi::communicator& comm, int root);
#endif
        };

        template <typename T>
        inline bool operator==(const T& lhs, const dict_value& rhs) { return rhs.compare(lhs)==0; }
        
        // template <typename T>
        // inline bool operator<(const T& lhs, const dict_value& rhs) {return false; }
        
        // template <typename T>
        // inline bool operator>(const T& lhs, const dict_value& rhs) {return false; }
        
        template <typename T>
        inline bool operator!=(const T& lhs, const dict_value& rhs) { return rhs.compare(lhs)!=0; }
        
        // template <typename T>
        // inline bool operator>=(const T& lhs, const dict_value& rhs) {return false; }
        
        // template <typename T>
        // inline bool operator<=(const T& lhs, const dict_value& rhs) {return false; }
        
        template <typename T>
        inline bool operator==(const dict_value& lhs, const T& rhs) { return lhs.compare(rhs)==0; }
        
        // template <typename T>
        // inline bool operator<(const dict_value& lhs, const T& rhs) {return false; }
        
        // template <typename T>
        // inline bool operator>(const dict_value& lhs, const T& rhs) {return false; }
        
        template <typename T>
        inline bool operator!=(const dict_value& lhs, const T& rhs) {return lhs.compare(rhs)!=0; }
        
        // template <typename T>
        // inline bool operator>=(const dict_value& lhs, const T& rhs) {return false; }
        
        // template <typename T>
        // inline bool operator<=(const dict_value& lhs, const T& rhs) {return false; }

        inline bool operator==(const dict_value& lhs, const dict_value& rhs) {return lhs.compare(rhs)==0; }
        
        // inline bool operator<(const dict_value& lhs, const dict_value& rhs) {return false; }
        
        // inline bool operator>(const dict_value& lhs, const dict_value& rhs) {return false; }
        
        inline bool operator!=(const dict_value& lhs, const dict_value& rhs) {return lhs.compare(rhs)!=0; }
        
        // inline bool operator>=(const dict_value& lhs, const dict_value& rhs) {return false; }
        
        // inline bool operator<=(const dict_value& lhs, const dict_value& rhs) {return false; }

        inline bool operator==(const dict_value& lhs, const char* rhs) {return lhs.compare(std::string(rhs))==0; }
        
        // inline bool operator<(const dict_value& lhs, const char* rhs) {return false; }
        
        // inline bool operator>(const dict_value& lhs, const char* rhs) {return false; }
        
        inline bool operator!=(const dict_value& lhs, const char* rhs) {return lhs.compare(std::string(rhs))!=0; }
        
        // inline bool operator>=(const dict_value& lhs, const char* rhs) {return false; }
        
        // inline bool operator<=(const dict_value& lhs, const char* rhs) {return false; }

        inline bool operator==(const char* lhs, const dict_value& rhs) {return rhs.compare(std::string(lhs))==0; }
        
        // inline bool operator<(const char* lhs, const dict_value& rhs) {return false; }
        
        // inline bool operator>(const char* lhs, const dict_value& rhs) {return false; }
        
        inline bool operator!=(const char* lhs, const dict_value& rhs) {return rhs.compare(std::string(lhs))!=0; }
        
        // inline bool operator>=(const char* lhs, const dict_value& rhs) {return false; }
        
        // inline bool operator<=(const char* lhs, const dict_value& rhs) {return false; }
        
    } // ::params_ns

#ifdef ALPS_HAVE_MPI
    namespace mpi {
        inline
        void broadcast(const alps::mpi::communicator &comm, alps::params_ns::dict_value& val, int root)
        {
            val.broadcast(comm, root);
        }
    }
#endif
    
} // ::alps

#endif /* ALPS_PARAMS_DICT_VALUE_HPP_a8ecbead92aa4a1995f43adfc6d0aae0 */
