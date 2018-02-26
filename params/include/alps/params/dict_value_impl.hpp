/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/**
   @file dict_value_impl.hpp
   Contains header-part implementation of dict_value.
   NOT TO BE INCLUDED DIRECTLY!
*/

#include <alps/utilities/short_print.hpp> // for streaming of vectors
#include <boost/type_index.hpp> // for pretty-printing user types in error messages
#include <boost/integer_traits.hpp>

#include <boost/static_assert.hpp> // FIXME:C++11 replace by std feature

#include <type_traits>

// namespace std {
//     // Printing of a vector
//     // FIXME: pollutes std:: namespace and is a bad practice, what if user has one's own vector printer???
//     template <typename T>
//     inline std::ostream& operator<<(std::ostream& strm, const std::vector<T>& vec)
//     {
//         typedef std::vector<T> vtype;
//         typedef typename vtype::const_iterator itype;

//         strm << "[";
//         itype it=vec.begin();
//         const itype end=vec.end();

//         if (end!=it) {
//             strm << *it;
//             for (++it; end!=it; ++it) {
//                 strm << ", " << *it;
//             }
//         }
//         strm << "]";

//         return strm;
//     }
// }

namespace alps {
    namespace params_ns {


        namespace detail {
            template <bool V> struct yes_no {};
            template <> struct yes_no<true> : public std::true_type { typedef bool yes; };
            template <> struct yes_no<false> : public std::false_type { typedef bool no; };

            template <typename T>
            struct is_bool : public yes_no<std::is_same<T,bool>::value> {};

            // bool type is NOT integral for the purposes of this code
            template <typename T>
            struct is_integral : public yes_no<std::is_integral<T>::value && !is_bool<T>::value> {};

            // type is allowed: bool, integral, char* or other supported
            // FIXME: should use `is_convertible`. Postponed till simplification refactoring
            template <typename T>
            struct is_allowed : public yes_no<
                is_bool<T>::value
                || std::is_same<char*, T>::value
                || std::is_same<const char*, T>::value
                || is_integral<T>::value
                || is_supported<T>::value> {};

            // signed value: integral and signed
            template <typename T>
            struct is_signed : public yes_no<std::is_signed<T>::value && is_integral<T>::value> {};

            // unsigned value: integral and unsigned
            template <typename T>
            struct is_unsigned : public yes_no<std::is_unsigned<T>::value && is_integral<T>::value> {};

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
                : public yes_no<is_integral<FROM>::value && std::is_floating_point<TO>::value>
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
                : public yes_no< std::is_floating_point<A>::value &&
                                 std::is_floating_point<B>::value &&
                                 !std::is_same<A,B>::value>
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


            // catch-all definition of pretty-printing for unknown types
            // FIXME: better design may be to define a separate class for it,
            //        in order to catch unintentional use of type_info<T> on unsupported types?
            template <typename T>
            struct type_info {
                static std::string pretty_name() {
                    return boost::typeindex::type_id<T>().pretty_name();
                }
            };

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
                        typedef typename std::make_unsigned<LHS_T>::type U_LHS_T;
                        const U_LHS_T max_num=boost::integer_traits<LHS_T>::const_max; // always possible
                        // compare 2 unsigned
                        if (val>max_num)
                            throw exception::value_mismatch("", "Integer overflow detected: unsigned integer too large");
                        return val;
                    }

                    /// Extracting signed integral type to an integral type
                    template <typename RHS_T>
                    LHS_T apply(const RHS_T& val, typename is_sig_to_intgl<RHS_T,LHS_T>::yes =true) const {
                        typedef typename std::make_signed<LHS_T>::type S_LHS_T;
                        typedef typename std::make_unsigned<LHS_T>::type U_LHS_T;
                        typedef typename std::make_unsigned<RHS_T>::type U_RHS_T;

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
                        std::string rhs_name=detail::type_info<RHS_T>::pretty_name();
                        std::string lhs_name=detail::type_info<LHS_T>::pretty_name();
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
                        std::string rhs_name=detail::type_info<RHS_T>::pretty_name();
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
                        std::string lhs_name=detail::type_info<LHS_T>::pretty_name();
                        std::string rhs_name=detail::type_info<RHS_T>::pretty_name();
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
                        typedef typename std::make_unsigned<LHS_T>::type U_LHS_T;
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
                        std::string lhs_name=detail::type_info<LHS_T>::pretty_name();
                        std::string rhs_name=detail::type_info<RHS_T>::pretty_name();
                        throw exception::type_mismatch("","Attempt to compare incompatible types "+
                                                       lhs_name + "<=>" + rhs_name);
                    }
                };

            } // ::visitor
        } //::detail

        template <typename F>
        typename F::result_type dict_value::apply_visitor(F& visitor) const {
            return boost::apply_visitor(visitor, val_);
        }

        template <typename F>
        typename F::result_type dict_value::apply_visitor(const F& visitor) const {
            return boost::apply_visitor(visitor, val_);
        }

        inline bool dict_value::empty() const {
            return val_.which()==0; // NOTE: relies on `None` being the first type
        }

        template <typename X>
        inline bool dict_value::isType() const {
            return apply_visitor(detail::visitor::check_type<X>());
        }

        template <typename T>
        inline const T& dict_value::operator=(const T& rhs) {
            val_=rhs;
            return rhs;
        }

        inline const char* dict_value::operator=(const char* rhs) {
            val_=std::string(rhs);
            return rhs;
        }

        template <typename T>
        inline T dict_value::as() const {
            BOOST_STATIC_ASSERT_MSG(detail::is_allowed<T>::value, "The type is not supported by dictionary");
            if (this->empty()) throw exception::uninitialized_value(name_,"Attempt to read uninitialized value");
            try {
                return apply_visitor(detail::visitor::getter<T>());
            } catch (exception::exception_base& exc) {
                exc.set_name(name_);
                throw;
            }
        }

        // template <typename T,
        //           typename std::enable_if<detail::is_allowed<T>::value, int> =0>
        // inline dict_value::operator T() const {
        //     // BOOST_STATIC_ASSERT_MSG(detail::is_allowed<T>::value, "The type is not supported by the dictionary");
        //     return as<T>();
        // }

        inline void dict_value::clear() { val_=None(); }

        template <typename T>
        inline int dict_value::compare(const T& rhs) const
        {
            BOOST_STATIC_ASSERT_MSG(detail::is_allowed<T>::value, "The type is not supported by dictionary");
            if (this->empty()) throw exception::uninitialized_value(name_,"Attempt to compare uninitialized value");
            try {
                return apply_visitor(detail::visitor::comparator<T>(rhs));
            } catch(exception::exception_base& exc) {
                exc.set_name(name_);
                throw;
            }
        }

        /// Const-access visitor to the bound value
        /** @param visitor functor should be callable as `R result=visitor(bound_value_const_ref)`
            @param dv  the dictionary value to access

            The functor type `F` must define typename `F::result_type`.
        */
        template <typename F>
        inline typename F::result_type apply_visitor(F& visitor, const dict_value& dv)
        {
            return dv.apply_visitor(visitor);
        }

        /// Const-access visitor to the bound value
        /** @param visitor functor should be callable as `R result=visitor(bound_value_const_ref)`
            @param dv  the dictionary value to access

            The functor type `F` must define typename `F::result_type`.
        */
        template <typename F>
        inline typename F::result_type apply_visitor(const F& visitor, const dict_value& dv)
        {
            return dv.apply_visitor(visitor);
        }

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

    } // params_ns::


#ifdef ALPS_HAVE_MPI
    namespace mpi {
        inline void broadcast(const alps::mpi::communicator &comm, alps::params_ns::dict_value& val, int root)
        {
            val.broadcast(comm, root);
        }
    }
#endif

} // alps::
