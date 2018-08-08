/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <alps/config.hpp>
#include <alps/hdf5/vector.hpp>

#include <alps/accumulators/wrappers.hpp>
// #include <alps/accumulators/feature/weight_holder.hpp>
#include <alps/accumulators/wrapper_set.hpp>

#include <alps/hdf5/archive.hpp>

#include <memory>

#include <boost/variant/variant.hpp>
#include <boost/variant/get.hpp>

#ifdef ALPS_HAVE_MPI
    #include <alps/utilities/boost_mpi.hpp>
#endif

#include <typeinfo>
#include <type_traits>
#include <stdexcept>

namespace alps {
    namespace accumulators {

        namespace detail {
            typedef std::string printable_type; ///<Implementation-defined printable type for results/accumulators

            template<typename T> struct add_base_wrapper_pointer {
                typedef std::shared_ptr<base_wrapper<T> > type;
            };

            template<typename... Types> struct make_variant_type {
              typedef boost::variant<typename add_base_wrapper_pointer<Types>::type...> type;
            };

            typedef typename make_variant_type<ALPS_ACCUMULATOR_VALUE_TYPES>::type variant_type;

            template<typename T, typename A> struct is_valid_argument : std::conditional<
                    std::is_scalar<A>::value
                  , typename std::is_convertible<T, A>::type
                  , typename std::is_same<T, A>::type
            >::type {};

            /// Check if LHS and RHS result types are allowed in binary OP
            /** @param LHSWT: left-hand side wrapper type
                @param RHSWT: right-hand side wrapper type
            */
            template <typename LHSWT, typename RHSWT>
            struct is_compatible_op
                : std::is_same<typename alps::numeric::scalar<typename LHSWT::value_type>::type,
                                 typename RHSWT::value_type>
            { };

            /// Safety check: verify that a pointer is valid.
            /** @note Throws on a failed check */
            // FIXME: better initialize the pointer with something reasonable to begin with?
            template <typename T>
            void check_ptr(const std::shared_ptr<T>& ptr) {
                if (!ptr) throw std::runtime_error("Uninitialized accumulator accessed");
            }

        } // detail::

        // TODO: merge with accumulator_wrapper, at least make common base ...
        class result_wrapper {
            public:

                // default constructor
                result_wrapper();

                // constructor from raw result
                template<typename T> result_wrapper(T arg)
                    : m_variant(typename detail::add_base_wrapper_pointer<typename value_type<T>::type>::type(
                        new derived_result_wrapper<T>(arg))
                      )
                {}

                // constructor from base_wrapper
                template<typename T> result_wrapper(base_wrapper<T> * arg)
                    : m_variant(typename detail::add_base_wrapper_pointer<T>::type(arg))
                {}

                // copy constructor
                result_wrapper(result_wrapper const & rhs);

                // constructor from hdf5
                result_wrapper(hdf5::archive & ar);

                // operator=
                result_wrapper & operator=(std::shared_ptr<result_wrapper> const & rhs);

            private:
                // Visitors that need access to m_variant
                struct assign_visitor;

            // get
            private:
                template<typename T> struct get_visitor: public boost::static_visitor<> {
                    template<typename X> void operator()(X const & /*arg*/) {
                        throw std::runtime_error(std::string("Cannot cast observable") + typeid(X).name() + " to base type: " + typeid(T).name() + ALPS_STACKTRACE);
                    }
                    void operator()(typename detail::add_base_wrapper_pointer<T>::type const & arg) { value = arg; }
                    typename detail::add_base_wrapper_pointer<T>::type value;
                };
            public:
                template <typename T> base_wrapper<T> & get() {
                    get_visitor<T> visitor;
                    boost::apply_visitor(visitor, m_variant);
                    return *visitor.value;
                }

            // extract
            private:
                template<typename A> struct extract_visitor: public boost::static_visitor<A*> {
                    template<typename T> A* operator()(T const & arg) { return &arg->template extract<A>(); }
                };
            public:
                template <typename A> A & extract() {
                    extract_visitor<A> visitor;
                    return *boost::apply_visitor(visitor, m_variant);
                }
                template <typename A> A const & extract() const {
                    extract_visitor<A> visitor;
                    return *boost::apply_visitor(visitor, m_variant);
                }

            // cast-to-other-result visitor
            private:
                /// Visitor class to use in cast<AFROM,ATO>() member function.
                /** AFROM, ATO are named accumulator template names (e.g. `NoBinningAccumulator`)
                    to cast from and to --- see the description of `cast()` member function */
                template<template<typename> class AFROM,
                         template<typename> class ATO>
                struct cast_visitor: public boost::static_visitor<result_wrapper> {
                    template<typename T> result_wrapper operator()(T const & arg) {
                        typedef typename value_type<typename T::element_type>::type value_type;
                        typedef typename AFROM<value_type>::result_type raw_result_from_type;
                        typedef typename ATO<value_type>::result_type raw_result_to_type;

                        const raw_result_from_type& from=arg->template extract<raw_result_from_type>();
                        const raw_result_to_type& to=dynamic_cast<const raw_result_to_type&>(from);
                        return result_wrapper(to);
                    }
                };
            public:
                /// Cast to the `result_wrapper` containing another raw result type, or throw.
                /** @tparam AFROM named accumulator template names (e.g., `FullBinningAccumulator`) to convert from
                    @tparam ATO named accumulator template names (e.g., `NoBinningAccumulator`) to convert to

                    Example:
                    @code
                        result_set rset;
                        // ....
                        const result_wrapper& r1=rset["no_binning"];
                        result_wrapper r2=rset["full_binning"].cast<FullBinningAccumulator,NoBinningAccumulator>();
                        result_wrapper rsum=r1+r2;
                    @endcode

                */
                template <template<typename> class AFROM, template<typename> class ATO>
                result_wrapper cast() const {
                    cast_visitor<AFROM,ATO> visitor;
                    return boost::apply_visitor(visitor, m_variant);
                }

                // count
                boost::uint64_t count() const;

            // mean, error
            #define ALPS_ACCUMULATOR_PROPERTY_PROXY(PROPERTY, TYPE)                                                 \
                private:                                                                                            \
                    template<typename T> struct PROPERTY ## _visitor: public boost::static_visitor<T> {             \
                        template<typename X> T apply(typename std::enable_if<                                     \
                            detail::is_valid_argument<typename TYPE <X>::type, T>::value, X const &         \
                        >::type arg) const {                                                                        \
                            return arg. PROPERTY ();                                                                \
                        }                                                                                           \
                        template<typename X> T apply(typename std::enable_if<!                                    \
                            detail::is_valid_argument<typename TYPE <X>::type, T>::value, X const &         \
                        >::type /*arg*/) const {                                                                        \
                            throw std::logic_error(std::string("cannot convert: ")                                  \
                                + typeid(typename TYPE <X>::type).name() + " to "                                   \
                                + typeid(T).name() + ALPS_STACKTRACE);                                              \
                        }                                                                                           \
                        template<typename X> T operator()(X const & arg) const {                                    \
                            return apply<typename X::element_type>(*arg);                                           \
                        }                                                                                           \
                    };                                                                                              \
                public:                                                                                             \
                    template<typename T> typename TYPE <base_wrapper<T> >::type PROPERTY () const {                 \
                        PROPERTY ## _visitor<typename TYPE <base_wrapper<T> >::type> visitor;                       \
                        return boost::apply_visitor(visitor, m_variant);                                            \
                    }
            ALPS_ACCUMULATOR_PROPERTY_PROXY(mean, mean_type)
            ALPS_ACCUMULATOR_PROPERTY_PROXY(error, error_type)
            ALPS_ACCUMULATOR_PROPERTY_PROXY(autocorrelation, autocorrelation_type)
            #undef ALPS_ACCUMULATOR_PROPERTY_PROXY

                // save
                void save(hdf5::archive & ar) const;

                // load
                void load(hdf5::archive & ar);

                // print
                void print(std::ostream & os, bool terse=false) const;

            // transform(T F(T))
            private:
                template<typename T> struct transform_1_visitor: public boost::static_visitor<> {
                    transform_1_visitor(boost::function<T(T)> f) : op(f) {}
                    template<typename X> void apply(typename std::enable_if<
                        detail::is_valid_argument<T, typename value_type<X>::type>::value, X &
                    >::type arg) const {
                        arg.transform(op);
                    }
                    template<typename X> void apply(typename std::enable_if<!
                        detail::is_valid_argument<T, typename value_type<X>::type>::value, X &
                    >::type /*arg*/) const {
                        throw std::logic_error(std::string("cannot convert: ") + typeid(T).name() + " to " + typeid(typename value_type<X>::type).name() + ALPS_STACKTRACE);
                    }
                    template<typename X> void operator()(X & arg) const {
                        apply<typename X::element_type>(*arg);
                    }
                    boost::function<T(T)> op;
                };
            public:
                template<typename T> result_wrapper transform(boost::function<T(T)> op) const {
                    result_wrapper clone(*this);
                    boost::apply_visitor(transform_1_visitor<T>(op), clone.m_variant);
                    return clone;
                }
                template<typename T> result_wrapper transform(T(*op)(T)) const {
                    return transform(boost::function<T(T)>(op));
                }

            public:

                // unary plus
                result_wrapper operator+ () const;

                // unary minus
                result_wrapper operator- () const;

                // operators
                // Naming conventions:
                //   Operation is `lhs_var AUGOP rhs_var`, where AUGOP is `+=` , `-=` etc.
                //   lhsvar contains a variant over LHSPT types
                //   rhsvar contains a variant over RHSPT types
                //   LHSPT: lhs (pointer) type, which is shared_ptr<LHSWT>
                //   LHSWT: lhs (base_wrapper<...>) type
                //   RHSPT: rhs (pointer) type, which is shared_ptr<RHSWT>
                //   RHSWT: rhs (base_wrapper<...>) type
                #define ALPS_ACCUMULATOR_OPERATOR_PROXY(OPNAME, AUGOPNAME, AUGOP, FUN)                              \
                    private:                                                                                        \
                        struct FUN ## _self_visitor;                                                                \
                    public:                                                                                         \
                    /** @brief Do AUGOP with another result   */                                                    \
                    result_wrapper & AUGOPNAME (result_wrapper const & rhs);                                        \
                    /** @brief Do AUGOP with a constant value */                                                    \
                    result_wrapper & AUGOPNAME (long double arg);                                                   \
                    result_wrapper OPNAME (result_wrapper const & arg) const;                                       \
                    /** @brief Visitor to do OP with RHS constant value */                                          \
                    result_wrapper OPNAME (long double arg) const;
                ALPS_ACCUMULATOR_OPERATOR_PROXY(operator+, operator+=, +=, add)
                ALPS_ACCUMULATOR_OPERATOR_PROXY(operator-, operator-=, -=, sub)
                ALPS_ACCUMULATOR_OPERATOR_PROXY(operator*, operator*=, *=, mul)
                ALPS_ACCUMULATOR_OPERATOR_PROXY(operator/, operator/=, /=, div)
                #undef ALPS_ACCUMULATOR_OPERATOR_PROXY

                // inverse
                result_wrapper inverse() const;

                result_wrapper sin () const;
                result_wrapper cos () const;
                result_wrapper tan () const;
                result_wrapper sinh () const;
                result_wrapper cosh () const;
                result_wrapper tanh () const;
                result_wrapper asin () const;
                result_wrapper acos () const;
                result_wrapper atan () const;
                result_wrapper abs () const;
                result_wrapper sqrt () const;
                result_wrapper log () const;
                result_wrapper sq () const;
                result_wrapper cb () const;
                result_wrapper cbrt () const;

            private:

                detail::variant_type m_variant;
        };

        inline result_wrapper operator+(long double arg1, result_wrapper const & arg2) {
            return arg2 + arg1;
        }
        inline result_wrapper operator-(long double arg1, result_wrapper const & arg2) {
            return -arg2 + arg1;
        }
        inline result_wrapper operator*(long double arg1, result_wrapper const & arg2) {
            return arg2 * arg1;
        }
        inline result_wrapper operator/(long double arg1, result_wrapper const & arg2) {
            return arg2.inverse() * arg1;
        }

        std::ostream & operator<<(std::ostream & os, const result_wrapper & arg);

        /// Return an "ostream-able" object to print result in a terse format
        detail::printable_type short_print(const result_wrapper& arg);

        /// Return an "ostream-able" object to print result in a verbose format
        detail::printable_type full_print(const result_wrapper& arg);

        /// Return the "raw result" of type A held in the result_wrapper m, or throw.
        template <typename A> A & extract(result_wrapper & m) {
            return m.extract<A>();
        }

        /// Return the "raw result" of type A held in the result_wrapper m, or throw.
        template <typename A> const A & extract(const result_wrapper & m) {
            return m.extract<A>();
        }

        /// Cast to the result_wrapper containing another raw result type, or throw.
        /** @tparam AFROM raw result type (e.g., `NoBinningAccumulator<double>``::``result_type`) to cast from
            @tparam ATO raw result type (e.g., `FullBinningAccumulator<double>``::``result_type`) to cast to

            Example:
            @code
                result_set rset;
                // ....
                const result_wrapper& r1=rset["no_binning"];
                result_wrapper r2=cast_raw<FullBinningAccumulator<double>::result_type,
                                           NoBinningAccumulator<double>::result_type>(rset["full_binning"]);
                result_wrapper rsum=r1+r2;
            @endcode
        */
        template <typename AFROM, typename ATO>
        result_wrapper cast_raw(const result_wrapper& res) {
            const AFROM& raw_res_from=extract<AFROM>(res);
            const ATO& raw_res_to=dynamic_cast<const ATO&>(raw_res_from);
            return result_wrapper(raw_res_to);
        }

        /// Cast to the result_wrapper containing another raw result type, or throw.
        /** @tparam AFROM named accumulator template name (e.g., `FullBinningAccumulator`) to convert from
            @tparam ATO named accumulator template name (e.g., `NoBinningAccumulator`) to convert to

            Example:
            @code
                result_set rset;
                // ....
                const result_wrapper& r1=rset["no_binning"];
                result_wrapper r2=cast<FullBinningAccumulator,NoBinningAccumulator>(rset["full_binning"]);
                result_wrapper rsum=r1+r2;
            @endcode

        */
        template <template<typename> class AFROM,
                  template<typename> class ATO>
        result_wrapper cast(const result_wrapper& res) {
            return res.cast<AFROM,ATO>();
        }

        result_wrapper sin (result_wrapper const & arg);
        result_wrapper cos (result_wrapper const & arg);
        result_wrapper tan (result_wrapper const & arg);
        result_wrapper sinh (result_wrapper const & arg);
        result_wrapper cosh (result_wrapper const & arg);
        result_wrapper tanh (result_wrapper const & arg);
        result_wrapper asin (result_wrapper const & arg);
        result_wrapper acos (result_wrapper const & arg);
        result_wrapper atan (result_wrapper const & arg);
        result_wrapper abs (result_wrapper const & arg);
        result_wrapper sqrt (result_wrapper const & arg);
        result_wrapper log (result_wrapper const & arg);
        result_wrapper sq (result_wrapper const & arg);
        result_wrapper cb (result_wrapper const & arg);
        result_wrapper cbrt (result_wrapper const & arg);

        class accumulator_wrapper {
            private:

                /// Check if the data is valid (not a 0-sized vector): Generic.
                template <typename T>
                static void check_nonempty_vector(const T&) {}

                /// Check if the data is valid (not a 0-sized vector): vector specialization.
                /** @note Throws on a failed check */
                template <typename T>
                static void check_nonempty_vector(const std::vector<T>& vec) {
                    if (vec.empty()) throw std::runtime_error("Zero-sized vector observables are not allowed");
                }

            public:
            /// default constructor
            accumulator_wrapper();

            /// constructor from raw accumulator
            template<typename T> accumulator_wrapper(T arg)
                : m_variant(typename detail::add_base_wrapper_pointer<typename value_type<T>::type>::type(
                    new derived_accumulator_wrapper<T>(arg))
                  )
            {}

            /// copy constructor
            /// @note The wrapped accumulator object is NOT copied!
            accumulator_wrapper(accumulator_wrapper const & rhs);

            // constructor from hdf5
            accumulator_wrapper(hdf5::archive & ar);

            // operator(T)
            private:
                template<typename T> struct call_1_visitor: public boost::static_visitor<> {
                    call_1_visitor(T const & v) : value(v) {}
                    template<typename X> void apply(typename std::enable_if<
                        detail::is_valid_argument<T, typename value_type<X>::type>::value, X &
                    >::type arg) const {
                        arg(value);
                    }
                    template<typename X> void apply(typename std::enable_if<!
                        detail::is_valid_argument<T, typename value_type<X>::type>::value, X &
                    >::type /*arg*/) const {
                        throw std::logic_error(std::string("cannot convert: ") + typeid(T).name() + " to " + typeid(typename value_type<X>::type).name() + ALPS_STACKTRACE);
                    }
                    template<typename X> void operator()(X & arg) const {
                        check_ptr(arg);
                        apply<typename X::element_type>(*arg);
                    }
                    T const & value;
                };
            public:
                template<typename T> void operator()(T const & value) {
                    check_nonempty_vector(value);
                    boost::apply_visitor(call_1_visitor<T>(value), m_variant);
                }
                template<typename T> accumulator_wrapper & operator<<(T const & value) {
                    (*this)(value);
                    return (*this);
                }

                /// Merge another accumulator into this one. @param rhs_acc  accumulator to merge.
                void merge(const accumulator_wrapper& rhs_acc);

                /// Returns a copy with the wrapped accumulator cloned
                accumulator_wrapper clone() const;

                /// Returns a pointer to a new-allocated copy with the wrapped accumulator cloned
                accumulator_wrapper* new_clone() const;

                // operator=
                accumulator_wrapper & operator=(std::shared_ptr<accumulator_wrapper> const & rhs);

                // count
                boost::uint64_t count() const;

            private:
                // Visitors that need access to m_variant
                struct merge_visitor;
                struct copy_visitor;
                struct assign_visitor;

            // get
            private:
                template<typename T> struct get_visitor: public boost::static_visitor<> {
                    template<typename X> void operator()(X const & /*arg*/) {
                        throw std::runtime_error(std::string("Cannot cast observable") + typeid(X).name() + " to base type: " + typeid(T).name() + ALPS_STACKTRACE);
                    }
                    void operator()(typename detail::add_base_wrapper_pointer<T>::type const & arg) { value = arg; }
                    typename detail::add_base_wrapper_pointer<T>::type value;
                };
            public:
                template <typename T> base_wrapper<T> & get() {
                    get_visitor<T> visitor;
                    boost::apply_visitor(visitor, m_variant);
                    check_ptr(visitor.value);
                    return *visitor.value;
                }

            // extract
            private:
                template<typename A> struct extract_visitor: public boost::static_visitor<A*> {
                    template<typename T> A* operator()(T const & arg) { check_ptr(arg);  return &arg->template extract<A>(); }
                };
            public:
                template <typename A> A & extract() {
                    extract_visitor<A> visitor;
                    return *boost::apply_visitor(visitor, m_variant);
                }

            // mean, error
            #define ALPS_ACCUMULATOR_PROPERTY_PROXY(PROPERTY, TYPE)                                                 \
                private:                                                                                            \
                    template<typename T> struct PROPERTY ## _visitor: public boost::static_visitor<T> {             \
                        template<typename X> T apply(typename std::enable_if<                                     \
                            detail::is_valid_argument<typename TYPE <X>::type, T>::value, X const &         \
                        >::type arg) const {                                                                        \
                            return arg. PROPERTY ();                                                                \
                        }                                                                                           \
                        template<typename X> T apply(typename std::enable_if<!                                    \
                            detail::is_valid_argument<typename TYPE <X>::type, T>::value, X const &         \
                        >::type /*arg*/) const {                                                                        \
                            throw std::logic_error(std::string("cannot convert: ")                                  \
                                + typeid(typename TYPE <X>::type).name() + " to "                                   \
                                + typeid(T).name() + ALPS_STACKTRACE);                                              \
                        }                                                                                           \
                        template<typename X> T operator()(X const & arg) const {                                    \
                            check_ptr(arg);                                                                         \
                            return apply<typename X::element_type>(*arg);                                           \
                        }                                                                                           \
                    };                                                                                              \
                public:                                                                                             \
                    template<typename T> typename TYPE <base_wrapper<T> >::type PROPERTY () const {                 \
                        PROPERTY ## _visitor<typename TYPE <base_wrapper<T> >::type> visitor;                       \
                        return boost::apply_visitor(visitor, m_variant);                                            \
                    }
            ALPS_ACCUMULATOR_PROPERTY_PROXY(mean, mean_type)
            ALPS_ACCUMULATOR_PROPERTY_PROXY(error, error_type)
            #undef ALPS_ACCUMULATOR_PROPERTY_PROXY

            // save
            void save(hdf5::archive & ar) const;
            // load
            void load(hdf5::archive & ar);

            // reset
            void reset() const;

            // result
            std::shared_ptr<result_wrapper> result() const;

            // print
            void print(std::ostream & os, bool terse=false) const;

#ifdef ALPS_HAVE_MPI
            void collective_merge(alps::mpi::communicator const & comm, int root);
#endif

            private:

                detail::variant_type m_variant;
        };

        std::ostream & operator<<(std::ostream & os, const accumulator_wrapper & arg);

        /// Return an "ostream-able" object to print accumulator in a terse format
        detail::printable_type short_print(const accumulator_wrapper& arg);

        /// Return an "ostream-able" object to print accumulator in a verbose format
        detail::printable_type full_print(const accumulator_wrapper& arg);

        template <typename A> A & extract(accumulator_wrapper & m) {
            return m.extract<A>();
        }

        void reset(accumulator_wrapper & arg);

        typedef impl::wrapper_set<accumulator_wrapper> accumulator_set;
        typedef impl::wrapper_set<result_wrapper> result_set;

    }
}
