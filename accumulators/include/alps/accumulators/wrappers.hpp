/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <alps/config.hpp>

#include <alps/accumulators/feature/mean.hpp>
#include <alps/accumulators/feature/error.hpp>
#include <alps/accumulators/feature/count.hpp>
#include <alps/accumulators/feature/max_num_binning.hpp>
#include <alps/accumulators/feature/binning_analysis.hpp>

#include <alps/hdf5/archive.hpp>

#include <boost/variant/variant.hpp>
#include <boost/variant/apply_visitor.hpp>

#include <typeinfo>
#include <type_traits>
#include <stdexcept>

namespace alps {
    namespace accumulators {

        template<typename A> class derived_wrapper;

        namespace detail {
            template<typename T> struct value_wrapper {
                typedef T value_type;
            };
        }

        template<typename T> class base_wrapper : public
            // impl::BaseWrapper<T, weight_tag,
            impl::BaseWrapper<T, max_num_binning_tag,
            impl::BaseWrapper<T, binning_analysis_tag,
            impl::BaseWrapper<T, error_tag,
            impl::BaseWrapper<T, mean_tag,
            impl::BaseWrapper<T, count_tag,
            detail::value_wrapper<T>
        // >
        > > > > > {

            public:
                typedef typename detail::value_wrapper<T>::value_type value_type;

                virtual ~base_wrapper() {}

                virtual void operator()(value_type const & value) = 0;
                // virtual void operator()(value_type const & value, detail::weight_variant_type const & weight) = 0;

                virtual void save(hdf5::archive & ar) const = 0;
                virtual void load(hdf5::archive & ar) = 0;

                virtual void print(std::ostream & os, bool terse) const = 0;
                virtual void reset() = 0;

                /// merge accumulators (defined in the derived classes)
                virtual void merge(const base_wrapper<T>&) = 0;
#ifdef ALPS_HAVE_MPI
                virtual void collective_merge(alps::mpi::communicator const & comm, int root) = 0;
#endif

                virtual base_wrapper * clone() const = 0;
                virtual base_wrapper * result() const = 0;

                template<typename A> A & extract() {
                    return dynamic_cast<derived_wrapper<A> &>(*this).extract();
                }
                template<typename A> A const & extract() const {
                    return dynamic_cast<derived_wrapper<A> const &>(*this).extract();
                }

            private:
                /* This machinery is to have `wrapped_value_type=base_wrapper< scalar_of_T > const &`
                   if T is a non-scalar type, and `wrapped_value_type=void*` if T is a scalar type.
                */
                template <typename X> struct wrap_value_type:
                    public std::conditional<
                        alps::is_scalar<X>::value,
                        void*,
                        base_wrapper<typename alps::numeric::scalar<X>::type> const & > {};

            protected:
                /// Either wrapped scalar<T>::type or unwrapped void*, depending on T
                typedef typename wrap_value_type<T>::type wrapped_scalar_value_type;

            public:

                virtual void operator+=(base_wrapper const &) = 0;
                virtual void operator-=(base_wrapper const &) = 0;
                virtual void operator*=(base_wrapper const &) = 0;
                virtual void operator/=(base_wrapper const &) = 0;

                virtual void operator+=(wrapped_scalar_value_type) = 0;
                virtual void operator-=(wrapped_scalar_value_type) = 0;
                virtual void operator*=(wrapped_scalar_value_type) = 0;
                virtual void operator/=(wrapped_scalar_value_type) = 0;


                // These virtual functions accept `long double`: it's the "widest" RHS scalar type.
                virtual void operator+=(long double) = 0;
                virtual void operator-=(long double) = 0;
                virtual void operator*=(long double) = 0;
                virtual void operator/=(long double) = 0;

                virtual void negate() = 0;
                virtual void inverse() = 0;

                virtual void sin() = 0;
                virtual void cos() = 0;
                virtual void tan() = 0;
                virtual void sinh() = 0;
                virtual void cosh() = 0;
                virtual void tanh() = 0;
                virtual void asin() = 0;
                virtual void acos() = 0;
                virtual void atan() = 0;
                virtual void abs() = 0;
                virtual void sqrt() = 0;
                virtual void log() = 0;
                virtual void sq() = 0;
                virtual void cb() = 0;
                virtual void cbrt() = 0;
        };

        namespace detail {
            template<typename A> class foundation_wrapper : public base_wrapper<typename value_type<A>::type> {
                public:
                    foundation_wrapper(A const & arg): m_data(arg) {}

                protected:
                    A m_data;
            };
        }

        template<typename A> class derived_wrapper : public
            // impl::DerivedWrapper<A, weight_tag,
            impl::DerivedWrapper<A, max_num_binning_tag,
            impl::DerivedWrapper<A, binning_analysis_tag,
            impl::DerivedWrapper<A, error_tag,
            impl::DerivedWrapper<A, mean_tag,
            impl::DerivedWrapper<A, count_tag,
        detail::foundation_wrapper<A>
        // >
        > > > > > {

            typedef typename detail::value_wrapper<typename value_type<A>::type>::value_type value_type;

            public:
                derived_wrapper()
                    :
                        // impl::DerivedWrapper<A, weight_tag,
                        impl::DerivedWrapper<A, max_num_binning_tag,
                        impl::DerivedWrapper<A, binning_analysis_tag,
                        impl::DerivedWrapper<A, error_tag,
                        impl::DerivedWrapper<A, mean_tag,
                        impl::DerivedWrapper<A, count_tag,
                    detail::foundation_wrapper<A>
                    // >
                    > > > > >()
                {}

                derived_wrapper(A const & arg)
                    :
                        // impl::DerivedWrapper<A, weight_tag,
                        impl::DerivedWrapper<A, max_num_binning_tag,
                        impl::DerivedWrapper<A, binning_analysis_tag,
                        impl::DerivedWrapper<A, error_tag,
                        impl::DerivedWrapper<A, mean_tag,
                        impl::DerivedWrapper<A, count_tag,
                    detail::foundation_wrapper<A>
                    // >
                    > > > > >(arg)
                {}

                A & extract() {
                    return this->m_data;
                }
                A const & extract() const {
                    return this->m_data;
                }

                void operator()(value_type const & value) {
                    this->m_data(value);
                }

            public:
                void save(hdf5::archive & ar) const {
                    ar[""] = this->m_data;
                   }
                void load(hdf5::archive & ar) {
                    ar[""] >> this->m_data;
                }

                void print(std::ostream & os, bool terse) const {
                    this->m_data.print(os, terse);
                }

                void reset() {
                    this->m_data.reset();
                }

                /// Merge the given accumulator into this accumulator @param rhs Accumulator to merge
                void merge(const base_wrapper<value_type>& rhs)
                {
                  this->m_data.merge(dynamic_cast<const derived_wrapper<A>&>(rhs).m_data);
                }

#ifdef ALPS_HAVE_MPI
                void collective_merge(
                      alps::mpi::communicator const & comm
                    , int root = 0
                ) {
                    this->m_data.collective_merge(comm, root);
                }

                void collective_merge(
                      alps::mpi::communicator const & comm
                    , int root = 0
                ) const {
                    this->m_data.collective_merge(comm, root);
                }
#endif
        };

        template<typename A> class derived_result_wrapper : public derived_wrapper<A> {
            private:
                typedef typename base_wrapper<typename value_type<A>::type>::wrapped_scalar_value_type wrapped_scalar_value_type;
            public:
                derived_result_wrapper(): derived_wrapper<A>() {}

                derived_result_wrapper(A const & arg): derived_wrapper<A>(arg) {}

                base_wrapper<typename value_type<A>::type> * clone() const {
                    return new derived_result_wrapper<A>(this->m_data);
                }
                base_wrapper<typename value_type<A>::type> * result() const {
                    throw std::runtime_error(std::string("A result(") + typeid(A).name() + ") cannot be converted to a result" + ALPS_STACKTRACE);
                    return NULL;
                }

                #define OPERATOR_PROXY(AUGOPNAME, AUGOP, AUGOPFN)                                            \
                    void AUGOPNAME(base_wrapper<typename value_type<A>::type> const & arg) {        \
                        this->m_data AUGOP arg.template extract<A>();                               \
                    }                                                   \
                    /** @brief A plug that has to be generated, but is never called */                                                    \
                    void do_##AUGOPFN(void*) {                           \
                        throw std::logic_error("This virtual method plug should never be called"); \
                    }                                                               \
                    template <typename W>                               \
                    void do_##AUGOPFN(W& arg) {                           \
                        this->m_data AUGOP arg.template extract<typename A::scalar_result_type>();          \
                    }                                                               \
                    void AUGOPNAME(wrapped_scalar_value_type arg) {        \
                        do_##AUGOPFN(arg);                               \
                    }                                                                               \
                    /* takes `long double`: it's the widest scalar numeric type */                  \
                    void AUGOPNAME(long double arg) {                                               \
                        this->m_data AUGOP arg;                                                     \
                    }
                OPERATOR_PROXY(operator+=, +=, add)
                OPERATOR_PROXY(operator-=, -=, sub)
                OPERATOR_PROXY(operator*=, *=, mul)
                OPERATOR_PROXY(operator/=, /=, div)
                #undef OPERATOR_PROXY

                void negate() {
                    this->m_data.negate();
                }
                void inverse() {
                    this->m_data.inverse();
                }

                #define FUNCTION_PROXY(FUN)                                                         \
                    void FUN () {                                                                   \
                        this->m_data. FUN ();                                                       \
                    }

                FUNCTION_PROXY(sin)
                FUNCTION_PROXY(cos)
                FUNCTION_PROXY(tan)
                FUNCTION_PROXY(sinh)
                FUNCTION_PROXY(cosh)
                FUNCTION_PROXY(tanh)
                FUNCTION_PROXY(asin)
                FUNCTION_PROXY(acos)
                FUNCTION_PROXY(atan)
                FUNCTION_PROXY(abs)
                FUNCTION_PROXY(sqrt)
                FUNCTION_PROXY(log)
                FUNCTION_PROXY(sq)
                FUNCTION_PROXY(cb)
                FUNCTION_PROXY(cbrt)

                #undef FUNCTION_PROXY
        };

        template<typename T, typename A> derived_result_wrapper<A> operator/(T arg, derived_result_wrapper<A> res) {
            return arg * res.inverse();
        }

        template<typename A> class derived_accumulator_wrapper : public derived_wrapper<A> {
            private:
                typedef typename base_wrapper<typename value_type<A>::type>::wrapped_scalar_value_type wrapped_scalar_value_type;
            public:
                derived_accumulator_wrapper(): derived_wrapper<A>() {}

                derived_accumulator_wrapper(A const & arg): derived_wrapper<A>(arg) {}

                base_wrapper<typename value_type<A>::type> * clone() const {
                    return new derived_accumulator_wrapper<A>(this->m_data);
                }
                base_wrapper<typename value_type<A>::type> * result() const {
                    return result_impl<A>();
                }

                void operator+=(base_wrapper<typename value_type<A>::type> const &) {
                    throw std::runtime_error("The Operator += is not implemented for accumulators, only for results" + ALPS_STACKTRACE);
                }
                void operator-=(base_wrapper<typename value_type<A>::type> const &) {
                    throw std::runtime_error("The Operator -= is not implemented for accumulators, only for results" + ALPS_STACKTRACE);
                }
                void operator*=(base_wrapper<typename value_type<A>::type> const &) {
                    throw std::runtime_error("The Operator *= is not implemented for accumulators, only for results" + ALPS_STACKTRACE);
                }
                void operator/=(base_wrapper<typename value_type<A>::type> const &) {
                    throw std::runtime_error("The Operator /= is not implemented for accumulators, only for results" + ALPS_STACKTRACE);
                }

                void operator+=(long double) {
                    throw std::runtime_error("The operator += is not implemented for accumulators, only for results" + ALPS_STACKTRACE);
                }
                void operator-=(long double) {
                    throw std::runtime_error("The operator -= is not implemented for accumulators, only for results" + ALPS_STACKTRACE);
                }
                void operator*=(long double) {
                    throw std::runtime_error("The operator *= is not implemented for accumulators, only for results" + ALPS_STACKTRACE);
                }
                void operator/=(long double) {
                    throw std::runtime_error("The operator /= is not implemented for accumulators, only for results" + ALPS_STACKTRACE);
                }

                void operator+=(wrapped_scalar_value_type /*arg*/) {
                    throw std::runtime_error("The Operator += is not implemented for accumulators, only for results" + ALPS_STACKTRACE);
                }
                void operator-=(wrapped_scalar_value_type /*arg*/) {
                    throw std::runtime_error("The Operator -= is not implemented for accumulators, only for results" + ALPS_STACKTRACE);
                }
                void operator*=(wrapped_scalar_value_type /*arg*/) {
                    throw std::runtime_error("The Operator *= is not implemented for accumulators, only for results" + ALPS_STACKTRACE);
                }
                void operator/=(wrapped_scalar_value_type /*arg*/) {
                    throw std::runtime_error("The Operator /= is not implemented for accumulators, only for results" + ALPS_STACKTRACE);
                }

                void negate() {
                    throw std::runtime_error("The function negate is not implemented for accumulators, only for results" + ALPS_STACKTRACE);
                }
                void inverse() {
                    throw std::runtime_error("The function inverse is not implemented for accumulators, only for results" + ALPS_STACKTRACE);
                }

                #define FUNCTION_PROXY(FUN)                                                                                                           \
                    void FUN () {                                                                                                                     \
                        throw std::runtime_error("The Function " #FUN " is not implemented for accumulators, only for results" + ALPS_STACKTRACE);    \
                    }

                FUNCTION_PROXY(sin)
                FUNCTION_PROXY(cos)
                FUNCTION_PROXY(tan)
                FUNCTION_PROXY(sinh)
                FUNCTION_PROXY(cosh)
                FUNCTION_PROXY(tanh)
                FUNCTION_PROXY(asin)
                FUNCTION_PROXY(acos)
                FUNCTION_PROXY(atan)
                FUNCTION_PROXY(abs)
                FUNCTION_PROXY(sqrt)
                FUNCTION_PROXY(log)
                FUNCTION_PROXY(sq)
                FUNCTION_PROXY(cb)
                FUNCTION_PROXY(cbrt)

                #undef FUNCTION_PROXY

            private:

                template<typename T> typename std::enable_if<has_result_type<T>::value, base_wrapper<typename value_type<A>::type> *>::type result_impl() const {
                    return new derived_result_wrapper<typename A::result_type>(this->m_data);
                }
                template<typename T> typename std::enable_if<!has_result_type<T>::value, base_wrapper<typename value_type<A>::type> *>::type result_impl() const {
                    throw std::runtime_error(std::string("The type ") + typeid(A).name() + " has no result_type" + ALPS_STACKTRACE);
                    return NULL;
                }

        };
    }
}
