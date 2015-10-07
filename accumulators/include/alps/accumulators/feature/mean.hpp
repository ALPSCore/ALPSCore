/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_ACCUMULATOR_MEAN_HPP
#define ALPS_ACCUMULATOR_MEAN_HPP

#include <alps/config.hpp>

#include <alps/accumulators/feature.hpp>
#include <alps/accumulators/parameter.hpp>
#include <alps/accumulators/feature/count.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/numeric/inf.hpp>
#include <alps/numeric/boost_array_functions.hpp>
#include <alps/numeric/check_size.hpp>
#include <alps/numeric/vector_functions.hpp>
#include <alps/utilities/stacktrace.hpp>
#include <alps/utilities/short_print.hpp>

#include <alps/type_traits/element_type.hpp>

#include <boost/mpl/if.hpp>
#include <boost/utility.hpp>
#include <boost/type_traits/is_scalar.hpp>
#include <boost/type_traits/is_integral.hpp>

#include <stdexcept>

namespace alps {
    namespace accumulators {
        // this should be called namespace tag { struct mean; }
        // but gcc <= 4.4 has lookup error, so name it different
        struct mean_tag;

        template<typename T> struct mean_type
            : public boost::mpl::if_<boost::is_integral<typename value_type<T>::type>, double, typename value_type<T>::type>
        {};

        template<typename T> struct has_feature<T, mean_tag> {
            template<typename R, typename C> static char helper(R(C::*)() const);
            template<typename C> static char check(boost::integral_constant<std::size_t, sizeof(helper(&C::mean))>*);
            template<typename C> static double check(...);
            typedef boost::integral_constant<bool, sizeof(char) == sizeof(check<T>(0))> type;
        };

        template<typename T> typename mean_type<T>::type mean(T const & arg) {
            return arg.mean();
        }

        namespace detail {

            template<typename A> typename boost::enable_if<
                  typename has_feature<A, mean_tag>::type
                , typename mean_type<A>::type
            >::type mean_impl(A const & acc) {
                return mean(acc);
            }

            template<typename A> typename boost::disable_if<
                  typename has_feature<A, mean_tag>::type
                , typename mean_type<A>::type
            >::type mean_impl(A const & acc) {
                throw std::runtime_error(std::string(typeid(A).name()) + " has no mean-method" + ALPS_STACKTRACE);
                return typename mean_type<A>::type();
            }
        }

        namespace impl {

            template<typename T, typename B> struct Accumulator<T, mean_tag, B> : public B {

                public:
                    typedef typename alps::accumulators::mean_type<B>::type mean_type;
                    typedef Result<T, mean_tag, typename B::result_type> result_type;

                    Accumulator(): B(), m_sum(T()) {}
                    Accumulator(Accumulator const & arg): B(arg), m_sum(arg.m_sum) {}

                    template<typename ArgumentPack> Accumulator(ArgumentPack const & args, typename boost::disable_if<is_accumulator<ArgumentPack>, int>::type = 0)
                        : B(args), m_sum(T())
                    {}

                    mean_type const mean() const {
                        using alps::numeric::operator/;

                        // TODO: make library for scalar type
                        typename alps::hdf5::scalar_type<mean_type>::type cnt = B::count();
                        
                        return mean_type(m_sum) / cnt;
                    }

                    using B::operator();
                    void operator()(T const & val) {
                        using alps::numeric::operator+=;
                        using alps::numeric::check_size;

                        B::operator()(val);
                        check_size(m_sum, val);
                        m_sum += val;
                    }

                    template<typename S> void print(S & os) const {
                        os << alps::short_print(mean());
                        B::print(os);
                    }

                    void save(hdf5::archive & ar) const {
                        B::save(ar);
                        ar["mean/value"] = mean();
                    }

                    void load(hdf5::archive & ar) { // TODO: make archive const
                        using alps::numeric::operator*;

                        B::load(ar);
                        mean_type mean;
                        ar["mean/value"] >> mean;
                        // TODO: make library for scalar type
                        typename alps::hdf5::scalar_type<mean_type>::type cnt = B::count();
                        m_sum = mean * cnt;
                    }

                    static std::size_t rank() { return B::rank() + 1; }
                    static bool can_load(hdf5::archive & ar) { // TODO: make archive const
                        using alps::hdf5::get_extent;
                        const char name[]="mean/value";
                        return B::can_load(ar)
                            && ar.is_data(name)
                            && ar.is_datatype<typename alps::hdf5::scalar_type<mean_type>::type>(name)
                            && boost::is_scalar<T>::value == ar.is_scalar(name)
                            && (boost::is_scalar<T>::value || get_extent(T()).size() == ar.dimensions(name))
                        ;
                    }

                    void reset() {
                        B::reset();
                        m_sum = T();
                    }

              /// Merge the sum (mean) of  given accumulator of type A into this sum (mean) @param rhs Accumulator to merge 
              template <typename A>
              void merge(const A& rhs)
              {
                using alps::numeric::operator+=;
                using alps::numeric::check_size;
                B::merge(rhs);
                check_size(m_sum,rhs.m_sum);
                m_sum += rhs.m_sum;
              }

#ifdef ALPS_HAVE_MPI
                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) {
                        if (comm.rank() == root) {
                            B::collective_merge(comm, root);
                            B::reduce_if(comm, T(m_sum), m_sum, std::plus<typename alps::hdf5::scalar_type<T>::type>(), root);
                        } else
                            const_cast<Accumulator<T, mean_tag, B> const *>(this)->collective_merge(comm, root);
                    }
                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) const {
                        B::collective_merge(comm, root);
                        if (comm.rank() == root)
                            throw std::runtime_error("A const object cannot be root" + ALPS_STACKTRACE);
                        else
                            B::reduce_if(comm, m_sum, std::plus<typename alps::hdf5::scalar_type<T>::type>(), root);
                    }
#endif
                protected:

                    T const & sum() const {
                        return m_sum;
                    }

                private:
                    T m_sum;
            };

            template<typename T, typename B> class Result<T, mean_tag, B> : public B {

                public:
                    typedef typename alps::accumulators::mean_type<B>::type mean_type;
                    typedef typename detail::make_scalar_result_type<impl::Result,T,mean_tag,B>::type scalar_result_type;
                    typedef Result<std::vector<T>, mean_tag, typename B::vector_result_type> vector_result_type;

                    Result()
                        : B()
                        , m_mean(mean_type()) 
                    {}

                    template<typename A> Result(A const & acc)
                        : B(acc)
                        , m_mean(detail::mean_impl(acc))
                    {}

                    mean_type const mean() const { 
                        return m_mean; 
                    }

                    template<typename S> void print(S & os) const {
                        os << alps::short_print(mean());
                        B::print(os);
                    }

                    void save(hdf5::archive & ar) const {
                        B::save(ar);
                        ar["mean/value"] = mean();
                    }

                    void load(hdf5::archive & ar) {
                        B::load(ar);
                        ar["mean/value"] >> m_mean;
                    }

                    static std::size_t rank() { return B::rank() + 1; }
                    static bool can_load(hdf5::archive & ar) { // TODO: make archive const
                        using alps::hdf5::get_extent;
                        const char name[]="mean/value";
                        return B::can_load(ar) 
                            && ar.is_data(name) 
                            && ar.is_datatype<typename alps::hdf5::scalar_type<mean_type>::type>(name)
                            && boost::is_scalar<T>::value == ar.is_scalar(name)
                            && (boost::is_scalar<T>::value || get_extent(T()).size() == ar.dimensions(name))
                        ;
                    }

                    template<typename U> void operator+=(U const & arg) { augadd(arg); }
                    template<typename U> void operator-=(U const & arg) { augsub(arg); }
                    template<typename U> void operator*=(U const & arg) { augmul(arg); }
                    template<typename U> void operator/=(U const & arg) { augdiv(arg); }
                    void negate() {
                        using alps::numeric::operator-;
                        m_mean = -m_mean;
                        B::negate();
                    }                    
                    void inverse() {
                        using alps::numeric::operator/;
                        // TODO: make library for scalar type
                        typename alps::hdf5::scalar_type<mean_type>::type one = 1;
                        m_mean = one / m_mean;
                        B::inverse();
                    }                    

                    #define NUMERIC_FUNCTION_IMPLEMENTATION(FUNCTION_NAME)              \
                        void FUNCTION_NAME () {                                         \
                            B:: FUNCTION_NAME ();                                       \
                            using std:: FUNCTION_NAME ;                                 \
                            using alps::numeric:: FUNCTION_NAME ;                  \
                            m_mean = FUNCTION_NAME (m_mean);                            \
                        }

                    NUMERIC_FUNCTION_IMPLEMENTATION(sin)
                    NUMERIC_FUNCTION_IMPLEMENTATION(cos)
                    NUMERIC_FUNCTION_IMPLEMENTATION(tan)
                    NUMERIC_FUNCTION_IMPLEMENTATION(sinh)
                    NUMERIC_FUNCTION_IMPLEMENTATION(cosh)
                    NUMERIC_FUNCTION_IMPLEMENTATION(tanh)
                    NUMERIC_FUNCTION_IMPLEMENTATION(asin)
                    NUMERIC_FUNCTION_IMPLEMENTATION(acos)
                    NUMERIC_FUNCTION_IMPLEMENTATION(atan)
                    NUMERIC_FUNCTION_IMPLEMENTATION(abs)
                    NUMERIC_FUNCTION_IMPLEMENTATION(sqrt)
                    NUMERIC_FUNCTION_IMPLEMENTATION(log)

                    #undef NUMERIC_FUNCTION_IMPLEMENTATION

                    #define NUMERIC_FUNCTION_IMPLEMENTATION(FUNCTION_NAME)          \
                        void FUNCTION_NAME () {                                     \
                            B:: FUNCTION_NAME ();                                   \
                            using alps::numeric:: FUNCTION_NAME ;                   \
                            using alps::numeric:: FUNCTION_NAME ;              \
                            m_mean = FUNCTION_NAME (m_mean);                        \
                        }

                    NUMERIC_FUNCTION_IMPLEMENTATION(sq)
                    NUMERIC_FUNCTION_IMPLEMENTATION(cb)
                    NUMERIC_FUNCTION_IMPLEMENTATION(cbrt)

                    #undef NUMERIC_FUNCTION_IMPLEMENTATION

                private:

                    mean_type m_mean;

                    #define NUMERIC_FUNCTION_OPERATOR(OP_NAME, OPEQ_NAME, OP, OP_TOKEN)                                                                                         \
                        template<typename U> void aug ## OP_TOKEN (U const & arg, typename boost::disable_if<boost::is_scalar<U>, int>::type = 0) {                             \
                            using alps::numeric:: OP_NAME ;                                                                                                                     \
                            m_mean = m_mean OP arg.mean();                                                                                                                      \
                            B:: OPEQ_NAME (arg);                                                                                                                                \
                        }                                                                                                                                                       \
                       template<typename U> void aug ## OP_TOKEN (U const & arg,                                                                                                \
                                                                  typename boost::enable_if<boost::is_scalar<U>, int>::type = 0) {                                              \
                            using alps::numeric:: OP_NAME ;                                                                                                                     \
                            m_mean = m_mean OP static_cast<typename alps::element_type<mean_type>::type>(arg);                                                                  \
                            B:: OPEQ_NAME (arg);                                                                                                                                \
                        }

                    NUMERIC_FUNCTION_OPERATOR(operator+, operator+=, +, add)
                    NUMERIC_FUNCTION_OPERATOR(operator-, operator-=, -, sub)
                    NUMERIC_FUNCTION_OPERATOR(operator*, operator*=, *, mul)
                    NUMERIC_FUNCTION_OPERATOR(operator/, operator/=, /, div)

                    #undef NUMERIC_FUNCTION_OPERATOR
            };

            template<typename T, typename B> class BaseWrapper<T, mean_tag, B> : public B {
                public:
                    virtual bool has_mean() const = 0;
                    virtual typename mean_type<B>::type mean() const = 0;
            };

            template<typename T, typename B> class DerivedWrapper<T, mean_tag, B> : public B {
                public:
                    DerivedWrapper(): B() {}
                    DerivedWrapper(T const & arg): B(arg) {}

                    bool has_mean() const { return has_feature<T, mean_tag>::type::value; }

                    typename mean_type<B>::type mean() const { return detail::mean_impl(this->m_data); }
            };

        }
    }
}

 #endif
