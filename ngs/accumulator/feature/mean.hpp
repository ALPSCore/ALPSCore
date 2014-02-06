/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 - 2013 by Mario Koenz <mkoenz@ethz.ch>                       *
 *                              Lukas Gamper <gamperl@gmail.com>                   *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef ALPS_NGS_ACCUMULATOR_MEAN_HPP
#define ALPS_NGS_ACCUMULATOR_MEAN_HPP

#include <alps/ngs/accumulator/feature.hpp>
#include <alps/ngs/accumulator/feature/count.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/ngs/numeric.hpp>
#include <alps/ngs/stacktrace.hpp>
#include <alps/ngs/short_print.hpp>

#include <boost/mpl/if.hpp>
#include <boost/utility.hpp>
#include <boost/type_traits/is_scalar.hpp>
#include <boost/type_traits/is_integral.hpp>

#include <stdexcept>

namespace alps {
    namespace accumulator {
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
                    typedef typename alps::accumulator::mean_type<B>::type mean_type;
                    typedef Result<T, mean_tag, typename B::result_type> result_type;

                    // TODO: implement using disable_if<Accumulator<...> > ...
                    // template<typename ArgumentPack> Accumulator(ArgumentPack const & args): B(args), m_sum(T()) {}

                    Accumulator(): B(), m_sum(T()) {}
                    Accumulator(Accumulator const & arg): B(arg), m_sum(arg.m_sum) {}

                    mean_type const mean() const {
                        using alps::ngs::numeric::operator/;

                        // TODO: make library for scalar type
                        typename alps::hdf5::scalar_type<mean_type>::type cnt = B::count();
                        return mean_type(m_sum) / cnt;
                    }

                    void operator()(T const & val) {
                        using alps::ngs::numeric::operator+=;
                        using alps::ngs::numeric::detail::check_size;

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
                        using alps::ngs::numeric::operator*;

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

                        return B::can_load(ar)
                            && ar.is_data("mean/value") 
                            && boost::is_scalar<T>::value == ar.is_scalar("mean/value")
                            && (boost::is_scalar<T>::value || get_extent(T()).size() == ar.dimensions("mean/value"))
                        ;
                    }

                    void reset() {
                        B::reset();
                        m_sum = T();
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
                    typedef typename alps::accumulator::mean_type<B>::type mean_type;

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

                        return B::can_load(ar) 
                            && ar.is_data("mean/value") 
                            && boost::is_scalar<T>::value == ar.is_scalar("mean/value")
                            && (boost::is_scalar<T>::value || get_extent(T()).size() == ar.dimensions("mean/value"))
                        ;
                    }

                    // TODO: implement -=, *=, /=
                    template<typename U> void operator+=(U const & arg) {

                        using alps::ngs::numeric::operator+;
                        using alps::ngs::numeric::operator*;
                        using alps::ngs::numeric::operator/;

                        // TODO: make library for scalar type
                        typename alps::hdf5::scalar_type<mean_type>::type cnt = B::count(), argcnt = arg.count();

                        m_mean = (cnt * m_mean + argcnt * arg.mean()) / (cnt + argcnt);
                        B::operator+=(arg);
                    }

/*
                    #define NUMERIC_FUNCTION_OPERATOR(OP_NAME, OP_USING, OP_SYMBOL)    \
                        template<typename U> void OP_NAME (U const & arg) {            \
                            B:: OP_NAME (arg);                                        \
                            OP_NAME ## _impl(arg);                                    \
                        }                                                            \
                        void OP_NAME (T const & arg) {                                \
                            using alps::ngs::numeric:: OP_USING ;                    \
                            B:: OP_NAME (arg);                                        \
                            m_mean OP_SYMBOL arg;                                    \
                        }

                    NUMERIC_FUNCTION_OPERATOR(addeq, operator+=, +=)
                    NUMERIC_FUNCTION_OPERATOR(subeq, operator-=, -=)
                    NUMERIC_FUNCTION_OPERATOR(muleq, operator*=, *=)
                    NUMERIC_FUNCTION_OPERATOR(diveq, operator/=, /=)

                    #undef NUMERIC_FUNCTION_OPERATOR
*/
                    #define NUMERIC_FUNCTION_IMPLEMENTATION(FUNCTION_NAME)            \
                        void FUNCTION_NAME () {                                     \
                            B:: FUNCTION_NAME ();                                    \
                            using std:: FUNCTION_NAME ;                                \
                            using alps::ngs::numeric:: FUNCTION_NAME ;                \
                            m_mean = FUNCTION_NAME (m_mean);                         \
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

                    #define NUMERIC_FUNCTION_IMPLEMENTATION(FUNCTION_NAME)            \
                        void FUNCTION_NAME () {                                         \
                            B:: FUNCTION_NAME ();                                    \
                            using alps::numeric:: FUNCTION_NAME ;                   \
                            using alps::ngs::numeric:: FUNCTION_NAME ;                \
                            m_mean = FUNCTION_NAME (m_mean);                         \
                        }

                    NUMERIC_FUNCTION_IMPLEMENTATION(sq)
                    NUMERIC_FUNCTION_IMPLEMENTATION(cb)
                    NUMERIC_FUNCTION_IMPLEMENTATION(cbrt)

                    #undef NUMERIC_FUNCTION_IMPLEMENTATION

                private:

                    mean_type m_mean;

                    // #define NUMERIC_FUNCTION_OPERATOR_IMPL(OP_NAME, OP_USING, OP_SYMBOL)                    \
                    //     template<typename U> void OP_NAME ## _impl(typename boost::disable_if<                \
                    //           typename boost::is_same<typename alps::hdf5::scalar_type<T>::type, U>::type    \
                    //         , U                                                                                \
                    //     >::type const & arg) {                                                                \
                    //         using alps::ngs::numeric:: OP_USING ;                                            \
                    //         m_mean OP_SYMBOL arg.mean();                                                    \
                    //     }                                                                                    \
                    //     template<typename U> void OP_NAME ## _impl(typename boost::enable_if<                \
                    //           typename boost::is_same<typename alps::hdf5::scalar_type<T>::type, U>::type    \
                    //         , typename alps::hdf5::scalar_type<T>::type                                        \
                    //     >::type arg) {                                                                        \
                    //         using alps::ngs::numeric:: OP_USING ;                                            \
                    //         m_mean OP_SYMBOL arg;                                                            \
                    //     }

                    // NUMERIC_FUNCTION_OPERATOR_IMPL(addeq, operator+=, +=)
                    // NUMERIC_FUNCTION_OPERATOR_IMPL(subeq, operator-=, -=)
                    // NUMERIC_FUNCTION_OPERATOR_IMPL(muleq, operator*=, *=)
                    // NUMERIC_FUNCTION_OPERATOR_IMPL(diveq, operator/=, /=)

                    // #undef NUMERIC_FUNCTION_OPERATOR_IMPL
            };

            template<typename B> class BaseWrapper<mean_tag, B> : public B {
                public:
                    virtual bool has_mean() const = 0;
            };

            template<typename T, typename B> class ResultTypeWrapper<T, mean_tag, B> : public B {
                public:
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
