/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_ACCUMULATOR_COUNT_HPP
#define ALPS_ACCUMULATOR_COUNT_HPP

#include <alps/config.hpp>

#include <alps/accumulators/feature.hpp>
#include <alps/accumulators/parameter.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/utilities/stacktrace.hpp>
#include <alps/utilities/short_print.hpp>

#include <boost/utility.hpp>
#include <boost/cstdint.hpp>

#include <stdexcept>

namespace alps {
    namespace accumulators {
        // this should be called namespace tag { struct count; }
        // but gcc <= 4.4 has lookup error, so name it different
        struct count_tag;

        template<typename T> struct count_type {
            typedef boost::uint64_t type;
        };

        template<typename T> struct has_feature<T, count_tag> {
            template<typename C> static char helper(typename count_type<T>::type (C::*)() const);
            template<typename C> static char check(boost::integral_constant<std::size_t, sizeof(helper(&C::count))>*);
            template<typename C> static double check(...);
            typedef boost::integral_constant<bool, sizeof(char) == sizeof(check<T>(0))> type;
        };

        template<typename T> typename count_type<T>::type count(T const & arg) {
            return arg.count();
        }

        namespace detail {

            template<typename A> typename boost::enable_if<
                typename has_feature<A, count_tag>::type, typename count_type<A>::type
            >::type count_impl(A const & acc) {
                return count(acc);
            }

            template<typename A> typename boost::disable_if<
                typename has_feature<A, count_tag>::type, typename count_type<A>::type
            >::type count_impl(A const & acc) {
                throw std::runtime_error(std::string(typeid(A).name()) + " has no count-method" + ALPS_STACKTRACE);
                return typename count_type<A>::type();
            }

        }

        namespace impl {

            template<typename T, typename B> class Result<T, count_tag, B> : public B {

                public:
                    typedef typename count_type<T>::type count_type;
                    typedef typename detail::make_scalar_result_type<impl::Result,T,count_tag,B>::type scalar_result_type;
                    typedef Result<std::vector<T>, count_tag, typename B::vector_result_type> vector_result_type;

                    Result()
                        : m_count(count_type())
                    {}

                    template<typename A> Result(A const & acc)
                        : m_count(detail::count_impl(acc))
                    {}

                    count_type count() const {
                        return m_count;
                    }

                    void operator()(T const &) {
                        throw std::runtime_error("No values can be added to a result" + ALPS_STACKTRACE);
                    }

                    template<typename W> void operator()(T const &, W) {
                        throw std::runtime_error("No values can be added to a result" + ALPS_STACKTRACE);
                    }

                    template<typename S> void print(S & os) const {
                        os << " #" << alps::short_print(count());
                    }

                    void save(hdf5::archive & ar) const {
                        if (m_count==0) {
                            throw std::logic_error("Attempt to save an empty result" + ALPS_STACKTRACE);
                        }
                        ar["count"] = m_count;
                    }

                    void load(hdf5::archive & ar) {
                        count_type cnt;
                        ar["count"] >> cnt;
                        if (cnt==0) {
                            throw std::runtime_error("Malformed archive containing an empty result"
                                                     + ALPS_STACKTRACE);
                        }
                        m_count=cnt;
                    }

                    static std::size_t rank() { return 1; }
                    static bool can_load(hdf5::archive & ar) { // TODO: make archive const
                        return ar.is_data("count");
                    }

                    template<typename U> void operator+=(U const & arg) { augadd(arg); }
                    template<typename U> void operator-=(U const & arg) { augsub(arg); }
                    template<typename U> void operator*=(U const & arg) { augmul(arg); }
                    template<typename U> void operator/=(U const & arg) { augdiv(arg); }

                    inline void reset() {
                        throw std::runtime_error("A result cannot be reseted" + ALPS_STACKTRACE);
                    }

                private:

                    // TODO: make macro ...
                    template<typename U> void augadd(U const & arg, typename boost::enable_if<boost::is_scalar<U>, int>::type = 0) {
                        if (m_count == 0)
                            throw std::runtime_error("The results needs measurements" + ALPS_STACKTRACE);
                        B::operator+=(arg);
                    }
                    template<typename U> void augadd(U const & arg, typename boost::disable_if<boost::is_scalar<U>, int>::type = 0) {
                        if (m_count == 0 || arg.count() == 0)
                            throw std::runtime_error("Both results needs measurements" + ALPS_STACKTRACE);
                        m_count = std::min(m_count,  arg.count());
                        B::operator+=(arg);
                    }

                    template<typename U> void augsub(U const & arg, typename boost::enable_if<boost::is_scalar<U>, int>::type = 0) {
                        if (m_count == 0)
                            throw std::runtime_error("The results needs measurements" + ALPS_STACKTRACE);
                        B::operator-=(arg);
                    }
                    template<typename U> void augsub(U const & arg, typename boost::disable_if<boost::is_scalar<U>, int>::type = 0) {
                        if (m_count == 0 || arg.count() == 0)
                            throw std::runtime_error("Both results needs measurements" + ALPS_STACKTRACE);
                        m_count = std::min(m_count,  arg.count());
                        B::operator-=(arg);
                    }

                    template<typename U> void augmul(U const & arg, typename boost::enable_if<boost::is_scalar<U>, int>::type = 0) {
                        if (m_count == 0)
                            throw std::runtime_error("The results needs measurements" + ALPS_STACKTRACE);
                        B::operator*=(arg);
                    }
                    template<typename U> void augmul(U const & arg, typename boost::disable_if<boost::is_scalar<U>, int>::type = 0) {
                        if (m_count == 0 || arg.count() == 0)
                            throw std::runtime_error("Both results needs measurements" + ALPS_STACKTRACE);
                        m_count = std::min(m_count,  arg.count());
                        B::operator*=(arg);
                    }

                    template<typename U> void augdiv(U const & arg, typename boost::enable_if<boost::is_scalar<U>, int>::type = 0) {
                        if (m_count == 0)
                            throw std::runtime_error("The results needs measurements" + ALPS_STACKTRACE);
                        B::operator/=(arg);
                    }
                    template<typename U> void augdiv(U const & arg, typename boost::disable_if<boost::is_scalar<U>, int>::type = 0) {
                        if (m_count == 0 || arg.count() == 0)
                            throw std::runtime_error("Both results needs measurements" + ALPS_STACKTRACE);
                        m_count = std::min(m_count,  arg.count());
                        B::operator/=(arg);
                    }

                    count_type m_count;
            };

            template<typename T, typename B> struct Accumulator<T, count_tag, B> : public B {

                public:
                    typedef typename count_type<T>::type count_type;
                    typedef Result<T, count_tag, typename B::result_type> result_type;

                    Accumulator(): m_count(count_type()) {}

                    Accumulator(Accumulator const & arg): m_count(arg.m_count) {}

                    template<typename ArgumentPack> Accumulator(ArgumentPack const & args, typename boost::disable_if<is_accumulator<ArgumentPack>, int>::type = 0)
                        : m_count(count_type())
                    {}

                    count_type count() const {
                        return m_count;
                    }

                    void operator()(T const &) {
                        ++m_count;
                    }
                    template<typename W> void operator()(T const &, W) {
                        throw std::runtime_error("Observable has no binary call operator" + ALPS_STACKTRACE);
                    }

                    template<typename S> void print(S & os) const {
                        os << " #" << alps::short_print(count());
                    }

                    void save(hdf5::archive & ar) const {
                        if (m_count==0) {
                            throw std::logic_error("Attempt to save an empty accumulator" + ALPS_STACKTRACE);
                        }
                        ar["count"] = m_count;
                    }

                    void load(hdf5::archive & ar) { // TODO: make archive const
                        count_type cnt;
                        ar["count"] >> cnt;
                        if (cnt==0) {
                            throw std::runtime_error("Malformed archive containing an empty accumulator"
                                                     + ALPS_STACKTRACE);
                        }
                        m_count=cnt;
                    }

                    static std::size_t rank() { return 1; }
                    static bool can_load(const hdf5::archive & ar) {
                        return ar.is_data("count");
                    }

                    inline void reset() {
                        m_count = 0;
                    }

              /// Merge the counter of the given accumulator of type A into this counter. @param rhs Accumulator to merge 
              template <typename A>
              void merge(const A& rhs)
              {
                m_count += rhs.m_count;
              }

#ifdef ALPS_HAVE_MPI
                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) {
                        if (comm.rank() == root)
                            alps::mpi::reduce(comm, m_count, m_count, std::plus<count_type>(), root);
                        else
                            const_cast<Accumulator<T, count_tag, B> const *>(this)->collective_merge(comm, root);
                    }
                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) const {
                        if (comm.rank() == root)
                            throw std::runtime_error("A const object cannot be root" + ALPS_STACKTRACE);
                        else
                            alps::mpi::reduce(comm, m_count, std::plus<count_type>(), root);
                    }
#endif

                private:
                    count_type m_count;
            };

            template<typename T, typename B> class BaseWrapper<T, count_tag, B> : public B {
                public:
                    virtual bool has_count() const = 0;
                    virtual boost::uint64_t count() const = 0;
            };

            template<typename T, typename B> class DerivedWrapper<T, count_tag, B> : public B {
                public:
                    DerivedWrapper(): B() {}
                    DerivedWrapper(T const & arg): B(arg) {}

                    bool has_count() const { return has_feature<T, count_tag>::type::value; }

                    typename boost::uint64_t count() const { return detail::count_impl(this->m_data); }
            };

        }
    }
}

 #endif
