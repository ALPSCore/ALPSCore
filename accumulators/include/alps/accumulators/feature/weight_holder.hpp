/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <alps/config.hpp>

#include <alps/accumulators/wrappers.hpp>
#include <alps/accumulators/feature/weight.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/utilities/stacktrace.hpp>
#include <alps/utilities/short_print.hpp>

#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>

#include <stdexcept>
#include <type_traits>

namespace alps {
    namespace accumulators {
        // this should be called namespace tag { template<typename T> struct weight_holder; }
        // but gcc <= 4.4 has lookup error, so name it different
        template<typename T> struct weight_holder_tag;

        namespace impl {

            template<typename T, typename W, typename B> struct Accumulator<T, weight_holder_tag<W>, B> : public B {

                public:
                    typedef W weight_type;
                    typedef Result<T, weight_holder_tag<W>, typename B::result_type> result_type;

                    // TODO: add external weight!

                    Accumulator(): B(), m_owner(true), m_weight(new ::alps::accumulators::derived_accumulator_wrapper<W>(W())) {}

                    Accumulator(Accumulator const & arg): B(arg), m_owner(arg.m_owner), m_weight(arg.m_weight) {}

                    template<typename ArgumentPack> Accumulator(ArgumentPack const & args, typename std::enable_if<!is_accumulator<ArgumentPack>::value, int>::type = 0)
                        : B(args), m_owner(true), m_weight(new ::alps::accumulators::derived_accumulator_wrapper<W>(W()))
                    {}

                    base_wrapper<T> const * weight() const {
                        // TODO: make library for scalar type
                        return m_weight.get();
                    }

                    void operator()(T const & val) {
                        // TODO: throw if weight is owned ...
                        B::operator()(val);
                    }

                    template<typename X> typename std::enable_if<std::conditional<
                          std::is_scalar<typename value_type<weight_type>::type>::value
                        , typename std::is_convertible<X, typename value_type<weight_type>::type>::type
                        , typename std::is_same<X, typename value_type<weight_type>::type>::type
                    >::value>::type operator()(T const & val, X const & weight) {
                        // TODO: how do we make sure, weight is updated only once?
                        B::operator()(val);
                        (m_weight->template extract<W>())(weight);
                    }

                    template<typename X> typename std::enable_if<!std::conditional<
                          std::is_scalar<typename value_type<weight_type>::type>::value
                        , typename std::is_convertible<X, typename value_type<weight_type>::type>::type
                        , typename std::is_same<X, typename value_type<weight_type>::type>::type
                    >::value>::type operator()(T const & /*val*/, X const & /*weight*/) {
                        throw std::runtime_error("Invalid type for binary call operator" + ALPS_STACKTRACE);
                    }

                    template<typename S> void print(S & os, bool terse=false) const {
                        B::print(os, terse);
                        os << ", weight: ";
                        m_weight->print(os, terse);
                    }

                    void save(hdf5::archive & ar) const {
                        B::save(ar);
                        ar["weight/value"] = *weight();
                    }

                    void load(hdf5::archive & ar) { // TODO: make archive const
                        B::load(ar);
                        ar["weight/value"] >> *m_weight;
                    }

                    static std::size_t rank() { return B::rank() + 1; }
                    static bool can_load(hdf5::archive & ar) { // TODO: make archive const
                        using alps::hdf5::get_extent;

                        ar.set_context("weight/value");
                        bool is = weight_type::can_load(ar);
                        ar.set_context("../..");

                        return is && B::can_load(ar);
                    }

                    void reset() {
                        B::reset();
                        m_weight->reset();
                    }

                   /// Merge placeholder \remark FIXME: always throws
                    template <typename A>
                    void merge(const A& /*rhs*/)
                    {
                      throw std::logic_error("Merging weight_holder accumulators is not yet implemented"
                                             +ALPS_STACKTRACE);
                    }

#ifdef ALPS_HAVE_MPI
                    void collective_merge(
                          alps::mpi::communicator const & comm
                        , int root
                    ) {
                        B::collective_merge(comm, root);
                        m_weight->collective_merge(comm, root);
                    }

                    void collective_merge(
                          alps::mpi::communicator const & comm
                        , int root
                    ) const {
                        B::collective_merge(comm, root);
                        m_weight->collective_merge(comm, root);
                    }
#endif

                    bool owns_weight() const {
                        return m_owner;
                    }

                private:
                    bool m_owner;
                    boost::shared_ptr< ::alps::accumulators::base_wrapper<typename value_type<weight_type>::type> > m_weight;
            };

            template<typename T, typename W, typename B> class Result<T, weight_holder_tag<W>, B> : public B {

                public:
                    typedef W weight_type;
                    typedef typename detail::make_scalar_result_type<impl::Result,T,weight_holder_tag<W>,B>::type scalar_result_type;

                    Result()
                        : B()
                        , m_owner(true)
                        , m_weight(new ::alps::accumulators::derived_result_wrapper<W>(W()))
                    {}

                    template<typename A> Result(A const & acc)
                        : B(acc)
                        , m_owner(acc.owns_weight())
                        // TODO: implement shared weight
                        , m_weight(acc.weight()->result())
                    {}

                    base_wrapper<typename value_type<weight_type>::type> const * weight() const {
                        return m_weight.get();
                    }

                    template<typename S> void print(S & os, bool terse=false) const {
                        B::print(os, terse);
                        os << ", weight: ";
                        m_weight->print(os, terse);
                    }

                    void save(hdf5::archive & ar) const {
                        B::save(ar);
                        ar["weight/value"] = *weight();
                    }

                    void load(hdf5::archive & ar) {
                        B::load(ar);
                        ar["weight/value"] >> *m_weight;
                    }

                    static std::size_t rank() { return B::rank() + 1; }
                    static bool can_load(hdf5::archive & ar) { // TODO: make archive const
                        using alps::hdf5::get_extent;

                        ar.set_context("weight/value");
                        bool is = weight_type::can_load(ar);
                        ar.set_context("../..");

                        return is && B::can_load(ar);
                    }

                protected:
                    bool m_owner;
                    boost::shared_ptr< ::alps::accumulators::base_wrapper<typename value_type<weight_type>::type> > m_weight;
            };

        }
    }
}
