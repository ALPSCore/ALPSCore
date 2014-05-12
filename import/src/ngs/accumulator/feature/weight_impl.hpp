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

#ifndef ALPS_NGS_ACCUMULATOR_WEIGHT_IMPL_HPP
#define ALPS_NGS_ACCUMULATOR_WEIGHT_IMPL_HPP

#include <alps/ngs/accumulator/wrappers.hpp>
#include <alps/ngs/accumulator/feature/weight.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/ngs/stacktrace.hpp>
#include <alps/ngs/short_print.hpp>

#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>

#include <stdexcept>

namespace alps {
    namespace accumulator {
        // this should be called namespace tag { template<typename T> struct weight_holder; }
        // but gcc <= 4.4 has lookup error, so name it different
        template<typename T> struct weight_holder_tag;

        namespace impl {

            template<typename T, typename W, typename B> struct Accumulator<T, weight_holder_tag<W>, B> : public B {

                public:
                    typedef W weight_type;
                    typedef Result<T, weight_holder_tag<W>, typename B::result_type> result_type;

                    // TODO: add external weight!

                    Accumulator(): B(), m_owner(true), m_weight(new ::alps::accumulator::derived_accumulator_wrapper<W>(W())) {}

                    Accumulator(Accumulator const & arg): B(arg), m_owner(arg.m_owner), m_weight(arg.m_weight) {}

                    template<typename ArgumentPack> Accumulator(ArgumentPack const & args, typename boost::disable_if<is_accumulator<ArgumentPack>, int>::type = 0)
                        : B(args), m_owner(true), m_weight(new ::alps::accumulator::derived_accumulator_wrapper<W>(W()))
                    {}

                    base_wrapper const * weight() const {
                        // TODO: make library for scalar type
                        return m_weight.get();
                    }

                    void operator()(T const & val) {
                        // TODO: throw if weight is owned ...
                        B::operator()(val);
                    }

                    void operator()(T const & val, typename value_type<weight_type>::type const & weight) {
                        // TODO: how do we make sure, weight is updated only once?
                        B::operator()(val);
                        (m_weight->extract<W>())(weight);
                    }

                    template<typename S> void print(S & os) const {
                        B::print(os);
                        os << ", weight: ";
                        m_weight->print(os);
                    }

                    // TODO: implement!
                    void save(hdf5::archive & ar) const {
                        B::save(ar);
                        ar["weight/value"] = *weight();
                    }

                    // TODO: implement!
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

#ifdef ALPS_HAVE_MPI
                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) {
                        B::collective_merge(comm, root);
                        m_weight->collective_merge(comm, root);
                    }

                    void collective_merge(
                          boost::mpi::communicator const & comm
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
                    boost::shared_ptr< ::alps::accumulator::base_wrapper> m_weight;
            };

            template<typename T, typename W, typename B> class Result<T, weight_holder_tag<W>, B> : public B {

                public:
                    typedef W weight_type;

                    Result()
                        : B()
                        , m_owner(true)
                        , m_weight(new ::alps::accumulator::derived_result_wrapper<W>(W()))
                    {}

                    template<typename A> Result(A const & acc)
                        : B(acc)
                        , m_owner(acc.owns_weight())
                        // TODO: implement shared weight
                        , m_weight(acc.weight()->result())
                    {}

                    base_wrapper const * weight() const {
                        return m_weight.get();
                    }

                    template<typename S> void print(S & os) const {
                        B::print(os);
                        os << ", weight: ";
                        m_weight->print(os);
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
                    boost::shared_ptr< ::alps::accumulator::base_wrapper> m_weight;
            };

        }
    }
}

 #endif
