/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/accumulators/accumulator.hpp>
#include <sstream>

namespace alps {
    namespace accumulators {

        //
        // constructors
        //

        accumulator_wrapper::accumulator_wrapper()
            : m_variant()
        {}

        accumulator_wrapper::accumulator_wrapper(accumulator_wrapper const & rhs)
            : m_variant(rhs.m_variant)
        {}

        accumulator_wrapper::accumulator_wrapper(hdf5::archive & ar) {
            ar[""] >> *this;
        }

        //
        // merge
        //

        /// Service class to access elements of a variant type
        struct accumulator_wrapper::merge_visitor: public boost::static_visitor<> {
            // The accumulator we want to merge (RHS):
            const accumulator_wrapper& rhs_acc;

            // Remember the RHS accumulator
            merge_visitor(const accumulator_wrapper& b): rhs_acc(b) {}

            // This is called by apply_visitor()
            template <typename P> // P can be dereferenced to base_wrapper<T>
            void operator()(P& lhs_ptr)
            {
                const P* rhs_ptr=boost::get<P>(& rhs_acc.m_variant);
                if (!rhs_ptr) throw std::runtime_error("Only accumulators of the same type can be merged"
                                                    + ALPS_STACKTRACE);
                detail::check_ptr(*rhs_ptr);
                lhs_ptr->merge(**rhs_ptr);
            }
        };
        void accumulator_wrapper::merge(const accumulator_wrapper& rhs_acc) {
            merge_visitor visitor(rhs_acc);
            boost::apply_visitor(visitor, m_variant);
        }

        //
        // clone / new_clone
        //

        struct accumulator_wrapper::copy_visitor: public boost::static_visitor<> {
            accumulator_wrapper& acc_wrap_;
            copy_visitor(accumulator_wrapper& aw): acc_wrap_(aw) {}

            template <typename T> // T is shared_ptr< base_wrapper<U> >
            void operator()(const T& val)
            {
                acc_wrap_.m_variant=T(val->clone());
            }
        };
        accumulator_wrapper accumulator_wrapper::clone() const
        {
            accumulator_wrapper result;
            copy_visitor vis(result);
            boost::apply_visitor(vis, m_variant);
            return result;
        }
        accumulator_wrapper* accumulator_wrapper::new_clone() const
        {
            accumulator_wrapper* result=new accumulator_wrapper();
            copy_visitor vis(*result);
            boost::apply_visitor(vis, m_variant);
            return result;
        }

        //
        // operator=
        //

        struct accumulator_wrapper::assign_visitor: public boost::static_visitor<> {
            assign_visitor(accumulator_wrapper * s): self(s) {}
            template<typename T> void operator()(T & arg) const {
                self->m_variant = arg;
            }
            mutable accumulator_wrapper * self;
        };
        accumulator_wrapper & accumulator_wrapper::operator=(boost::shared_ptr<accumulator_wrapper> const & rhs) {
            boost::apply_visitor(assign_visitor(this), rhs->m_variant);
            return *this;
        }

        //
        // count
        //

        struct count_visitor: public boost::static_visitor<boost::uint64_t> {
            template<typename T> boost::uint64_t operator()(T const & arg) const {
                detail::check_ptr(arg);
                return arg->count();
            }
        };
        boost::uint64_t accumulator_wrapper::count() const {
            count_visitor visitor;
            return boost::apply_visitor(visitor, m_variant);
        }

        //
        // save
        //

        struct save_visitor: public boost::static_visitor<> {
            save_visitor(hdf5::archive & a): ar(a) {}
            template<typename T> void operator()(T & arg) const {
                detail::check_ptr(arg);
                ar[""] = *arg;
            }
            hdf5::archive & ar;
        };
        void accumulator_wrapper::save(hdf5::archive & ar) const {
            boost::apply_visitor(save_visitor(ar), m_variant);
        }

        //
        // load
        //

        struct load_visitor: public boost::static_visitor<> {
            load_visitor(hdf5::archive & a): ar(a) {}
            template<typename T> void operator()(T const & arg) const {
                detail::check_ptr(arg);
                ar[""] >> *arg;
            }
            hdf5::archive & ar;
        };
        void accumulator_wrapper::load(hdf5::archive & ar) {
            boost::apply_visitor(load_visitor(ar), m_variant);
        }

        //
        // reset
        //

        struct reset_visitor: public boost::static_visitor<> {
            template<typename T> void operator()(T const & arg) const {
                detail::check_ptr(arg);
                arg->reset();
            }
        };
        void accumulator_wrapper::reset() const {
            boost::apply_visitor(reset_visitor(), m_variant);
        }

        //
        // result
        //

        struct result_visitor: public boost::static_visitor<> {
            template<typename T> void operator()(T const & arg) {
                detail::check_ptr(arg);
                value = boost::shared_ptr<result_wrapper>(new result_wrapper(arg->result()));
            }
            boost::shared_ptr<result_wrapper> value;
        };
        boost::shared_ptr<result_wrapper> accumulator_wrapper::result() const {
            result_visitor visitor;
            boost::apply_visitor(visitor, m_variant);
            return visitor.value;
        }

        //
        // print
        //

        struct print_visitor: public boost::static_visitor<> {
            print_visitor(std::ostream & o, bool t): os(o), terse(t) {}
            template<typename T> void operator()(T const & arg) const {
                detail::check_ptr(arg);
                arg->print(os, terse);
            }
            std::ostream & os;
            bool terse;
        };
        void accumulator_wrapper::print(std::ostream & os, bool terse) const {
            boost::apply_visitor(print_visitor(os, terse), m_variant);
        }

        //
        // collective_merge
        //

#ifdef ALPS_HAVE_MPI
        struct collective_merge_visitor: public boost::static_visitor<> {
            collective_merge_visitor(alps::mpi::communicator const & c, int r): comm(c), root(r) {}
            template<typename T> void operator()(T & arg) const { arg->collective_merge(comm, root); }
            template<typename T> void operator()(T const & arg) const { arg->collective_merge(comm, root); }
            alps::mpi::communicator const & comm;
            int root;
        };

        void accumulator_wrapper::collective_merge(alps::mpi::communicator const & comm, int root) {
            boost::apply_visitor(collective_merge_visitor(comm, root), m_variant);
        }
        void accumulator_wrapper::collective_merge(alps::mpi::communicator const & comm, int root) const {
            boost::apply_visitor(collective_merge_visitor(comm, root), m_variant);
        }
#endif

        //
        // Free functions
        //

        std::ostream & operator<<(std::ostream & os, const accumulator_wrapper & arg) {
            arg.print(os, false); // verbose (non-terse) printing by default
            return os;
        }
        detail::printable_type short_print(const accumulator_wrapper& arg)
        {
            std::ostringstream ostr;
            arg.print(ostr,true);
            return ostr.str();
        }

        detail::printable_type full_print(const accumulator_wrapper& arg)
        {
            std::ostringstream ostr;
            arg.print(ostr,false);
            return ostr.str();
        }

        void reset(accumulator_wrapper & arg) {
            return arg.reset();
        }

    }
}
