/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_WRAPED_ACCUMULATORS_HPP
#define ALPS_WRAPED_ACCUMULATORS_HPP

#include <alps/config.hpp>

#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/archive.hpp>

#ifdef ALPS_HAVE_MPI
    #include <alps/utilities/boost_mpi.hpp>
#endif

#include <alps/accumulators/wrapper_set.hpp>

#include <string>
#include <iostream>
#include <typeinfo>
#include <stdexcept>

namespace alps {
    namespace accumulators {

        class ALPS_DECL accumulator_wrapper;

        namespace wrapped {

            class ALPS_DECL virtual_accumulator_wrapper {
                public:
                    // default constructor
                    virtual_accumulator_wrapper();

                    // constructor from raw accumulator
                    virtual_accumulator_wrapper(accumulator_wrapper * arg);

                    // copy constructor
                    virtual_accumulator_wrapper(virtual_accumulator_wrapper const & rhs);

                    // constructor from hdf5
                    virtual_accumulator_wrapper(hdf5::archive & ar);

                    virtual ~virtual_accumulator_wrapper();

                    // operator()
                    // TODO: use ALPS_ACCUMULATOR_VALUE_TYPES
                    virtual_accumulator_wrapper & operator()(double const & value);

                    template<typename T> virtual_accumulator_wrapper & operator<<(T const & value) {
                        (*this)(value);
                        return (*this);
                    }

                    /// Merge another accumulator into this one. @param rhs  accumulator to merge.
                    void merge(const virtual_accumulator_wrapper & rhs);

                    virtual_accumulator_wrapper & operator=(boost::shared_ptr<virtual_accumulator_wrapper> const & rhs);

                    // get
                    // template <typename T> base_wrapper<T> & get() {
                    //     get_visitor<T> visitor;
                    //     boost::apply_visitor(visitor, m_variant);
                    //     return *visitor.value;
                    // }

                    // extract
                    // template <typename A> A & extract() {
                    //     throw std::logic_error(std::string("unknown type : ") + typeid(A).name() + ALPS_STACKTRACE);
                    // }
                    // template <> MeanAccumulatorDouble & extract<MeanAccumulatorDouble>();

                    // count
                    boost::uint64_t count() const;

                // // mean, error
                // #define ALPS_ACCUMULATOR_PROPERTY_PROXY(PROPERTY, TYPE)                                                 \
                //     private:                                                                                            \
                //         template<typename T> struct PROPERTY ## _visitor: public boost::static_visitor<> {              \
                //             template<typename X> void apply(typename boost::enable_if<                                  \
                //                 typename detail::is_valid_argument<typename TYPE <X>::type, T>::type, X const &         \
                //             >::type arg) const {                                                                        \
                //                 value = arg. PROPERTY ();                                                               \
                //             }                                                                                           \
                //             template<typename X> void apply(typename boost::disable_if<                                 \
                //                 typename detail::is_valid_argument<typename TYPE <X>::type, T>::type, X const &         \
                //             >::type arg) const {                                                                        \
                //                 throw std::logic_error(std::string("cannot convert: ")                                  \
                //                     + typeid(typename TYPE <X>::type).name() + " to "                                   \
                //                     + typeid(T).name() + ALPS_STACKTRACE);                                              \
                //             }                                                                                           \
                //             template<typename X> void operator()(X const & arg) const {                                 \
                //                 apply<typename X::element_type>(*arg);                                                  \
                //             }                                                                                           \
                //             mutable T value;                                                                            \
                //         };                                                                                              \
                //     public:                                                                                             \
                //         template<typename T> typename TYPE <base_wrapper<T> >::type PROPERTY () const {                 \
                //             PROPERTY ## _visitor<typename TYPE <base_wrapper<T> >::type> visitor;                       \
                //             boost::apply_visitor(visitor, m_variant);                                                   \
                //             return visitor.value;                                                                       \
                //         }
                // ALPS_ACCUMULATOR_PROPERTY_PROXY(mean, mean_type)
                // ALPS_ACCUMULATOR_PROPERTY_PROXY(error, error_type)
                // #undef ALPS_ACCUMULATOR_FUNCTION_PROXY

                    // save
                    void save(hdf5::archive & ar) const;

                    // load
                    void load(hdf5::archive & ar);

                    // reset
                    void reset() const;

                    // result
                    // boost::shared_ptr<result_wrapper> result() const;

                    // print
                    void print(std::ostream & os) const;

#ifdef ALPS_HAVE_MPI
                    // collective_merge
                    void collective_merge(boost::mpi::communicator const & comm, int root);
                    void collective_merge(boost::mpi::communicator const & comm, int root) const;
#endif

                private:

                    std::ptrdiff_t * m_cnt;
                    accumulator_wrapper * m_ptr;
            };

            inline std::ostream & operator<<(std::ostream & os, const virtual_accumulator_wrapper & arg) {
                arg.print(os);
                return os;
            }

            // template <typename A> A & extract(virtual_accumulator_wrapper & m) {
            //     return m.extract<A>();
            // }

            inline void ALPS_DECL reset(virtual_accumulator_wrapper & arg) {
                return arg.reset();
            }

        }
    }

    // TODO: take type from variant type
    template<typename T> struct MeanAccumulator {
        public:
            MeanAccumulator(std::string const & name): m_name(name) {}
            std::string const & name() const { return m_name; }
        private:
            std::string m_name;
    };

    typedef accumulators::impl::wrapper_set<accumulators::wrapped::virtual_accumulator_wrapper> accumulator_set;
    // typedef impl::wrapper_set<result_wrapper> result_set;

    accumulator_set & operator<<(accumulator_set & set, const MeanAccumulator<double> & arg);
}

#endif