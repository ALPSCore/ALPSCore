/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <alps/config.hpp>

#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/archive.hpp>

#ifdef ALPS_HAVE_MPI
    #include <alps/utilities/boost_mpi.hpp>
#endif

#include <alps/accumulators/wrapper_set.hpp>

#include <boost/preprocessor/tuple/to_seq.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#include <string>
#include <iostream>
#include <typeinfo>
#include <stdexcept>

#ifndef ALPS_ACCUMULATOR_VALUE_TYPES_SEQ
    #define ALPS_ACCUMULATOR_VALUE_TYPES_SEQ BOOST_PP_TUPLE_TO_SEQ(ALPS_ACCUMULATOR_VALUE_TYPES_SIZE, (ALPS_ACCUMULATOR_VALUE_TYPES))
#endif

namespace alps {
    namespace accumulators {

        class result_wrapper;
        class accumulator_wrapper;

        namespace wrapped {

            template<typename accumulator_type> class virtual_result_wrapper {
                public:

                    // default constructor
                    virtual_result_wrapper();

                    // constructor from raw accumulator
                    virtual_result_wrapper(result_wrapper * arg);

                    // copy constructor
                    virtual_result_wrapper(virtual_result_wrapper const & rhs);

                    // constructor from hdf5
                    virtual_result_wrapper(hdf5::archive & ar);

                    virtual ~virtual_result_wrapper();

                    // count
                    boost::uint64_t count() const;

                private:

                    #define ALPS_ACCUMULATOR_MEAN_IMPL(r, data, T)  \
                        T mean_impl(T) const;
                    BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_MEAN_IMPL, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
                    #undef ALPS_ACCUMULATOR_MEAN_IMPL

                    #define ALPS_ACCUMULATOR_ERROR_IMPL(r, data, T)  \
                        T error_impl(T) const;
                    BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_ERROR_IMPL, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
                    #undef ALPS_ACCUMULATOR_ERROR_IMPL

                public:

                    // mean
                    template<typename T> T mean() const {
                        return mean_impl(T());
                    }

                    // error
                    template<typename T> T error() const {
                        return mean_impl(T());
                    }

                    // save
                    void save(hdf5::archive & ar) const;

                    // load
                    void load(hdf5::archive & ar);

                    // print
                    void print(std::ostream & os) const;

                private:

                    std::ptrdiff_t * m_cnt;
                    result_wrapper * m_ptr;
            };

            class virtual_accumulator_wrapper {
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

                    // operator(), operator<<
                    #define ALPS_ACCUMULATOR_OPERATOR_CALL(r, data, T)              \
                        virtual_accumulator_wrapper & operator()(T const & value);  \
                        virtual_accumulator_wrapper & operator<<(T const & value) { \
                            (*this)(value);                                         \
                            return (*this);                                         \
                        }
                    BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_OPERATOR_CALL, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
                    #undef ALPS_ACCUMULATOR_OPERATOR_CALL

                    /// Merge another accumulator into this one. @param rhs  accumulator to merge.
                    void merge(const virtual_accumulator_wrapper & rhs);

                    virtual_accumulator_wrapper & operator=(boost::shared_ptr<virtual_accumulator_wrapper> const & rhs);

                    // count
                    boost::uint64_t count() const;

                private:

                    #define ALPS_ACCUMULATOR_MEAN_IMPL(r, data, T)  \
                        T mean_impl(T) const;
                    BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_MEAN_IMPL, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
                    #undef ALPS_ACCUMULATOR_MEAN_IMPL

                    #define ALPS_ACCUMULATOR_ERROR_IMPL(r, data, T)  \
                        T error_impl(T) const;
                    BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_ERROR_IMPL, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
                    #undef ALPS_ACCUMULATOR_ERROR_IMPL

                public:

                    // mean
                    template<typename T> T mean() const {
                        return mean_impl(T());
                    }

                    // error
                    template<typename T> T error() const {
                        return mean_impl(T());
                    }

                    // save
                    void save(hdf5::archive & ar) const;

                    // load
                    void load(hdf5::archive & ar);

                    // reset
                    void reset() const;

                    // result
                    boost::shared_ptr<virtual_result_wrapper<virtual_accumulator_wrapper> > result() const;

                    // print
                    void print(std::ostream & os) const;

#ifdef ALPS_HAVE_MPI
                    // collective_merge
                    void collective_merge(alps::mpi::communicator const & comm, int root);
                    void collective_merge(alps::mpi::communicator const & comm, int root) const;
#endif

                private:

                    std::ptrdiff_t * m_cnt;
                    accumulator_wrapper * m_ptr;
            };

            inline std::ostream & operator<<(std::ostream & os, const virtual_accumulator_wrapper & arg) {
                arg.print(os);
                return os;
            }

            inline void reset(virtual_accumulator_wrapper & arg) {
                return arg.reset();
            }

        }
    }

    typedef accumulators::impl::wrapper_set<accumulators::wrapped::virtual_accumulator_wrapper> accumulator_set;
    typedef accumulators::impl::wrapper_set<accumulators::wrapped::virtual_result_wrapper<accumulators::wrapped::virtual_accumulator_wrapper> > result_set;

    template<typename T> struct MeanAccumulator {
        public:
            MeanAccumulator(std::string const & name): m_name(name) {}
            std::string const & name() const { return m_name; }
        private:
            std::string m_name;
    };

    template<typename T> struct NoBinningAccumulator {
        public:
            NoBinningAccumulator(std::string const & name): m_name(name) {}
            std::string const & name() const { return m_name; }
        private:
            std::string m_name;
    };

    template<typename T> struct LogBinningAccumulator {
        public:
            LogBinningAccumulator(std::string const & name): m_name(name) {}
            std::string const & name() const { return m_name; }
        private:
            std::string m_name;
    };

    template<typename T> struct FullBinningAccumulator {
        public:
            FullBinningAccumulator(std::string const & name): m_name(name) {}
            std::string const & name() const { return m_name; }
        private:
            std::string m_name;
    };

    #define ALPS_ACCUMULATOR_ADD_ACCUMULATOR(r, data, T)                                            \
        accumulator_set & operator<<(accumulator_set & set, const MeanAccumulator< T > & arg);      \
        accumulator_set & operator<<(accumulator_set & set, const NoBinningAccumulator< T > & arg); \
        accumulator_set & operator<<(accumulator_set & set, const LogBinningAccumulator< T > & arg); \
        accumulator_set & operator<<(accumulator_set & set, const FullBinningAccumulator< T > & arg);
    BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_ADD_ACCUMULATOR, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
    #undef ALPS_ACCUMULATOR_ADD_ACCUMULATOR
}

