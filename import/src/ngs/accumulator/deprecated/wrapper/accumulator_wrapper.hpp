/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NGS_ALEA_ACCUMULATOR_WRAPPER_HPP
#define ALPS_NGS_ALEA_ACCUMULATOR_WRAPPER_HPP

#include <alps/ngs/stacktrace.hpp>
#include <alps/ngs/alea/accumulator.hpp>
#include <alps/ngs/alea/wrapper/base_wrapper.hpp>
#include <alps/ngs/alea/wrapper/result_wrapper.hpp>
#include <alps/ngs/alea/wrapper/derived_wrapper.hpp>
#include <alps/ngs/alea/wrapper/result_type_wrapper.hpp>
 
#ifdef ALPS_HAVE_MPI
    #include <alps/ngs/boost_mpi.hpp>
#endif

#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>

#include <typeinfo> //used in add_value
#include <stdexcept>

namespace alps {
    namespace accumulator {
        namespace detail {

            // class that holds the base_accumulator_wrapper pointer
            class accumulator_wrapper {
                public:
                    template<typename T> 
                    accumulator_wrapper(T arg)
                        : base_(new derived_accumulator_wrapper<T>(arg))
                    {}

                    accumulator_wrapper(accumulator_wrapper const & arg)
                        : base_(arg.base_->clone()) 
                    {}

                    accumulator_wrapper(hdf5::archive & ar) {
                        if (ar.is_attribute("@ownsweight"))
                            throw std::runtime_error("Weighted observables are not Implemented" + ALPS_STACKTRACE);
                        else if (ar.is_data("timeseries/data"))
                            create_accumulator<features<tag::max_num_binning> >(ar);
                        else if (ar.is_data("mean/error"))
                            create_accumulator<features<tag::error> >(ar);
                        else if (ar.is_data("mean/value"))
                            create_accumulator<features<tag::mean> >(ar);
                        else
                            throw std::runtime_error("Unable to load accumulator" + ALPS_STACKTRACE);
                    }

                //------------------- normal input -------------------
                    template<typename T> void operator()(T const & value) {
                        (*base_)(value); 
                    }
                    template<typename T> accumulator_wrapper & operator<<(T const & value) {
                        (*this)(value);
                        return (*this);
                    }
                //------------------- input with weight-------------------
                    template<typename T, typename W> 
                    void operator()(T const & value, W const & weight) {
                        (*base_)(value, weight);
                    }

                    template<typename T> result_type_accumulator_wrapper<T> & get() const {
                        return base_->get<T>();
                    }

                    friend std::ostream& operator<<(std::ostream &out, const accumulator_wrapper& wrapper);

                    template <typename T> T & extract() const {
                        return (dynamic_cast<derived_accumulator_wrapper<T> &>(*base_)).accum_;
                    }

                    boost::uint64_t count() const {
                        return base_->count();
                    }

                    void save(hdf5::archive & ar) const {
                        ar[""] = *base_;
                    }

                    void load(hdf5::archive & ar) {
                        ar[""] >> *base_;
                    }

                    inline void reset() {
                        base_->reset();
                    }

                    boost::shared_ptr<result_wrapper> result() const {
                        return boost::shared_ptr<result_wrapper>(new result_wrapper(base_->result()));
                    }

#ifdef ALPS_HAVE_MPI
                    inline void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) {
                        base_->collective_merge(comm, root);
                    }

                    inline void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) const {
                        base_->collective_merge(comm, root);
                    }
#endif
                private:

                    template <typename Features> void create_accumulator(hdf5::archive & ar) {
                    // TODO: implement other types ...
                    #define ALPS_NGS_ACCUMULATOR_WRAPPER_FOREACH_NATIVE_TYPE(CALLBACK)      \
/*                        CALLBACK(char)                                                      \
                        CALLBACK(signed char)                                               \
                        CALLBACK(unsigned char)                                             \
                        CALLBACK(short)                                                     \
                        CALLBACK(unsigned short)                                            \
                        CALLBACK(int)                                                       \
                        CALLBACK(unsigned)                                                  \
                        CALLBACK(long)                                                      \
                        CALLBACK(unsigned long)                                             \
                        CALLBACK(long long)                                                 \
                        CALLBACK(unsigned long long)                                        \
                        CALLBACK(float)                                                     \
*/                        CALLBACK(double)                                                    \
                        CALLBACK(long double)                                               \
/*                        CALLBACK(bool) 
*/

                        if (ar.is_attribute("@valuetype")) {
                            #define ALPS_NGS_ACCUMULATOR_WRAPPER_FIND_TYPE_VALUETYPE(T)     \
                                if (ar.is_datatype<T>("@valuetype"))                        \
                                    create_accumulator_typed<T, Features>(ar);
                            ALPS_NGS_ACCUMULATOR_WRAPPER_FOREACH_NATIVE_TYPE(ALPS_NGS_ACCUMULATOR_WRAPPER_FIND_TYPE_VALUETYPE)
                            #undef ALPS_NGS_ACCUMULATOR_WRAPPER_FIND_TYPE_VALUETYPE
                        } else {
                            #define ALPS_NGS_ACCUMULATOR_WRAPPER_FIND_TYPE_MEAN_VALUE(T)    \
                                if (ar.is_datatype<T>("mean/value"))                        \
                                    create_accumulator_typed<T, Features>(ar);
                            ALPS_NGS_ACCUMULATOR_WRAPPER_FOREACH_NATIVE_TYPE(ALPS_NGS_ACCUMULATOR_WRAPPER_FIND_TYPE_MEAN_VALUE)
                            #undef ALPS_NGS_ACCUMULATOR_WRAPPER_FIND_TYPE_MEAN_VALUE
                            #undef ALPS_NGS_ACCUMULATOR_WRAPPER_FOREACH_NATIVE_TYPE
                        }
                        throw std::runtime_error("Unknown HDF5 datatype" + ALPS_STACKTRACE);
                    }

                    template <typename T, typename Features> void create_accumulator_typed(hdf5::archive & ar) {
                        boost::shared_ptr<base_accumulator_wrapper> base_;
                        if (ar.is_attribute("@valuetype") ? ar.is_scalar("@valuetype") : ar.is_scalar("mean/value"))
                            base_ = boost::shared_ptr<detail::base_accumulator_wrapper>(new detail::derived_accumulator_wrapper<accumulator<T, Features> >);
                        else
                            switch (ar.is_attribute("@valuetype") ? ar.dimensions("@valuetype") : ar.dimensions("mean/value")) {
                                case 1:
                                    base_ = boost::shared_ptr<detail::base_accumulator_wrapper>(
                                        new detail::derived_accumulator_wrapper<accumulator<std::vector<T>, Features> >
                                    );
                                    break;
                                case 2:
                                    base_ = boost::shared_ptr<detail::base_accumulator_wrapper>(
                                        new detail::derived_accumulator_wrapper<accumulator<alps::multi_array<T, 2>, Features> >
                                    );
                                    break;
                                case 3:
                                    base_ = boost::shared_ptr<detail::base_accumulator_wrapper>(
                                        new detail::derived_accumulator_wrapper<accumulator<alps::multi_array<T, 3>, Features> >
                                    );
                                    break;
                                case 4:
                                    base_ = boost::shared_ptr<detail::base_accumulator_wrapper>(
                                        new detail::derived_accumulator_wrapper<accumulator<alps::multi_array<T, 4>, Features> >
                                    );
                                    break;
                                case 5:
                                    base_ = boost::shared_ptr<detail::base_accumulator_wrapper>(
                                        new detail::derived_accumulator_wrapper<accumulator<alps::multi_array<T, 5>, Features> >
                                    );
                                    break;
                                default:
                                    throw std::runtime_error("More than 5 dimensional value types are not Implemented" + ALPS_STACKTRACE);
                            }
                    }

                    boost::shared_ptr<base_accumulator_wrapper> base_;
            };

            inline std::ostream & operator<<(std::ostream & out, const accumulator_wrapper & m) {
                m.base_->print(out);
                return out;
            }
        }

        template <typename Accum> Accum & extract(detail::accumulator_wrapper & m) {
            return m.extract<Accum>();
        }

        template <typename Accum> inline boost::uint64_t count(Accum const & arg) {
            return arg.count();
        }

        template <typename Accum> void reset(Accum & arg) {
            return arg.reset();
        }
    }
}
#endif
