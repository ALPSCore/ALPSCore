/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NGS_ALEA_DETAIL_ERROR_HPP
#define ALPS_NGS_ALEA_DETAIL_ERROR_HPP

#include <alps/ngs/alea/feature/mean.hpp>
#include <alps/ngs/alea/feature/feature_traits.hpp>
#include <alps/ngs/alea/feature/generate_property.hpp>

#include <alps/ngs/short_print.hpp>
#include <alps/ngs/numeric/array.hpp>
#include <alps/ngs/numeric/detail.hpp>
#include <alps/ngs/numeric/vector.hpp>

#include <alps/multi_array.hpp>

#ifdef ALPS_HAVE_MPI
    #include <alps/ngs/boost_mpi.hpp>
#endif

#include <cmath>
namespace alps {
    namespace accumulator {
        //=================== error proxy ===================
        //=================== error trait ===================
        template <typename T>
        struct error_type
        {
            typedef typename mean_type<T>::type type;
        };
        //=================== error implementation ===================
        namespace detail
        {
            //set up the dependencies for the tag::error-Implementation
            template<> 
            struct Dependencies<tag::error> 
            {
                typedef MakeList<tag::mean>::type type;
            };

            template<typename base_type> 
            class AccumulatorImplementation<tag::error, base_type> : public base_type 
            {
                typedef typename base_type::value_type value_type_loc;
				typedef typename alps::accumulator::error_type<value_type_loc>::type error_type;
                typedef AccumulatorImplementation<tag::error, base_type> ThisType;
                
                public:

                    AccumulatorImplementation<tag::error, base_type>(ThisType const & arg): base_type(arg), sum2_(arg.sum2_) {}
                    
                    template<typename ArgumentPack> AccumulatorImplementation<tag::error, base_type>(
                          ArgumentPack const & args
                        , typename boost::disable_if<boost::is_base_of<ThisType, ArgumentPack>, int>::type = 0
                    )
                        : base_type(args)
                        , sum2_() 
                    {}
                    
                    inline error_type const error() const {
                        using alps::ngs::numeric::sqrt;
                        using std::sqrt;
                        using alps::ngs::numeric::operator/;
                        using alps::ngs::numeric::operator-;
                        using alps::ngs::numeric::operator*;

                        return sqrt((sum2_ / (typename alps::hdf5::scalar_type<value_type_loc>::type)base_type::count() - base_type::mean() * base_type::mean()) 
                            / ((typename alps::hdf5::scalar_type<value_type_loc>::type)base_type::count() - 1));
                    }
                    
                    inline void operator ()(value_type_loc const & val) {
                        using alps::ngs::numeric::operator*;
                        using alps::ngs::numeric::operator+=;
                        using alps::ngs::numeric::detail::check_size;
                        
                        check_size(sum2_, val);
                        base_type::operator()(val);
                        sum2_ += val * val;
                    }

                    inline ThisType& operator <<(value_type_loc const & val) {
                        (*this)(val);
                        return (*this);
                    }
                    
                    template<typename Stream> inline void print(Stream & os) {
                        base_type::print(os);
                        os << "tag::error: " << alps::short_print(error()) << " " << std::endl;
                    }

                    void save(hdf5::archive & ar) const {
                        base_type::save(ar);
                        ar["mean/error"] = error();
                    }

                    void load(hdf5::archive & ar) {
                        using alps::ngs::numeric::operator*;
                        using alps::ngs::numeric::operator+;

                        base_type::load(ar);
                        error_type error;
                        ar["mean/error"] >> error;
                        sum2_ = (
                              error * error * (typename alps::hdf5::scalar_type<error_type>::type)(base_type::count() - 1) 
                            + base_type::mean() * base_type::mean()
                        ) * (typename alps::hdf5::scalar_type<error_type>::type)base_type::count();
                    }

                    inline void reset() {
                        sum2_ = error_type();
                        base_type::reset();
                    }

#ifdef ALPS_HAVE_MPI
                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) {
                        base_type::collective_merge(comm, root);
                        if (comm.rank() == root)
                            base_type::reduce_if(comm, sum2_, sum2_, std::plus<typename alps::hdf5::scalar_type<error_type>::type>(), root);
                        else
                            const_cast<ThisType const *>(this)->collective_merge(comm, root);
                    }

                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) const {
                        base_type::collective_merge(comm, root);
                        if (comm.rank() == root)
                            throw std::runtime_error("A const object cannot be root" + ALPS_STACKTRACE);
                        else
                            base_type::reduce_if(comm, sum2_, std::plus<typename alps::hdf5::scalar_type<error_type>::type>(), root);
                    }
#endif

                private:
                    error_type sum2_;
            };

            template<typename base_type> class ResultImplementation<tag::error, base_type> : public base_type {

                typedef typename error_type<typename base_type::value_type>::type error_type;

                public:

                    template<typename Accumulator> ResultImplementation(Accumulator const & accum)
                        : base_type(accum)
                        , error_(accum.error())
                    {}

                    inline error_type const error() const { 
                        return error_;
                    }
// TODO: implement!
                protected:
                    error_type error_;
            };            

        }

        //=================== call GENERATE_PROPERTY macro ===================
        GENERATE_PROPERTY(error, tag::error)

    }
}
#endif
