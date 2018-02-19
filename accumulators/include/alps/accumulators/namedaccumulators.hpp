/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <alps/accumulators/accumulator.hpp>

namespace alps {
    namespace accumulators {

        namespace detail {

            template<typename A> struct AccumulatorBase {
                typedef A accumulator_type;
                typedef typename A::result_type result_type;

                /// Named-argument constructor: takes `name`, forwards the ArgumentPack to the wrapped accumulator constructor
                template<typename ArgumentPack>
                AccumulatorBase(const ArgumentPack& args,
                                typename std::enable_if<!std::is_base_of<AccumulatorBase,ArgumentPack>::value,int>::type =0)
                    : name(args[accumulator_name])
                    , wrapper(new accumulator_wrapper(A(args)))
                {}

                /// Copy constructor: clones the wrapped accumulator
                template<typename ArgumentPack>
                AccumulatorBase(const ArgumentPack& rhs,
                                typename std::enable_if<std::is_base_of<AccumulatorBase,ArgumentPack>::value,int>::type =0)
                    : name(rhs.name),
                      wrapper(boost::shared_ptr<accumulator_wrapper>(rhs.wrapper->new_clone()))
                { }

                /// Adds value directly to this named accumulator
                template <typename T>
                const AccumulatorBase& operator<<(const T& value) const
                {
                    (*wrapper) << value;
                    return *this;
                }

                /// Returns a shared pointer to the result associated with this named accumulator
                boost::shared_ptr<result_wrapper> result() const
                {
                    return wrapper->result();
                }

                /// Assignment operator: duplicates the wrapped accumulator.
                AccumulatorBase& operator=(const AccumulatorBase& rhs)
                {
                    // Self-assignment is handled correctly (albeit inefficiently)
                    this->name=rhs.name;
                    this->wrapper = boost::shared_ptr<accumulator_wrapper>(rhs.wrapper->new_clone());
                    return *this;
                }

#ifdef ALPS_HAVE_MPI
                /// Collective MPI merge.
                // FIXME! TODO: Introduce a proper "interface"-like superclass? with all relevant accumulator methods?
                void collective_merge(alps::mpi::communicator const & comm, int root)
                {
                    this->wrapper->collective_merge(comm,root);
                }
#endif

                std::string name;
                boost::shared_ptr<accumulator_wrapper> wrapper;
            }; // end struct AccumulatorBase
        } // end namespace detail


        template<typename T> struct MeanAccumulator : public detail::AccumulatorBase<
            impl::Accumulator<T, mean_tag, impl::Accumulator<T, count_tag, impl::AccumulatorBase<T> > >
        > {
            typedef typename impl::Accumulator<T, mean_tag, impl::Accumulator<T, count_tag, impl::AccumulatorBase<T> > > accumulator_type;
            typedef typename accumulator_type::result_type result_type;
            BOOST_PARAMETER_CONSTRUCTOR(
                MeanAccumulator,
                (detail::AccumulatorBase<accumulator_type>),
                accumulator_keywords,
                    (required (_accumulator_name, (std::string)))
            )

            MeanAccumulator& operator=(const MeanAccumulator& rhs);
            MeanAccumulator(const MeanAccumulator& rhs);
        };

        template<typename T> struct NoBinningAccumulator : public detail::AccumulatorBase<
            typename impl::Accumulator<T, error_tag, typename MeanAccumulator<T>::accumulator_type>
        > {
            typedef typename impl::Accumulator<T, error_tag, typename MeanAccumulator<T>::accumulator_type> accumulator_type;
            typedef typename accumulator_type::result_type result_type;
            BOOST_PARAMETER_CONSTRUCTOR(
                NoBinningAccumulator,
                (detail::AccumulatorBase<accumulator_type>),
                accumulator_keywords,
                    (required (_accumulator_name, (std::string)))
            )
            NoBinningAccumulator& operator=(const NoBinningAccumulator& rhs);
            NoBinningAccumulator(const NoBinningAccumulator& rhs);
        };

        template<typename T> struct LogBinningAccumulator : public detail::AccumulatorBase<
            typename impl::Accumulator<T, binning_analysis_tag, typename NoBinningAccumulator<T>::accumulator_type>
        > {
            typedef typename impl::Accumulator<T, binning_analysis_tag, typename NoBinningAccumulator<T>::accumulator_type> accumulator_type;
            typedef typename accumulator_type::result_type result_type;
            BOOST_PARAMETER_CONSTRUCTOR(
                LogBinningAccumulator,
                (detail::AccumulatorBase<accumulator_type>),
                accumulator_keywords,
                    (required (_accumulator_name, (std::string)))
            )
            LogBinningAccumulator& operator=(const LogBinningAccumulator& rhs);
            LogBinningAccumulator(const LogBinningAccumulator& rhs);
            /// Data type corresponding to autocorrelation
            typedef typename autocorrelation_type<accumulator_type>::type autocorrelation_type;
            /// Returns autocorrelation for this accumulator.
            autocorrelation_type tau() const;
        };

        template<typename T> struct FullBinningAccumulator : public detail::AccumulatorBase<
            typename impl::Accumulator<T, max_num_binning_tag, typename LogBinningAccumulator<T>::accumulator_type>
        > {
            typedef typename impl::Accumulator<T, max_num_binning_tag, typename LogBinningAccumulator<T>::accumulator_type> accumulator_type;
            typedef typename accumulator_type::result_type result_type;
            BOOST_PARAMETER_CONSTRUCTOR(
                FullBinningAccumulator,
                (detail::AccumulatorBase<accumulator_type>),
                accumulator_keywords,
                    (required (_accumulator_name, (std::string)))
                    (optional
                        (_max_bin_number, (std::size_t))
                    )
            )
            FullBinningAccumulator& operator=(const FullBinningAccumulator& rhs);
            FullBinningAccumulator(const FullBinningAccumulator& rhs);

            /// Data type corresponding to autocorrelation
            typedef typename autocorrelation_type<accumulator_type>::type autocorrelation_type;
            /// Returns autocorrelation for this accumulator.
            autocorrelation_type tau() const;
        };

        #define ALPS_ACCUMULATOR_REGISTER_OPERATOR(A)                                                               \
            template<typename T> accumulator_set & operator<<(accumulator_set & set, const A <T> & arg);

        ALPS_ACCUMULATOR_REGISTER_OPERATOR(MeanAccumulator)
        ALPS_ACCUMULATOR_REGISTER_OPERATOR(NoBinningAccumulator)
        ALPS_ACCUMULATOR_REGISTER_OPERATOR(LogBinningAccumulator)
        ALPS_ACCUMULATOR_REGISTER_OPERATOR(FullBinningAccumulator)
        #undef ALPS_ACCUMULATOR_REGISTER_OPERATOR

    }
}

