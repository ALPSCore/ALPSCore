/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_ACCUMULATOR_NAMEDACCUMULATOR_HPP
#define ALPS_ACCUMULATOR_NAMEDACCUMULATOR_HPP

#include <alps/accumulators/accumulator.hpp>

namespace alps {
    namespace accumulators {

        namespace detail {

            template<typename T> struct AccumulatorBase {
                typedef T accumulator_type;
                typedef typename T::result_type result_type;

                template<typename ArgumentPack> AccumulatorBase(ArgumentPack const& args) 
                    : name(args[accumulator_name])
                    , wrapper(new accumulator_wrapper(T(args)))
                {}

                std::string name;
                boost::shared_ptr<accumulator_wrapper> wrapper;
            };

        }

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

        }; 

        #define ALPS_ACCUMULATOR_REGISTER_OPERATOR(A)                                                               \
            template<typename T> inline accumulator_set & operator<<(accumulator_set & set, const A <T> & arg) {    \
                set.insert(arg.name, arg.wrapper);                                                                  \
                return set;                                                                                         \
            }

        ALPS_ACCUMULATOR_REGISTER_OPERATOR(MeanAccumulator)
        ALPS_ACCUMULATOR_REGISTER_OPERATOR(NoBinningAccumulator)
        ALPS_ACCUMULATOR_REGISTER_OPERATOR(LogBinningAccumulator)
        ALPS_ACCUMULATOR_REGISTER_OPERATOR(FullBinningAccumulator)
        #undef ALPS_ACCUMULATOR_REGISTER_OPERATOR

        namespace detail {

            inline void register_predefined_serializable_type() {
                #define ALPS_ACCUMULATOR_REGISTER_ACCUMULATOR(A)                                                    \
                    accumulator_set::register_serializable_type<A::accumulator_type>(true);                         \
                    result_set::register_serializable_type<A::result_type>(true);

                #define ALPS_ACCUMULATOR_REGISTER_TYPE(T)                                                           \
                    ALPS_ACCUMULATOR_REGISTER_ACCUMULATOR(MeanAccumulator<T>)                                       \
                    ALPS_ACCUMULATOR_REGISTER_ACCUMULATOR(NoBinningAccumulator<T>)                                  \
                    ALPS_ACCUMULATOR_REGISTER_ACCUMULATOR(LogBinningAccumulator<T>)                                 \
                    ALPS_ACCUMULATOR_REGISTER_ACCUMULATOR(FullBinningAccumulator<T>)

                // TODO: ues ALPS_ACCUMULATOR_VALUE_TYPES and iterate over it
                ALPS_ACCUMULATOR_REGISTER_TYPE(float)
                ALPS_ACCUMULATOR_REGISTER_TYPE(double)
                ALPS_ACCUMULATOR_REGISTER_TYPE(long double)
                ALPS_ACCUMULATOR_REGISTER_TYPE(std::vector<float>)
                ALPS_ACCUMULATOR_REGISTER_TYPE(std::vector<double>)
                ALPS_ACCUMULATOR_REGISTER_TYPE(std::vector<long double>)

                #undef ALPS_ACCUMULATOR_REGISTER_TYPE
                #undef ALPS_ACCUMULATOR_REGISTER_ACCUMULATOR
            }
        }
    }
}

 #endif
