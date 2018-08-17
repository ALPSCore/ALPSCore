/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <boost/preprocessor/tuple/to_seq.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#include <alps/config.hpp>

#include <alps/accumulators/namedaccumulators.hpp>

#define ALPS_ACCUMULATOR_VALUE_TYPES_SEQ BOOST_PP_TUPLE_TO_SEQ(ALPS_ACCUMULATOR_VALUE_TYPES_SIZE, (ALPS_ACCUMULATOR_VALUE_TYPES))

namespace alps {
    namespace accumulators {

        //
        // MeanAccumulator
        //

        template<typename T>
        MeanAccumulator<T>& MeanAccumulator<T>::operator=(const MeanAccumulator& rhs)
        {
            return static_cast<MeanAccumulator&>(*static_cast<base_type*>(this)=rhs);
        }

        template<typename T>
        MeanAccumulator<T>::MeanAccumulator(const MeanAccumulator& rhs) :
            detail::AccumulatorBase<accumulator_type>(rhs) {}

        #define ALPS_ACCUMULATOR_INST_MEAN_ACCUMULATOR(r, data, T) \
            template struct MeanAccumulator<T>;
        BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_INST_MEAN_ACCUMULATOR, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)

        //
        // NoBinningAccumulator
        //

        template<typename T>
        NoBinningAccumulator<T>& NoBinningAccumulator<T>::operator=(const NoBinningAccumulator& rhs)
        {
            return static_cast<NoBinningAccumulator&>(*static_cast<base_type*>(this)=rhs);
        }

        template<typename T>
        NoBinningAccumulator<T>::NoBinningAccumulator(const NoBinningAccumulator& rhs) :
            detail::AccumulatorBase<accumulator_type>(rhs) {}

        #define ALPS_ACCUMULATOR_INST_NO_BINNING_ACCUMULATOR(r, data, T) \
            template struct NoBinningAccumulator<T>;
        BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_INST_NO_BINNING_ACCUMULATOR, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)

        //
        // LogBinningAccumulator
        //

        template<typename T>
        LogBinningAccumulator<T>& LogBinningAccumulator<T>::operator=(const LogBinningAccumulator& rhs)
        {
                return static_cast<LogBinningAccumulator&>(*static_cast<base_type*>(this)=rhs);
        }

        template<typename T>
        LogBinningAccumulator<T>::LogBinningAccumulator(const LogBinningAccumulator& rhs) :
            detail::AccumulatorBase<accumulator_type>(rhs) {}

        template<typename T>
        auto LogBinningAccumulator<T>::tau() const -> autocorrelation_type
        {
            return this->wrapper->template extract<accumulator_type>().autocorrelation();
        }

        #define ALPS_ACCUMULATOR_INST_LOG_BINNING_ACCUMULATOR(r, data, T) \
            template struct LogBinningAccumulator<T>;
        BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_INST_LOG_BINNING_ACCUMULATOR, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)

        //
        // FullBinningAccumulator
        //

        template<typename T>
        FullBinningAccumulator<T>& FullBinningAccumulator<T>::operator=(const FullBinningAccumulator& rhs)
        {
            return static_cast<FullBinningAccumulator&>(*static_cast<base_type*>(this)=rhs);
        }

        template<typename T>
        FullBinningAccumulator<T>::FullBinningAccumulator(const FullBinningAccumulator& rhs) :
            detail::AccumulatorBase<accumulator_type>(rhs) {}

        template<typename T>
        auto FullBinningAccumulator<T>::tau() const -> autocorrelation_type
        {
            return this->wrapper->template extract<accumulator_type>().autocorrelation();
        }

        #define ALPS_ACCUMULATOR_INST_FULL_BINNING_ACCUMULATOR(r, data, T) \
            template struct FullBinningAccumulator<T>;
        BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_INST_FULL_BINNING_ACCUMULATOR, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)

        //
        // operator<<
        //

        #define ALPS_ACCUMULATOR_DEFINE_OPERATOR(A)                                                          \
            template<typename T> accumulator_set & operator<<(accumulator_set & set, const A <T> & arg) {    \
                set.insert(arg.name, arg.wrapper);                                                           \
                return set;                                                                                  \
            }
        ALPS_ACCUMULATOR_DEFINE_OPERATOR(MeanAccumulator)
        ALPS_ACCUMULATOR_DEFINE_OPERATOR(NoBinningAccumulator)
        ALPS_ACCUMULATOR_DEFINE_OPERATOR(LogBinningAccumulator)
        ALPS_ACCUMULATOR_DEFINE_OPERATOR(FullBinningAccumulator)

        #define ALPS_ACCUMULATOR_INST_OPERATOR_TYPE(r, data, T)                                              \
            template accumulator_set & operator<<(accumulator_set & set, const MeanAccumulator <T> &);       \
            template accumulator_set & operator<<(accumulator_set & set, const NoBinningAccumulator <T> &);  \
            template accumulator_set & operator<<(accumulator_set & set, const LogBinningAccumulator <T> &); \
            template accumulator_set & operator<<(accumulator_set & set, const FullBinningAccumulator <T> &);
        BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_INST_OPERATOR_TYPE, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
    }
}
