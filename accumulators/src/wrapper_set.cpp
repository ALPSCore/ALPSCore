/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/accumulators.hpp>

namespace alps {
    namespace accumulators {
        namespace detail {

            void register_predefined_serializable_type() {
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
