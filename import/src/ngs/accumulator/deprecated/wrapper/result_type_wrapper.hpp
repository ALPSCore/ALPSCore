/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NGS_ALEA_RESULT_TYPE_WRAPPER_HPP
#define ALPS_NGS_ALEA_RESULT_TYPE_WRAPPER_HPP

#include <alps/ngs/alea/features.hpp>
#include <alps/ngs/alea/wrapper/base_wrapper.hpp>

namespace alps {
    namespace accumulator {
        namespace detail {

            template <typename ValueType> class result_type_result_wrapper: public base_result_wrapper {
                public:
                    typedef ValueType value_type;
                    virtual ~result_type_result_wrapper() {}
                    virtual typename mean_type<value_type>::type mean() const = 0;
                    virtual bool has_mean() const = 0;
                    virtual typename error_type<value_type>::type error() const = 0;
                    virtual bool has_error() const = 0;
                    virtual typename tau_type<value_type>::type tau() const = 0;
                    virtual bool has_tau() const = 0;
            };            

            template <typename ValueType> class result_type_accumulator_wrapper: public base_accumulator_wrapper {
                public:
                    typedef ValueType value_type;
                    virtual ~result_type_accumulator_wrapper() {}
                    virtual typename mean_type<value_type>::type mean() const = 0;
                    virtual bool has_mean() const = 0;
                    virtual typename error_type<value_type>::type error() const = 0;
                    virtual bool has_error() const = 0;
                    virtual typename fixed_size_binning_type<value_type>::type fixed_size_binning() const = 0;
                    virtual bool has_fixed_size_binning() const = 0;
                    virtual typename max_num_binning_type<value_type>::type max_num_binning() const = 0;
                    virtual bool has_max_num_binning() const = 0;
                    virtual typename log_binning_type<value_type>::type log_binning() const = 0;
                    virtual bool has_log_binning() const = 0;
                    virtual typename autocorrelation_type<value_type>::type autocorrelation() const = 0;
                    virtual bool has_autocorrelation() const = 0;
                    virtual typename converged_type<value_type>::type converged() const = 0;
                    virtual bool has_converged() const = 0;
                    virtual typename tau_type<value_type>::type tau() const = 0;
                    virtual bool has_tau() const = 0;
                    virtual typename weight_type<value_type>::type weight() const = 0;
                    virtual bool has_weight() const = 0;
                    virtual typename histogram_type<value_type>::type histogram() const = 0;
                    virtual bool has_histogram() const = 0;
            };
        }
    }
}
#endif
