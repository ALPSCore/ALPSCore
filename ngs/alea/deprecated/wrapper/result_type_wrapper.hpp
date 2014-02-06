/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 - 2013 by Mario Koenz <mkoenz@ethz.ch>                       *
 *                              Lukas Gamper <gamperl@gmail.com>                   *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

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
