/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
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

#ifndef ALPS_NGS_MEASUREMENTS_BINNING_HPP
#define ALPS_NGS_MEASUREMENTS_BINNING_HPP

namespace alps {
    namespace masurements {

        template <typename Base> class binning {

            public:

                binning<Base> operator<<(Base::value_type const & x) {
                    Base::operator<<(x);
                    partial_bin_ += x;
                    if (++partial_count_ == bin_size_) {
                        bins_.push(partial_bin_);
                        partial_bin_ = Base::value_type();
                        partial_count_ = 0;
                        if (bins_.size() == max_bin_number_)
                            collect_bins((bins_.size() - 1) / max_bin_number_ + 1);
                    }
                }

                void set_bin_number(std::size_t bin_number) {
                    collect_bins((bins_.size() - 1) / bin_number + 1);
                }

                std::size_t bin_number() {
                    return bins_.size();
                }

                std::size_t max_bin_number() {
                    return max_bin_number_;
                }

                void set_bin_size(unsigned long bin_size) {
                    collect_bins((bin_size - 1) / bin_size_ + 1);
                    bin_size_ = bin_size;
                }

                std::size_t bin_size() {
                    return bin_size_;
                }

                Base::value_type const & bin(std::size_t index) {
                    return bins_[index];
                }

                std::vector<Base::value_type> const & bins() {
                    return bins_;
                }

            private:

                void collect_bins(std::size_t how_many) {
                    using boost::numeric::operators::operator+;
                    using boost::numeric::operators::operator/;
                    if (bins_.empty() || how_many <= 1) 
                        return;
                    std::size_t new_bins = bins_.size() / how_many;
                    for (std::size_t i = 0; i < new_bins; ++i) {
                        bins_[i] = bins_[how_many * i];
                        for (std::size_t j = 1; j < how_many; ++j)
                            bins_[i] = bins_[i] + bins_[how_many * i + j];
                    }
                    bins_.resize(new_bins);
                    bin_size_ *= how_many;
                }

                unsigned long bin_size_;
                unsigned long partial_count_;
                unsigned long max_bin_number_;
                std::vector<Base::value_type> bins_;
                Base::value_type partial_bin_;

        };

    }
}

#endif
