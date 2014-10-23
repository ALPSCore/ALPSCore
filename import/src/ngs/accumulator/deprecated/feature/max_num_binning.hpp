/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NGS_ALEA_DETAIL_MAX_NUM_BINNING_HPP
#define ALPS_NGS_ALEA_DETAIL_MAX_NUM_BINNING_HPP

#include <alps/ngs/alea/feature/mean.hpp>
#include <alps/ngs/alea/feature/feature_traits.hpp>
#include <alps/ngs/alea/feature/generate_property.hpp>

#include <alps/ngs/alea/accumulators/arguments.hpp>

#include <vector>
#include <ostream>
#include <cmath>
#include <algorithm>

namespace alps {
    namespace accumulator {
        //=================== max_num_binning proxy ===================
        template<typename value_type> class max_num_binning_proxy_type {
            typedef typename mean_type<value_type>::type mean_type;
            typedef typename std::vector<value_type>::size_type size_type;
        public:
            max_num_binning_proxy_type()
                : bin_(std::vector<mean_type>()) 
            {}
            
            max_num_binning_proxy_type(
                std::vector<mean_type> const & bin , size_type const & bin_number
            )
                : bin_(bin)
                , bin_number_(bin_number)
            {}
            
            inline std::vector<mean_type> const & bins() const {
                return bin_;
            }
            
            inline size_type const & bin_number() const {
                return bin_number_;
            }
            
            template<typename T> friend std::ostream & operator<<(std::ostream & os, max_num_binning_proxy_type<T> const & arg);
        private:
            std::vector<mean_type> const & bin_;
            size_type bin_number_;
        };

        template<typename T> inline std::ostream & operator<<(std::ostream & os, max_num_binning_proxy_type<T> const & arg) {
            os << "max_num_binning_proxy" << std::endl;
            return os;
        };
        //=================== max_num_binning trait ===================
        template <typename T> struct max_num_binning_type {
            typedef max_num_binning_proxy_type<T> type;
        };
        //=================== max_num_binning implementation ===================
        namespace detail {
            //set up the dependencies for the tag::max_num_binning-Implementation
            template<> struct Dependencies<tag::max_num_binning> {
                typedef MakeList<tag::mean, tag::error>::type type;
            };

            template<typename base_type> class AccumulatorImplementation<tag::max_num_binning, base_type> : public base_type {
                typedef typename base_type::value_type value_type_loc;
                typedef typename max_num_binning_type<value_type_loc>::type num_bin_type;
				typedef typename std::size_t size_type;
				typedef typename alps::accumulator::mean_type<value_type_loc>::type mean_type;
                typedef AccumulatorImplementation<tag::max_num_binning, base_type> ThisType;

                public:
                    AccumulatorImplementation<tag::max_num_binning, base_type>(ThisType const & arg)
                        : base_type(arg)
                        , bin_(arg.bin_)
                        , partial_(arg.partial_)
                        , elements_in_bin_(arg.elements_in_bin_)
                        , pos_in_partial_(arg.pos_in_partial_)
                        , max_bin_num_(arg.max_bin_num_) 
                    {}
                    //TODO: set right default value 
                    
                    template<typename ArgumentPack>
                    AccumulatorImplementation<tag::max_num_binning, base_type>(
                          ArgumentPack const & args
                        , typename boost::disable_if<boost::is_base_of<ThisType, ArgumentPack>, int>::type = 0
                    )
                        : base_type(args)
                        , partial_()
                        , elements_in_bin_(1)
                        , pos_in_partial_(0)
                        , max_bin_num_(args[bin_num | 128]) //change doc if manipulated
                    {}
                    
                    inline num_bin_type const max_num_binning() const {
                        return max_num_binning_proxy_type<value_type_loc>(bin_, max_bin_num_);
                    }
              
                    inline void operator()(value_type_loc const & val) {
                        using namespace alps::numeric;
                        using alps::numeric::detail::check_size;
                        
                        base_type::operator()(val);
                        
                        check_size(partial_, val);
                        partial_ += val;
                        ++pos_in_partial_;
                        
                        if(pos_in_partial_ == elements_in_bin_) {
                            if(bin_.size() >= max_bin_num_) {
                                if(max_bin_num_ % 2 == 1) {
                                    partial_ += bin_[max_bin_num_ - 1];
                                    pos_in_partial_ += elements_in_bin_;
                                }
                                
                                for(unsigned int i = 0; i < max_bin_num_ / 2; ++i) //the rounding down here is intentional
                                    bin_[i] = (bin_[2 * i] + bin_[2 * i + 1]) / (typename alps::hdf5::scalar_type<mean_type>::type)2;
                                
                                bin_.erase(bin_.begin() + max_bin_num_ / 2, bin_.end());
                                
                                elements_in_bin_ *= 2;
                                
                                if(pos_in_partial_ == elements_in_bin_) {
                                    bin_.push_back(partial_ / (typename alps::hdf5::scalar_type<value_type_loc>::type)elements_in_bin_);
                                    partial_ = value_type_loc();
                                    pos_in_partial_ = 0;
                                }
                            } else {
                                bin_.push_back(partial_ / (typename alps::hdf5::scalar_type<value_type_loc>::type)elements_in_bin_);
                                partial_ = value_type_loc();
                                pos_in_partial_ = 0;
                            }
                        }
                    }

                    inline ThisType& operator<<(value_type_loc const & val) {
                        (*this)(val);
                        return (*this);
                    }

                    template<typename Stream> inline void print(Stream & os) {
                        base_type::print(os);
                        os << "MaxBinningNumber: MaxBinNumber: " << max_bin_num_ << std::endl;
                        
                        //~ os << std::endl;
                        //~ for (unsigned int i = 0; i < bin_.size(); ++i)
                        //~ {
                            //~ os << "bin[" << i << "] = " << bin_[i] << std::endl;
                        //~ }
                    }

                    void save(hdf5::archive & ar) const {
                        base_type::save(ar);
                        if (base_type::count()) {
                            ar["timeseries/partialbin"] = partial_;
                            ar["timeseries/partialbin/@count"] = pos_in_partial_;
                            ar["timeseries/data"] = bin_;
                            ar["timeseries/data/@binningtype"] = "linear";
                            ar["timeseries/data/@minbinsize"] = elements_in_bin_; // TODO: what should we put here?
                            ar["timeseries/data/@binsize"] = elements_in_bin_;
                            ar["timeseries/data/@maxbinnum"] = max_bin_num_;
                        } else {
                            ar["timeseries/data"] = bin_;
                            ar["timeseries/data/@binningtype"] = "linear";
                            ar["timeseries/data/@minbinsize"] = elements_in_bin_; // TODO: what should we put here?
                            ar["timeseries/data/@binsize"] = elements_in_bin_;
                            ar["timeseries/data/@maxbinnum"] = max_bin_num_;
                        }
                    }

                    void load(hdf5::archive & ar) {
                        base_type::load(ar);
                        ar["timeseries/data"] = bin_;
                        ar["timeseries/data/@binsize"] = elements_in_bin_;
                        ar["timeseries/data/@maxbinnum"] = max_bin_num_;
                        if (ar.is_data("timeseries/partialbin")) {
                            ar["timeseries/partialbin"] = partial_;
                            ar["timeseries/partialbin/@count"] = pos_in_partial_;
                        }
                    }

                    inline void reset() {
                        base_type::reset();
                        bin_.clear();
                        partial_ = value_type_loc();
                        elements_in_bin_ = 0;
                        pos_in_partial_ = 0;
                    }

#ifdef ALPS_HAVE_MPI
                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) {
                        base_type::collective_merge(comm, root);
                        if (comm.rank() == root) {
                            if (!bin_.empty()) {
                                std::vector<mean_type> local_bins(bin_);
                                elements_in_bin_ = partition_bins(comm, local_bins);
                                bin_.resize(local_bins.size());
                                base_type::reduce_if(comm, local_bins, bin_, std::plus<typename alps::hdf5::scalar_type<mean_type>::type>(), 0);
                            }
                        } else
                            const_cast<ThisType const *>(this)->collective_merge(comm, root);
                    }

                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) const {
                        base_type::collective_merge(comm, root);
                        if (comm.rank() == root)
                            throw std::runtime_error("A const object cannot be root" + ALPS_STACKTRACE);
                        else if (!bin_.empty()) {
                            std::vector<mean_type> local_bins(bin_);
                            partition_bins(comm, local_bins);
                            base_type::reduce_if(comm, local_bins, std::plus<typename alps::hdf5::scalar_type<mean_type>::type>(), root);
                        }
                    }

                private:
                    std::size_t partition_bins (boost::mpi::communicator const & comm, std::vector<mean_type> & local_bins) const {
                        using alps::numeric::operator+;
                        using alps::numeric::operator/;
                        boost::uint64_t elements_in_local_bins = boost::mpi::all_reduce(comm, elements_in_bin_, boost::mpi::maximum<boost::uint64_t>());
                        size_type howmany = (elements_in_local_bins - 1) / elements_in_bin_ + 1;
                        if (howmany > 1) {
                            size_type newbins = local_bins.size() / howmany;
                            for (size_type i = 0; i < newbins; ++i) {
                                local_bins[i] = local_bins[howmany * i];
                                for (size_type j = 1; j < howmany; ++j)
                                    local_bins[i] = local_bins[i] + local_bins[howmany * i + j];
                                local_bins[i] = local_bins[i] / (boost::uint64_t)howmany;
                            }
                            local_bins.resize(newbins);
                        }
                        return elements_in_local_bins;
                    }
#endif
                private:
                    std::vector<mean_type> bin_;
                    value_type_loc partial_;
                    size_type elements_in_bin_;
                    size_type pos_in_partial_;
                    size_type const max_bin_num_;
            };

            template<typename base_type> class ResultImplementation<tag::max_num_binning, base_type> : public base_type {

                public:

                    template<typename Accumulator> ResultImplementation(Accumulator const & accum)
                        : base_type(accum)
                    {}
            };

        }

        //=================== call GENERATE_PROPERTY macro ===================
        GENERATE_PROPERTY(max_num_binning, tag::max_num_binning)

    }
}
#endif
