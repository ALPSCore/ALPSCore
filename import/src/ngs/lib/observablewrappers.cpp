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

#include <alps/ngs/observablewrappers.hpp>
// #ifdef ALPS_NGS_USE_NEW_ALEA
//     #include <alps/ngs/alea.hpp>
// #endif

namespace alps {

    namespace ngs {

        namespace detail {

            std::string ObservableWapper::getName() const {
                return _name;
            }

            uint32_t ObservableWapper::getBinnum() const {
                return _binnum;
            }

            std::string SignedObservableWapper::getSign() const {
                return _sign;
            }

        }
        
        //TODO
        alps::mcobservables & operator<< (alps::mcobservables & set, RealObservable const & obs) {
            set.create_RealObservable(obs.getName(), obs.getBinnum());
            return set;
        }

        // #ifdef ALPS_NGS_USE_NEW_ALEA
        //     alps::accumulator::accumulator_set & operator<< (alps::accumulator::accumulator_set & set, RealObservable const & obs) {
        //         using namespace alps::accumulator::tag;
                
        //         typedef accumulator::accumulator<double, accumulator::features<mean, error, max_num_binning> > accum_type;
        //         typedef accumulator::detail::accumulator_wrapper wrapper_type;
                
        //         set.insert(obs.getName(), boost::shared_ptr<wrapper_type>(new wrapper_type(accum_type(accumulator::bin_num = obs.getBinnum()))));

        //         return set;
        //     }
            
        //     alps::accumulator::accumulator_set & operator<< (alps::accumulator::accumulator_set & set, RealVectorObservable const & obs) {
        //         using namespace alps::accumulator::tag;
                
        //         typedef accumulator::accumulator<std::vector<double>, accumulator::features<mean, error, max_num_binning> > accum_type;
        //         typedef accumulator::detail::accumulator_wrapper wrapper_type;
                
        //         set.insert(obs.getName(), boost::shared_ptr<wrapper_type>(new wrapper_type(accum_type(accumulator::bin_num = obs.getBinnum()))));

        //         return set;
        //     }
            
        //     alps::accumulator::accumulator_set & operator<< (alps::accumulator::accumulator_set & set, SimpleRealObservable const & obs) {
        //         using namespace alps::accumulator::tag;
                
        //         typedef accumulator::accumulator<double, accumulator::features<mean, error> > accum_type;
        //         typedef accumulator::detail::accumulator_wrapper wrapper_type;
                
        //         set.insert(obs.getName(), boost::shared_ptr<wrapper_type>(new wrapper_type(accum_type(accumulator::bin_num = obs.getBinnum()))));
                
        //         return set;
        //     }
            
        //     alps::accumulator::accumulator_set & operator<< (alps::accumulator::accumulator_set & set, SimpleRealVectorObservable const & obs) {
        //         using namespace alps::accumulator::tag;
                
        //         typedef accumulator::accumulator<std::vector<double>, accumulator::features<mean, error> > accum_type;
        //         typedef accumulator::detail::accumulator_wrapper wrapper_type;
                
        //         set.insert(obs.getName(), boost::shared_ptr<wrapper_type>(new wrapper_type(accum_type(accumulator::bin_num = obs.getBinnum()))));
                
        //         return set;
        //     }
        // #endif        

        alps::mcobservables & operator<< (alps::mcobservables & set, RealVectorObservable const & obs) {
            set.create_RealVectorObservable(obs.getName(), obs.getBinnum());
            return set;
        }

        alps::mcobservables & operator<< (alps::mcobservables & set, SimpleRealObservable const & obs) {
            set.create_SimpleRealObservable(obs.getName());
            return set;
        }

        alps::mcobservables & operator<< (alps::mcobservables & set, SimpleRealVectorObservable const & obs) {
            set.create_SimpleRealVectorObservable(obs.getName());
            return set;
        }

        alps::mcobservables & operator<< (alps::mcobservables & set, SignedRealObservable const & obs) {
            set.create_SignedRealObservable(obs.getName(), obs.getSign(), obs.getBinnum());
            return set;
        }

        alps::mcobservables & operator<< (alps::mcobservables & set, SignedRealVectorObservable const & obs) {
            set.create_SignedRealVectorObservable(obs.getName(), obs.getSign(), obs.getBinnum());
            return set;
        }

        alps::mcobservables & operator<< (alps::mcobservables & set, SignedSimpleRealObservable const & obs) {
            set.create_SignedSimpleRealObservable(obs.getName(), obs.getSign());
            return set;
        }

        alps::mcobservables & operator<< (alps::mcobservables & set, SignedSimpleRealVectorObservable const & obs) {
            set.create_SignedSimpleRealVectorObservable(obs.getName(), obs.getSign());
            return set;
        }

        alps::mcobservables & operator<< (alps::mcobservables & set, RealTimeSeriesObservable const & obs) {
            set.create_RealTimeSeriesObservable(obs.getName());
            return set;
        }

        alps::mcobservables & operator<< (alps::mcobservables & set, RealVectorTimeSeriesObservable const & obs) {
          set.create_RealTimeSeriesObservable(obs.getName());
          return set;
        }
    };

}
