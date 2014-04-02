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

#ifndef ALPS_NGS_OBSERVABLEWRAPPERS_HPP
#define ALPS_NGS_OBSERVABLEWRAPPERS_HPP

#include <alps/ngs/mcobservables.hpp>
// #ifdef ALPS_NGS_USE_NEW_ALEA
//     #include <alps/ngs/alea/accumulator_set.hpp>
// #endif

#include <string>

namespace alps {

    namespace ngs {

        namespace detail {

            class ObservableWapper {
                public:

                    ObservableWapper(std::string const & name, uint32_t binnum = 0): _name(name), _binnum(binnum) {}
                    std::string getName() const;
                    uint32_t getBinnum() const;

                protected:

                    std::string _name;
                    uint32_t _binnum;
            };


            class SignedObservableWapper : public ObservableWapper {
                public:

                    SignedObservableWapper(std::string const & name, std::string const & sign, uint32_t binnum = 0): ObservableWapper(name, binnum), _sign(sign) {}
                    std::string getSign() const;

                private:

                    std::string _sign;
            };

        }

        class ALPS_DECL RealObservable : public detail::ObservableWapper {
            public:
                RealObservable(std::string const & name, uint32_t binnum = 0): ObservableWapper(name,binnum) {}
        };

        // #ifdef ALPS_NGS_USE_NEW_ALEA
        //     ALPS_DECL alps::accumulator::accumulator_set & operator<< (alps::accumulator::accumulator_set & set, RealObservable const & obs);
        // #endif
        ALPS_DECL alps::mcobservables & operator<< (alps::mcobservables & set, RealObservable const & obs);

        class ALPS_DECL RealVectorObservable : public detail::ObservableWapper {
            public:
                RealVectorObservable(std::string const & name, uint32_t binnum = 0): ObservableWapper(name,binnum) {}
        };

        // #ifdef ALPS_NGS_USE_NEW_ALEA
        //     ALPS_DECL alps::accumulator::accumulator_set & operator<< (alps::accumulator::accumulator_set & set, RealVectorObservable const & obs);
        // #endif
        ALPS_DECL alps::mcobservables & operator<< (alps::mcobservables & set, RealVectorObservable const & obs);

        class ALPS_DECL SimpleRealObservable : public detail::ObservableWapper {
            public:
                SimpleRealObservable(std::string const & name): ObservableWapper(name) {}
        };
        
        // #ifdef ALPS_NGS_USE_NEW_ALEA
        //     ALPS_DECL alps::accumulator::accumulator_set & operator<< (alps::accumulator::accumulator_set & set, SimpleRealObservable const & obs);
        // #endif
        ALPS_DECL alps::mcobservables & operator<< (alps::mcobservables & set, SimpleRealObservable const & obs);


        class ALPS_DECL SimpleRealVectorObservable : public detail::ObservableWapper {
            public:
                SimpleRealVectorObservable(std::string const & name): ObservableWapper(name) {}
        };

        // #ifdef ALPS_NGS_USE_NEW_ALEA
        //     ALPS_DECL alps::accumulator::accumulator_set & operator<< (alps::accumulator::accumulator_set & set, SimpleRealVectorObservable const & obs);
        // #endif
        ALPS_DECL alps::mcobservables & operator<< (alps::mcobservables & set, SimpleRealVectorObservable const & obs);

        class ALPS_DECL SignedRealObservable : public detail::SignedObservableWapper {
            public:
                SignedRealObservable(std::string const & name, std::string const & sign = "Sign", uint32_t binnum = 0): SignedObservableWapper(name, sign, binnum) {}
        };

        ALPS_DECL alps::mcobservables & operator<< (alps::mcobservables & set, SignedRealObservable const & obs);

        class ALPS_DECL SignedRealVectorObservable : public detail::SignedObservableWapper {
            public:
                SignedRealVectorObservable(std::string const & name, std::string const & sign = "Sign", uint32_t binnum = 0): SignedObservableWapper(name, sign, binnum) {}
        };

        ALPS_DECL alps::mcobservables & operator<< (alps::mcobservables & set, SignedRealVectorObservable const & obs);

        class ALPS_DECL SignedSimpleRealObservable : public detail::SignedObservableWapper {
            public:
                SignedSimpleRealObservable(std::string const & name, std::string const & sign = "Sign"): SignedObservableWapper(name, sign) {}
        };

        ALPS_DECL alps::mcobservables & operator<< (alps::mcobservables & set, SignedSimpleRealObservable const & obs);

        class ALPS_DECL SignedSimpleRealVectorObservable : public detail::SignedObservableWapper {
            public:
                SignedSimpleRealVectorObservable(std::string const & name, std::string const & sign = "Sign"): SignedObservableWapper(name, sign) {}
        };

        ALPS_DECL alps::mcobservables & operator<< (alps::mcobservables & set, SignedSimpleRealVectorObservable const & obs);

        class ALPS_DECL RealTimeSeriesObservable : public detail::ObservableWapper {
            public:
                RealTimeSeriesObservable(std::string const & name): ObservableWapper(name) {}
        };

        ALPS_DECL alps::mcobservables & operator<< (alps::mcobservables & set, RealTimeSeriesObservable const & obs);

        class ALPS_DECL RealVectorTimeSeriesObservable : public detail::ObservableWapper {
          public:
          RealVectorTimeSeriesObservable(std::string const & name): ObservableWapper(name) {}
        };
      
        ALPS_DECL alps::mcobservables & operator<< (alps::mcobservables & set, RealVectorTimeSeriesObservable const & obs);

    };

}

#endif
