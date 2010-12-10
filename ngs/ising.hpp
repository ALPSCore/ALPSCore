/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2010 by Lukas Gamper <gamperl -at- gmail.com>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

#include <alps/ngs.hpp>

class simulation_type : public alps::mcbase {
    public:
        simulation_type(parameters_type const & params, std::size_t seed_offset = 0)
            : alps::mcbase(params, seed_offset)
            , length(params["L"])
            , beta(1. / double(params["T"]))
            , sweeps(0)
            , thermalization_sweeps(int(params["THERMALIZATION"]))
            , total_sweeps(int(params["SWEEPS"]))
            , spins(length)
        {
            for(int i = 0; i < length; ++i)
                spins[i] = (random() < 0.5 ? 1 : -1);
            results << alps::RealObservable("Unused");
            results << alps::SimpleRealObservable("EnergySimple");
            results << alps::RealObservable("Energy");
            results << alps::RealObservable("Magnetization");
            results << alps::RealObservable("Magnetization^2");
            results << alps::RealObservable("Magnetization^4");
            results << alps::SimpleRealVectorObservable("CorrelationsSimple");
            results << alps::RealVectorObservable("Correlations");
        }
        void do_update() {
            for (int j = 0; j < length; ++j) {
                using std::exp;
                int i = int(double(length) * random());
                int right = ( i + 1 < length ? i + 1 : 0 );
                int left = ( i - 1 < 0 ? length - 1 : i - 1 );
                double p = exp( 2. * beta * spins[i] * ( spins[right] + spins[left] ));
                if ( p >= 1. || random() < p )
                    spins[i] =- spins[i];
            }
        };
        void do_measurements() {
            sweeps++;
            if (sweeps > thermalization_sweeps) {
                tmag = 0;
                ten = 0;
                corr.resize(length, 0.);
                for (int i = 0; i < length; ++i) {
                    tmag += spins[i];
                    ten += -spins[i] * spins[ i + 1 < length ? i + 1 : 0 ];
                    for (int d = 0; d < length; ++d)
                        corr[d] += spins[i] * spins[( i + d ) % length ];
                }
                corr /= double(length);
                ten /= length;
                tmag /= length;
                results["EnergySimple"] << ten;
                results["Energy"] << ten;
                results["Magnetization"] << tmag;
                results["Magnetization^2"] << tmag * tmag;
                results["Magnetization^4"] << tmag * tmag * tmag * tmag;
                results["CorrelationsSimple"] << corr;
                results["Correlations"] << corr;
            }
        };
        double fraction_completed() const {
            return (sweeps < thermalization_sweeps ? 0. : ( sweeps - thermalization_sweeps ) / double(total_sweeps));
        }
    private:
        int length;
        int sweeps;
        int thermalization_sweeps;
        int total_sweeps;
        double beta;
        double tmag;
        double ten;
        std::vector<int> spins;
        std::valarray<double> corr;
};
