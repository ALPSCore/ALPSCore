/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2003 by Brigitte Surer and Jan Gukelberger                        *
 * Copyright (C) 2010 - 2012 by Lukas Gamper <gamperl@gmail.com>                   *
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

#include <alps/scheduler/montecarlo.h>
#include <alps/alea.h>
#include <boost/random.hpp>
#include <boost/multi_array.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>

class Simulation
{
public:
    Simulation(double beta,size_t L, std::string output_file)
    :   eng_(42)
    ,   rng_(eng_, dist_)
    ,   L_(L)
    ,   beta_(beta)
    ,   spins_(boost::extents[L][L])
    ,   energy_("E")
    ,   magnetization_("m")
    ,   abs_magnetization_("|m|")
    ,   m2_("m^2")
    ,   m4_("m^4")
    ,   filename_(output_file)
    {
        // Init exponential map
        for(int E = -4; E <= 4; E += 2)
            exp_table_[E] = exp(2*beta*E);

        // Init random spin configuration
        for(size_t i = 0; i < L; ++i)
        {
            for(size_t j = 0; j < L; ++j)
                spins_[i][j] = 2 * randint(2) - 1;
        }
    }

    void run(size_t ntherm,size_t n)
    {
        thermalization_ = ntherm;
        sweeps_=n;
        // Thermalize for ntherm steps
        while(ntherm--)
            step();

        // Run n steps
        while(n--)
        {
            step();
            measure();
        }
        
        //save the observables 
        save(filename_);
        
        // Print observables
        std::cout << abs_magnetization_;
        std::cout << energy_.name() << ":\t" << energy_.mean()
            << " +- " << energy_.error() << ";\ttau = " << energy_.tau() 
            << ";\tconverged: " << alps::convergence_to_text(energy_.converged_errors()) 
            << std::endl;
        std::cout << magnetization_.name() << ":\t" << magnetization_.mean()
            << " +- " << magnetization_.error() << ";\ttau = " << magnetization_.tau() 
            << ";\tconverged: " << alps::convergence_to_text(magnetization_.converged_errors())
            << std::endl;
    }
    void step()
    {
        for(size_t s = 0; s < L_*L_; ++s)
        {
            // Pick random site k=(i,j)
            int i = randint(L_);
            int j = randint(L_);
            
            // Measure local energy e = -s_k * sum_{l nn k} s_l
            int e = spins_[(i-1+L_)%L_][j] + spins_[(i+1)%L_][j] +
            spins_[i][(j-1+L_)%L_] + spins_[i][(j+1)%L_];
            e *= -spins_[i][j];
            
            // Flip s_k with probability exp(2 beta e)
            if(e > 1 || rng_() < exp_table_[e])
                spins_[i][j] = -spins_[i][j];
        }
    }
    void measure()
    {
        int E = 0; // energy
        int M = 0; // magnetization
        for(size_t i = 0; i < L_; ++i)
        {
            for(size_t j = 0; j < L_; ++j)
            {
                E -= spins_[i][j]*(spins_[(i+1)%L_][j] + spins_[i][(j+1)%L_]);
                M += spins_[i][j];
            }
        }
        
        // Add sample to observables
        energy_ << E/double(L_*L_);
        double m = M/double(L_*L_);
        magnetization_ << m;
        abs_magnetization_ << std::abs(M)/double(L_*L_);
        m2_ << m*m;
        m4_ << m*m*m*m;
    }
    
    void save(std::string const & filename){
        alps::hdf5::archive ar(filename, "wm");
        ar << alps::make_pvp("/simulation/results/"+energy_.representation(), energy_);
        ar << alps::make_pvp("/simulation/results/"+magnetization_.representation(), magnetization_);
        ar << alps::make_pvp("/simulation/results/"+abs_magnetization_.representation(), abs_magnetization_);
        ar << alps::make_pvp("/simulation/results/"+m2_.representation(), m2_);
        ar << alps::make_pvp("/simulation/results/"+m4_.representation(), m4_);
        ar << alps::make_pvp("/parameters/L", L_);
        ar << alps::make_pvp("/parameters/BETA", beta_);
        ar << alps::make_pvp("/parameters/SWEEPS", sweeps_);
        ar << alps::make_pvp("/parameters/THERMALIZATION", thermalization_);
    }
    
    protected:
    // Random int from the interval [0,max)
    int randint(int max) const
    {
        return static_cast<int>(max * rng_());
    }

private:
    typedef boost::mt19937 engine_type;
    typedef boost::uniform_real<> distribution_type;
    typedef boost::variate_generator<engine_type&, distribution_type> rng_type;
    engine_type eng_;
    distribution_type dist_;
    mutable rng_type rng_;

    size_t L_;
    double beta_;
    size_t sweeps_;
    size_t thermalization_;
    boost::multi_array<int,2> spins_;
    std::map< int, double > exp_table_;

    alps::RealObservable energy_;
    alps::RealObservable magnetization_;
    alps::RealObservable abs_magnetization_;
    alps::RealObservable m2_;
    alps::RealObservable m4_;
    
    std::string filename_;
};


int main(int,char**)
{
    size_t L = 16;    // Linear lattice size
    size_t N = 5000;    // # of simulation steps

    std::cout << "# L: " << L << " N: " << N << std::endl;

    // Scan beta range [0,1] in steps of 0.1
    for(double beta = 0.; beta <= 1.; beta += .1)
    {
        std::cout << "----------" << std::endl;
        std::cout << "beta = " << beta << std::endl;
        std::stringstream output_name;
        output_name << "ising.L_" << L << "beta_" << beta <<".h5";
        Simulation sim(beta,L, output_name.str());
        sim.run(N/2,N);
    }

    return 0;
}
