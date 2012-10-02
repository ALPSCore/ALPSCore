/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 - 2012 by Mario Koenz <mkoenz@ethz.ch>                       *
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

#define BOOST_TEST_MODULE alps::ngs::alea

#include <alps/ngs.hpp>

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_CASE(test_log_bin_in_modular_accum)
{
    alps::alea::accumulator<int, alps::alea::features<alps::alea::tag::log_binning> > acci; //Default 128
    
    acci << 2;
    acci << 6;
    
    //~ BOOST_REQUIRE( alps::alea::max_num_bin(acci) == 128);
    
    
    alps::alea::accumulator<double, alps::alea::features<alps::alea::tag::log_binning> > accd;
    
    for(int i = 0; i < 96; ++i)
    {
        accd << 1.;
    }
    
    std::vector<double> vec;
    vec = log_bin(accd).bins();
    
    for(int i = 0; i < vec.size(); ++i)
    {
        std::cout << vec[i] << std::endl;
    }
        
    //~ BOOST_REQUIRE( alps::alea::max_num_bin(accd) == 10);
}
