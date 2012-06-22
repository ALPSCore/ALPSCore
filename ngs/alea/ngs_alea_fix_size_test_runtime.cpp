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


#include <alps/ngs.hpp>

//these two flags will create the int main() together with unit_test.hpp
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_CASE(test_fixed_size_bin_in_modular_accum)
{
    alps::alea::accumulator<int, alps::alea::features<alps::alea::tag::fixed_size_binning> > acci; //Default 128
    
    acci << 2;
    acci << 6;
    
    BOOST_REQUIRE( alps::alea::fixed_size_bin(acci).bin_size() == 128);
    
    
    alps::alea::accumulator<double, alps::alea::features<alps::alea::tag::fixed_size_binning> > accd(alps::alea::bin_size = 10);

    for(int i = 0; i < 100; ++i)
    {
        accd << 0.1*i;
    }
    
        
        
    BOOST_REQUIRE( alps::alea::fixed_size_bin(accd).bin_size() == 10);
    alps::alea::fixed_size_bin(accd);
    accd.fixed_size_bin();
}
