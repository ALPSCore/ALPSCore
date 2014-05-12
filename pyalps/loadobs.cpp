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

#include <alps/hdf5/archive.hpp>
#include <alps/utility/encode.hpp>
#include <alps/alea.h>

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>


int main() {
    alps::hdf5::archive iar("loadobs.h5");

    std::vector<std::string> list = iar.list_children("/simulation/results");
    for (std::vector<std::string>::const_iterator it = list.begin(); it != list.end(); ++it) {
        iar.set_context("/simulation/results/" + iar.encode_segment(*it));
        if (iar.is_scalar("/simulation/results/" + iar.encode_segment(*it) + "/mean/value")) {
            alps::alea::mcdata<double> obs;
            obs.load(iar);
            std::cout << *it << " " << obs << std::endl;
        } else {
            alps::alea::mcdata<std::vector<double> > obs;
            obs.load(iar);
            std::cout << *it << " " << obs << std::endl;
        }
    }
}
