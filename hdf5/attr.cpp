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

#include "creator.hpp"

int main() {
    std::string const filename = "test.h5";
    if (boost::filesystem::exists(boost::filesystem::path(filename)))
        boost::filesystem::remove(boost::filesystem::path(filename));
    bool result = true;
    {
        alps::hdf5::oarchive oar(filename);
        oar.serialize("/data");
    }
    for (std::size_t i = 0; result && i < 32; ++i) {
        TYPE random_write(creator< TYPE >::random());
        TYPE empty_write(creator< TYPE >::empty());
        TYPE special_write(creator< TYPE >::special());
        {
            alps::hdf5::oarchive oar(filename);
            oar
                << alps::make_pvp("/data/@random", random_write)
                << alps::make_pvp("/data/@empty", empty_write)
                << alps::make_pvp("/data/@special", special_write)
            ;
        }
        {
            alps::hdf5::iarchive iar(filename);
            TYPE random_read(creator<TYPE>::random(iar));
            TYPE empty_read(creator<TYPE>::empty(iar));
            TYPE special_read(creator<TYPE>::special(iar));
            iar
                >> alps::make_pvp("/data/@random", random_read)
                >> alps::make_pvp("/data/@empty", empty_read)
                >> alps::make_pvp("/data/@special", special_read)
            ;
            result = equal(random_write, random_read) && equal(empty_write, empty_read) && equal(special_write, special_read);
            destructor<TYPE>::apply(random_read);
            destructor<TYPE>::apply(empty_read);
            destructor<TYPE>::apply(special_read);
        }
        destructor<TYPE>::apply(random_write);
        destructor<TYPE>::apply(empty_write);
        destructor<TYPE>::apply(special_write);
    }
    boost::filesystem::remove(boost::filesystem::path(filename));
    std::cout << (result ? "SUCCESS" : "FAILURE") << std::endl;
    return result ? EXIT_SUCCESS : EXIT_FAILURE;
}
