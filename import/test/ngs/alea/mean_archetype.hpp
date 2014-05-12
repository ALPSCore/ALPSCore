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


#ifndef TEST_NGS_MEAN_ARCHETYPE_HEADER
#define TEST_NGS_MEAN_ARCHETYPE_HEADER

#include <alps/hdf5/archive.hpp>
#include <iostream>

using namespace std;

struct mean_archetype
{
    mean_archetype() {}
    
    mean_archetype operator+(mean_archetype const & arg)
    {
        return mean_archetype();
    }
    
    mean_archetype(int const & arg){}
    
    mean_archetype operator/(double const & arg)
    {
        return mean_archetype();
    }
    
    mean_archetype operator=(mean_archetype rhs)
    {
        return mean_archetype();
    }
    
    mean_archetype operator+=(mean_archetype rhs)
    {
        return mean_archetype();
    }

    void save(alps::hdf5::archive & ar) const {}
    void load(alps::hdf5::archive & ar) {}
};

mean_archetype operator*(mean_archetype const & arg, mean_archetype const & arg2)
{
    return mean_archetype();
}

mean_archetype operator/(mean_archetype const & arg, mean_archetype const & arg2)
{
    return mean_archetype();
}

ostream & operator<<(ostream & out, mean_archetype arg)
{
    out << "mean_archetype";
    return out;
}
#endif // TEST_NGS_MEAN_ARCHETYPE_HEADER
