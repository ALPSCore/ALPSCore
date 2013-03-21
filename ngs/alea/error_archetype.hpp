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


#ifndef TEST_NGS_ERROR_ARCHETYPE_HEADER
#define TEST_NGS_ERROR_ARCHETYPE_HEADER

#include "mean_archetype.hpp"

#include <iostream>

using namespace std;

struct error_archetype
{
    error_archetype() {}
    
    error_archetype operator+(error_archetype const & arg)
    {
        return error_archetype();
    }
    
    error_archetype(int const & arg){}
    
    error_archetype operator/(double const & arg) const
    {
        return error_archetype();
    }
    
    error_archetype operator=(error_archetype rhs)
    {
        return error_archetype();
    }
    
    error_archetype operator+=(error_archetype rhs)
    {
        return error_archetype();
    }

    void save(alps::hdf5::archive & ar) const {}
    void load(alps::hdf5::archive & ar) {}
};

error_archetype operator*(error_archetype const & arg, error_archetype const & arg2)
{
    return error_archetype();
}

error_archetype operator/(error_archetype const & arg, error_archetype const & arg2)
{
    return error_archetype();
}

error_archetype operator-(error_archetype const & arg, error_archetype const & arg2)
    {
        return error_archetype();
    }

ostream & operator<<(ostream & out, error_archetype arg)
{
    out << "error_archetype";
    return out;
}

error_archetype sqrt(error_archetype)
{
    return error_archetype();
}
#endif // TEST_NGS_ERROR_ARCHETYPE_HEADER
