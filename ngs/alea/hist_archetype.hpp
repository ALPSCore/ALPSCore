/*****************************************************************************
 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations
 *
 * ALPS Libraries
 *
 * Copyright (C) 1997-2011 by Lukas Gamper
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
#ifndef HIST_ARCHETYPE_HEADER
#define HIST_ARCHETYPE_HEADER

#include <iostream>

using namespace std;

struct hist_archetype
{
    hist_archetype() {}
    
    operator double() const
    {
        return double();
    }
    
    hist_archetype operator+(hist_archetype const & arg) const
    {
        return hist_archetype();
    }
    hist_archetype operator-(hist_archetype const & arg) const
    {
        return hist_archetype();
    }
    
    hist_archetype(int const & arg){}
    
    hist_archetype operator/(double const & arg) const
    {
        return hist_archetype();
    }
    
    hist_archetype operator=(hist_archetype rhs)
    {
        return hist_archetype();
    }
    
    hist_archetype operator+=(hist_archetype rhs)
    {
        return hist_archetype();
    }
};
hist_archetype operator*(hist_archetype const & arg, hist_archetype const & arg2)
{
    return hist_archetype();
}
hist_archetype operator*(hist_archetype const & arg, unsigned int const & arg2)
{
    return hist_archetype();
}

hist_archetype operator*(int const & arg, hist_archetype const & arg2)
{
    return hist_archetype();
}

ostream & operator<<(ostream & out, hist_archetype arg)
{
    out << "hist_archetype";
    return out;
}
#endif // HIST_ARCHETYPE_HEADER
