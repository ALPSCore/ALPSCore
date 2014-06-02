/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
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
