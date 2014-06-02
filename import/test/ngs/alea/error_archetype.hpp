/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */


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
