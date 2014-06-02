/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */


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
