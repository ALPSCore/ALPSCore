/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/utility/bitops.hpp>

int main()
{
    uint32_t ui  = 5;
    uint64_t uli = 5;
    ui   <<= sizeof(uint32_t)*8-4;
    uli  <<= sizeof(uint64_t)*8-4;

    bool succ = alps::popcnt(ui)  == 2;
    succ     &= alps::popcnt(uli) == 2;
    return succ ? 0 : -1;
}
