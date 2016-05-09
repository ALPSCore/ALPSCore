////////Make sure we use NDEBUG and BOOST_DISABLE_ASSERTS!!
#pragma once
#ifndef ALPS_GF_DEBUG
#ifndef BOOST_DISABLE_ASSERTS
#warning "BOOST_DISABLE_ASSERTS is not defined in GF. This leads to very slow code. Define ALPS_GF_DEBUG to avoid this warning."
#endif //BOOST_DISABLE_ASSERTS

#ifndef NDEBUG
#warning "NDEBUG is not defined in GF. This leads to very slow code. Define ALPS_GF_DEBUG to suppress this warning"
#endif //NDEBUG

#endif //ALPS_GF_DEBUG
