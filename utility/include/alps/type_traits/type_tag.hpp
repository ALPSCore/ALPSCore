/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_TYPE_TRAITS_TYPE_TAG_H
#define ALPS_TYPE_TRAITS_TYPE_TAG_H

#include <alps/config.h>
#include <boost/mpl/int.hpp>
#include <complex>
#include <string>

namespace alps {

template <class T>
struct type_tag /* : public boost::mpl::int_<-1> */ {};

#define DEFINE_TYPE_TAG(TYPE,TAG) \
template<> struct type_tag< TYPE > : public boost::mpl::int_<TAG> {};

DEFINE_TYPE_TAG(float,0)
DEFINE_TYPE_TAG(double,1)
DEFINE_TYPE_TAG(long double,2)
DEFINE_TYPE_TAG(std::complex<float>,3)
DEFINE_TYPE_TAG(std::complex<double>,4)
DEFINE_TYPE_TAG(std::complex<long double>,5)
DEFINE_TYPE_TAG(int16_t,6)
DEFINE_TYPE_TAG(int32_t,7)
#ifndef BOOST_NO_INT64_T
DEFINE_TYPE_TAG(int64_t,8)
#endif
DEFINE_TYPE_TAG(uint16_t,9)
DEFINE_TYPE_TAG(uint32_t,10)
#ifndef BOOST_NO_INT64_T
DEFINE_TYPE_TAG(uint64_t,11)
#endif
DEFINE_TYPE_TAG(int8_t,12)
DEFINE_TYPE_TAG(uint8_t,13)
DEFINE_TYPE_TAG(std::string,14)
DEFINE_TYPE_TAG(bool,15)
#undef DEFINE_TYPE_TAG

} // namespace alps

#endif // ALPS_TYPE_TRAITS_TYPE_TAG_H
