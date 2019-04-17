/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <string>

#include <alps/serialization/core.hpp>

namespace alps { namespace serialization {

/**
 * Allows RAII-type use of groups in serializer.
 *
 * Enters the specified group on construction of the sentry, and automatically
 * leaves the group when the object is destroyed.  This also allows recovery in
 * the case of soft exceptions.
 *
 *     void write_to_group(serializer &s, std::string name) {
 *         internal::serializer_sentry group(s, name);
 *         serialize(s, "first_item", 42);
 *         serialize(s, "second_item", 4711);
 *     } // exits group here
 */
struct serializer_sentry
{
    serializer_sentry(serializer &ser, const std::string &group)
        : ser_(ser)
        , group_(group)
    {
        if (group != "")
            ser_.enter(group);
    }

    ~serializer_sentry()
    {
        if (group_ != "")
            ser_.exit();
    }

private:
    serializer &ser_;
    std::string group_;
};

/**
 * Allows RAII-type use of groups in deserializer.
 *
 * Enters the specified group on construction of the sentry, and automatically
 * leaves the group when the object is destroyed.  This also allows recovery in
 * the case of soft exceptions.
 */
struct deserializer_sentry
{
    deserializer_sentry(deserializer &ser, const std::string &group)
        : ser_(ser)
        , group_(group)
    {
        if (group != "")
            ser_.enter(group);
    }

    ~deserializer_sentry()
    {
        if (group_ != "")
            ser_.exit();
    }

private:
    deserializer &ser_;
    std::string group_;
};

}}
