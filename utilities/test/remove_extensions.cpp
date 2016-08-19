/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <alps/utilities/remove_extensions.hpp>

#include <gtest/gtest.h>

TEST(RemoveExtensionsTest, SimpleName)
{
    EXPECT_EQ("name", alps::remove_extensions("name"));
}

TEST(RemoveExtensionsTest, SimpleNameExt)
{
    EXPECT_EQ("name", alps::remove_extensions("name.ext"));
}

TEST(RemoveExtensionsTest, SimpleName2Ext)
{
    EXPECT_EQ("name", alps::remove_extensions("name.ext.another_ext"));
}

TEST(RemoveExtensionsTest, SimpleName3Ext)
{
    EXPECT_EQ("name", alps::remove_extensions("name.ext.another_ext.third_ext"));
}

TEST(RemoveExtensionsTest, CurPathName)
{
    EXPECT_EQ("./name", alps::remove_extensions("./name"));
}

TEST(RemoveExtensionsTest, CurPathNameExt)
{
    EXPECT_EQ("./name", alps::remove_extensions("./name.ext"));
}

TEST(RemoveExtensionsTest, CurPathName2Ext)
{
    EXPECT_EQ("./name", alps::remove_extensions("./name.ext.another_ext"));
}

TEST(RemoveExtensionsTest, CurPathName3Ext)
{
    EXPECT_EQ("./name", alps::remove_extensions("./name.ext.another_ext.third_ext"));
}

TEST(RemoveExtensionsTest, FullPathName)
{
    EXPECT_EQ("/path/to/name", alps::remove_extensions("/path/to/name"));
}

TEST(RemoveExtensionsTest, FullPathNameExt)
{
    EXPECT_EQ("/path/to/name", alps::remove_extensions("/path/to/name.ext"));
}

TEST(RemoveExtensionsTest, FullPathName2Ext)
{
    EXPECT_EQ("/path/to/name", alps::remove_extensions("/path/to/name.ext.another_ext"));
}

TEST(RemoveExtensionsTest, FullPathName3Ext)
{
    EXPECT_EQ("/path/to/name", alps::remove_extensions("/path/to/name.ext.another_ext.third_ext"));
}

TEST(RemoveExtensionsTest, FullDotPathName)
{
    EXPECT_EQ("/path/some.myext/to/name", alps::remove_extensions("/path/some.myext/to/name"));
}

TEST(RemoveExtensionsTest, FullDotPathNameExt)
{
    EXPECT_EQ("/path/some.myext/to/name", alps::remove_extensions("/path/some.myext/to/name.ext"));
}

TEST(RemoveExtensionsTest, FullDotPathName2Ext)
{
    EXPECT_EQ("/path/some.myext/to/name", alps::remove_extensions("/path/some.myext/to/name.ext.another_ext"));
}

TEST(RemoveExtensionsTest, FullDotPathName3Ext)
{
    EXPECT_EQ("/path/some.myext/to/name", alps::remove_extensions("/path/some.myext/to/name.ext.another_ext.third_ext"));
}

