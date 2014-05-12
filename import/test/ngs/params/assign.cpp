/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
 *                              Matthias Troyer <troyer@comp-phys.org>             *
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

#include <alps/ngs/params.hpp>

int main() {

    alps::params parms;
    parms["char"] = static_cast<char>(1);
    parms["signed char"] = static_cast<signed char>(1);
    parms["unsigned char"] = static_cast<unsigned char>(1);
    parms["short"] = static_cast<short>(1);
    parms["unsigned short"] = static_cast<unsigned short>(1);
    parms["int"] = static_cast<int>(1);
    parms["unsigned"] = static_cast<unsigned>(1);
    parms["long"] = static_cast<long>(1);
    parms["unsigned long"] = static_cast<unsigned long>(1);
    parms["long long"] = static_cast<long long>(1);
    parms["unsigned long long"] = static_cast<unsigned long long>(1);
    parms["float"] = static_cast<float>(1);
    parms["double"] = static_cast<double>(1);
    parms["long double"] = static_cast<long double>(1);
    parms["bool"] = static_cast<bool>(1);
    parms["std::string"] = std::string("asdf");

    std::cout << parms << std::endl;
    return 0;
}
