/*****************************************************************************
 *
 * ALPS DMFT Project - BLAS Compatibility headers
 *  BLAS headers for accessing BLAS from C++.
 *
 * Copyright (C) 2010 Matthias Troyer <gtroyer@ethz.ch>
 *
 *
* This software is part of the ALPS Applications, published under the ALPS
* Application License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Application License along with
* the ALPS Applications; see the file LICENSE.txt. If not, the license is also
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

#include <boost/numeric/bindings/blas.hpp>
#include<vector>

// provide overloads for types where blas can be used        

namespace blas{

#define IMPLEMENT_FOR_REAL_BLAS_TYPES(F) F(float) F(double)

#define IMPLEMENT_FOR_COMPLEX_BLAS_TYPES(F) \
F(std::complex<float>) \
F(std::complex<double>)

#define IMPLEMENT_FOR_ALL_BLAS_TYPES(F) \
IMPLEMENT_FOR_REAL_BLAS_TYPES(F) \
IMPLEMENT_FOR_COMPLEX_BLAS_TYPES(F) 
} // namespace
