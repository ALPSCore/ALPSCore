/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
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

#ifndef ALPS_NGS_MACROS_HPP
#define ALPS_NGS_MACROS_HPP

#include <sstream>
#include <stdexcept>

#define ALPS_NGS_STRINGIFY(arg) ALPS_NGS_STRINGIFY_HELPER(arg)

#define ALPS_NGS_STRINGIFY_HELPER(arg) #arg

#define ALPS_NGS_THROW_ERROR(error, message)                                                                                                   \
	{                                                                                                                                          \
		std::ostringstream buffer;                                                                                                             \
		buffer << "Error in " << __FILE__ << " on " << ALPS_NGS_STRINGIFY(__LINE__) << " in " << __FUNCTION__ << ":" << std::endl << message;  \
		throw ( error (buffer.str()));                                                                                                         \
	}

#define ALPS_NGS_THROW_OUT_OF_RANGE(message)                                                                                                   \
	ALPS_NGS_THROW_ERROR(std::out_of_range, message)

#define ALPS_NGS_THROW_RANGE_ERROR(message)                                                                                                    \
	ALPS_NGS_THROW_ERROR(std::range_error, message)

#define ALPS_NGS_THROW_RUNTIME_ERROR(message)                                                                                                  \
	ALPS_NGS_THROW_ERROR(std::runtime_error, message)

#define ALPS_NGS_THROW_LOGIC_ERROR(message)                                                                                                    \
	ALPS_NGS_THROW_ERROR(std::logic_error, message)

#define ALPS_NGS_THROW_INVALID_ARGUMENT(message)                                                                                               \
	ALPS_NGS_THROW_ERROR(std::invalid_argument, message)

#endif