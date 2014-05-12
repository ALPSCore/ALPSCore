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

#ifndef ALPS_NGS_DETAIL_TCPSESSION_HPP
#define ALPS_NGS_DETAIL_TCPSESSION_HPP

#include <boost/bind.hpp>
#include <boost/asio.hpp>
#include <boost/lexical_cast.hpp>

#include <cstdlib>

namespace alps {

    namespace detail {

        class tcpsession {

            public:

                typedef std::map<std::string, boost::function<std::string()> > actions_type;

                tcpsession(actions_type & a, boost::asio::io_service & io_service)
                    : actions(a)
                    , socket(io_service)
                {}

                boost::asio::ip::tcp::socket & get_socket() {
                    return socket;
                }

                void start(boost::system::error_code const & error = boost::system::error_code()) {
                    if (!error)
                        socket.async_read_some(
                              boost::asio::buffer(data, 1028)
                            , boost::bind(
                                  &tcpsession::handler
                                , this
                                , boost::asio::placeholders::error
                                , boost::asio::placeholders::bytes_transferred
                            )
                        );
                    else
                        delete this;
                }

            private:

                void handler(const boost::system::error_code& error, boost::uint32_t size) {
                    if (!error) {
                        std::string action(data, size);
                        std::string response;
                        if (actions.find(action) != actions.end())
                            response = actions[action]();
                        else
                            response = "Unknown action: " + action;
                        std::memcpy(data + 4, response.c_str(), size = response.size());
                        std::memcpy(data, &size, 4);
                        boost::asio::async_write(
                              socket
                            , boost::asio::buffer(data, 1028)
                            , boost::bind(&tcpsession::start, this, boost::asio::placeholders::error)
                        );
                    } else
                        delete this;
                }

                actions_type & actions;
                char data[1028];
                boost::asio::ip::tcp::socket socket;
        };
    }
}

#endif
