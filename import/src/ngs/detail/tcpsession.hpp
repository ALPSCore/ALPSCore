/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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
