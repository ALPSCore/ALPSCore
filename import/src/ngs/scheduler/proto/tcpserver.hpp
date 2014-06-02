/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NGS_TCPSERVER_HPP
#define ALPS_NGS_TCPSERVER_HPP

#include <alps/ngs/detail/tcpsession.hpp>

#include <boost/bind.hpp>
#include <boost/asio.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>

#include <map>
#include <string>

namespace alps {

    class tcpserver {
        
        public:
        
            typedef std::map<std::string, boost::function<std::string()> > actions_type;

            tcpserver(short port, actions_type const & a = actions_type())
                : actions(a)
                , io_service()
                , acceptor(io_service, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port))
            {
                start();
            }
        
            void add_action(std::string const & name, boost::function<std::string()> const & action) {
                actions[name] = action;
            }
        
            void poll() {
                io_service.poll();
            }
        
            void stop() {
                io_service.stop();
            }

        private:

            void start() {
                detail::tcpsession * new_sess = new detail::tcpsession(actions, io_service); // TODO: use shared_ptr and manage the sockets manually
                acceptor.async_accept(
                      new_sess->get_socket()
                    , boost::bind(&tcpserver::handler, this, new_sess, boost::asio::placeholders::error)
                );
            }

            void handler(detail::tcpsession * new_sess, boost::system::error_code const & error) {
                if (!error)
                    new_sess->start();
                else
                    delete new_sess;
                start();
            }

            actions_type actions;
            boost::asio::io_service io_service;
            boost::asio::ip::tcp::acceptor acceptor;
    };

}

#endif
