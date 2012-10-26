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
