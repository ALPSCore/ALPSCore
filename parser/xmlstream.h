/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2009 by Synge Todo <wistaria@comp-phys.org>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
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

/* $Id$ */

#ifndef ALPS_PARSER_XMLSTREAM_H
#define ALPS_PARSER_XMLSTREAM_H

// for MSVC
#if defined(_MSC_VER)
# pragma warning(disable:4251)
#include <complex>
template <class T>
bool _isnan(std::complex<T> const& x)
{
  return _isnan(x.real()) || _isnan(x.imag());
}
template <class T>
bool _finite(std::complex<T> const& x)
{
  return _finite(x.real()) || _isnan(x.imag());
}
#endif

#include <alps/config.h>
#include <alps/parser/xmlattributes.h>

#include <boost/config.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/lexical_cast.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stack>
#include <string>
#include <complex>
#ifdef BOOST_MSVC
# include <float.h>
#endif

namespace alps {

namespace detail {

struct header_t
{
header_t(const std::string& enc = "")
  : version("1.0"), encoding(enc) {}
std::string version;
std::string encoding;
};

struct start_tag_t
{
  start_tag_t(const std::string& n) : name(n) {}
  std::string name;
};

struct end_tag_t
{
  end_tag_t(const std::string& n = "") : name(n) {}
  std::string name;
};

struct attribute_t
{
  template<class T>
  attribute_t(const std::string& n, const T& v) : attr(n, v) {}
  XMLAttribute attr;
};

struct stylesheet_t
{
  stylesheet_t(const std::string& n) : url(n) {}
  std::string url;
};

struct pi_t : public start_tag_t
{
  pi_t(const std::string& n) : start_tag_t(n) {}
};

} // namespace detail

class ALPS_DECL oxstream
{
public:
  oxstream();
  oxstream(std::ostream& os, uint32_t incr = 2);
  oxstream(const boost::filesystem::path& file, uint32_t incr = 2);
  ~oxstream();

  oxstream& operator<<(const detail::header_t& c);
  oxstream& operator<<(const detail::start_tag_t& c);
  oxstream& operator<<(const detail::end_tag_t& c);
  oxstream& operator<<(const detail::stylesheet_t& c);
  oxstream& operator<<(const detail::attribute_t& c);
  oxstream& operator<<(const detail::pi_t& c);
  oxstream& start_comment();
  oxstream& end_comment();
  oxstream& start_cdata();
  oxstream& end_cdata();
  oxstream& no_linebreak();
  oxstream& endl() { *this << "\n"; return *this; }

  oxstream& operator<<(const XMLAttribute& c);
  oxstream& operator<<(const XMLAttributes& c);

  oxstream& operator<<(const std::string& t) {
    return text_str(t);
  }
  oxstream& operator<<(const char t) {
    return text_str(std::string(1,t));
  }
  oxstream& operator<<(const char * t) {
    return text_str(t);
  }

  // operator<< for intrinsic types

# define ALPS_XMLSTREAM_DO_TYPE(T) \
  oxstream& operator<<(const T t) \
  { return text_str(boost::lexical_cast<std::string, T>(t)); }
  ALPS_XMLSTREAM_DO_TYPE(bool)
  ALPS_XMLSTREAM_DO_TYPE(signed char)
  ALPS_XMLSTREAM_DO_TYPE(unsigned char)
  ALPS_XMLSTREAM_DO_TYPE(short)
  ALPS_XMLSTREAM_DO_TYPE(unsigned short)
  ALPS_XMLSTREAM_DO_TYPE(int)
  ALPS_XMLSTREAM_DO_TYPE(unsigned int)
  ALPS_XMLSTREAM_DO_TYPE(long)
  ALPS_XMLSTREAM_DO_TYPE(unsigned long)
# ifdef BOOST_HAS_LONG_LONG
  ALPS_XMLSTREAM_DO_TYPE(long long)
  ALPS_XMLSTREAM_DO_TYPE(unsigned long long)
# endif
  ALPS_XMLSTREAM_DO_TYPE(float)
  ALPS_XMLSTREAM_DO_TYPE(double)
  ALPS_XMLSTREAM_DO_TYPE(long double)

# undef ALPS_XMLSTREAM_DO_TYPE

  template <class T>
  oxstream& operator<<(const std::complex<T>& t) {
    return text_str(boost::lexical_cast<std::string>(t));
  }

  // for manipulators
  template<class T>
  oxstream& operator<<(T (*fn)(const std::string&)) {
    return (*this << fn(std::string()));
  }
  oxstream& operator<<(oxstream& (*fn)(oxstream& oxs)) { return fn(*this); }

  std::ostream& stream() { return os_; }

protected:
  oxstream& text_str(const std::string& text);

  void output(bool close = false);
  void output_offset();

private:
  enum Context { NotSpecified, StartTag, PI, Text, Comment, Cdata };

  boost::filesystem::ofstream of_;
  std::ostream& os_;
  std::stack<std::pair<std::string, bool> > stack_;
  XMLAttributes attr_;
  Context context_;
  bool linebreak_;
  uint32_t offset_;
  uint32_t offset_incr_;
};

// manipulators

inline detail::header_t header(const std::string& enc) {
  return detail::header_t(enc);
}

inline detail::stylesheet_t stylesheet(const std::string& url) {
  return detail::stylesheet_t(url);
}

inline detail::pi_t processing_instruction(const std::string& name) {
  return detail::pi_t(name);
}

inline detail::start_tag_t start_tag(const std::string& name) {
  return detail::start_tag_t(name);
}

inline detail::end_tag_t end_tag(const std::string& name = "") {
  return detail::end_tag_t(name);
}

template<class T>
inline detail::attribute_t attribute(const std::string& name, const T& value) {
  return detail::attribute_t(name, value);
}

inline detail::attribute_t xml_namespace(const std::string& name,
                                         const std::string& url) {
  return detail::attribute_t("xmlns:" + name, url);
}

inline oxstream& start_comment(oxstream& oxs) { return oxs.start_comment(); }

inline oxstream& end_comment(oxstream& oxs) { return oxs.end_comment(); }

inline oxstream& start_cdata(oxstream& oxs) { return oxs.start_comment(); }

inline oxstream& end_cdata(oxstream& oxs) { return oxs.end_comment(); }

inline oxstream& no_linebreak(oxstream& oxs) { return oxs.no_linebreak(); }

// replace "<", "&", etc to entities
ALPS_DECL std::string convert(const std::string& str);

template<class T>
inline std::string precision(const T& d, int n)
{
  std::ostringstream stream;
#ifndef BOOST_MSVC
  stream << std::setprecision(n) << d;
#else
  if (_finite(d)) {
    stream << std::setprecision(n) << d;
  } else {
    if (_isnan(d)) {
      stream << "nan";
    } else {
      stream << "inf"; // (d > 0 ? "inf" : "-inf");
    }
  }
#endif
  return stream.str();
}

} // namespace alps

namespace std {

inline alps::oxstream& endl(alps::oxstream& oxs) { return oxs.endl(); }

}

#endif // ALPS_PARSER_XMLSTREAM_H
