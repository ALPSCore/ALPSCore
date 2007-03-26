#ifndef OSIRIS_STD_STACK_HPP
#define OSIRIS_STD_STACK_HPP

#include <alps/config.h>
#include <alps/osiris/dump.h>
#include <alps/osiris/std/impl.h>

#include <stack>

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class T, class Sequence>
inline alps::IDump& operator>>(alps::IDump& dump, std::stack<T,Sequence>& x)
{
  Sequence helper;
  alps::detail::loadContainer(helper,x);
  while(!helper.empty()) {
  	x.push(helper.back());
    helper.pop_back();
  }
  return dump;
}

template <class T, class Sequence>
inline alps::ODump& operator<<(alps::ODump& dump, const std::stack<T,Sequence>& x)
{
  std::stack<T,Sequence> cphelper(x);
  Sequence sqhelper;
  while(!cphelper.empty()) {
    sqhelper.push_back(cphelper.top());
    cphelper.pop();
  }
  alps::detail::saveContainer(dump,sqhelper);
  return dump;
}          

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // OSIRIS_STD_STACK_HPP
