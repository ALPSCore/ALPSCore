#include <alps/config.h>
#include <cstddef>
#include <iostream>

int main() {

#define DO_TYPE(T) \
  std::cout << "size of "#T" is " << sizeof(T) << std::endl;

  DO_TYPE(bool)
  DO_TYPE(char)
  DO_TYPE(short)
  DO_TYPE(int)
  DO_TYPE(long)
  DO_TYPE(long long)
  DO_TYPE(float)
  DO_TYPE(double)
  DO_TYPE(long double)

  DO_TYPE(int8_t)
  DO_TYPE(int16_t)
  DO_TYPE(int32_t)
  DO_TYPE(int64_t)

  DO_TYPE(std::size_t)
  DO_TYPE(std::ptrdiff_t)
}
