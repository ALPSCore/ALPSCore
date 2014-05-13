#include_directories(${PROJECT_BINARY_DIR}/include)
#include_directories(${PROJECT_SOURCE_DIR}/include)

set (test_src
    vmusage
    bitops
    )

foreach(test ${test_src})
  add_executable(${test} ${test}.cpp)
  add_dependencies(${test} alps)
  target_link_libraries(${test} ${PROJECT_NAME})
  add_test(${test} ${test})
  #set_property(TEST ${test} PROPERTY LABELS utility)
endforeach(test)
