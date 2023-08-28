add_executable(run_tests test.cc)
install(TARGETS run_tests)

target_include_directories(run_tests PUBLIC ../src)
target_link_libraries(run_tests PRIVATE 3crowns)
add_test(test_3crowns run_tests)
