add_executable(run_tests test_minictc.cc)
install(TARGETS run_tests)

target_include_directories(run_tests PUBLIC ../src)
target_link_libraries(run_tests PRIVATE minictc)
add_test(test_minictc run_tests)
