find_package(GTest REQUIRED)

function(my_add_test name)
    add_executable(test_${name} "${name}.cpp")
    target_link_libraries(test_${name} PRIVATE GTest::gtest_main dt)
endfunction()

my_add_test(image_view)