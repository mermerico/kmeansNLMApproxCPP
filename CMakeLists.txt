cmake_minimum_required(VERSION 3.10)

project(kMeansNLMApprox)
  
add_executable(testNLM testNLM.cpp)

#Add Eigen
target_include_directories(testNLM PUBLIC ../eigen-3.3.7)

#Add dlib
add_subdirectory(../dlib-19.19/dlib dlib_build)
target_link_libraries(testNLM PUBLIC dlib::dlib)

#Add OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(testNLM PUBLIC OpenMP::OpenMP_CXX)
endif()

# Add Google Profiling Tools
#set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
#find_package(Gperftools)
#target_link_libraries(testNLM PUBLIC ${GPERFTOOLS_LIBRARIES})
#target_include_directories(testNLM PUBLIC ${GPERFTOOLS_INCLUDE_DIR})

if (MSVC)
	set_property(TARGET testNLM PROPERTY COMPILE_FLAGS "/std:c++latest")
else()
	set_property(TARGET testNLM PROPERTY CXX_STANDARD 17)
endif()

#set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -march=haswell -mtune=native")