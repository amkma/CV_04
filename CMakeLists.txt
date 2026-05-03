# ═══════════════════════════════════════════════════════════════
#  CV_04 – Build configuration
#
#  Prerequisites:
#    • OpenCV 4.x   – set OpenCV_DIR if not in default location
#    • Python 3.12+  – with pybind11 installed in the venv
#
#  Build:
#    cmake -S . -B build -G "Visual Studio 18 2026" -A x64
#    cmake --build build --config Release
# ═══════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.15)
project(CV_Four LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC AND MSVC_VERSION GREATER_EQUAL 1950 AND NOT DEFINED OpenCV_RUNTIME)
	set(OpenCV_RUNTIME vc16)
endif()

find_package(OpenCV REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
	COMMAND ${Python3_EXECUTABLE} -m pybind11 --includes
	OUTPUT_VARIABLE PYBIND11_INCLUDE_FLAGS
	OUTPUT_STRIP_TRAILING_WHITESPACE
)

separate_arguments(PYBIND11_INCLUDE_FLAGS)
set(PYBIND11_INCLUDE_DIRS)
foreach(flag IN LISTS PYBIND11_INCLUDE_FLAGS)
	if(flag MATCHES "^-I(.+)")
		list(APPEND PYBIND11_INCLUDE_DIRS "${CMAKE_MATCH_1}")
	endif()
endforeach()

execute_process(
	COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.pyd')"
	OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
	OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_library(cv_core MODULE core/cv_core.cpp)

target_include_directories(cv_core PRIVATE
	${OpenCV_INCLUDE_DIRS}
	${Python3_INCLUDE_DIRS}
	${PYBIND11_INCLUDE_DIRS}
	${CMAKE_SOURCE_DIR}/core
)

target_link_libraries(cv_core PRIVATE ${OpenCV_LIBS} ${Python3_LIBRARIES})

set_target_properties(cv_core PROPERTIES
	PREFIX ""
	SUFFIX "${PYTHON_EXT_SUFFIX}"
)

# Copy the .pyd to project root so Python can import it
add_custom_command(TARGET cv_core POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		"$<TARGET_FILE:cv_core>"
		"${CMAKE_SOURCE_DIR}/cv_core${PYTHON_EXT_SUFFIX}"
)
