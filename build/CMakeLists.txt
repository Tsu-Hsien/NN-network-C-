cmake_minimum_required(VERSION 3.0)
project(main)

aux_source_directory(../src DIR_TOOT_SRCS)

include_directories("../lib/OpenCV348/include")
link_directories("../lib/OpenCV348/x64/mingw/lib")
FILE (GLOB DIR_LIB "../lib/OpenCV348/x64/mingw/lib/*.a")


SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../bin)

add_executable(${PROJECT_NAME} ${DIR_TOOT_SRCS})

target_link_libraries(${PROJECT_NAME} ${DIR_LIB})
