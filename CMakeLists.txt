cmake_minimum_required(VERSION 3.1.0)
project(GPUJobSystem)

set(CMAKE_CXX_STANDARD 17)

include_directories($ENV{VULKAN_SDK}/include)


# library
set(CommonSrc src/Job.cpp
              src/JobManager.cpp)

add_library(GPUJobSystem ${CommonSrc})
target_link_libraries(GPUJobSystem $ENV{VULKAN_SDK}/Lib/vulkan-1.lib)


# examples
set(Shaders fibonacci
            sum
            edgedetect)

foreach(shader ${Shaders})
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/examples/shaders/${shader}.spv
        COMMAND glslc ${CMAKE_CURRENT_SOURCE_DIR}/examples/shaders/${shader}.comp
                      -o ${CMAKE_CURRENT_SOURCE_DIR}/examples/shaders/${shader}.spv
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/examples/shaders/${shader}.comp
    )
    add_custom_target(shader_${shader} ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/examples/shaders/${shader}.spv)
endforeach(shader ${Shader})


set(Examples simple_task
             multiple_invokations
             img_kernel)

foreach(examplename ${Examples})
    add_executable(${examplename} examples/${examplename}.cpp)
    target_include_directories(${examplename} PUBLIC 3rd_party/stb
                                              PUBLIC src)
    target_link_libraries(${examplename} GPUJobSystem)
endforeach(examplename ${Examples})

