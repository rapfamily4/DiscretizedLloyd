cmake_minimum_required(VERSION 3.16)
project(voronoi)

# Functions
function(copy_folder target source dest)
    add_custom_command(
        TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${source} ${dest}
    )
endfunction()

# C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch libraries
include(FetchContent)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG v2.5.0
)
FetchContent_Declare(
	glm
	GIT_REPOSITORY	https://github.com/g-truc/glm.git
	GIT_TAG 	bf71a834948186f4097caa076cd2663c69a10e1e #refs/tags/1.0.1
)
FetchContent_MakeAvailable(libigl glm)

# Enable libigl modules
igl_include(opengl)
igl_include(glfw)
igl_include(imgui)
igl_include(embree)

# Options
# option (LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui" ON)

# Source files
file(GLOB SRC_FILES ./src/*.cpp)
add_executable(${PROJECT_NAME} ${SRC_FILES})

# Include directories
target_include_directories(${PROJECT_NAME} BEFORE PUBLIC ./src/)

# Link igl (and modules) to your project
target_link_libraries(${PROJECT_NAME} PUBLIC 
    glm::glm
    igl::opengl
    igl::glfw
    igl::imgui
    igl::embree
)

# Copying models into build
copy_folder(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/models ${CMAKE_BINARY_DIR}/models)