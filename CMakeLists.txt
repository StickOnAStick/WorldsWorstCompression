cmake_minimum_required(VERSION 4.2.3)
project(AutoCompressor C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Torch REQUIRED)
find_package(CLI11 REQUIRED)

# LIBRARIES
add_library(autoencoder_lib
    src/models/autoencoder/autoencoder.cpp
)
target_include_directories(autoencoder_lib PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(autoencoder_lib PUBLIC ${TORCH_LIBRARIES})

# EXECUTABLES
add_executable(
    ${PROJECT_NAME} 
    src/main.cpp 
)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

add_executable(
    ${PROJECT_NAME}_train 
    src/train.cpp
)
target_compile_features(${PROJECT_NAME}_train PRIVATE cxx_std_17)

add_executable(
    ${PROJECT_NAME}_infer 
    src/infer.cpp
)
target_compile_features(${PROJECT_NAME}_infer PRIVATE cxx_std_17)

# CLI
target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE 
    autoencoder_lib
    CLI11::CLI11
)

# TRAIN 
target_link_libraries(
    ${PROJECT_NAME}_train
    PRIVATE 
    autoencoder_lib
    CLI11::CLI11
)

# INFER
target_link_libraries(
    ${PROJECT_NAME}_infer
    PRIVATE 
    autoencoder_lib
    CLI11::CLI11
)