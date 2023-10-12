cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(eeenc
  LANGUAGES CXX
)

add_executable(eeenc 
  "src/main.cpp"
  "src/chunks.hpp"
)
target_include_directories(eeenc PUBLIC inc)
target_compile_features(eeenc PUBLIC cxx_std_20)

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(AMD64|x86_64)$")
  # AVX2 implementation (if you're running an 11+ year old PC at this point, I won't bother)
  target_sources(eeenc PUBLIC "src/chunks_avx2.cpp")
  if (${CMAKE_CXX_COMPILER_ID} MATCHES "^(GNU|(Apple)?Clang)$")
    # MSVC auto-enables AVX2, so we only do this on GNU-compatible compilers
    set_source_files_properties("src/chunks_avx2.cpp" PROPERTIES COMPILE_FLAGS "-mavx2 -mbmi2")
  endif()
else()
  message(FATAL_ERROR "Unsupported architecture!!")
endif()