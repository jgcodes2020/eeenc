#include <cstddef>

#if defined(__GNUC__)
#define force_inline inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define force_inline inline __forceinline
#else
#define force_inline inline
#endif
namespace eeenc {
  inline constexpr size_t chunk_len = 65536;
  
  alignas(32) inline char b_data[chunk_len / 8] = {};
  alignas(32) inline char e_data[chunk_len] = {};
  
  size_t encode_chunk(size_t len);
  size_t decode_chunk(size_t len);
}