#include "chunks.hpp"

#include <emmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <x86intrin.h>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <stdexcept>

template <std::unsigned_integral T>
static inline void prn(__m256i value) {
  auto flags = std::cout.flags();
  
  constexpr size_t len = 32 / sizeof(T);
  T array[len];
  _mm256_storeu_si256((__m256i*) array, value);
  std::cout << "[";
  for (int i = 0; i < len; i++) {
    if (i != 0)
      std::cout << ", ";
    std::cout << (uint64_t) array[i];
  }
  std::cout << "]\n";
}

template <std::unsigned_integral T>
static inline void prn(__m128i value) {
  auto flags = std::cout.flags();
  
  constexpr size_t len = 16 / sizeof(T);
  T array[len];
  _mm_storeu_si128((__m128i_u*) array, value);
  std::cout << "[";
  for (int i = 0; i < len; i++) {
    if (i != 0)
      std::cout << ", ";
    std::cout << (uint64_t) array[i];
  }
  std::cout << "]\n";
}

size_t eeenc::encode_chunk(size_t len) {
  const __m256i all_ones = _mm256_set1_epi8(0xFF);

  const __m256i pshufb_reg = _mm256_setr_epi16(
    0x8000, 0x8000, 0x8000, 0x8000, 0x8001, 0x8001, 0x8001, 0x8001, 0x8002,
    0x8002, 0x8002, 0x8002, 0x8003, 0x8003, 0x8003, 0x8003
  );
  const __m256i mask1_reg = _mm256_setr_epi16(
    0x00C0, 0x0030, 0x000C, 0x0003, 0x00C0, 0x0030, 0x000C, 0x0003, 0x00C0,
    0x0030, 0x000C, 0x0003, 0x00C0, 0x0030, 0x000C, 0x0003
  );
  const __m256i mul_reg = _mm256_setr_epi16(
    0x0201, 0x0804, 0x2010, 0x8040, 0x0201, 0x0804, 0x2010, 0x8040, 0x0201,
    0x0804, 0x2010, 0x8040, 0x0201, 0x0804, 0x2010, 0x8040
  );
  const __m256i mask2_reg = _mm256_set1_epi8(0x80);
  const __m256i econv_reg = _mm256_set1_epi8(0x45);

// Takes a uint in the low word and generates its representation in E-nary.
// Latency: 13 cycles.
#define AVX2_EXPAND_CHUNKS(chunk)                                   \
  do {                                                              \
    /* Broadcast each byte to 4 corresponding ushorts */            \
    chunk = _mm256_broadcastd_epi32(_mm256_castsi256_si128(chunk)); \
    chunk = _mm256_shuffle_epi8(chunk, pshufb_reg);                 \
    /* Mask off bit pairs from MSB to LSB for each byte */          \
    chunk = _mm256_and_si256(chunk, mask1_reg);                     \
    /* Shift and distribute bits to MSBs of each byte */            \
    chunk = _mm256_mullo_epi16(chunk, mul_reg);                     \
    chunk = _mm256_and_si256(chunk, mask2_reg);                     \
    /* Convert MSBs to Es */                                        \
    chunk = _mm256_srli_epi32(chunk, 2);                            \
    chunk = _mm256_or_si256(chunk, econv_reg);                      \
  } while (false)

  char* end = b_data + len;
  char* ip  = b_data;
  char* op  = e_data;

  // unrolled, 16/loop
  for (; (ip + 16) <= end; ip += 16, op += 128) {
    __m256i chunk1, chunk2, chunk3, chunk4;
    // load 16 bytes, invert and broadcast to all chunks
    chunk4 = _mm256_castsi128_si256(_mm_load_si128((__m128i*) ip));
    chunk4 = _mm256_xor_si256(chunk4, all_ones);
    chunk1 = chunk2 = chunk3 = chunk4;
    // each chunk can handle 4 bytes
    chunk2 = _mm256_bsrli_epi128(chunk2, 4);
    chunk3 = _mm256_bsrli_epi128(chunk3, 8);
    chunk4 = _mm256_bsrli_epi128(chunk4, 12);
    // process chunks. Using macros encourages instruction reordering.
    AVX2_EXPAND_CHUNKS(chunk1);
    AVX2_EXPAND_CHUNKS(chunk2);
    AVX2_EXPAND_CHUNKS(chunk3);
    AVX2_EXPAND_CHUNKS(chunk4);
    // write all output bytes
    _mm256_store_si256((__m256i*) (op), chunk1);
    _mm256_store_si256((__m256i*) (op + 32), chunk2);
    _mm256_store_si256((__m256i*) (op + 64), chunk3);
    _mm256_store_si256((__m256i*) (op + 96), chunk4);
  }
  // single batch at a time, 4/loop
  for (; (ip + 4) <= end; ip += 4, op += 32) {
    __m256i chunk;
    // load 4 bytes, invert
    chunk = _mm256_castsi128_si256(_mm_loadu_si32(ip));
    chunk = _mm256_xor_si256(chunk, all_ones);
    // process chunk
    AVX2_EXPAND_CHUNKS(chunk);
    // write output bytes
    _mm256_store_si256((__m256i*) (op), chunk);
  }
  // use SWAR for the final few bytes.
  for (; ip < end; ip += 1, op += 8) {
    uint64_t val = (uint64_t) (uint8_t) *ip;
    val = _bswap64(_pdep_u64(val, UINT64_C(0x2020'2020'2020'2020)));
    val = val ^ UINT64_C(0x6565'6565'6565'6565);
    *((uint64_t*) op) = val;
  }

  return op - e_data;
#undef AVX2_EXPAND_CHUNKS
}

inline __m256i dec_chunk(__m256i chunk) {
  const __m256i cmask_reg = _mm256_set1_epi8(0xDF);
  const __m256i check_reg = _mm256_set1_epi8(0x45);
  const __m256i madd_reg = _mm256_setr_epi8(
    128, 64, 32, 16, 8, 4, 2, 1, 128, 64, 32, 16, 8, 4, 2, 1, 
    128, 64, 32, 16, 8, 4, 2, 1, 128, 64, 32, 16, 8, 4, 2, 1
  );
  const __m256i perm_reg = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
  
  __m256i cmp_chunk = _mm256_and_si256(chunk, cmask_reg);
  cmp_chunk = _mm256_cmpeq_epi8(cmp_chunk, check_reg);
  if (!_mm256_testc_si256(cmp_chunk, _mm256_set1_epi8(0xFF)))
    throw std::invalid_argument("Invalid e-nary (not e or E)");
  
  chunk = _mm256_andnot_si256(cmask_reg, chunk);
  chunk = _mm256_srli_epi32(chunk, 5);
  
  chunk = _mm256_maddubs_epi16(madd_reg, chunk);
  chunk = _mm256_sad_epu8(chunk, _mm256_setzero_si256());
  
  chunk = _mm256_permutevar8x32_epi32(chunk, perm_reg);

  return chunk;
}

size_t eeenc::decode_chunk(size_t len) {
  const __m128i shuf1_reg = _mm_setr_epi8(
    0x00, 0x04, 0x08, 0x0C, 0x80, 0x80, 0x80, 0x80, 
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
  );
  const __m128i shuf2_reg = _mm_setr_epi8(
    0x80, 0x80, 0x80, 0x80, 0x00, 0x04, 0x08, 0x0C, 
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
  );
  const __m128i shuf3_reg = _mm_setr_epi8(
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
    0x00, 0x04, 0x08, 0x0C, 0x80, 0x80, 0x80, 0x80
  );
  const __m128i shuf4_reg = _mm_setr_epi8(
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
    0x80, 0x80, 0x80, 0x80, 0x00, 0x04, 0x08, 0x0C
  );

#define AVX2_COMPRESS_CHUNKS(chunk) chunk = dec_chunk(chunk)

  if (len % 8 != 0)
    throw std::invalid_argument("Invalid e-nary");

  char* end = e_data + len;
  char* ip  = e_data;
  char* op  = b_data;

  for (; (ip + 128) <= end; ip += 128, op += 16) {
    __m256i chunk1, chunk2, chunk3, chunk4;
    __m128i chunk1s, chunk2s, chunk3s, chunk4s;
    chunk1 = _mm256_load_si256((__m256i*) (ip));
    chunk2 = _mm256_load_si256((__m256i*) (ip + 32));
    chunk3 = _mm256_load_si256((__m256i*) (ip + 64));
    chunk4 = _mm256_load_si256((__m256i*) (ip + 96));

    AVX2_COMPRESS_CHUNKS(chunk1);
    AVX2_COMPRESS_CHUNKS(chunk2);
    AVX2_COMPRESS_CHUNKS(chunk3);
    AVX2_COMPRESS_CHUNKS(chunk4);

    chunk1s = _mm256_castsi256_si128(chunk1);
    chunk2s = _mm256_castsi256_si128(chunk2);
    chunk3s = _mm256_castsi256_si128(chunk3);
    chunk4s = _mm256_castsi256_si128(chunk4);

    chunk1s = _mm_shuffle_epi8(chunk1s, shuf1_reg);
    chunk2s = _mm_shuffle_epi8(chunk2s, shuf2_reg);
    chunk3s = _mm_shuffle_epi8(chunk3s, shuf3_reg);
    chunk4s = _mm_shuffle_epi8(chunk4s, shuf4_reg);

    chunk1s = _mm_or_si128(chunk1s, chunk2s);
    chunk3s = _mm_or_si128(chunk3s, chunk4s);
    chunk1s = _mm_or_si128(chunk1s, chunk3s);
    
    chunk1s = _mm_xor_si128(chunk1s, _mm_set1_epi8(0xFF));
    _mm_store_si128((__m128i*) op, chunk1s);
  }
  for (; (ip + 32) <= end; ip += 32, op += 4) {
    __m256i chunk1;
    __m128i chunk1s;
    
    chunk1 = _mm256_loadu_si256((__m256i*) (ip));
    
    AVX2_COMPRESS_CHUNKS(chunk1);
    chunk1s = _mm256_castsi256_si128(chunk1);
    chunk1s = _mm_shuffle_epi8(chunk1s, shuf1_reg);

    chunk1s = _mm_xor_si128(chunk1s, _mm_set1_epi8(0xFF));
    _mm_storeu_si32(op, chunk1s);
  }
  for (; (ip + 8) <= end; ip += 8, op += 1) {
    uint64_t val = *((uint64_t*) ip);
    if ((val & UINT64_C(0xDFDF'DFDF'DFDF'DFDF)) != UINT64_C(0x4545'4545'4545'4545)) {
      throw std::invalid_argument("Invalid e-nary");
    }
    val = _pext_u64(_bswap64(val), UINT64_C(0x2020'2020'2020'2020));
    val ^= UINT64_C(0x0000'0000'0000'00FF);
    *op = (uint8_t) val;
  }
  
  return op - b_data;
}