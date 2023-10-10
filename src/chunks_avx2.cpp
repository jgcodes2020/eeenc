#include "chunks.hpp"

#include <immintrin.h>
#include <cstdint>

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

  char* end = input_data + len;
  char* ip  = input_data;
  char* op  = output_data;

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
  // final batch, 1-3 remaining
  if (ip < end) {
    __m256i chunk;
    // load 4 bytes, invert
    chunk = _mm256_castsi128_si256(_mm_loadu_si32(ip));
    chunk = _mm256_xor_si256(chunk, all_ones);
    // process chunk
    AVX2_EXPAND_CHUNKS(chunk);
    // write output bytes
    _mm256_store_si256((__m256i*) (op), chunk);
    // move output pointer
    op += (end - ip) * 8;
  }

  return op - output_data;
}

#undef AVX2_EXPAND_CHUNKS