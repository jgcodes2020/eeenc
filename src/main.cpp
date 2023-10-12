#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include "chunks.hpp"

#include "mtap/mtap.hpp"

static inline void simd_encode() {
  while (!feof(stdin)) {
    size_t read = fread(eeenc::b_data, 1, sizeof(eeenc::b_data), stdin);
    size_t to_write = eeenc::encode_chunk(read);
    fwrite(eeenc::e_data, 1, to_write, stdout);
  }
}
static inline void simd_decode() {
  while (!feof(stdin)) {
    size_t read = fread(eeenc::e_data, 1, sizeof(eeenc::e_data), stdin);
    size_t to_write = eeenc::decode_chunk(read);
    fwrite(eeenc::b_data, 1, to_write, stdout);
  }
}

int main(int argc, const char* argv[]) {
  bool decode = false;
  mtap::parser {
    mtap::option<"-d", 0>([&]() {
      decode = true;
    })
  }.parse(argc, argv);
  
  if (decode)
    simd_decode();
  else
    simd_encode();
}