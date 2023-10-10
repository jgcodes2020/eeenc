#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include "chunks.hpp"

#include "mtap/mtap.hpp"

static inline void simd_encode() {
  while (!feof(stdin)) {
    size_t read = fread(eeenc::input_data, 1, sizeof(eeenc::input_data), stdin);
    size_t to_write = eeenc::encode_chunk(read);
    fwrite(eeenc::output_data, 1, to_write, stdout);
  }
}

int main(int argc, const char* argv[]) {
  bool decode;
  mtap::parser {
    mtap::option<"-d", 0>([&]() {
      decode = true;
    })
  }.parse(argc, argv);
  
  simd_encode();
}